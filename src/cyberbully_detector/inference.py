from __future__ import annotations
import json
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

from .config import BEST_MODEL_DIR, META_PATH
from .utils import severity_from_category_and_conf
from .preprocessing import preprocess


class Predictor:
    def __init__(self):
        with open(META_PATH, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.type = self.meta.get('type')
        if self.type == 'ml':
            from joblib import load
            obj = load(BEST_MODEL_DIR / self.meta['path'])
            self.vectorizer = obj['vectorizer']
            self.le = obj['label_encoder']
            self.model = obj['model']
        elif self.type == 'dl':
            import torch
            from .models.dl import LSTMClassifier, TextCNN
            self.le_classes = self.meta.get('label_classes')
            self.label2id = {l: i for i, l in enumerate(self.le_classes)}
            self.vocab = self.meta.get('vocab')
            winner_type = self.meta.get('winner', 'lstm')  # Default to lstm if not specified
            
            if winner_type == 'cnn':
                self.model = TextCNN(vocab_size=len(self.vocab), num_classes=len(self.le_classes))
            else:  # lstm or other
                self.model = LSTMClassifier(vocab_size=len(self.vocab), num_classes=len(self.le_classes))
                
            self.model.load_state_dict(torch.load(BEST_MODEL_DIR / self.meta['path'], map_location='cpu'))
            self.model.eval()
        elif self.type == 'transformer':
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.hf_dir = BEST_MODEL_DIR / self.meta['path']
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.hf_dir)
            self.id2label = {int(k): v for k, v in self.meta.get('id2label', {}).items()}
        else:
            raise RuntimeError('Unknown model type')

    def predict(self, text: str) -> Dict[str, Any]:
        clean = preprocess(text)
        if self.type == 'ml':
            X = self.vectorizer.transform([clean])
            probs = getattr(self.model, 'predict_proba', None)
            if probs is None:
                dec = self.model.decision_function(X)
                if dec.ndim == 1:
                    dec = np.vstack([-dec, dec]).T
                e = np.exp(dec - dec.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
            else:
                p = probs(X)
            idx = int(np.argmax(p[0]))
            label = self.le.inverse_transform([idx])[0]
            conf = float(p[0][idx])
        elif self.type == 'dl':
            import torch
            from .models.dl import encode
            x = encode(clean, self.vocab)[:128]
            x = x + [0] * (128 - len(x))
            xb = torch.tensor([x], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(xb)
                p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(p))
            label = self.le_classes[idx]
            conf = float(p[idx])
        else:  # transformer
            import torch
            inputs = self.tokenizer(clean, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(p))
            label = self.id2label.get(idx, str(idx))
            conf = float(p[idx])
        severity = severity_from_category_and_conf(label, conf)
        category = label
        confidence = conf
        return {
            "prediction": "bullying" if label != "not_cyberbullying" else "not",
            "category": category,
            "severity": severity,
            "confidence": confidence,
            "probs": p.tolist()
        }

    def explain_lime(self, text: str, num_features: int = 8) -> List[Dict[str, Any]]:
        try:
            from lime.lime_text import LimeTextExplainer
        except Exception:
            return []
        clean = preprocess(text)
        class_names = []
        if self.type == 'ml':
            class_names = list(self.le.classes_)
            def predict_proba(texts):
                X = self.vectorizer.transform([preprocess(t) for t in texts])
                probs = getattr(self.model, 'predict_proba', None)
                if probs is None:
                    dec = self.model.decision_function(X)
                    if dec.ndim == 1:
                        dec = np.vstack([-dec, dec]).T
                    e = np.exp(dec - dec.max(axis=1, keepdims=True))
                    p = e / e.sum(axis=1, keepdims=True)
                    return p
                return probs(X)
        else:
            if self.type == 'transformer':
                class_names = [self.id2label[i] for i in range(self.model.num_labels)]
                def predict_proba(texts):
                    import torch
                    inputs = self.tokenizer([preprocess(t) for t in texts], return_tensors='pt', truncation=True, padding=True, max_length=128)
                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                        p = torch.softmax(logits, dim=1).cpu().numpy()
                    return p
            else:  # dl
                class_names = self.le_classes
                def predict_proba(texts):
                    import torch
                    from .models.dl import encode
                    xs = []
                    for t in texts:
                        c = preprocess(t)
                        arr = encode(c, self.vocab)[:128]
                        arr = arr + [0] * (128 - len(arr))
                        xs.append(arr)
                    xb = torch.tensor(xs, dtype=torch.long)
                    with torch.no_grad():
                        p = torch.softmax(self.model(xb), dim=1).cpu().numpy()
                    return p
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(clean, predict_proba, num_features=num_features)
        return [{"token": t, "weight": float(w)} for t, w in exp.as_list()]
