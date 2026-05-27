from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from joblib import dump

from .config import MODEL_REGISTRY_DIR, BEST_MODEL_DIR, META_PATH, DEFAULT_TRANSFORMER_MODELS, EPOCHS_DL, EPOCHS_TRANSFORMER
from .data import load_all_data, train_val_test_split
from .features import build_tfidf_and_labels
from .models.ml import train_logreg, train_svm, save_ml_pipeline
from .models.dl import (build_vocab, TextDataset, LSTMClassifier, TextCNN, train_model, evaluate,
                         EnhancedLSTMClassifier, BERTClassifier, BERTDataset, train_bert_model, evaluate_bert)
from .utils import set_seed
from .model_comparison import (
    compute_detailed_metrics,
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    save_all_model_metrics,
    create_comparison_plots,
    generate_comparison_report
)


def select_best(run_results):
    best = None
    for r in run_results:
        f1 = r["metrics"].get("f1_macro")
        if best is None or f1 > best["metrics"].get("f1_macro"):
            best = r
    return best


def persist_best(best: Dict[str, Any], le, vectorizer=None, vocab=None):
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"winner": best["name"], "type": best["type"], "metrics": best["metrics"]}
    if best["type"] == "ml":
        save_ml_pipeline(BEST_MODEL_DIR / "ml_pipeline.joblib", vectorizer, le, best["model"]) 
        meta.update({"path": "ml_pipeline.joblib", "label_classes": list(le.classes_)})
    elif best["type"] == "dl":
        from torch import save
        save(best["model"].state_dict(), BEST_MODEL_DIR / "dl_model.pt")
        meta.update({"path": "dl_model.pt", "label_classes": list(le.classes_), "vocab": vocab})
    elif best["type"] == "transformer":
        model = best["hf_model"]
        tok = best["tokenizer"]
        model.save_pretrained(BEST_MODEL_DIR / "hf_model")
        tok.save_pretrained(BEST_MODEL_DIR / "hf_model")
        meta.update({"path": "hf_model", "label2id": best["label2id"], "id2label": best["id2label"]})
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_training(sample_frac: float = 1.0):
    set_seed()
    df = load_all_data(sample_frac=sample_frac)
    df_train, df_val, df_test = train_val_test_split(df)

    # Classical ML
    vect, le, X_train, y_train, X_val, y_val, X_test, y_test = build_tfidf_and_labels(df_train, df_val, df_test)
    res_lr = train_logreg(X_train, y_train, X_val, y_val)
    res_svm = train_svm(X_train, y_train, X_val, y_val)

    results = [
        {"name": res_lr.name, "type": "ml", "model": res_lr.model, "metrics": res_lr.metrics},
        {"name": res_svm.name, "type": "ml", "model": res_svm.model, "metrics": res_svm.metrics},
    ]

    # DL models
    try:
        from torch.utils.data import DataLoader
        import torch
        vocab = build_vocab(df_train["text_clean"].tolist())
        n_classes = len(le.classes_)
        train_ds = TextDataset(df_train["text_clean"].tolist(), le.transform(df_train["category"]).tolist(), vocab)
        val_ds = TextDataset(df_val["text_clean"].tolist(), le.transform(df_val["category"]).tolist(), vocab)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Original LSTM
        lstm = LSTMClassifier(vocab_size=len(vocab), num_classes=n_classes)
        train_model(lstm, train_loader, val_loader, epochs=EPOCHS_DL, device=device)
        f1, rep = evaluate(lstm, val_loader, device=device)
        results.append({"name": "lstm", "type": "dl", "model": lstm, "metrics": rep, "vocab": vocab})

        # Enhanced LSTM with attention
        enhanced_lstm = EnhancedLSTMClassifier(vocab_size=len(vocab), num_classes=n_classes)
        train_model(enhanced_lstm, train_loader, val_loader, epochs=EPOCHS_DL, device=device)
        f1, rep = evaluate(enhanced_lstm, val_loader, device=device)
        results.append({"name": "enhanced_lstm", "type": "dl", "model": enhanced_lstm, "metrics": rep, "vocab": vocab})

        # CNN
        cnn = TextCNN(vocab_size=len(vocab), num_classes=n_classes)
        train_model(cnn, train_loader, val_loader, epochs=EPOCHS_DL, device=device)
        f1, rep = evaluate(cnn, val_loader, device=device)
        results.append({"name": "cnn", "type": "dl", "model": cnn, "metrics": rep, "vocab": vocab})
        
        # Standalone BERT model
        try:
            from transformers import AutoTokenizer
            bert_model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            
            bert_train_ds = BERTDataset(df_train["text_clean"].tolist(), 
                                       le.transform(df_train["category"]).tolist(), 
                                       tokenizer)
            bert_val_ds = BERTDataset(df_val["text_clean"].tolist(), 
                                     le.transform(df_val["category"]).tolist(), 
                                     tokenizer)
            bert_train_loader = DataLoader(bert_train_ds, batch_size=16, shuffle=True)
            bert_val_loader = DataLoader(bert_val_ds, batch_size=32)
            
            bert_classifier = BERTClassifier(num_classes=n_classes, model_name=bert_model_name)
            train_bert_model(bert_classifier, bert_train_loader, bert_val_loader, 
                           epochs=EPOCHS_DL, device=device)
            f1, rep = evaluate_bert(bert_classifier, bert_val_loader, device=device)
            results.append({"name": "bert_custom", "type": "dl", "model": bert_classifier, 
                          "metrics": rep, "vocab": None, "tokenizer": tokenizer})
        except Exception as e:
            print(f"Standalone BERT training failed: {e}")
            
    except Exception as e:
        print(f"DL models training failed: {e}")

    # Transformers
    try:
        from .models.transformers import train_transformer
        label2id = {l: i for i, l in enumerate(le.classes_)}
        for key, mname in DEFAULT_TRANSFORMER_MODELS.items():
            hfres = train_transformer(df_train, df_val, mname, label2id, epochs=EPOCHS_TRANSFORMER)
            results.append({
                "name": key,
                "type": "transformer",
                "metrics": {"f1_macro": hfres.metrics.get("f1_macro")},
                "label2id": hfres.label2id,
                "id2label": hfres.id2label,
                "hf_model": hfres.model,
                "tokenizer": hfres.tokenizer
            })
    except Exception:
        pass

    # Generate comprehensive model comparison
    print("\n" + "="*80)
    print("Generating comprehensive model comparison metrics...")
    print("="*80)
    
    all_detailed_metrics = []
    label_names = list(le.classes_)
    
    # Collect predictions on test set for all models
    for result in results:
        try:
            model_name = result["name"]
            model_type = result["type"]
            print(f"\nEvaluating {model_name} on test set...")
            
            if model_type == "ml":
                # ML models
                y_pred = result["model"].predict(X_test)
                y_true = y_test
                
            elif model_type == "dl":
                # DL models (LSTM, CNN, Enhanced LSTM, BERT)
                import torch
                from torch.utils.data import DataLoader
                
                model = result["model"]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                model.eval()
                
                # Check if it's BERT or regular DL model
                if "bert" in model_name.lower() and result.get("tokenizer"):
                    # BERT model
                    tokenizer = result["tokenizer"]
                    test_ds = BERTDataset(df_test["text_clean"].tolist(),
                                         le.transform(df_test["category"]).tolist(),
                                         tokenizer)
                    test_loader = DataLoader(test_ds, batch_size=32)
                    
                    y_pred_list = []
                    with torch.no_grad():
                        for batch in test_loader:
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            logits = model(input_ids, attention_mask)
                            pred = logits.argmax(dim=1)
                            y_pred_list.extend(pred.cpu().tolist())
                    y_pred = np.array(y_pred_list)
                else:
                    # Regular DL models (LSTM, CNN)
                    vocab = result.get("vocab")
                    test_ds = TextDataset(df_test["text_clean"].tolist(),
                                         le.transform(df_test["category"]).tolist(),
                                         vocab)
                    test_loader = DataLoader(test_ds, batch_size=64)
                    
                    y_pred_list = []
                    with torch.no_grad():
                        for xb, yb in test_loader:
                            xb = xb.to(device)
                            logits = model(xb)
                            pred = logits.argmax(dim=1)
                            y_pred_list.extend(pred.cpu().tolist())
                    y_pred = np.array(y_pred_list)
                
                y_true = le.transform(df_test["category"])
                
            elif model_type == "transformer":
                # HuggingFace transformers
                from datasets import Dataset
                model_hf = result["hf_model"]
                tokenizer = result["tokenizer"]
                
                test_texts = df_test["text_clean"].tolist()
                test_labels = le.transform(df_test["category"]).tolist()
                
                # Tokenize
                test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
                
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_hf.to(device)
                model_hf.eval()
                
                y_pred_list = []
                batch_size = 32
                for i in range(0, len(test_texts), batch_size):
                    batch_encodings = {
                        k: torch.tensor(v[i:i+batch_size]).to(device)
                        for k, v in test_encodings.items()
                    }
                    with torch.no_grad():
                        outputs = model_hf(**batch_encodings)
                        pred = outputs.logits.argmax(dim=1)
                        y_pred_list.extend(pred.cpu().tolist())
                
                y_pred = np.array(y_pred_list)
                y_true = np.array(test_labels)
            
            else:
                continue
            
            # Compute detailed metrics
            detailed_metrics = compute_detailed_metrics(y_true, y_pred, model_name, label_names)
            all_detailed_metrics.append(detailed_metrics)
            
            # Plot confusion matrices
            cm = np.array(detailed_metrics["confusion_matrix"])
            plot_confusion_matrix(cm, label_names, model_name)
            plot_normalized_confusion_matrix(cm, label_names, model_name)
            
            print(f"✓ {model_name}: F1={detailed_metrics['f1_macro']:.4f}, Accuracy={detailed_metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"✗ Failed to evaluate {result['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save all metrics
    if all_detailed_metrics:
        save_all_model_metrics(all_detailed_metrics)
        create_comparison_plots(all_detailed_metrics)
        report = generate_comparison_report(all_detailed_metrics)
        print("\n" + report)
        print("\n✅ Model comparison complete! Check the comparisons directory for detailed results.")
    
    # pick best
    best = select_best(results)
    # persist best
    if best["type"] == "ml":
        persist_best(best, le, vectorizer=vect)
    elif best["type"] == "dl":
        persist_best(best, le, vocab=best.get("vocab"))
    elif best["type"] == "transformer":
        persist_best(best, le)

    return best
