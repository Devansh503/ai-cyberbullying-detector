from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter


def build_vocab(texts: List[str], min_freq: int = 2, max_size: int = 30000) -> Dict[str, int]:
    counter = Counter()
    for s in texts:
        for t in s.split():
            counter[t] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(t, 1) for t in text.split()]


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        x = encode(self.texts[idx], self.vocab)[: self.max_len]
        x = x + [0] * (self.max_len - len(x))
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class BERTDataset(Dataset):
    """Dataset for BERT-based models with tokenization"""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int = 128, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)
    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        h = out[:, -1, :]
        return self.fc(h)


class EnhancedLSTMClassifier(nn.Module):
    """Enhanced LSTM with attention mechanism and dropout for better regularization"""
    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int = 128, hidden: int = 128, 
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=n_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden * 2, 1)
        
        # Classification layers with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden * 2, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)
        
    def attention_net(self, lstm_output):
        """Apply attention mechanism to LSTM outputs"""
        # lstm_output: (batch_size, seq_len, hidden*2)
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)  # (batch, seq_len, 1)
        attn_applied = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden*2)
        return attn_applied, attn_weights
        
    def forward(self, x):
        e = self.emb(x)
        e = self.emb_dropout(e)
        lstm_out, _ = self.lstm(e)  # (batch, seq_len, hidden*2)
        
        # Apply attention
        attn_output, _ = self.attention_net(lstm_out)
        
        # Classification
        h = self.dropout(attn_output)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        return self.fc2(h)


class BERTClassifier(nn.Module):
    """Standalone BERT classifier using transformers library with custom head"""
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", 
                 dropout: float = 0.3, hidden_dim: int = 256):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Custom classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.bert_hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        
        # Classification head
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int = 128, filters: int = 100, kernels: List[int] = [3,4,5]):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, k) for k in kernels])
        self.fc = nn.Linear(filters * len(kernels), num_classes)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)  # B x E x T
        c = [torch.relu(conv(e)) for conv in self.convs]
        p = [torch.max(ci, dim=2)[0] for ci in c]
        h = torch.cat(p, dim=1)
        return self.fc(h)


@dataclass
class DLResult:
    name: str
    model: nn.Module
    vocab: Dict[str, int]
    metrics: Dict[str, Any]


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Tuple[float, Dict[str, Any]]:
    from sklearn.metrics import f1_score, classification_report
    model.eval()
    ys, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            ys.extend(yb.cpu().tolist())
            yp.extend(pred.cpu().tolist())
    f1 = f1_score(ys, yp, average='macro')
    rep = classification_report(ys, yp, output_dict=True, zero_division=0)
    return f1, {"f1_macro": f1, "report": rep}


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 3, lr: float = 1e-3, device: str = "cpu") -> Dict[str, Any]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    best = {"f1": -math.inf, "state": None}
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        f1, _ = evaluate(model, val_loader, device)
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.cpu() for k, v in model.state_dict().items()}}
    if best["state"]:
        model.load_state_dict(best["state"])
    return {"best_f1": best["f1"]}


def evaluate_bert(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Tuple[float, Dict[str, Any]]:
    """Evaluate BERT model with proper input handling"""
    from sklearn.metrics import f1_score, classification_report
    model.eval()
    ys, yp = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            pred = logits.argmax(dim=1)
            
            ys.extend(labels.cpu().tolist())
            yp.extend(pred.cpu().tolist())
    f1 = f1_score(ys, yp, average='macro')
    rep = classification_report(ys, yp, output_dict=True, zero_division=0)
    return f1, {"f1_macro": f1, "report": rep}


def train_bert_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 3, lr: float = 2e-5, device: str = "cpu") -> Dict[str, Any]:
    """Train BERT model with proper optimization settings"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    best = {"f1": -math.inf, "state": None}
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            opt.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            
        f1, _ = evaluate_bert(model, val_loader, device)
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.cpu() for k, v in model.state_dict().items()}}
    
    if best["state"]:
        model.load_state_dict(best["state"])
    return {"best_f1": best["f1"]}
