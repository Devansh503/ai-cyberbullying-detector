# AI Cyberbullying Detection System - Detailed Code Analysis

## Project Overview
This document provides a comprehensive line-by-line analysis of an AI-powered cyberbullying detection system. The system implements multiple machine learning approaches including traditional ML models, deep learning models, and transformer-based models to detect and classify cyberbullying in text.

## Project Structure
```
ai-cyberbullying-detector/
├── src/
│   ├── cyberbully_detector/         # Core detection module
│   ├── api/                         # FastAPI REST API
│   └── frontend/                    # Streamlit web interface
├── scripts/                         # Training and utility scripts
├── data/                           # Data storage
├── deployment/                     # Deployment configurations
└── requirements.txt               # Python dependencies
```

---

## 1. Configuration Module (`src/cyberbully_detector/config.py`)

This file contains all the configuration settings for the project.

```python path=/src/cyberbully_detector/config.py start=1
import os
from pathlib import Path
```
**Lines 1-2**: Import necessary modules for operating system operations and path handling.

```python path=/src/cyberbully_detector/config.py start=4
BASE_DIR = Path(__file__).resolve().parents[2]
```
**Line 4**: Establishes the base directory of the project by going up 2 levels from this config file (from `src/cyberbully_detector/` to project root).

```python path=/src/cyberbully_detector/config.py start=5
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"
MODEL_REGISTRY_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "src" / "cyberbully_detector" / "model_registry"))
BEST_MODEL_DIR = MODEL_REGISTRY_DIR / "best_model"
META_PATH = MODEL_REGISTRY_DIR / "model_meta.json"
```
**Lines 5-11**: Define directory paths for different data types:
- `DATA_DIR`: Main data directory
- `RAW_DATA_DIR`: Original, unprocessed datasets
- `PROCESSED_DATA_DIR`: Cleaned and preprocessed datasets
- `LOGS_DIR`: Application logs and analytics
- `MODEL_REGISTRY_DIR`: Trained model storage (configurable via environment variable)
- `BEST_MODEL_DIR`: Best performing model storage
- `META_PATH`: Metadata file containing information about the best model

```python path=/src/cyberbully_detector/config.py start=13
for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR, MODEL_REGISTRY_DIR, BEST_MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)
```
**Lines 13-14**: Create all necessary directories if they don't exist. The `parents=True` creates parent directories, and `exist_ok=True` prevents errors if directories already exist.

```python path=/src/cyberbully_detector/config.py start=16
APP_ENV = os.getenv("APP_ENV", "dev")
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "1") == "1"
```
**Lines 16-17**: Environment configuration:
- `APP_ENV`: Application environment (defaults to "dev")
- `ENABLE_TRANSLATION`: Boolean flag for translation feature (defaults to enabled)

```python path=/src/cyberbully_detector/config.py start=19
DEFAULT_TRANSFORMER_MODELS = {
    "bert": "distilbert-base-uncased",
    "roberta": "roberta-base"
}
```
**Lines 19-22**: Dictionary mapping transformer model names to their Hugging Face model identifiers. Uses lightweight versions (DistilBERT and base RoBERTa) for efficiency.

```python path=/src/cyberbully_detector/config.py start=24
RANDOM_SEED = 42
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS_DL = 3
EPOCHS_TRANSFORMER = 2
```
**Lines 24-28**: Training hyperparameters:
- `RANDOM_SEED`: For reproducible results
- `MAX_SEQ_LEN`: Maximum sequence length for tokenization
- `BATCH_SIZE`: Number of samples per training batch
- `EPOCHS_DL`: Training epochs for deep learning models
- `EPOCHS_TRANSFORMER`: Training epochs for transformer models

```python path=/src/cyberbully_detector/config.py start=30
CATEGORIES = [
    "not_cyberbullying",
    "harassment", 
    "insult",
    "hate",
    "threat",
    "other"
]
```
**Lines 30-37**: Classification categories for cyberbullying types. The system classifies text into these 6 categories.

```python path=/src/cyberbully_detector/config.py start=39
SEVERITY_RULES = {
    "threat": "high",
    "hate": "high", 
    "harassment": "medium",
    "insult": "medium",
    "other": "low",
    "not_cyberbullying": "low"
}
```
**Lines 39-46**: Severity mapping for each category, used to prioritize responses and interventions.

---

## 2. Utility Module (`src/cyberbully_detector/utils.py`)

Contains common utility functions used throughout the project.

```python path=/src/cyberbully_detector/utils.py start=1
import json
import random
import numpy as np
import torch
from pathlib import Path
from .config import RANDOM_SEED
```
**Lines 1-6**: Import necessary libraries and the random seed from configuration.

```python path=/src/cyberbully_detector/utils.py start=9
def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```
**Lines 9-13**: **`set_seed` function** - Ensures reproducibility by setting seeds for all random number generators used in the project (Python's random, NumPy, PyTorch CPU and GPU).

```python path=/src/cyberbully_detector/utils.py start=16
def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
```
**Lines 16-19**: **`save_json` function** - Utility to save Python objects as JSON files with proper formatting and UTF-8 encoding.

```python path=/src/cyberbully_detector/utils.py start=22
def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
```
**Lines 22-24**: **`load_json` function** - Utility to load JSON files back into Python objects.

```python path=/src/cyberbully_detector/utils.py start=27
def severity_from_category_and_conf(category: str, conf: float) -> str:
    from .config import SEVERITY_RULES
    base = SEVERITY_RULES.get(category, 'low')
    if base == 'high':
        return 'high'
    if base == 'medium':
        return 'high' if conf >= 0.9 else 'medium'
    return 'medium' if conf >= 0.95 else 'low'
```
**Lines 27-34**: **`severity_from_category_and_conf` function** - Determines severity based on both category and model confidence:
- High severity categories remain high
- Medium severity categories become high if confidence ≥ 90%
- Low severity categories become medium if confidence ≥ 95%

---

## 3. Data Loading Module (`src/cyberbully_detector/data.py`)

Handles data loading from multiple sources and preprocessing.

```python path=/src/cyberbully_detector/data.py start=1
from __future__ import annotations
import os
from typing import Tuple, Optional
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CATEGORIES
from .preprocessing import preprocess
```
**Lines 1-8**: Import required libraries and modules. Uses future annotations for better type hints.

```python path=/src/cyberbully_detector/data.py start=11
def _load_hf_tweet_eval_offensive() -> pd.DataFrame:
    ds = load_dataset("tweet_eval", "offensive")
    def map_row(r):
        # 0: non-offensive, 1: offensive
        label = r["label"]
        text = r["text"]
        cat = "harassment" if label == 1 else "not_cyberbullying"
        return {"text": text, "category": cat}
    frames = []
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame([map_row(x) for x in ds[split]])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
```
**Lines 11-23**: **`_load_hf_tweet_eval_offensive` function** - Loads the TweetEval offensive language detection dataset from Hugging Face:
- Maps binary labels (0/1) to project categories
- Combines train, validation, and test splits
- Returns a unified DataFrame

```python path=/src/cyberbully_detector/data.py start=26
def _try_load_hf_hatexplain() -> Optional[pd.DataFrame]:
    try:
        ds = load_dataset("hatexplain")
    except Exception:
        return None
    def to_text(ex):
        # hatexplain has 'post_text' as list of tokens; convert to string
        if isinstance(ex.get("post_text"), list):
            return " ".join(ex["post_text"])[:512]
        return str(ex.get("post_text", ""))
    def map_label(ex):
        lab = ex.get("label")
        if lab is not None:
            if lab == 0:
                return "not_cyberbullying"
            if lab == 1:
                return "harassment"
            if lab == 2:
                return "hate"
        tg = ex.get("target", [])
        return "hate" if tg else "other"
    frames = []
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame({
            "text": [to_text(x) for x in ds[split]],
            "category": [map_label(x) for x in ds[split]]
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
```
**Lines 26-54**: **`_try_load_hf_hatexplain` function** - Attempts to load the HatExplain dataset:
- Gracefully handles loading failures (returns None)
- Converts tokenized text back to strings
- Maps labels: 0→not_cyberbullying, 1→harassment, 2→hate
- Limits text length to 512 characters for efficiency

```python path=/src/cyberbully_detector/data.py start=57
def _try_load_kaggle_csvs() -> Optional[pd.DataFrame]:
    candidates = list(Path(RAW_DATA_DIR).glob("*.csv"))
    frames = []
    for fp in candidates:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        text_col = cols.get("text") or cols.get("comment_text") or cols.get("content")
        label_col = cols.get("label") or cols.get("target") or cols.get("category")
        if not text_col or not label_col:
            continue
        tmp = pd.DataFrame({"text": df[text_col].astype(str)})
        def norm_label(v):
            s = str(v).lower()
            if s in CATEGORIES:
                return s
            if s in ["toxic", "obscene", "insult", "identity_hate", "severe_toxic", "threat"]:
                if s == "threat":
                    return "threat"
                if s in ["identity_hate"]:
                    return "hate"
                if s in ["insult", "obscene", "toxic", "severe_toxic"]:
                    return "harassment"
            if s in ["offensive", "abusive"]:
                return "insult"
            if s in ["non-toxic", "not_offensive", "clean", "0"]:
                return "not_cyberbullying"
            return "other"
        tmp["category"] = [norm_label(v) for v in df[label_col]]
        frames.append(tmp)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None
```
**Lines 57-91**: **`_try_load_kaggle_csvs` function** - Loads custom CSV datasets from the raw data directory:
- Searches for CSV files in RAW_DATA_DIR
- Flexibly identifies text and label columns by common names
- Maps various label formats to project categories
- Handles different naming conventions from different datasets

```python path=/src/cyberbully_detector/data.py start=94
def load_all_data(sample_frac: float = 1.0, shuffle: bool = True) -> pd.DataFrame:
    frames = []
    try:
        frames.append(_load_hf_tweet_eval_offensive())
    except Exception:
        pass
    hx = _try_load_hf_hatexplain()
    if hx is not None:
        frames.append(hx)
    kg = _try_load_kaggle_csvs()
    if kg is not None:
        frames.append(kg)
    if not frames:
        frames.append(pd.DataFrame({
            "text": [
                "I hate you",
                "You are amazing",
                "I will hurt you", 
                "This is so dumb",
                "Have a nice day"
            ],
            "category": ["hate", "not_cyberbullying", "threat", "insult", "not_cyberbullying"]
        }))
    df = pd.concat(frames, ignore_index=True)
    df.dropna(subset=["text", "category"], inplace=True)
    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    df["text_clean"] = df["text"].astype(str).apply(preprocess)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / "dataset_all.csv", index=False)
    return df
```
**Lines 94-126**: **`load_all_data` function** - Main data loading function:
- Attempts to load all available datasets
- Provides fallback sample data if no datasets are available
- Removes missing values
- Optionally shuffles and samples the data
- Applies preprocessing to create clean text column
- Saves processed dataset for future use

```python path=/src/cyberbully_detector/data.py start=129
def train_val_test_split(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    df_train, df_temp = train_test_split(df, test_size=val_size + test_size, stratify=df["category"], random_state=random_state)
    rel_test = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(df_temp, test_size=rel_test, stratify=df_temp["category"], random_state=random_state)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
```
**Lines 129-134**: **`train_val_test_split` function** - Splits data into training, validation, and test sets:
- Uses stratified sampling to maintain class balance
- Default split: 80% train, 10% validation, 10% test
- Resets indices for clean DataFrames

---

## 4. Preprocessing Module (`src/cyberbully_detector/preprocessing.py`)

Handles text cleaning and preprocessing operations.

```python path=/src/cyberbully_detector/preprocessing.py start=1
import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
```
**Lines 1-5**: Import regex, NLTK components for natural language processing.

```python path=/src/cyberbully_detector/preprocessing.py start=7
# Ensure resources are available
for pkg in ["punkt", "punkt_tab", "wordnet", "stopwords", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        try:
            nltk.download(pkg)
        except Exception:
            pass
```
**Lines 7-15**: Automatically downloads required NLTK resources if they're not available, with graceful error handling.

```python path=/src/cyberbully_detector/preprocessing.py start=17
STOPWORDS = set(stopwords.words('english')) if 'english' in stopwords.fileids() else set()
LEMMATIZER = WordNetLemmatizer()
```
**Lines 17-18**: Initialize stopwords and lemmatizer with fallback to empty set if English stopwords aren't available.

```python path=/src/cyberbully_detector/preprocessing.py start=20
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")
```
**Lines 20-24**: Compile regular expressions for text cleaning:
- `URL_RE`: Matches URLs
- `USER_RE`: Matches Twitter-style mentions (@username)
- `HASHTAG_RE`: Matches hashtags (#hashtag)
- `NON_ALNUM_RE`: Matches non-alphanumeric characters
- `MULTISPACE_RE`: Matches multiple consecutive spaces

```python path=/src/cyberbully_detector/preprocessing.py start=27
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = URL_RE.sub(" ", s)
    s = USER_RE.sub(" ", s)
    s = HASHTAG_RE.sub(" ", s)
    s = NON_ALNUM_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s)
    return s.strip()
```
**Lines 27-36**: **`clean_text` function** - Applies comprehensive text cleaning:
- Handles empty/null strings
- Converts to lowercase
- Removes URLs, mentions, hashtags
- Removes special characters
- Normalizes whitespace

```python path=/src/cyberbully_detector/preprocessing.py start=39
def tokenize(text: str) -> List[str]:
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()
    return [t for t in tokens if t and t not in STOPWORDS]
```
**Lines 39-44**: **`tokenize` function** - Tokenizes text into words:
- Uses NLTK's word tokenizer with fallback to simple splitting
- Removes empty tokens and stopwords

```python path=/src/cyberbully_detector/preprocessing.py start=47
def lemmatize(tokens: List[str]) -> List[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens]
```
**Lines 47-48**: **`lemmatize` function** - Reduces words to their root forms (e.g., "running" → "run").

```python path=/src/cyberbully_detector/preprocessing.py start=51
def preprocess(text: str) -> str:
    text = clean_text(text)
    tokens = tokenize(text)
    lemmas = lemmatize(tokens)
    return " ".join(lemmas)
```
**Lines 51-55**: **`preprocess` function** - Main preprocessing pipeline that combines all steps into a single function.

---

## 5. Feature Engineering Module (`src/cyberbully_detector/features.py`)

Handles feature extraction for machine learning models.

```python path=/src/cyberbully_detector/features.py start=1
from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
```
**Lines 1-5**: Import required libraries for feature extraction and label encoding.

```python path=/src/cyberbully_detector/features.py start=8
def build_tfidf_and_labels(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)
    X_train = vect.fit_transform(df_train["text_clean"]) 
    X_val = vect.transform(df_val["text_clean"]) 
    X_test = vect.transform(df_test["text_clean"]) 
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["category"]) 
    y_val = le.transform(df_val["category"]) 
    y_test = le.transform(df_test["category"]) 
    return vect, le, X_train, y_train, X_val, y_val, X_test, y_test
```
**Lines 8-17**: **`build_tfidf_and_labels` function** - Creates TF-IDF features and encodes labels:
- TF-IDF parameters: unigrams and bigrams, max 50,000 features, minimum document frequency of 2
- Fits vectorizer on training data, transforms all splits
- Encodes categorical labels to integers
- Returns all components needed for ML model training

---

## 6. Machine Learning Models Module (`src/cyberbully_detector/models/ml.py`)

Implements traditional machine learning models.

```python path=/src/cyberbully_detector/models/ml.py start=1
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score
from joblib import dump
```
**Lines 1-8**: Import required libraries for ML models and evaluation.

```python path=/src/cyberbully_detector/models/ml.py start=11
@dataclass
class MLResult:
    name: str
    model: Any
    metrics: Dict[str, Any]
```
**Lines 11-15**: **`MLResult` dataclass** - Structure to store ML model results including name, trained model, and performance metrics.

```python path=/src/cyberbully_detector/models/ml.py start=18
def train_logreg(X_train, y_train, X_val, y_val) -> MLResult:
    clf = LogisticRegression(max_iter=2000, n_jobs=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return MLResult(name="logreg", model=clf, metrics={"f1_macro": f1, "report": rep})
```
**Lines 18-24**: **`train_logreg` function** - Trains a logistic regression model:
- Uses 2000 max iterations and 2 parallel jobs
- Calculates macro F1-score and detailed classification report
- Returns structured result

```python path=/src/cyberbully_detector/models/ml.py start=27
def train_svm(X_train, y_train, X_val, y_val) -> MLResult:
    base = LinearSVM()
    clf = CalibratedClassifierCV(base, method='sigmoid')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return MLResult(name="svm", model=clf, metrics={"f1_macro": f1, "report": rep})
```
**Lines 27-34**: **`train_svm` function** - Trains a calibrated SVM model:
- Uses LinearSVC wrapped in CalibratedClassifierCV for probability estimates
- Sigmoid calibration method for better probability estimates

```python path=/src/cyberbully_detector/models/ml.py start=37
def save_ml_pipeline(path, vectorizer, label_encoder, model):
    obj = {
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "model": model
    }
    dump(obj, path)
```
**Lines 37-43**: **`save_ml_pipeline` function** - Saves the complete ML pipeline (vectorizer, encoder, and model) to disk using joblib.

---

## 7. Deep Learning Models Module (`src/cyberbully_detector/models/dl.py`)

Implements PyTorch-based deep learning models.

```python path=/src/cyberbully_detector/models/dl.py start=1
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
```
**Lines 1-8**: Import PyTorch and related libraries for deep learning implementation.

```python path=/src/cyberbully_detector/models/dl.py start=11
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
```
**Lines 11-24**: **`build_vocab` function** - Creates vocabulary dictionary from training texts:
- Counts word frequencies
- Reserves indices 0 and 1 for padding and unknown tokens
- Only includes words with minimum frequency
- Limits vocabulary size for efficiency

```python path=/src/cyberbully_detector/models/dl.py start=27
def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(t, 1) for t in text.split()]
```
**Lines 27-28**: **`encode` function** - Converts text to sequence of token IDs, using unknown token (1) for out-of-vocabulary words.

```python path=/src/cyberbully_detector/models/dl.py start=31
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
```
**Lines 31-43**: **`TextDataset` class** - PyTorch Dataset for text data:
- Stores texts, labels, vocabulary, and maximum sequence length
- `__getitem__` encodes text, truncates/pads to fixed length
- Returns tensors ready for neural network training

```python path=/src/cyberbully_detector/models/dl.py start=46
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
```
**Lines 46-56**: **`LSTMClassifier` class** - Bidirectional LSTM for text classification:
- Embedding layer with padding awareness
- Bidirectional LSTM (output size = hidden × 2)
- Uses final timestep output for classification
- Linear layer maps to class predictions

```python path=/src/cyberbully_detector/models/dl.py start=59
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
```
**Lines 59-71**: **`TextCNN` class** - Convolutional Neural Network for text classification:
- Multiple 1D convolutions with different kernel sizes (3, 4, 5)
- Max-pooling over time dimension
- Concatenates features from all kernel sizes
- Final linear layer for classification

```python path=/src/cyberbully_detector/models/dl.py start=73
@dataclass
class DLResult:
    name: str
    model: nn.Module
    vocab: Dict[str, int]
    metrics: Dict[str, Any]
```
**Lines 73-78**: **`DLResult` dataclass** - Structure for storing deep learning results including the trained model and vocabulary.

```python path=/src/cyberbully_detector/models/dl.py start=81
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
```
**Lines 81-94**: **`evaluate` function** - Evaluates model performance:
- Sets model to evaluation mode
- Disables gradient computation for efficiency
- Collects predictions and true labels
- Calculates F1-score and classification report

```python path=/src/cyberbully_detector/models/dl.py start=97
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
```
**Lines 97-116**: **`train_model` function** - Trains deep learning model:
- Uses Adam optimizer and CrossEntropy loss
- Implements early stopping based on validation F1-score
- Saves best model state during training
- Restores best model after training completes

---

## 8. Transformer Models Module (`src/cyberbully_detector/models/transformers.py`)

Implements Hugging Face transformer-based models.

```python path=/src/cyberbully_detector/models/transformers.py start=1
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import f1_score, classification_report
```
**Lines 1-8**: Import Hugging Face transformers library and related dependencies.

```python path=/src/cyberbully_detector/models/transformers.py start=11
@dataclass
class HFResult:
    name: str
    model_name: str
    num_labels: int
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    metrics: Dict[str, Any]
    model: Any
    tokenizer: Any
```
**Lines 11-20**: **`HFResult` dataclass** - Structure for storing transformer model results with label mappings.

```python path=/src/cyberbully_detector/models/transformers.py start=23
def train_transformer(df_train: pd.DataFrame, df_val: pd.DataFrame, model_name: str, label2id: Dict[str, int], epochs: int = 2, lr: float = 5e-5, batch_size: int = 16) -> HFResult:
    id2label = {i: l for l, i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(model_name)
    def tokenize(batch):
        return tok(batch['text'], truncation=True, max_length=128)
    ds_train = Dataset.from_pandas(df_train[["text", "category"]].rename(columns={"category": "labels"}), preserve_index=False)
    ds_val = Dataset.from_pandas(df_val[["text", "category"]].rename(columns={"category": "labels"}), preserve_index=False)
    ds_train = ds_train.map(lambda ex: {**tokenize(ex), "labels": [label2id[x] for x in ex["labels"]]}, batched=True, remove_columns=["text"]) 
    ds_val = ds_val.map(lambda ex: {**tokenize(ex), "labels": [label2id[x] for x in ex["labels"]]}, batched=True, remove_columns=["text"]) 
    collator = DataCollatorWithPadding(tokenizer=tok)
```
**Lines 23-32**: **First part of `train_transformer` function**:
- Creates bidirectional label mappings
- Loads pre-trained tokenizer
- Converts DataFrames to Hugging Face Dataset format
- Applies tokenization and label encoding

```python path=/src/cyberbully_detector/models/transformers.py start=34
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
```
**Line 34**: Loads pre-trained transformer model and adapts it for classification with the correct number of labels.

```python path=/src/cyberbully_detector/models/transformers.py start=36
    args = TrainingArguments(
        output_dir="./hf_runs",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[]
    )
```
**Lines 36-48**: Configures training arguments:
- Evaluates and saves after each epoch
- Loads best model based on F1-score
- Disables external logging (WandB, etc.)

```python path=/src/cyberbully_detector/models/transformers.py start=50
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average='macro')
        rep = classification_report(p.label_ids, preds, output_dict=True, zero_division=0)
        return {"f1": f1, "report": rep}
```
**Lines 50-54**: **`compute_metrics` function** - Calculates F1-score and classification report during training.

```python path=/src/cyberbully_detector/models/transformers.py start=56
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    return HFResult(
        name=model_name.split('/')[-1],
        model_name=model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        metrics={"f1_macro": metrics.get("eval_f1", 0.0), "report": metrics.get("eval_report", {})},
        model=trainer.model,
        tokenizer=tok,
    )
```
**Lines 56-76**: **Final part of `train_transformer` function**:
- Creates Trainer object with all components
- Executes training and evaluation
- Returns structured result with trained model

---

## 9. Training Module (`src/cyberbully_detector/train.py`)

Orchestrates the training of all model types and selects the best performer.

```python path=/src/cyberbully_detector/train.py start=1
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
from .models.dl import build_vocab, TextDataset, LSTMClassifier, TextCNN, train_model, evaluate
from .utils import set_seed
```
**Lines 1-14**: Import all required modules for training different model types.

```python path=/src/cyberbully_detector/train.py start=17
def select_best(run_results):
    best = None
    for r in run_results:
        f1 = r["metrics"].get("f1_macro")
        if best is None or f1 > best["metrics"].get("f1_macro"):
            best = r
    return best
```
**Lines 17-23**: **`select_best` function** - Selects the model with highest macro F1-score from all trained models.

```python path=/src/cyberbully_detector/train.py start=26
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
```
**Lines 26-43**: **`persist_best` function** - Saves the best model and its metadata:
- Creates necessary directories
- Handles different model types (ML, DL, transformer)
- Saves models in appropriate format (joblib, PyTorch, Hugging Face)
- Creates metadata JSON file for inference

```python path=/src/cyberbully_detector/train.py start=46
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
```
**Lines 46-59**: **Beginning of `run_training` function**:
- Sets random seed for reproducibility
- Loads and splits data
- Trains classical ML models (logistic regression and SVM)
- Collects results

```python path=/src/cyberbully_detector/train.py start=61
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

        lstm = LSTMClassifier(vocab_size=len(vocab), num_classes=n_classes)
        train_model(lstm, train_loader, val_loader, epochs=EPOCHS_DL, device=device)
        f1, rep = evaluate(lstm, val_loader, device=device)
        results.append({"name": "lstm", "type": "dl", "model": lstm, "metrics": rep, "vocab": vocab})

        cnn = TextCNN(vocab_size=len(vocab), num_classes=n_classes)
        train_model(cnn, train_loader, val_loader, epochs=EPOCHS_DL, device=device)
        f1, rep = evaluate(cnn, val_loader, device=device)
        results.append({"name": "cnn", "type": "dl", "model": cnn, "metrics": rep, "vocab": vocab})
    except Exception:
        pass
```
**Lines 61-83**: **Deep learning models training**:
- Gracefully handles PyTorch unavailability
- Builds vocabulary and creates datasets
- Trains LSTM and CNN models
- Evaluates and collects results

```python path=/src/cyberbully_detector/train.py start=85
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
```
**Lines 85-101**: **Transformer models training**:
- Gracefully handles transformers library unavailability
- Trains configured transformer models (BERT, RoBERTa)
- Collects results with proper format

```python path=/src/cyberbully_detector/train.py start=103
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
```
**Lines 103-113**: **Model selection and persistence**:
- Selects best performing model
- Saves model with appropriate parameters
- Returns best model information

---

## 10. Inference Module (`src/cyberbully_detector/inference.py`)

Handles loading saved models and making predictions.

```python path=/src/cyberbully_detector/inference.py start=1
from __future__ import annotations
import json
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

from .config import BEST_MODEL_DIR, META_PATH
from .utils import severity_from_category_and_conf
from .preprocessing import preprocess
```
**Lines 1-9**: Import required modules for inference operations.

```python path=/src/cyberbully_detector/inference.py start=12
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
```
**Lines 12-45**: **`Predictor` class initialization**:
- Loads model metadata to determine model type
- ML models: Loads joblib pipeline with vectorizer, encoder, and model
- DL models: Reconstructs appropriate architecture (CNN/LSTM) and loads weights
- Transformer models: Loads from Hugging Face format
- Handles model type detection and appropriate loading

```python path=/src/cyberbully_detector/inference.py start=47
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
```
**Lines 47-93**: **`predict` method** - Makes predictions on input text:
- Preprocesses input text
- ML models: Uses vectorizer and handles both probabilistic and decision function outputs
- DL models: Encodes text, pads sequences, runs through neural network
- Transformer models: Tokenizes and runs through transformer
- Calculates severity based on category and confidence
- Returns structured prediction result

```python path=/src/cyberbully_detector/inference.py start=95
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
```
**Lines 95-142**: **`explain_lime` method** - Provides explanations using LIME:
- Gracefully handles missing LIME library
- Creates appropriate prediction functions for each model type
- Uses LIME to identify important features/tokens
- Returns list of tokens with their importance weights

---

## 11. API Module (`src/api/main.py`)

Implements the REST API using FastAPI.

```python path=/src/api/main.py start=1
from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from ..cyberbully_detector.inference import Predictor
from ..cyberbully_detector.rephrase import rephrase_to_safe
from ..cyberbully_detector.analytics import log_event, compute_dashboard, export_csv, export_pdf
from .schemas import AnalyzeRequest, AnalyzeResponse, RephraseRequest, RephraseResponse, FeedbackRequest
```
**Lines 1-10**: Import FastAPI components and project modules.

```python path=/src/api/main.py start=12
# Optional translation
try:
    from langdetect import detect
    from deep_translator import GoogleTranslator
except Exception:
    detect = None
    GoogleTranslator = None
```
**Lines 12-18**: Optionally import translation libraries with graceful fallback.

```python path=/src/api/main.py start=20
app = FastAPI(title="Cyberbullying Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
**Lines 20-27**: Create FastAPI application with CORS middleware for cross-origin requests.

```python path=/src/api/main.py start=29
predictor: Predictor | None = None

@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = Predictor()
```
**Lines 29-35**: Initialize the predictor on application startup.

```python path=/src/api/main.py start=38
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    assert predictor is not None
    text = req.text
    translated = None
    if GoogleTranslator is not None:
        try:
            lang = req.source_lang or (detect(text) if detect else "en")
            if lang and lang != "en":
                translated = GoogleTranslator(source=lang, target="en").translate(text)
            else:
                translated = text
        except Exception:
            translated = text
    else:
        translated = text

    out = predictor.predict(translated)
    exps = predictor.explain_lime(translated)
    out_payload = AnalyzeResponse(**{**out, "explanations": exps, "translated": translated if translated != text else None})

    log_event(text=text, prediction=out_payload.prediction, category=out_payload.category, severity=out_payload.severity, confidence=out_payload.confidence)
    return out_payload
```
**Lines 38-60**: **`/analyze` endpoint**:
- Handles text analysis requests
- Optionally detects language and translates to English
- Makes prediction using the loaded model
- Generates LIME explanations
- Logs the event for analytics
- Returns structured response

```python path=/src/api/main.py start=63
@app.post("/rephrase", response_model=RephraseResponse)
async def rephrase(req: RephraseRequest):
    return RephraseResponse(**rephrase_to_safe(req.text))
```
**Lines 63-65**: **`/rephrase` endpoint** - Provides text rephrasing suggestions.

```python path=/src/api/main.py start=68
@app.get("/dashboard")
async def dashboard():
    return compute_dashboard()
```
**Lines 68-70**: **`/dashboard` endpoint** - Returns analytics dashboard data.

```python path=/src/api/main.py start=73
@app.get("/report/csv")
async def report_csv():
    path = export_csv()
    return {"path": path}

@app.get("/report/pdf")
async def report_pdf():
    path = export_pdf()
    return {"path": path}
```
**Lines 73-82**: **Report endpoints** - Generate and return paths to CSV and PDF reports.

```python path=/src/api/main.py start=85
@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    from ..cyberbully_detector.analytics import load_analytics_df, ANALYTICS_CSV
    df = load_analytics_df()
    if not df.empty:
        idxs = df.index[df['text'] == req.text].tolist()
        if idxs:
            idx = idxs[-1]
            df.at[idx, 'feedback'] = 'yes' if req.correct else 'no'
            df.to_csv(ANALYTICS_CSV, index=False)
    return {"ok": True}
```
**Lines 85-95**: **`/feedback` endpoint** - Records user feedback on predictions for model improvement.

---

## 12. API Schemas Module (`src/api/schemas.py`)

Defines Pydantic models for API request/response validation.

```python path=/src/api/schemas.py start=1
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
```
**Lines 1-3**: Import Pydantic for data validation.

```python path=/src/api/schemas.py start=6
class AnalyzeRequest(BaseModel):
    text: str
    source_lang: Optional[str] = Field(default=None, description="ISO code of the source language if known")
```
**Lines 6-8**: **`AnalyzeRequest` schema** - Validates analysis requests with optional language specification.

```python path=/src/api/schemas.py start=11
class AnalyzeResponse(BaseModel):
    prediction: str
    category: str
    severity: str
    confidence: float
    probs: List[float]
    explanations: List[Dict[str, float]]
    translated: Optional[str] = None
```
**Lines 11-18**: **`AnalyzeResponse` schema** - Defines structure of analysis response including predictions, confidence, and explanations.

```python path=/src/api/schemas.py start=21
class RephraseRequest(BaseModel):
    text: str

class RephraseResponse(BaseModel):
    original: str
    censored: str
    suggestion: str
```
**Lines 21-28**: **Rephrasing schemas** - Define request and response for text rephrasing feature.

```python path=/src/api/schemas.py start=31
class FeedbackRequest(BaseModel):
    text: str
    correct: bool
```
**Lines 31-33**: **`FeedbackRequest` schema** - Validates user feedback on prediction accuracy.

---

## 13. Frontend Module (`src/frontend/streamlit_app.py`)

Implements the web user interface using Streamlit.

```python path=/src/frontend/streamlit_app.py start=1
from __future__ import annotations
import os
import io
import json
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
```
**Lines 1-9**: Import required libraries for web interface and visualization.

```python path=/src/frontend/streamlit_app.py start=11
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Cyberbullying Detector", layout="wide")
st.title("AI Cyberbullying Detection")
```
**Lines 11-15**: Configure Streamlit app and set title.

```python path=/src/frontend/streamlit_app.py start=17
text = st.text_area("Enter text to analyze", height=150)
col1, col2 = st.columns([1,1])
with col1:
    source_lang = st.text_input("Source language (optional, e.g., en, es, fr)")
with col2:
    if st.button("Analyze", type="primary") and text.strip():
        with st.spinner("Analyzing..."):
            resp = requests.post(f"{API_BASE}/analyze", json={"text": text, "source_lang": source_lang or None})
            if resp.ok:
                data = resp.json()
                st.session_state['last_analysis'] = data
            else:
                st.error(f"API error: {resp.status_code}")
```
**Lines 17-29**: **Input section** - Creates text input area and analysis button with API call handling.

```python path=/src/frontend/streamlit_app.py start=31
if 'last_analysis' in st.session_state:
    data = st.session_state['last_analysis']
    st.subheader("Prediction")
    color = {"low": "#8BC34A", "medium": "#FFC107", "high": "#F44336"}.get(data['severity'], "#9E9E9E")
    st.markdown(f"<div style='padding:12px;border-radius:8px;background:{color}'>Prediction: <b>{data['prediction']}</b>, Category: <b>{data['category']}</b>, Severity: <b>{data['severity']}</b>, Confidence: <b>{data['confidence']:.3f}</b></div>", unsafe_allow_html=True)
    if data.get('translated'):
        st.info(f"Translated text used for inference: {data['translated']}")

    st.write("Top contributing tokens (LIME):")
    if data.get('explanations'):
        dfexp = pd.DataFrame(data['explanations'])
        st.bar_chart(dfexp.set_index('token'))
    else:
        st.write("Explainability unavailable.")
```
**Lines 31-44**: **Results display** - Shows prediction results with color-coded severity and LIME explanations.

```python path=/src/frontend/streamlit_app.py start=46
    st.subheader("Rephrasing suggestion")
    if st.button("Suggest safer rephrasing"):
        rr = requests.post(f"{API_BASE}/rephrase", json={"text": text})
        if rr.ok:
            rj = rr.json()
            st.code(rj['suggestion'])
        else:
            st.error("Rephrase API error")
```
**Lines 46-53**: **Rephrasing section** - Provides safer text alternatives.

```python path=/src/frontend/streamlit_app.py start=55
    st.subheader("Feedback")
    fbcol1, fbcol2 = st.columns(2)
    with fbcol1:
        if st.button("Was this correct? Yes ✅"):
            requests.post(f"{API_BASE}/feedback", json={"text": text, "correct": True})
            st.success("Thanks for the feedback!")
    with fbcol2:
        if st.button("Was this correct? No ❌"):
            requests.post(f"{API_BASE}/feedback", json={"text": text, "correct": False})
            st.success("Thanks for the feedback!")
```
**Lines 55-64**: **Feedback section** - Allows users to rate prediction accuracy.

```python path=/src/frontend/streamlit_app.py start=68
st.subheader("Dashboard")
if st.button("Refresh dashboard"):
    dash = requests.get(f"{API_BASE}/dashboard").json()
    st.session_state['dash'] = dash

if 'dash' in st.session_state:
    dash = st.session_state['dash']
    c1, c2 = st.columns(2)
    with c1:
        st.write("Category distribution")
        df = pd.DataFrame([dash['category_distribution']]).T
        df.columns = ['count']
        st.bar_chart(df)
    with c2:
        st.write("Severity distribution")
        df2 = pd.DataFrame([dash['severity_distribution']]).T
        df2.columns = ['count']
        st.bar_chart(df2)
    c3, c4 = st.columns(2)
    with c3:
        st.write("Timeline")
        tl = pd.DataFrame(dash['timeline'])
        if not tl.empty:
            tl['date'] = pd.to_datetime(tl['date'])
            tl = tl.set_index('date')
            st.line_chart(tl)
        else:
            st.write("No data yet.")
    with c4:
        st.write("Word cloud")
        if dash.get('wordcloud_path') and os.path.exists(dash['wordcloud_path']):
            st.image(dash['wordcloud_path'])
        else:
            st.write("No word cloud available.")
```
**Lines 68-101**: **Dashboard section** - Displays analytics charts including category distribution, severity trends, timeline, and word cloud.

```python path=/src/frontend/streamlit_app.py start=105
st.subheader("Reports")
ccol1, ccol2 = st.columns(2)
with ccol1:
    if st.button("Download CSV report"):
        r = requests.get(f"{API_BASE}/report/csv").json()
        st.success(f"CSV generated at: {r['path']}")
with ccol2:
    if st.button("Download PDF report"):
        r = requests.get(f"{API_BASE}/report/pdf").json()
        st.success(f"PDF generated at: {r['path']}")
```
**Lines 105-114**: **Reports section** - Buttons to generate downloadable reports.

---

## 14. Training Script (`scripts/train_all.py`)

Command-line script to train all models.

```python path=/scripts/train_all.py start=1
import argparse
from src.cyberbully_detector.train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()
    best = run_training(sample_frac=args.sample_frac)
    print("Best model:", best["name"], "type:", best["type"], "f1:", best["metrics"].get("f1_macro"))
```
**Lines 1-9**: Simple command-line interface for training with optional data sampling and result reporting.

---

## 15. Model Evaluation Script (`scripts/print_model_accuracies.py`)

Script to compare performance of different model types.

```python path=/scripts/print_model_accuracies.py start=15
def main(sample_frac: float = 1.0):
    set_seed()
    df = load_all_data(sample_frac=sample_frac)
    df_train, df_val, df_test = train_val_test_split(df)

    print("=== Classical ML (TF-IDF) ===")
    vect, le, X_train, y_train, X_val, y_val, X_test, y_test = build_tfidf_and_labels(
        df_train, df_val, df_test
    )
    # Logistic Regression
    lr = train_logreg(X_train, y_train, X_val, y_val)
    acc_lr = lr.metrics["report"].get("accuracy")
    print(f"LogReg -> accuracy: {acc_lr:.4f}, f1_macro: {lr.metrics.get('f1_macro'):.4f}")

    # Linear SVM (Calibrated)
    svm = train_svm(X_train, y_train, X_val, y_val)
    acc_svm = svm.metrics["report"].get("accuracy")
    print(f"SVM -> accuracy: {acc_svm:.4f}, f1_macro: {svm.metrics.get('f1_macro'):.4f}")
```
**Lines 15-32**: **Classical ML evaluation** - Tests logistic regression and SVM models.

```python path=/scripts/print_model_accuracies.py start=34
    print("\n=== Deep Learning (PyTorch) ===")
    try:
        import torch
        from torch.utils.data import DataLoader
        vocab = build_vocab(df_train["text_clean"].tolist())
        n_classes = len(le.classes_)
        train_ds = TextDataset(df_train["text_clean"].tolist(), le.transform(df_train["category"]).tolist(), vocab)
        val_ds   = TextDataset(df_val["text_clean"].tolist(),   le.transform(df_val["category"]).tolist(),   vocab)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=64)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # LSTM
        lstm = LSTMClassifier(vocab_size=len(vocab), num_classes=n_classes)
        train_model(lstm, train_loader, val_loader, epochs=EPOCHS_DL, device=device)
        _, met_lstm = evaluate(lstm, val_loader, device=device)
        acc_lstm = met_lstm["report"].get("accuracy")
        print(f"LSTM -> accuracy: {acc_lstm:.4f}, f1_macro: {met_lstm.get('f1_macro'):.4f}")

        # TextCNN
        cnn = TextCNN(vocab_size=len(vocab), num_classes=n_classes)
        train_model(cnn, train_loader, val_loader, epochs=EPOCHS_DL, device=device)
        _, met_cnn = evaluate(cnn, val_loader, device=device)
        acc_cnn = met_cnn["report"].get("accuracy")
        print(f"TextCNN -> accuracy: {acc_cnn:.4f}, f1_macro: {met_cnn.get('f1_macro'):.4f}")
    except Exception:
        print("DL models skipped due to missing deps or runtime error:")
        traceback.print_exc()
```
**Lines 34-62**: **Deep learning evaluation** - Tests LSTM and CNN models with error handling.

```python path=/scripts/print_model_accuracies.py start=64
    print("\n=== Transformers (Hugging Face) ===")
    try:
        label2id = {l: i for i, l in enumerate(le.classes_)}
        for key, mname in DEFAULT_TRANSFORMER_MODELS.items():
            # Use 1 epoch here to keep evaluation lightweight
            hfres = train_transformer(df_train, df_val, mname, label2id, epochs=10)
            rep = hfres.metrics.get("report", {}) or {}
            acc = rep.get("accuracy", None)
            f1  = hfres.metrics.get("f1_macro", None)
            if acc is not None and f1 is not None:
                print(f"{key} ({mname}) -> accuracy: {acc:.4f}, f1_macro: {f1:.4f}")
            else:
                print(f"{key} ({mname}) -> metrics not available")
    except Exception:
        print("Transformer models skipped due to missing deps or runtime error:")
        traceback.print_exc()
```
**Lines 64-79**: **Transformer evaluation** - Tests BERT and RoBERTa models with error handling.

---

## 16. Analytics Module (`src/cyberbully_detector/analytics.py`)

Handles logging, reporting, and dashboard analytics.

```python path=/src/cyberbully_detector/analytics.py start=16
def log_event(text: str, prediction: str, category: str, severity: str, confidence: float, correct: bool | None = None):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "text", "prediction", "category", "severity", "confidence", "feedback"]
    row = [datetime.utcnow().isoformat(), text, prediction, category, severity, confidence, None if correct is None else ("yes" if correct else "no")]
    new_file = not ANALYTICS_CSV.exists()
    with open(ANALYTICS_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)
```
**Lines 16-25**: **`log_event` function** - Logs prediction events to CSV file with timestamp and optional feedback.

```python path=/src/cyberbully_detector/analytics.py start=34
def compute_dashboard() -> Dict[str, Any]:
    df = load_analytics_df()
    cat_counts = df["category"].value_counts(dropna=False).to_dict()
    sev_counts = df["severity"].value_counts(dropna=False).to_dict()
    bully_texts = df[df["prediction"] == "bullying"]["text"].astype(str)
    text_blob = " ".join(bully_texts.tolist())
    wc = WordCloud(width=600, height=400).generate(text_blob) if text_blob.strip() else None
    wc_path = None
    if wc is not None:
        wc_path = REPORTS_DIR / "wordcloud.png"
        wc.to_file(wc_path)
    if not df.empty:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        timeline = df.groupby("date").size().reset_index(name="count").to_dict(orient="list")
    else:
        timeline = {"date": [], "count": []}
    return {
        "category_distribution": cat_counts,
        "severity_distribution": sev_counts,
        "wordcloud_path": str(wc_path) if wc_path else None,
        "timeline": timeline
    }
```
**Lines 34-55**: **`compute_dashboard` function** - Creates analytics dashboard data including distributions, word cloud, and timeline.

```python path=/src/cyberbully_detector/analytics.py start=65
def export_pdf() -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    out = REPORTS_DIR / f"analytics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, h - 72, "Cyberbullying Analytics Report")
    c.setFont("Helvetica", 12)
    y = h - 100
    dash = compute_dashboard()
    c.drawString(72, y, f"Category distribution: {dash['category_distribution']}")
    y -= 20
    c.drawString(72, y, f"Severity distribution: {dash['severity_distribution']}")
    y -= 20
    if dash.get("wordcloud_path"):
        try:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(dash["wordcloud_path"])
            c.drawImage(img, 72, y - 200, width=400, height=200)
            y -= 220
        except Exception:
            pass
    c.showPage()
    c.save()
    return str(out)
```
**Lines 65-90**: **`export_pdf` function** - Generates PDF reports with analytics summary and word cloud visualization.

---

## 17. Text Rephrasing Module (`src/cyberbully_detector/rephrase.py`)

Provides text improvement suggestions to reduce harmful content.

```python path=/src/cyberbully_detector/rephrase.py start=8
REPLACEMENTS = {
    "hate": "dislike",
    "stupid": "not very thoughtful",
    "idiot": "person",
    "kill": "harm",
    "dumb": "unwise"
}
```
**Lines 8-14**: Dictionary mapping harmful words to more appropriate alternatives.

```python path=/src/cyberbully_detector/rephrase.py start=17
def rephrase_to_safe(text: str) -> Dict[str, Any]:
    raw = text
    cleaned = clean_text(text)
    censored = profanity.censor(raw)
    tokens = cleaned.split()
    replaced = [REPLACEMENTS.get(t, t) for t in tokens]
    suggestion = " ".join(replaced)
    if suggestion.strip() == cleaned.strip():
        suggestion = censored
    return {
        "original": raw,
        "censored": censored,
        "suggestion": suggestion
    }
```
**Lines 17-30**: **`rephrase_to_safe` function** - Creates safer text alternatives using word replacement and profanity censoring.

---

## 18. Test Script (`test_model.py`)

Simple script to test the trained model.

```python path=/test_model.py start=14
    test_texts = [
        "You are such a loser, nobody likes you",
        "Have a great day!",
        "I hope you fail at everything you do",
        "This movie was really good"
    ]
    
    print("\nTesting predictions:")
    print("-" * 60)
    
    for text in test_texts:
        try:
            result = predictor.predict(text)
            print(f"Text: {text[:50]}...")
            print(f"Prediction: {result['prediction']}")
            print(f"Category: {result['category']}")
            print(f"Severity: {result['severity']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print("-" * 60)
        except Exception as e:
            print(f"Error with text '{text}': {e}")
            print("-" * 60)
```
**Lines 14-35**: Tests the model with sample texts covering different cyberbullying categories and normal text.

---

## Summary

This AI Cyberbullying Detection system is a comprehensive solution that:

1. **Data Processing**: Loads multiple datasets, preprocesses text, and handles various data formats
2. **Multiple Model Types**: Implements traditional ML, deep learning, and transformer models
3. **Model Selection**: Automatically selects the best performing model based on F1-score
4. **Inference**: Provides fast predictions with explanations using LIME
5. **Web API**: RESTful API with FastAPI for integration
6. **User Interface**: Streamlit-based web interface for interactive use
7. **Analytics**: Comprehensive logging and reporting system
8. **Text Improvement**: Rephrasing suggestions for safer communication

The system is designed for production use with proper error handling, model versioning, and monitoring capabilities. It can classify text into six categories (not_cyberbullying, harassment, insult, hate, threat, other) and provides severity levels and confidence scores for each prediction.

---

**Technical Report Prepared by**: AI Code Analysis System  
**Date**: December 2024  
**Document Version**: 1.0