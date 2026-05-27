from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score
from joblib import dump


@dataclass
class MLResult:
    name: str
    model: Any
    metrics: Dict[str, Any]


def train_logreg(X_train, y_train, X_val, y_val) -> MLResult:
    clf = LogisticRegression(max_iter=2000, n_jobs=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return MLResult(name="logreg", model=clf, metrics={"f1_macro": f1, "report": rep})


def train_svm(X_train, y_train, X_val, y_val) -> MLResult:
    base = LinearSVC()
    clf = CalibratedClassifierCV(base, method='sigmoid')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return MLResult(name="svm", model=clf, metrics={"f1_macro": f1, "report": rep})


def save_ml_pipeline(path, vectorizer, label_encoder, model):
    obj = {
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "model": model
    }
    dump(obj, path)
