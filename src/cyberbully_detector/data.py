from __future__ import annotations
import os
from typing import Tuple, Optional
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CATEGORIES
from .preprocessing import preprocess


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


def train_val_test_split(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    df_train, df_temp = train_test_split(df, test_size=val_size + test_size, stratify=df["category"], random_state=random_state)
    rel_test = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(df_temp, test_size=rel_test, stratify=df_temp["category"], random_state=random_state)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
