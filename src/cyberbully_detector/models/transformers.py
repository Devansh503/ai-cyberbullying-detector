from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import f1_score, classification_report


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

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)

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

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average='macro')
        rep = classification_report(p.label_ids, preds, output_dict=True, zero_division=0)
        return {"f1": f1, "report": rep}

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
