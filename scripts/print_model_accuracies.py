from __future__ import annotations
import traceback

from src.cyberbully_detector.data import load_all_data, train_val_test_split
from src.cyberbully_detector.features import build_tfidf_and_labels
from src.cyberbully_detector.models.ml import train_logreg, train_svm
from src.cyberbully_detector.models.dl import (
    build_vocab, TextDataset, LSTMClassifier, TextCNN, train_model, evaluate
)
from src.cyberbully_detector.models.transformers import train_transformer
from src.cyberbully_detector.config import DEFAULT_TRANSFORMER_MODELS, EPOCHS_DL, EPOCHS_TRANSFORMER
from src.cyberbully_detector.utils import set_seed


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


if __name__ == "__main__":
    # Use a smaller fraction to keep runtime reasonable
    main(sample_frac=0.3)
