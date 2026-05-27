import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"
MODEL_REGISTRY_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "src" / "cyberbully_detector" / "model_registry"))
BEST_MODEL_DIR = MODEL_REGISTRY_DIR / "best_model"
META_PATH = MODEL_REGISTRY_DIR / "model_meta.json"

for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR, MODEL_REGISTRY_DIR, BEST_MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

APP_ENV = os.getenv("APP_ENV", "dev")
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "1") == "1"

DEFAULT_TRANSFORMER_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-base-uncased",  # Using base for memory efficiency, change to bert-large-uncased if resources allow
    "roberta": "roberta-base",
    "albert": "albert-base-v2"
}

RANDOM_SEED = 42
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS_DL = 3
EPOCHS_TRANSFORMER = 2

CATEGORIES = [
    "not_cyberbullying",
    "harassment",
    "insult",
    "hate",
    "threat",
    "other"
]

SEVERITY_RULES = {
    "threat": "high",
    "hate": "high",
    "harassment": "medium",
    "insult": "medium",
    "other": "low",
    "not_cyberbullying": "low"
}
