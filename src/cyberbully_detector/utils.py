import json
import random
import numpy as np
import torch
from pathlib import Path
from .config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def severity_from_category_and_conf(category: str, conf: float) -> str:
    from .config import SEVERITY_RULES
    base = SEVERITY_RULES.get(category, 'low')
    if base == 'high':
        return 'high'
    if base == 'medium':
        return 'high' if conf >= 0.9 else 'medium'
    return 'medium' if conf >= 0.95 else 'low'
