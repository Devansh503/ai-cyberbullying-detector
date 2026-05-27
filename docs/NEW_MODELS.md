# New Models Documentation

## Overview
This document describes the newly added models to the AI Cyberbullying Detector project, including enhanced LSTM and standalone BERT implementations.

## Models Added

### 1. Enhanced LSTM Classifier (`EnhancedLSTMClassifier`)

**Location:** `src/cyberbully_detector/models/dl.py`

**Features:**
- **Attention Mechanism**: Applies attention weights to LSTM outputs for better focus on relevant text parts
- **Dropout Regularization**: Multiple dropout layers (embedding dropout, LSTM dropout, FC dropout) to prevent overfitting
- **Multi-layer Architecture**: 2-layer bidirectional LSTM by default
- **Deep Classification Head**: Two fully-connected layers with ReLU activation

**Parameters:**
- `vocab_size`: Size of vocabulary
- `num_classes`: Number of output classes
- `emb_dim`: Embedding dimension (default: 128)
- `hidden`: Hidden layer size (default: 128)
- `n_layers`: Number of LSTM layers (default: 2)
- `dropout`: Dropout probability (default: 0.3)

**Advantages over original LSTM:**
- Better handling of long sequences through attention
- Reduced overfitting with dropout
- Improved feature extraction with deeper architecture

### 2. Standalone BERT Classifier (`BERTClassifier`)

**Location:** `src/cyberbully_detector/models/dl.py`

**Features:**
- **Pretrained BERT Base**: Uses `bert-base-uncased` (or configurable)
- **Custom Classification Head**: Two-layer FC network with dropout
- **Fine-tuning Support**: BERT layers can be fine-tuned during training
- **Flexible Architecture**: Customizable hidden dimensions and dropout

**Parameters:**
- `num_classes`: Number of output classes
- `model_name`: Pretrained BERT model name (default: "bert-base-uncased")
- `dropout`: Dropout probability (default: 0.3)
- `hidden_dim`: Hidden layer dimension (default: 256)

**Advantages:**
- More control over BERT architecture than HuggingFace's AutoModelForSequenceClassification
- Custom classification head for better task-specific learning
- Can be easily extended with additional layers or features

### 3. Additional BERT Variants (via Transformers)

**Location:** `src/cyberbully_detector/config.py`

**Available Models:**
- `distilbert`: distilbert-base-uncased (lightweight, faster)
- `bert-base`: bert-base-uncased (standard BERT)
- `bert-large`: bert-base-uncased (placeholder, can upgrade to bert-large-uncased)
- `roberta`: roberta-base (RoBERTa variant)
- `albert`: albert-base-v2 (ALBERT variant)

## Training Pipeline Integration

All models are automatically integrated into the training pipeline in `train.py`:

1. **Enhanced LSTM**: Trained alongside original LSTM and CNN
2. **Standalone BERT**: Trained as a custom deep learning model
3. **Transformer Variants**: Trained using HuggingFace Trainer API

The best performing model across all types (ML, DL, Transformer) is automatically selected and saved.

## Usage

### Running Training with New Models

```bash
# Navigate to project root
cd F:/Major/ai-cyberbullying-detector

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run training (will include all new models)
python -m src.cyberbully_detector.train
```

### Using Models in Code

**Enhanced LSTM:**
```python
from src.cyberbully_detector.models.dl import EnhancedLSTMClassifier, build_vocab, TextDataset
import torch

# Build vocabulary and dataset
vocab = build_vocab(texts)
dataset = TextDataset(texts, labels, vocab)

# Create and train model
model = EnhancedLSTMClassifier(vocab_size=len(vocab), num_classes=6)
# ... training code
```

**Standalone BERT:**
```python
from src.cyberbully_detector.models.dl import BERTClassifier, BERTDataset, train_bert_model
from transformers import AutoTokenizer

# Create tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = BERTDataset(texts, labels, tokenizer)

# Create and train model
model = BERTClassifier(num_classes=6)
# ... training code
```

## Performance Comparison

After training, compare model performance:

- **Original LSTM**: Baseline bidirectional LSTM
- **Enhanced LSTM**: +Attention +Dropout +Deeper architecture
- **CNN**: Convolutional approach for comparison
- **Standalone BERT**: Custom BERT with controllable head
- **Transformer Variants**: Various pretrained transformers

The system automatically selects the best model based on F1 macro score on validation set.

## Model Architecture Details

### Enhanced LSTM Architecture
```
Input Text
  ↓
Embedding Layer (with dropout)
  ↓
Bidirectional LSTM (2 layers, with dropout)
  ↓
Attention Mechanism
  ↓
FC Layer (hidden_dim, ReLU, dropout)
  ↓
Output Layer (num_classes)
```

### Standalone BERT Architecture
```
Input Text (tokenized)
  ↓
BERT Encoder (bert-base-uncased)
  ↓
[CLS] Token Extraction
  ↓
Dropout
  ↓
FC Layer (hidden_dim=256, ReLU)
  ↓
Dropout
  ↓
Output Layer (num_classes)
```

## Training Hyperparameters

**Enhanced LSTM:**
- Learning rate: 1e-3
- Optimizer: Adam
- Epochs: 3 (configurable via EPOCHS_DL)
- Batch size: 32

**Standalone BERT:**
- Learning rate: 2e-5
- Optimizer: AdamW
- Epochs: 3 (configurable via EPOCHS_DL)
- Batch size: 16

## Notes

- All models use the same train/val/test split for fair comparison
- GPU acceleration is automatically used if available
- Models are saved only if they achieve the best performance
- The standalone BERT has lower batch size due to memory requirements
- You can modify `bert-large` to use actual `bert-large-uncased` if you have sufficient GPU memory

## Troubleshooting

**Out of Memory Error:**
- Reduce batch size in `train.py`
- Use smaller BERT variant (distilbert)
- Reduce hidden dimensions

**Slow Training:**
- Use GPU if available
- Use distilbert instead of bert-base
- Reduce number of epochs

**Poor Performance:**
- Increase training epochs
- Adjust learning rate
- Try different BERT variants
- Check data quality and preprocessing
