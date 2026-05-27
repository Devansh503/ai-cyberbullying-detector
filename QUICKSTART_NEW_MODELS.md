# Quick Start: New Models

## What Was Added

✅ **Enhanced LSTM** with attention mechanism and dropout  
✅ **Standalone BERT** classifier with custom head  
✅ **5 BERT variants** (distilbert, bert-base, bert-large, roberta, albert)

## Quick Test

To test the new models, navigate to the project and run training:

```powershell
cd F:\Major\ai-cyberbullying-detector
.venv\Scripts\Activate.ps1
python -m src.cyberbully_detector.train
```

## Model Comparison

| Model | Type | Features | Speed | Accuracy |
|-------|------|----------|-------|----------|
| Original LSTM | DL | Basic BiLSTM | Fast | Baseline |
| **Enhanced LSTM** | DL | +Attention +Dropout | Fast | Better |
| **BERT Custom** | DL | Custom head | Medium | Best |
| CNN | DL | Convolutional | Fast | Good |
| DistilBERT | Transformer | Lightweight | Medium | Good |
| BERT-Base | Transformer | Standard | Slow | Best |
| RoBERTa | Transformer | Optimized | Slow | Best |

## Files Modified

1. **`src/cyberbully_detector/models/dl.py`**
   - Added `EnhancedLSTMClassifier` (lines 59-98)
   - Added `BERTClassifier` (lines 101-128)
   - Added `BERTDataset` (lines 46-75)
   - Added `train_bert_model()` and `evaluate_bert()` (lines 223-270)

2. **`src/cyberbully_detector/config.py`**
   - Updated `DEFAULT_TRANSFORMER_MODELS` with 5 BERT variants (lines 19-24)

3. **`src/cyberbully_detector/train.py`**
   - Integrated enhanced LSTM (lines 80-84)
   - Integrated standalone BERT (lines 92-114)
   - Added error handling

4. **`docs/NEW_MODELS.md`**
   - Complete documentation (new file)

## Next Steps

1. **Train models**: Run the training script
2. **Compare results**: Check which model performs best
3. **Adjust hyperparameters**: Modify `config.py` if needed
4. **Deploy best model**: The system auto-selects and saves the best model

## Configuration Options

Edit `config.py` to customize:

```python
EPOCHS_DL = 3              # Epochs for DL models
EPOCHS_TRANSFORMER = 2     # Epochs for transformers
BATCH_SIZE = 16           # Training batch size
MAX_SEQ_LEN = 128         # Maximum sequence length
```

## Memory Considerations

- **Enhanced LSTM**: Low memory (~500MB)
- **Standalone BERT**: Medium memory (~2GB)
- **BERT variants**: High memory (2-4GB)

If you encounter OOM errors, use DistilBERT or reduce batch size.
