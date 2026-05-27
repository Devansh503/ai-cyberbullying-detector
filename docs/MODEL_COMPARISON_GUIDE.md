# Model Comparison & Metrics Guide

## Overview
This guide explains how to view comprehensive model performance metrics including confusion matrices, precision, recall, and F1 scores for all trained models.

## Features

### 📊 Metrics Generated
For each model, the following metrics are automatically computed and saved:

1. **Overall Metrics**
   - Accuracy
   - Precision (Macro & Weighted)
   - Recall (Macro & Weighted)
   - F1 Score (Macro & Weighted)

2. **Per-Class Metrics**
   - Precision per category
   - Recall per category
   - F1 Score per category
   - Support (number of samples) per category

3. **Confusion Matrices**
   - Raw confusion matrix (counts)
   - Normalized confusion matrix (percentages)

4. **Visual Comparisons**
   - Side-by-side metric comparisons
   - Grouped bar charts
   - Individual confusion matrix plots

## How to Generate Metrics

### Step 1: Run Training
The metrics are automatically generated when you run model training:

```powershell
cd F:\Major\ai-cyberbullying-detector
.venv\Scripts\Activate.ps1
python -m src.cyberbully_detector.train
```

During training, after all models are trained, you'll see:
```
================================================================================
Generating comprehensive model comparison metrics...
================================================================================

Evaluating logreg on test set...
✓ logreg: F1=0.8542, Accuracy=0.8631

Evaluating svm on test set...
✓ svm: F1=0.8598, Accuracy=0.8701

... (continues for all models)

✅ Model comparison complete! Check the comparisons directory for detailed results.
```

### Step 2: View Results

#### Option A: Files (Automatic)
All metrics are saved to: `src/cyberbully_detector/model_registry/comparisons/`

**Generated Files:**
- `model_metrics.json` - Complete metrics in JSON format
- `model_metrics.csv` - Summary table (easy to open in Excel)
- `model_metrics_detailed.csv` - Per-class metrics for all models
- `comparison_report.txt` - Text summary report
- `model_comparison.png` - Visual comparison of all metrics
- `model_comparison_grouped.png` - Grouped bar chart comparison
- `confusion_matrices/` - Directory containing confusion matrix plots for each model

#### Option B: Frontend (Interactive)
1. Start the API server:
   ```powershell
   cd F:\Major\ai-cyberbullying-detector
   uvicorn src.api.main:app --reload
   ```

2. Start the frontend:
   ```powershell
   streamlit run src/frontend/streamlit_app.py
   ```

3. In the browser:
   - Scroll down to "📊 Model Performance Comparison" section
   - Click "Load Model Comparison" button
   - View interactive metrics, confusion matrices, and plots

#### Option C: API (Programmatic)
Access metrics via REST API:

```python
import requests

# Get all model comparison data
response = requests.get("http://localhost:8000/models/comparison")
data = response.json()

# Get specific model metrics
response = requests.get("http://localhost:8000/models/metrics/bert_custom")
model_data = response.json()
```

## Understanding the Metrics

### Confusion Matrix
Shows the relationship between predicted and actual labels:
- **Diagonal values**: Correct predictions
- **Off-diagonal values**: Misclassifications
- **Normalized version**: Shows percentages for easier interpretation

**Example:**
```
                  Predicted
               0    1    2    3
Actual    0  [150   5    2    1]  ← 150 correctly classified as class 0
          1  [  3  120   8    2]  ← 120 correctly classified as class 1
          2  [  1   10  135   4]  ← Most accurate class
          3  [  0    2    5   93]  ← Some confusion with class 2
```

### Precision
**Precision = True Positives / (True Positives + False Positives)**

Answers: "Of all samples predicted as class X, how many were actually class X?"
- High precision → Low false positive rate
- Important when false positives are costly

### Recall
**Recall = True Positives / (True Positives + False Negatives)**

Answers: "Of all actual class X samples, how many did we correctly identify?"
- High recall → Low false negative rate
- Important when false negatives are costly

### F1 Score
**F1 = 2 × (Precision × Recall) / (Precision + Recall)**

Harmonic mean of precision and recall:
- Balances both metrics
- Best overall performance indicator
- Used to select the best model

## Model Comparison Example

### Sample Output in comparison_report.txt:
```
================================================================================
MODEL COMPARISON REPORT
================================================================================

🏆 BEST MODEL: bert_custom
   F1 Score (Macro): 0.9234
   Accuracy: 0.9311

MODELS RANKED BY F1 SCORE:
--------------------------------------------------------------------------------
Rank   Model                Accuracy     Precision    Recall       F1 Score    
--------------------------------------------------------------------------------
1      bert_custom          0.9311       0.9245       0.9223       0.9234      
2      distilbert           0.9187       0.9156       0.9134       0.9145      
3      enhanced_lstm        0.8923       0.8901       0.8887       0.8894      
4      roberta              0.9123       0.9098       0.9089       0.9093      
5      lstm                 0.8756       0.8734       0.8712       0.8723      
6      svm                  0.8701       0.8689       0.8578       0.8598      
7      cnn                  0.8645       0.8623       0.8601       0.8612      
8      logreg               0.8631       0.8612       0.8523       0.8542      

================================================================================
```

## Frontend Features

### 1. Overall Performance Metrics Table
- Shows all models ranked by F1 score
- Highlights the best performing model
- Color-coded for easy identification

### 2. Performance Comparison Plots
- **Model Comparison**: 4-panel view (Accuracy, Precision, Recall, F1)
- **Grouped Bar Chart**: All metrics side-by-side for each model

### 3. Confusion Matrices
- Select any model from dropdown
- View both raw and normalized versions side-by-side
- High-resolution plots for detailed analysis

### 4. Per-Class Performance
- Select model to see detailed per-class metrics
- Shows precision, recall, F1, and support for each category
- Identifies which classes are easiest/hardest to predict

### 5. Download Links
- Direct paths to CSV and text reports
- Easy to share or import into other tools

## Interpreting Results

### Which Model is Best?
- **Highest F1 Score (Macro)**: Best overall balance
- Check confusion matrix for specific weaknesses
- Consider per-class metrics if certain categories are more important

### Common Patterns
1. **BERT/Transformers**: Usually highest performance but slowest
2. **Enhanced LSTM**: Good balance of speed and accuracy
3. **Original LSTM/CNN**: Fast but lower accuracy
4. **ML Models (SVM/LogReg)**: Fastest, baseline performance

### What if a Model Has Low Performance?
Check:
1. **Confusion matrix**: Which classes are confused?
2. **Per-class metrics**: Are specific categories problematic?
3. **Training data**: Is there class imbalance?
4. **Hyperparameters**: May need tuning

## Troubleshooting

### No Metrics Available
**Cause**: Training hasn't been run yet
**Solution**: Run `python -m src.cyberbully_detector.train`

### Images Not Showing in Frontend
**Cause**: File paths may be incorrect
**Solution**: Check that files exist in `model_registry/comparisons/`

### API Returns Empty Data
**Cause**: Metrics JSON file doesn't exist
**Solution**: Re-run training to generate metrics

### Confusion Matrix Plots are Blank
**Cause**: matplotlib/seaborn configuration issue
**Solution**: Ensure matplotlib backend is properly configured

## Advanced Usage

### Custom Metric Computation
```python
from src.cyberbully_detector.model_comparison import compute_detailed_metrics
import numpy as np

# Your predictions
y_true = [0, 1, 2, 0, 1]
y_pred = [0, 1, 1, 0, 1]
label_names = ["class0", "class1", "class2"]

metrics = compute_detailed_metrics(y_true, y_pred, "my_model", label_names)
print(f"F1 Score: {metrics['f1_macro']:.4f}")
```

### Generate Custom Plots
```python
from src.cyberbully_detector.model_comparison import (
    plot_confusion_matrix,
    plot_normalized_confusion_matrix
)
from sklearn.metrics import confusion_matrix
import numpy as np

# Your data
y_true = [...]
y_pred = [...]
cm = confusion_matrix(y_true, y_pred)

# Plot
plot_confusion_matrix(cm, label_names, "my_model", save_path="my_cm.png")
```

## Files Reference

| File | Description | Format |
|------|-------------|--------|
| `model_metrics.json` | All metrics for all models | JSON |
| `model_metrics.csv` | Summary comparison table | CSV |
| `model_metrics_detailed.csv` | Per-class metrics | CSV |
| `comparison_report.txt` | Human-readable summary | Text |
| `model_comparison.png` | 4-panel comparison plot | PNG |
| `model_comparison_grouped.png` | Grouped bar chart | PNG |
| `{model}_confusion_matrix.png` | Raw confusion matrix | PNG |
| `{model}_confusion_matrix_normalized.png` | Normalized CM | PNG |

## Best Practices

1. **Always run on test set**: Metrics are computed on held-out test data
2. **Check multiple metrics**: Don't rely on accuracy alone
3. **Examine confusion matrices**: Understand model mistakes
4. **Compare per-class**: Some models excel at specific categories
5. **Consider trade-offs**: Speed vs. accuracy, precision vs. recall
6. **Iterate**: Use insights to improve models

## Next Steps

After reviewing metrics:
1. Select the best model for your use case
2. Deploy the best model (automatically selected and saved)
3. Monitor performance on real data
4. Retrain periodically with new data
5. Compare new models against baseline

For questions or issues, check the main documentation or create an issue in the project repository.
