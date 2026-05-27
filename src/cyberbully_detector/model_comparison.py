from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    accuracy_score
)

from .config import MODEL_REGISTRY_DIR, CATEGORIES


COMPARISON_DIR = MODEL_REGISTRY_DIR / "comparisons"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON = COMPARISON_DIR / "model_metrics.json"
METRICS_CSV = COMPARISON_DIR / "model_metrics.csv"
CONFUSION_MATRICES_DIR = COMPARISON_DIR / "confusion_matrices"
CONFUSION_MATRICES_DIR.mkdir(parents=True, exist_ok=True)


def compute_detailed_metrics(y_true: List[int], y_pred: List[int], model_name: str, 
                             label_names: List[str]) -> Dict[str, Any]:
    """
    Compute comprehensive metrics including confusion matrix, precision, recall, F1
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        label_names: List of label names
        
    Returns:
        Dictionary with all metrics
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=label_names, 
                                   output_dict=True, zero_division=0)
    
    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            label_names[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1_score": float(f1_per_class[i]),
                "support": int(report[label_names[i]]["support"])
            }
            for i in range(len(label_names))
        },
        "classification_report": report
    }


def plot_confusion_matrix(cm: np.ndarray, label_names: List[str], model_name: str, 
                         save_path: Path = None) -> Path:
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix array
        label_names: List of label names
        model_name: Name of the model
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    if save_path is None:
        save_path = CONFUSION_MATRICES_DIR / f"{model_name}_confusion_matrix.png"
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_normalized_confusion_matrix(cm: np.ndarray, label_names: List[str], 
                                     model_name: str, save_path: Path = None) -> Path:
    """
    Plot and save normalized confusion matrix (percentages)
    """
    if save_path is None:
        save_path = CONFUSION_MATRICES_DIR / f"{model_name}_confusion_matrix_normalized.png"
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Percentage'})
    plt.title(f'Normalized Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def save_all_model_metrics(all_metrics: List[Dict[str, Any]]) -> None:
    """
    Save all model metrics to JSON and CSV files
    
    Args:
        all_metrics: List of metric dictionaries for all models
    """
    # Save to JSON
    with open(METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Create summary DataFrame for CSV
    summary_data = []
    for metrics in all_metrics:
        row = {
            "Model": metrics["model_name"],
            "Accuracy": metrics["accuracy"],
            "Precision (Macro)": metrics["precision_macro"],
            "Recall (Macro)": metrics["recall_macro"],
            "F1 Score (Macro)": metrics["f1_macro"],
            "Precision (Weighted)": metrics["precision_weighted"],
            "Recall (Weighted)": metrics["recall_weighted"],
            "F1 Score (Weighted)": metrics["f1_weighted"],
        }
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values("F1 Score (Macro)", ascending=False)
    df_summary.to_csv(METRICS_CSV, index=False)
    
    # Also save detailed per-class metrics
    detailed_csv = COMPARISON_DIR / "model_metrics_detailed.csv"
    detailed_data = []
    for metrics in all_metrics:
        for class_name, class_metrics in metrics["per_class_metrics"].items():
            row = {
                "Model": metrics["model_name"],
                "Class": class_name,
                "Precision": class_metrics["precision"],
                "Recall": class_metrics["recall"],
                "F1 Score": class_metrics["f1_score"],
                "Support": class_metrics["support"]
            }
            detailed_data.append(row)
    
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(detailed_csv, index=False)


def create_comparison_plots(all_metrics: List[Dict[str, Any]]) -> None:
    """
    Create comparison plots for all models
    """
    # Extract data
    models = [m["model_name"] for m in all_metrics]
    accuracies = [m["accuracy"] for m in all_metrics]
    precisions = [m["precision_macro"] for m in all_metrics]
    recalls = [m["recall_macro"] for m in all_metrics]
    f1_scores = [m["f1_macro"] for m in all_metrics]
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)[::-1]
    models = [models[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy
    axes[0, 0].barh(models, accuracies, color='skyblue')
    axes[0, 0].set_xlabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlim([0, 1])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # Precision
    axes[0, 1].barh(models, precisions, color='lightcoral')
    axes[0, 1].set_xlabel('Precision (Macro)', fontsize=12)
    axes[0, 1].set_title('Model Precision Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlim([0, 1])
    for i, v in enumerate(precisions):
        axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # Recall
    axes[1, 0].barh(models, recalls, color='lightgreen')
    axes[1, 0].set_xlabel('Recall (Macro)', fontsize=12)
    axes[1, 0].set_title('Model Recall Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlim([0, 1])
    for i, v in enumerate(recalls):
        axes[1, 0].text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # F1 Score
    axes[1, 1].barh(models, f1_scores, color='gold')
    axes[1, 1].set_xlabel('F1 Score (Macro)', fontsize=12)
    axes[1, 1].set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlim([0, 1])
    for i, v in enumerate(f1_scores):
        axes[1, 1].text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='skyblue')
    ax.bar(x - 0.5*width, precisions, width, label='Precision', color='lightcoral')
    ax.bar(x + 0.5*width, recalls, width, label='Recall', color='lightgreen')
    ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color='gold')
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison - All Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "model_comparison_grouped.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_comparison_report(all_metrics: List[Dict[str, Any]]) -> str:
    """
    Generate a text report summarizing model comparison
    """
    report_lines = ["=" * 80]
    report_lines.append("MODEL COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Sort by F1 score
    sorted_metrics = sorted(all_metrics, key=lambda x: x["f1_macro"], reverse=True)
    
    # Best model
    best = sorted_metrics[0]
    report_lines.append(f"🏆 BEST MODEL: {best['model_name']}")
    report_lines.append(f"   F1 Score (Macro): {best['f1_macro']:.4f}")
    report_lines.append(f"   Accuracy: {best['accuracy']:.4f}")
    report_lines.append("")
    
    # All models summary
    report_lines.append("MODELS RANKED BY F1 SCORE:")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Rank':<6} {'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    report_lines.append("-" * 80)
    
    for rank, m in enumerate(sorted_metrics, 1):
        report_lines.append(
            f"{rank:<6} {m['model_name']:<20} "
            f"{m['accuracy']:<12.4f} {m['precision_macro']:<12.4f} "
            f"{m['recall_macro']:<12.4f} {m['f1_macro']:<12.4f}"
        )
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_path = COMPARISON_DIR / "comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text


def load_saved_metrics() -> List[Dict[str, Any]]:
    """Load previously saved metrics from JSON file"""
    if METRICS_JSON.exists():
        with open(METRICS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []
