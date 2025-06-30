# This file will contain evaluation and visualization utility functions.

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

def comprehensive_evaluation(model, test_loader, device):
    """
    Perform comprehensive evaluation on test set (following training code logic)
    Returns all metrics and predictions
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_true_labels = []

    print("\n🔍 Running comprehensive test evaluation...")

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Evaluating on test set"):
            data, targets = data.to(device), targets.to(device)

            # Get model outputs (logits)
            outputs = model(data)

            # Handle different output shapes
            if outputs.dim() == 2:
                outputs = outputs[:, -1]  # Take last time step for sequences
            if targets.dim() == 2:
                targets = targets[:, -1]  # Take last time step for sequences

            # Convert to probabilities using sigmoid
            probabilities = torch.sigmoid(outputs)

            # Get binary predictions
            predictions = (probabilities > 0.5).float()

            # Store results
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_probabilities.extend(probabilities.detach().cpu().numpy())
            all_true_labels.extend(targets.detach().cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    return y_true, y_pred, y_prob

def calculate_all_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob)
    }

    return metrics

def display_metrics_table(metrics):
    """Display metrics in a formatted table"""
    print("\n📊 COMPREHENSIVE TEST METRICS")
    print("="*50)

    # Create DataFrame for better display
    metrics_df = pd.DataFrame([
        ['Accuracy', f"{metrics['accuracy']:.4f}"],
        ['Balanced Accuracy', f"{metrics['balanced_accuracy']:.4f}"],
        ['Precision', f"{metrics['precision']:.4f}"],
        ['Recall (Sensitivity)', f"{metrics['recall']:.4f}"],
        ['F1-Score', f"{metrics['f1_score']:.4f}"],
        ['AUROC', f"{metrics['auroc']:.4f}"],
        ['AUPRC', f"{metrics['auprc']:.4f}"]
    ], columns=['Metric', 'Value'])
    print(metrics_df.to_string(index=False))
    print("="*50)

    # Calculate dataset imbalance info
    return metrics_df

def create_comprehensive_plots(y_true, y_pred, y_prob, metrics, model_config):
    """Create comprehensive visualization plots"""
    fig = plt.figure(figsize=(20, 15))

    combination_name = model_config.get('combination_name', 'Unknown Model')

    # 1. Confusion Matrix
    plt.subplot(3, 4, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No AKI', 'AKI'],
                yticklabels=['No AKI', 'AKI'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix\\n{combination_name}', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 2. ROC Curve
    plt.subplot(3, 4, 2)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {metrics["auroc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    plt.subplot(3, 4, 3)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2,
             label=f'PR (AUC = {metrics["auprc"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve', fontsize=12, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # 4. Metrics Bar Chart
    plt.subplot(3, 4, 4)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC', 'AUPRC']
    metrics_values = [metrics['accuracy'], metrics['precision'],
                     metrics['recall'], metrics['f1_score'],
                     metrics['auroc'], metrics['auprc']]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange', 'purple']
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
    plt.ylabel('Score')
    plt.title(f'Classification Metrics', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    # 5. Prediction Distribution
    plt.subplot(3, 4, 5)
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='No AKI', color='lightblue', density=True)
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='AKI', color='lightcoral', density=True)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, label='Threshold (0.5)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(f'Prediction Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Dataset Statistics
    plt.subplot(3, 4, 6)
    n_total = len(y_true)
    n_positive = np.sum(y_true)
    n_negative = n_total - n_positive

    sizes = [n_negative, n_positive]
    labels = [f'No AKI\\n({n_negative:,})', f'AKI\\n({n_positive:,})']
    colors = ['lightblue', 'lightcoral']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Dataset Distribution', fontsize=12, fontweight='bold')

    # 7. Performance vs. Different Thresholds
    plt.subplot(3, 4, 7)
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1s = [], [], []

    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        if len(np.unique(y_pred_thresh)) > 1:
            prec = precision_score(y_true, y_pred_thresh, zero_division=0)
            rec = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1_thresh = f1_score(y_true, y_pred_thresh, zero_division=0)
        else:
            prec, rec, f1_thresh = 0, 0, 0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1_thresh)

    plt.plot(thresholds, precisions, label='Precision', color='blue', alpha=0.8)
    plt.plot(thresholds, recalls, label='Recall', color='green', alpha=0.8)
    plt.plot(thresholds, f1s, label='F1-Score', color='red', alpha=0.8)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Used Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Metrics vs. Threshold', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. Clinical Insights (Text Summary)
    plt.subplot(3, 4, 8)
    plt.axis('off')

    # Calculate additional clinical metrics
    # Need confusion matrix again for specificity and npv if not passed directly
    cm = confusion_matrix(y_true, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    sensitivity = metrics['recall']
    ppv = metrics['precision']
    npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0

    n_total = len(y_true)
    n_positive = np.sum(y_true)

    clinical_text = f'''🏥 CLINICAL PERFORMANCE

📊 Key Metrics:
• Sensitivity: {sensitivity:.3f}
• Specificity: {specificity:.3f}
• PPV: {ppv:.3f}
• NPV: {npv:.3f}

📈 Dataset Info:
• Total Patients: {n_total:,}
• AKI Cases: {n_positive:,} ({n_positive/n_total:.1%})
• Imbalance Ratio: {(n_total - n_positive)/n_positive:.1f}:1

🎯 Model Performance:
• AUROC: {metrics["auroc"]:.4f}
• AUPRC: {metrics["auprc"]:.4f}
• Balanced Acc: {metrics["balanced_accuracy"]:.4f}
'''

    plt.text(0.05, 0.95, clinical_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.show()

    return fig

def generate_research_findings(metrics, model_config, y_true, y_prob):
    '''Generate comprehensive research findings'''

    print("="*80)
    print("🔬 COMPREHENSIVE RESEARCH FINDINGS")
    print("="*80)

    # Extract configuration details
    config_details = model_config.get('configuration', {})
    combination_name = model_config.get('combination_name', 'Unknown')

    print(f"\\n📋 EVALUATED MODEL CONFIGURATION:")
    print(f"   • Model Name: {combination_name}")
    print(f"   • Prediction Horizon: {config_details.get('HORIZON', 'Unknown')}h")
    print(f"   • Architecture: LSTM with Attention Mechanism")
    print(f"   • Pre-trained Embeddings: {config_details.get('PRETRAINED_EMBEDDINGS', {}).get('USE', 'Unknown')}")
    print(f"   • Weighted Loss: {config_details.get('WEIGHTED', 'Unknown')}")

    print(f"\\n🎯 PERFORMANCE ANALYSIS:")
    print(f"   • AUROC: {metrics['auroc']:.4f}")
    print(f"   • AUPRC: {metrics['auprc']:.4f} (Critical for imbalanced datasets)")
    print(f"   • Sensitivity/Recall: {metrics['recall']:.4f} (Ability to detect AKI)")

    # Calculate specificity - requires confusion matrix or similar
    # Using a placeholder calculation based on AUROC if CM is not available
    # This is an approximation and should ideally use the confusion matrix
    specificity_approx = 1 - (metrics['recall']/(metrics['auroc']*2 - 1)) if metrics['auroc'] > 0.5 and (metrics['auroc']*2 - 1) != 0 else 0
    # If confusion matrix is available, use it
    try:
        # Need y_pred for confusion matrix, which is not passed here.
        # Re-calculate or assume y_pred can be derived from y_prob (using 0.5 threshold)
        y_pred = (y_prob > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        print(f"   • Specificity: {specificity:.4f}")
    except Exception as e:
        print(f"   • Specificity (Approx): {specificity_approx:.4f} (Could not calculate exact specificity: {e})")


    print(f"   • Precision: {metrics['precision']:.4f} (Positive predictive value)")

    # Clinical interpretation
    print(f"\\n🏥 CLINICAL INTERPRETATION:")

    # AUROC interpretation
    if metrics['auroc'] >= 0.9:
        auroc_interpretation = "Excellent discrimination ability"
    elif metrics['auroc'] >= 0.8:
        auroc_interpretation = "Good discrimination ability"
    elif metrics['auroc'] >= 0.7:
        auroc_interpretation = "Fair discrimination ability"
    else:
        auroc_interpretation = "Poor discrimination ability"

    print(f"   • AUROC ({metrics['auroc']:.3f}): {auroc_interpretation}")

    # AUPRC interpretation (more important for imbalanced datasets)
    baseline_auprc = np.mean(y_true)  # Random classifier performance
    auprc_improvement = (metrics['auprc'] - baseline_auprc) / baseline_auprc * 100 if baseline_auprc > 0 else float('inf')

    print(f"   • AUPRC ({metrics['auprc']:.3f}): {auprc_improvement:.1f}% improvement over random")

    # Sensitivity analysis
    if metrics['recall'] >= 0.9:
        sensitivity_interpretation = "Excellent - Catches most AKI cases"
    elif metrics['recall'] >= 0.8:
        sensitivity_interpretation = "Good - Catches most AKI cases with few false negatives"
    elif metrics['recall'] >= 0.7:
        sensitivity_interpretation = "Moderate - May miss some AKI cases"
    else:
        sensitivity_interpretation = "Poor - Misses many AKI cases (high false negative rate)"

    print(f"   • Sensitivity ({metrics['recall']:.3f}): {sensitivity_interpretation}")

    # Precision analysis
    if metrics['precision'] >= 0.5:
        precision_interpretation = "Good - Low false positive rate"
    elif metrics['precision'] >= 0.3:
        precision_interpretation = "Moderate - Some false positives expected"
    else:
        precision_interpretation = "Poor - High false positive rate"

    print(f"   • Precision ({metrics['precision']:.3f}): {precision_interpretation}")

    print(f"\\n💡 KEY INSIGHTS:")

    # Pre-trained embeddings impact
    if config_details.get('PRETRAINED_EMBEDDINGS', {}).get('USE', False):
        print(f"   ✅ Pre-trained medical embeddings are being utilized")
        print(f"      → Leverages existing medical knowledge for better representation learning")

        if config_details.get('PRETRAINED_EMBEDDINGS', {}).get('FREEZE', False):
            unfreeze_epoch = config_details.get('UNFREEZE_EPOCH', None)
            if unfreeze_epoch:
                print(f"      → Progressive unfreezing strategy (unfreeze at epoch {unfreeze_epoch})")
                print(f"      → Balances knowledge preservation with task adaptation")
            else:
                print(f"      → Frozen embeddings preserve pre-trained medical knowledge")
        else:
            print(f"      → Fine-tuned embeddings adapt to AKI prediction task")
    else:
        print(f"   ℹ️ Using randomly initialized embeddings")
        print(f"      → Learning medical representations from scratch")

    # Weighted loss impact
    if config_details.get('WEIGHTED', False):
        print(f"   ⚖️ Weighted loss function addresses class imbalance")
        print(f"      → Gives higher importance to rare AKI cases")
        print(f"      → Critical for clinical deployment where missing AKI is costly")
    else:
        print(f"   ℹ️ Standard (unweighted) loss function used")
        print(f"      → May struggle with severe class imbalance")

    # Performance context
    print(f"\\n📊 PERFORMANCE IN CONTEXT:")
    n_total = len(y_true)
    n_positive = np.sum(y_true)
    imbalance_ratio = (n_total - n_positive) / n_positive if n_positive > 0 else float('inf')

    print(f"   • Dataset: {n_total:,} patients ({n_positive:,} AKI cases, {imbalance_ratio:.1f}:1 imbalance)")
    print(f"   • Challenge: Severe class imbalance makes this a difficult prediction task")
    print(f"   • Clinical Impact: Early AKI prediction can enable preventive interventions")

    print("="*80)
