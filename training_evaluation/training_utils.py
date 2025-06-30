"""
Training and Evaluation Utilities for AKI Prediction LSTM in Google Colab
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score
from datetime import datetime

# Import project modules
from utils.modelIO import save_model
from utils.helpers import get_n_param, array
from models.losses import BCE


def create_example_data(n_samples=100, n_tokens=1000, sequence_length=48, has_aki_ratio=0.1):
    """Create synthetic example data for demonstration."""
    print(f"Creating example data: {n_samples} patients, {sequence_length}h sequences...")
    
    # Create synthetic token vocabulary
    token2index = {'<PAD>': 0, '<UNK>': 1}
    
    # Add medical event tokens
    medical_events = ['heart_rate_normal', 'heart_rate_high', 'creatinine_normal', 'creatinine_elevated']
    
    for event in medical_events:
        for i in range(5):
            token2index[f"{event}_bin_{i}"] = len(token2index)
    
    # Pad vocabulary to desired size
    while len(token2index) < n_tokens:
        token2index[f"token_{len(token2index)}"] = len(token2index)
    
    # Create synthetic data
    X = torch.zeros(n_samples, sequence_length, 2)
    Y = torch.zeros(n_samples)
    
    for i in range(n_samples):
        X[i, :, 0] = torch.arange(sequence_length, dtype=torch.float32)
        
        if i < int(n_samples * has_aki_ratio):
            Y[i] = 1.0
            for t in range(sequence_length):
                if t < sequence_length * 0.7:
                    token_id = np.random.choice([2, 3, 8, 9])
                else:
                    token_id = np.random.choice([16, 17, 20, 21])
                X[i, t, 1] = token_id
        else:
            Y[i] = 0.0
            for t in range(sequence_length):
                token_id = np.random.choice([2, 3, 8, 9])
                X[i, t, 1] = token_id
    
    print(f"✅ Created example data: X={X.shape}, Y={Y.shape}, vocab={len(token2index)}")
    return X, Y, token2index


def create_dataloaders(X, Y, batch_size=64):
    """Create train/validation/test data loaders."""
    from torch.utils.data import TensorDataset, DataLoader
    
    n_samples = X.shape[0]
    indices = torch.randperm(n_samples)
    
    train_end = int(0.7 * n_samples)
    val_end = train_end + int(0.15 * n_samples)
    
    splits = {
        'train': indices[:train_end],
        'valid': indices[train_end:val_end],
        'test': indices[val_end:]
    }
    
    dataloaders = {}
    for split, split_indices in splits.items():
        dataset = TensorDataset(X[split_indices], Y[split_indices])
        dataloaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=0
        )
        print(f"Created {split} dataloader: {len(dataset)} samples")
    
    return dataloaders


def train_model(model, dataloaders, config, device, save_dir=None):
    """Train a single model."""
    print(f"🚀 Starting training on {device}")
    print(f"📊 Model parameters: {get_n_param(model):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
    loss_f = BCE()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_metric = 0.
    patience_counter = 0
    train_history = []
    
    for epoch in range(1, config.get('EPOCHS', 50) + 1):
        # Training phase
        model.train()
        train_loss = 0.
        
        for X, Y in tqdm(dataloaders['train'], desc=f"Epoch {epoch}", leave=False):
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            output = model(X)
            loss = loss_f(output, Y, is_train=True, storer=defaultdict(list))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(dataloaders['train'])
        
        # Validation phase
        val_metrics, _, _ = run_evaluation(model, dataloaders['valid'], loss_f, device)
        
        val_auroc = val_metrics.get('auroc', 0.0) or 0.0
        val_auprc = val_metrics.get('auprc', 0.0) or 0.0
        
        current_val_metric = val_auprc if config.get('OPTIMIZATION_METRIC', 'AUPRC') == 'AUPRC' else val_auroc
        scheduler.step(current_val_metric)
        
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Valid AUROC: {val_auroc:.4f} | Valid AUPRC: {val_auprc:.4f}")
        
        train_history.append({
            'epoch': epoch, 'train_loss': avg_train_loss, 'val_auroc': val_auroc, 'val_auprc': val_auprc
        })
        
        # Early stopping
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            patience_counter = 0
            if save_dir:
                save_model(model, save_dir, filename='best_model.pt')
            print(f"✅ New best model! Metric: {best_val_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.get('EARLY_STOPPING_PATIENCE', 5):
                print(f"⏹️ Early stopping at epoch {epoch}")
                break
    
    # Final test evaluation
    test_metrics, y_pred, y_prob = comprehensive_evaluation(model, dataloaders['test'], device)
    
    return {
        'best_val_metric': best_val_metric,
        'train_history': train_history,
        'test_metrics': test_metrics,
        'config': config
    }


def run_evaluation(model, loader, loss_f, device, phase='valid'):
    """Run evaluation with metrics."""
    model.eval()
    epoch_loss = 0.
    y_preds_list = []
    y_trues_list = []

    with torch.no_grad():
        for data, y_true in loader:
            data, y_true = data.to(device), y_true.to(device)
            y_pred = model(data)
            loss = loss_f(y_pred, y_true, is_train=False, storer=defaultdict(list))
            
            epoch_loss += loss.item() if torch.isfinite(loss) else 0
            y_preds_list.append(y_pred.detach().cpu())
            y_trues_list.append(y_true.detach().cpu())

    avg_loss = epoch_loss / len(loader)
    final_metrics = {'loss': avg_loss, 'auroc': None, 'auprc': None}
    
    try:
        y_preds_all = torch.cat(y_preds_list, dim=0)
        y_trues_all = torch.cat(y_trues_list, dim=0)
        
        y_pred_np = array(y_preds_all)
        y_true_np = array(y_trues_all)
        
        if y_pred_np.ndim == 2: 
            y_pred_np = y_pred_np[:, -1]
        if y_true_np.ndim == 2: 
            y_true_np = y_true_np[:, -1]
            
        probs = 1 / (1 + np.exp(-y_pred_np))
        
        if len(np.unique(y_true_np)) > 1:
            final_metrics['auroc'] = roc_auc_score(y_true_np, probs)
            final_metrics['auprc'] = average_precision_score(y_true_np, probs)
    except:
        pass

    return final_metrics, y_preds_all, y_trues_all


def comprehensive_evaluation(model, test_loader, device):
    """Comprehensive evaluation on test set."""
    model.eval()
    all_probabilities = []
    all_true_labels = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            if outputs.dim() == 2:
                outputs = outputs[:, -1]
            if targets.dim() == 2:
                targets = targets[:, -1]
            
            probabilities = torch.sigmoid(outputs)
            
            all_probabilities.extend(probabilities.detach().cpu().numpy())
            all_true_labels.extend(targets.detach().cpu().numpy())
    
    y_true = np.array(all_true_labels)
    y_prob = np.array(all_probabilities)
    y_pred = (y_prob > 0.5).astype(int)
    
    test_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob)
    }
    
    print(f"✅ Test Results:")
    for metric, value in test_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    return test_metrics, y_pred, y_prob
