"""
Configuration file for AKI Prediction LSTM Training and Evaluation in Google Colab
"""

import os
import torch
from utils.helpers import set_seed

# =============================================================================
# BASIC CONFIGURATION
# =============================================================================

# Fixed Configuration
T_HOURS = 24  # Input hours
N_BINS = 20   # Number of bins
SEED = 0      # Random seed
MAX_LEN = 10000 # Max sequence length
EPOCHS = 50 # Max epochs (reduced for Colab)
BATCH_SIZE = 64  # Reduced batch size for Colab
EARLY_STOPPING_PATIENCE = 5

MODEL_TYPE = 'AKI' # AKI binary classification task  
DT = 1.0 # Time step (hours)
DYNAMIC = False # Predict only at end of sequence

# Fixed embedding dimension (word2vec embedding size)
LATENT_DIM = 128  # Always 128 for word2vec embeddings
WEIGHT_DECAY = 1e-05  # Fixed weight decay

# Optimization Metric Configuration
OPTIMIZATION_METRIC = 'AUPRC'  # Choose: 'AUPRC' (default for imbalanced) or 'AUROC'

# Progress Display Configuration
DISABLE_PROGRESS_BARS = False  # Enable for Colab interactive display

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Default hyperparameters for single model training
DEFAULT_CONFIG = {
    'HORIZON': 12,  # 12-hour prediction horizon
    'LR': 0.001,
    'LATENT_DIM': LATENT_DIM,
    'HIDDEN_DIM': 128,
    'P_DROPOUT': 0.1,
    'WEIGHT_DECAY': WEIGHT_DECAY,
    'WEIGHTED': True,  # Use weighted loss for imbalanced data
    'PRETRAINED_EMBEDDINGS': {
        'USE': False,  # Set to True if you have pre-trained embeddings
        'EMBEDDINGS_PATH': None,
        'VOCAB_PATH': None,
        'FREEZE': False
    },
    'UNFREEZE_EPOCH': None
}

# =============================================================================
# COLAB PATHS (will be set up in notebook)
# =============================================================================

# These will be set by the notebook based on Colab environment
DATA_PATH = '/sise/robertmo-group/Eldar/projects/AKI_prediction_lstm/data_24h_new'
MODELS_PATH = '/content/models'
RESULTS_PATH = '/sise/robertmo-group/Eldar/projects/AKI_prediction_lstm/results_24h'
EXAMPLE_DATA_PATH = '/content/example_data'

# =============================================================================
# DEVICE SETUP
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(SEED)

print(f"Configuration:")
print(f"  Device: {device}")
print(f"  Model Type: {MODEL_TYPE}")
print(f"  Input Hours: {T_HOURS}")
print(f"  Embedding Dimension: {LATENT_DIM}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Optimization Metric: {OPTIMIZATION_METRIC}")
print(f"  Progress Bars: {'Enabled' if not DISABLE_PROGRESS_BARS else 'Disabled'}")

# =============================================================================
# EXAMPLE DATA CONFIGURATION
# =============================================================================

# Configuration for example patient data
EXAMPLE_PATIENT_CONFIG = {
    'n_events': 50,  # Number of events for example patient
    'sequence_length': 48,  # 48 hours of data
    'has_aki': True,  # Whether example patient develops AKI
    'aki_onset_hour': 36  # Hour when AKI occurs (if has_aki=True)
} 