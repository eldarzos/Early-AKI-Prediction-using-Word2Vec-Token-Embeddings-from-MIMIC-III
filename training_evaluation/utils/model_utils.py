# This file will contain model loading and configuration utility functions.

import torch
import torch.nn as nn
import json
import os
import sys

# Try to import project modules with fallbacks
try:
    from .helpers import get_n_param
    print("✅ Successfully imported get_n_param from helpers")
except ImportError:
    try:
        from helpers import get_n_param
        print("✅ Successfully imported get_n_param from helpers (absolute import)")
    except ImportError:
        print("⚠️ Could not import get_n_param, using fallback")
        def get_n_param(model): 
            if hasattr(model, 'parameters'):
                return sum(p.numel() for p in model.parameters())
            return 0

try:
    from models.models import init_model
    print("✅ Successfully imported init_model from models")
except ImportError:
    print("⚠️ Could not import init_model from models, using dummy function")
    def init_model(model_type, n_tokens, latent_dim, hidden_dim, p_dropout, weighted, dynamic):
        print("❌ Dummy init_model called - actual model loading will fail")
        return nn.Module()

try:
    from .data_utils import load_token_mapping
    print("✅ Successfully imported load_token_mapping from data_utils")
except ImportError:
    try:
        from data_utils import load_token_mapping
        print("✅ Successfully imported load_token_mapping from data_utils (absolute import)")
    except ImportError:
        print("⚠️ Could not import load_token_mapping, defining fallback")
        def load_token_mapping(use_pretrained=True, vocab_path=None):
            # Fallback token mapping function
            import numpy as np
            token_map_path = vocab_path if use_pretrained and vocab_path else '/content/word2vec_embeddings/word2vec_token2idx_10epoches.npy'
            print(f"📚 Loading token mapping from: {token_map_path}")
            
            if os.path.exists(token_map_path):
                token_data = np.load(token_map_path, allow_pickle=True)
                if isinstance(token_data, np.ndarray) and token_data.dtype == object:
                    token2index = token_data.item()
                else:
                    token2index = token_data
                print(f"✅ Loaded token mapping with {len(token2index)} tokens")
                return token2index
            else:
                raise FileNotFoundError(f"Token mapping not found at {token_map_path}")

# Config class for device handling
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

def load_model_and_config(model_dir):
    """Load trained model and its configuration (following training code logic)"""
    # Load configuration
    config_path = os.path.join(model_dir, 'combination_config.json')
    print(f"📂 Loading config from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Load final results if available
    results_path = os.path.join(model_dir, 'final_results.json')
    print(f"📂 Looking for results at: {results_path}")
    final_results = None
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            final_results = json.load(f)

    # Extract configuration details (same as training code)
    combination_config = model_config.get('configuration', {})
    fixed_config = model_config.get('fixed_config', {})

    # Load token mapping to get n_tokens
    try:
        if combination_config.get('PRETRAINED_EMBEDDINGS', {}).get('USE', False):
            # Use the load_token_mapping function
            token2index = load_token_mapping(use_pretrained=True, vocab_path='/content/word2vec_embeddings/word2vec_token2idx_10epoches.npy')
        else:
            # For non-pretrained, try to use a placeholder or default
            print("⚠️ Non-pretrained embeddings not fully supported in this Colab version")
            token2index = load_token_mapping(use_pretrained=True, vocab_path='/content/word2vec_embeddings/word2vec_token2idx_10epoches.npy')

        n_tokens = len(token2index)
        print(f"📚 Loaded vocabulary with {n_tokens} tokens")
    except Exception as e:
        print(f"⚠️ Could not load token mapping: {e}")
        # Use default if token mapping fails
        n_tokens = 10000
        print(f"⚠️ Using default n_tokens: {n_tokens}")

    # Initialize model architecture (exactly like in training code)
    try:
        print("🏗️ Initializing model architecture...")
        model = init_model(
            model_type=fixed_config.get('MODEL_TYPE', 'Mortality'),
            n_tokens=n_tokens,
            latent_dim=combination_config.get('LATENT_DIM', 128),
            hidden_dim=combination_config.get('HIDDEN_DIM', 128),
            p_dropout=combination_config.get('P_DROPOUT', 0.1),
            weighted=combination_config.get('WEIGHTED', True),
            dynamic=fixed_config.get('DYNAMIC', False)
        ).to(config.DEVICE)

        print(f"✅ Model architecture initialized: {model.__class__.__name__}")
        print(f"🔧 Model parameters: {get_n_param(model):,}")

    except Exception as e:
        print(f"❌ Error initializing model architecture: {e}")
        print("🔄 Attempting fallback model loading...")
        raise

    # Load model weights
    model_path = os.path.join(model_dir, 'best_model.pt')
    print(f"📂 Loading model weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Load the saved model data
        loaded_data = torch.load(model_path, map_location=config.DEVICE)
        print(f"🔍 Loaded object type: {type(loaded_data)}")

        if hasattr(loaded_data, 'state_dict'):
            # It's a complete model object
            model.load_state_dict(loaded_data.state_dict())
            print("✅ Loaded from complete model object")
        elif isinstance(loaded_data, dict) and any('weight' in k or 'bias' in k for k in loaded_data.keys()):
            # It's a state dict
            model.load_state_dict(loaded_data)
            print("✅ Loaded from state dict")
        elif hasattr(loaded_data, 'eval'):
            # It's already a complete model, use it directly
            model = loaded_data.to(config.DEVICE)
            print("✅ Used complete model object directly")
        else:
            print(f"⚠️ Unknown model format: {type(loaded_data)}")
            print("🔄 Attempting direct state dict load...")
            model.load_state_dict(loaded_data)

        # Set model to evaluation mode
        model.eval()
        print("🎯 Model set to evaluation mode")

    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        raise

    print(f"✅ Successfully loaded model from: {model_dir}")
    print(f"📋 Configuration: {model_config.get('combination_name', 'Unknown')}")

    if final_results:
        print(f"📊 Previous test AUROC: {final_results['test_metrics']['auroc']:.4f}")
        print(f"📊 Previous test AUPRC: {final_results['test_metrics']['auprc']:.4f}")

    return model, model_config, final_results

def display_model_info(model, model_config):
    """Display comprehensive model information"""
    print("="*60)
    print("🏥 MODEL INFORMATION")
    print("="*60)

    # Configuration details
    combination_config = model_config.get('configuration', {})

    print(f"📋 Combination Name: {model_config.get('combination_name', 'Unknown')}")
    print(f"🔢 Combination Index: {model_config.get('combination_idx', 'Unknown')}")
    print(f"📅 Timestamp: {model_config.get('timestamp', 'Unknown')}")

    print(f"\n🎯 HYPERPARAMETERS:")
    print(f"   • Horizon: {combination_config.get('HORIZON', 'Unknown')}h")
    print(f"   • Learning Rate: {combination_config.get('LR', 'Unknown')}")
    print(f"   • Hidden Dimension: {combination_config.get('HIDDEN_DIM', 'Unknown')}")
    print(f"   • Dropout: {combination_config.get('P_DROPOUT', 'Unknown')}")
    print(f"   • Weighted Loss: {combination_config.get('WEIGHTED', 'Unknown')}")

    print(f"\n🧠 EMBEDDING CONFIGURATION:")
    emb_config = combination_config.get('PRETRAINED_EMBEDDINGS', {})
    print(f"   • Use Pre-trained: {emb_config.get('USE', 'Unknown')}")
    print(f"   • Freeze Embeddings: {emb_config.get('FREEZE', 'Unknown')}")
    print(f"   • Unfreeze Epoch: {combination_config.get('UNFREEZE_EPOCH', 'Never')}")

    # Model architecture - safely handle different model formats
    print(f"\n🏗️ MODEL ARCHITECTURE:")

    try:
        if hasattr(model, 'parameters') and callable(getattr(model, 'parameters')):
            # It's a proper PyTorch model
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"   • Total Parameters: {total_params:,}")
            print(f"   • Trainable Parameters: {trainable_params:,}")
            print(f"   • Model Type: {model.__class__.__name__}")
        elif isinstance(model, dict):
            # It's a state dict
            total_params = sum(param.numel() for param in model.values() if isinstance(param, torch.Tensor))
            print(f"   • Total Parameters: {total_params:,}")
            print(f"   • Model Type: State Dict (dict)")
            print(f"   • State Dict Keys: {len(model)} layers")
        else:
            print(f"   • Model Type: {type(model)}")
            print(f"   • Cannot determine parameter count for this model format")

    except Exception as e:
        print(f"   • Error analyzing model architecture: {e}")
        print(f"   • Model Type: {type(model)}")

    print("="*60)
