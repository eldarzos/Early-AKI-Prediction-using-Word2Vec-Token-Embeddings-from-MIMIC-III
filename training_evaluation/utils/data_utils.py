# This file will contain data loading and preprocessing utility functions.

import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader

# Assuming config is available or passed as argument
# from .config import config # If config is in a separate file and needs importing

def load_test_arrays(array_test_path):
    """Load test data arrays for a specific horizon"""
    arrays = {}
    print(f"📂 Loading test arrays from: {array_test_path}")
    data = np.load(array_test_path, allow_pickle=True)
    arrays['test'] = {
        'X': data['X'],
        'Y': data['Y'],
        'paths': data['paths']
    }
    print(f"✅ Loaded test: X={data['X'].shape}, Y={data['Y'].shape}")

    return arrays

def load_token_mapping(use_pretrained=True, vocab_path=None):
    """Load token to index mapping"""
    # Assuming config is available or passed as argument
    # if use_pretrained and vocab_path:
    #     token_map_path = vocab_path
    #     print(f"📚 Loading PRE-TRAINED token mapping from: {token_map_path}")
    # else:
    #     token_map_path = os.path.join(config.DICTIONARIES_DIR, f'{config.T_HOURS}_{config.SEED}_{config.N_BINS}-token2index.npy')
    #     print(f"📚 Loading ORIGINAL token mapping from: {token_map_path}")

    # Use the hardcoded path from the notebook for now
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


def create_test_dataloader(arrays, batch_size=256):
    """Create DataLoader for test set"""
    if arrays['test'] is not None:
        # Don't move data to device here - let the training loop handle it
        X = torch.FloatTensor(arrays['test']['X'])
        Y = torch.FloatTensor(arrays['test']['Y'])

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"✅ Created test dataloader with {len(dataset)} samples")
        return dataloader
    else:
        raise ValueError("No test data available")
