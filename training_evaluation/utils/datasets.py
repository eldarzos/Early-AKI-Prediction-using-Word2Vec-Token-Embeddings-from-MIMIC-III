"""Datasets module for loading pre-split EHR data."""

import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pickle


def get_improved_file_paths(data_root, horizon, input_hours=12, n_bins=20, seed=0, 
                           splits=['train', 'valid', 'test']):
    """
    Get file paths for improved preprocessing outputs with specific horizon.
    
    Parameters:
    -----------
    data_root : str
        Root directory containing processed data
    horizon : str
        Prediction horizon (e.g., "6h", "12h", "24h", "48h")
    input_hours : int
        Input time window in hours
    n_bins : int
        Number of quantization bins
    seed : int
        Random seed used in preprocessing
    splits : list
        List of splits to get paths for
        
    Returns:
    --------
    dict
        Dictionary with file paths for arrays, splits, and token map
    """
    
    # Directory paths
    arrays_dir = os.path.join(data_root, 'arrays_improved')
    splits_dir = os.path.join(data_root, 'splits_improved') 
    dict_dir = os.path.join(data_root, 'dictionaries')
    
    # Token map path (same for all horizons)
    token_map_path = os.path.join(dict_dir, f'{input_hours}_{seed}_{n_bins}-token2index.npy')
    
    # Array and split paths for each split
    paths = {
        'token_map': token_map_path,
        'arrays': {},
        'splits': {}
    }
    
    for split in splits:
        # Array file: e.g., "12_0_20-24h-train-arrays.npz"
        array_filename = f'{input_hours}_{seed}_{n_bins}-{horizon}-{split}-arrays.npz'
        paths['arrays'][split] = os.path.join(arrays_dir, array_filename)
        
        # Split file: e.g., "0-12-24h-train.csv"
        split_filename = f'{seed}-{input_hours}-{horizon}-{split}.csv'
        paths['splits'][split] = os.path.join(splits_dir, split_filename)
    
    return paths


def get_dataloaders_for_horizon(data_root, horizon, input_hours=12, n_bins=20, seed=0,
                               validation=True, t_hours=48, dt=1.0, dynamic=True,
                               shuffle=True, pin_memory=True, batch_size=128,
                               logger=logging.getLogger(__name__)):
    """
    Get DataLoaders for a specific prediction horizon using improved preprocessing outputs.
    
    This is a convenience function that automatically constructs the correct file paths
    for the improved preprocessing and calls the standard get_dataloaders function.
    
    Parameters:
    -----------
    data_root : str
        Root directory containing processed data
    horizon : str
        Prediction horizon (e.g., "6h", "12h", "24h", "48h")
    input_hours : int
        Input time window in hours (should match preprocessing)
    n_bins : int
        Number of quantization bins (should match preprocessing)
    seed : int
        Random seed (should match preprocessing)
    validation : bool
        If True, return train+valid loaders. If False, return test loader
    t_hours, dt, dynamic : int, float, bool
        Parameters for EHR dataset
    shuffle, pin_memory, batch_size : bool, bool, int
        DataLoader parameters
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    tuple
        (train_loader, valid_loader) if validation=True
        (test_loader, None) if validation=False
    """
    
    logger.info(f"Loading data for horizon: {horizon}")
    
    # Get file paths for this horizon
    paths = get_improved_file_paths(data_root, horizon, input_hours, n_bins, seed)
    
    # Check if improved preprocessing files exist
    if validation:
        required_files = [
            paths['token_map'],
            paths['arrays']['train'], 
            paths['arrays']['valid'],
            paths['splits']['train'],
            paths['splits']['valid']
        ]
    else:
        required_files = [
            paths['token_map'],
            paths['arrays']['test'],
            paths['splits']['test']
        ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logger.error(f"Missing improved preprocessing files for horizon {horizon}:")
        for f in missing_files:
            logger.error(f"  - {f}")
        raise FileNotFoundError(f"Missing files for horizon {horizon}. Run improved preprocessing first.")
    
    logger.info(f"✓ All required files found for horizon {horizon}")
    
    # Use horizon-specific array files directly
    if validation:
        # For validation, we need to combine train and valid arrays into single files
        # since get_dataloaders expects single array file + split files
        return get_dataloaders_improved_combined(
            paths, validation=True, t_hours=t_hours, dt=dt, dynamic=dynamic,
            shuffle=shuffle, pin_memory=pin_memory, batch_size=batch_size, logger=logger
        )
    else:
        # For testing, we can load the test array directly
        return get_dataloaders(
            array_path=paths['arrays']['test'],
            token_map_path=paths['token_map'],
            test_split_path=paths['splits']['test'],
            validation=False,
            t_hours=t_hours, dt=dt, dynamic=dynamic,
            shuffle=shuffle, pin_memory=pin_memory, batch_size=batch_size,
            logger=logger
        )


def get_dataloaders_improved_combined(paths, validation=True, **kwargs):
    """
    Helper function to combine separate train/valid arrays for horizon-specific loading.
    
    Since the improved preprocessing creates separate array files for each split,
    but the original get_dataloaders expects a single array file + split files,
    this function combines the arrays as needed.
    """
    logger = kwargs.get('logger', logging.getLogger(__name__))
    
    try:
        # Load separate array files
        train_arrays = np.load(paths['arrays']['train'], allow_pickle=True)
        valid_arrays = np.load(paths['arrays']['valid'], allow_pickle=True)
        
        # Combine arrays
        X_combined = np.concatenate([train_arrays['X'], valid_arrays['X']], axis=0)
        Y_combined = np.concatenate([train_arrays['Y'], valid_arrays['Y']], axis=0)
        paths_combined = np.concatenate([train_arrays['paths'], valid_arrays['paths']], axis=0)
        
        # Create temporary combined array file path
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_array_path = os.path.join(temp_dir, 'combined_arrays.npz')
        
        # Save combined arrays
        np.savez(temp_array_path, X=X_combined, Y=Y_combined, paths=paths_combined)
        
        logger.info(f"Combined arrays: X={X_combined.shape}, Y={Y_combined.shape}")
        
        # Use original get_dataloaders with combined arrays
        result = get_dataloaders(
            array_path=temp_array_path,
            token_map_path=paths['token_map'],
            train_split_path=paths['splits']['train'],
            valid_split_path=paths['splits']['valid'],
            validation=True,
            **kwargs
        )
        
        # Clean up temporary file
        os.remove(temp_array_path)
        os.rmdir(temp_dir)
        
        return result
        
    except Exception as e:
        logger.error(f"Error combining arrays: {e}")
        raise

class EHR(Dataset):
    """
    EHR Dataset - Stores pre-processed sequences and labels.

    Parameters
    ----------
    X: numpy.ndarray
        Array containing patient sequences for this split, shape (n_patients, max_len, 2)
    Y: numpy.ndarray
        Array containing patient outcomes for this split, shape (n_patients,)
    n_tokens_expected: int
        The expected vocabulary size (max token index + 1) based on the token map.
    t_hours: int, optional
        Time horizon (used for dynamic label tiling if applicable).
    dt: float, optional
        Time step between intervals (used for dynamic label tiling if applicable).
    dynamic: bool, optional
        Whether the model expects dynamically tiled labels.
    logger: logging.Logger, optional
    """
    def __init__(self, X, Y, n_tokens_expected, t_hours=48, dt=1.0, dynamic=True,
                 logger=logging.getLogger(__name__)):

        self.logger = logger
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_original = torch.tensor(Y, dtype=torch.float32) # Store original labels

        # --- Data Validation and Clamping ---
        if self.X.shape[0] > 0: # Only process if there's data
            token_indices = self.X[:, :, 1]
            # Handle potential NaNs introduced during padding or processing
            token_indices = torch.nan_to_num(token_indices, nan=0.0) # Replace NaN with 0 (padding index)

            max_id = int(token_indices.max().item())
            min_id = int(token_indices.min().item())
            logger.info(f"🔍 Initial Token ID range in loaded data: min={min_id}, max={max_id}")
            logger.info(f"  Expected vocabulary size (n_tokens): {n_tokens_expected}")

            # Clamp values to be within the valid range [0, n_tokens_expected - 1]
            # This prevents IndexError in the embedding layer.
            # Values outside this range indicate an issue in preprocessing/tokenization.
            if max_id >= n_tokens_expected or min_id < 0:
                logger.warning(f"⚠️ Token IDs found outside expected range [0, {n_tokens_expected-1}]. Clamping values.")
                self.X[:, :, 1] = torch.clamp(token_indices, min=0, max=n_tokens_expected - 1)
                # Log new range after clamping
                max_id_clamped = int(self.X[:, :, 1].max().item())
                min_id_clamped = int(self.X[:, :, 1].min().item())
                logger.info(f"  Clamped Token ID range: min={min_id_clamped}, max={max_id_clamped}")
            else:
                 self.X[:, :, 1] = token_indices # Assign back NaN-handled tensor
                 logger.info(f"  Token IDs are within the expected range.")
        else:
             logger.warning("⚠️ EHR Dataset initialized with zero samples.")


        # --- Dynamic Label Tiling ---
        if dynamic:
            # Tile the original Y label across time steps if needed by the model
            num_intervals = int(t_hours / dt)
            if self.Y_original.ndim == 1 and self.Y_original.shape[0] > 0:
                 # Reshape Y to (n_patients, 1) before tiling
                 self.Y = self.Y_original.unsqueeze(1).repeat(1, num_intervals)
            elif self.Y_original.shape[0] == 0:
                 # Handle empty case: create empty tensor with correct dimensions
                 self.Y = torch.empty((0, num_intervals), dtype=torch.float32)
            else:
                 # Y might already be tiled or have unexpected shape
                 logger.warning(f"Dynamic label tiling requested, but Y shape is {self.Y_original.shape}. Using original Y.")
                 self.Y = self.Y_original
        else:
            self.Y = self.Y_original # Use original labels if not dynamic

        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Returns the sequence and the (potentially tiled) label
        return self.X[idx], self.Y[idx]


def get_dataloaders(array_path, token_map_path, # Paths to data and token map
                    train_split_path=None, valid_split_path=None, test_split_path=None, # Paths to split files
                    validation=True, # If True, load train+valid; If False, load test
                    t_hours=48, dt=1.0, dynamic=True, # Dataset parameters
                    shuffle=True, pin_memory=True, batch_size=128, # DataLoader parameters
                    logger=logging.getLogger(__name__)):
    """
    Creates PyTorch DataLoaders based on pre-defined data splits.

    Parameters:
        array_path (str): Path to the .npz file containing 'X', 'Y', 'paths' arrays.
        token_map_path (str): Path to the .npy file containing the token2index map.
        train_split_path (str, optional): Path to train split CSV file. Required if validation=True.
        valid_split_path (str, optional): Path to validation split CSV file. Required if validation=True.
        test_split_path (str, optional): Path to test split CSV file. Required if validation=False.
        validation (bool): If True, return (train_loader, valid_loader). If False, return (test_loader, None).
        t_hours, dt, dynamic: Parameters passed to the EHR Dataset class.
        shuffle, pin_memory, batch_size: Parameters for the DataLoader.
        logger: Logger instance.

    Returns:
        tuple: (train_loader, valid_loader) or (test_loader, None)
    """
    pin_memory = pin_memory and torch.cuda.is_available()

    # --- Load Full Data Arrays and Token Map ---
    try:
        logger.info(f"Loading full dataset arrays from: {array_path}")
        # Use np.load directly for .npz file
        arrs = np.load(array_path, allow_pickle=True)
        X_all = arrs['X']
        Y_all = arrs['Y']
        Paths_all = arrs['paths'] # Relative paths from array creation step
        logger.info(f" Loaded data shapes: X={X_all.shape}, Y={Y_all.shape}, Paths={Paths_all.shape}")

        logger.info(f"Loading token map from: {token_map_path}")
        token2index = np.load(token_map_path, allow_pickle=True).item()
        n_tokens = len(token2index)
        logger.info(f" Token map loaded. Vocabulary size (n_tokens): {n_tokens}")

    except FileNotFoundError as e:
        logger.error(f"Error: Required file not found - {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data arrays or token map: {e}")
        raise

    # Create a mapping from relative path to index in the full arrays
    path_to_idx = {path: i for i, path in enumerate(Paths_all)}

    # --- Load Data Based on Splits ---
    if validation:
        if not train_split_path or not valid_split_path:
            raise ValueError("train_split_path and valid_split_path are required when validation=True")

        # Load train split
        try:
            train_df = pd.read_csv(train_split_path)
            train_paths = train_df['Paths'].tolist()
            # Get indices corresponding to train paths
            train_indices = [path_to_idx[p] for p in train_paths if p in path_to_idx]
            if len(train_indices) != len(train_paths):
                 logger.warning(f"Mismatch between paths in {train_split_path} and paths in {array_path}")
            X_train, Y_train = X_all[train_indices], Y_all[train_indices]
            logger.info(f"Loaded training set: {len(X_train)} samples based on {train_split_path}")
        except Exception as e:
            logger.error(f"Error loading training data based on {train_split_path}: {e}")
            raise

        # Load validation split
        try:
            valid_df = pd.read_csv(valid_split_path)
            valid_paths = valid_df['Paths'].tolist()
            # Get indices corresponding to validation paths
            valid_indices = [path_to_idx[p] for p in valid_paths if p in path_to_idx]
            if len(valid_indices) != len(valid_paths):
                 logger.warning(f"Mismatch between paths in {valid_split_path} and paths in {array_path}")
            X_valid, Y_valid = X_all[valid_indices], Y_all[valid_indices]
            logger.info(f"Loaded validation set: {len(X_valid)} samples based on {valid_split_path}")
        except Exception as e:
            logger.error(f"Error loading validation data based on {valid_split_path}: {e}")
            raise

        # Create Datasets
        train_dataset = EHR(X_train, Y_train, n_tokens, t_hours, dt, dynamic, logger)
        valid_dataset = EHR(X_valid, Y_valid, n_tokens, t_hours, dt, dynamic, logger)

        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      pin_memory=pin_memory,
                                      num_workers=4 if pin_memory else 0) # Use workers if pinning memory
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=False, # No shuffle for validation
                                      pin_memory=pin_memory,
                                      num_workers=4 if pin_memory else 0)

        return train_dataloader, valid_dataloader

    else: # Load test set
        if not test_split_path:
            raise ValueError("test_split_path is required when validation=False")

        # Load test split
        try:
            test_df = pd.read_csv(test_split_path)
            test_paths = test_df['Paths'].tolist()
            # Get indices corresponding to test paths
            test_indices = [path_to_idx[p] for p in test_paths if p in path_to_idx]
            if len(test_indices) != len(test_paths):
                 logger.warning(f"Mismatch between paths in {test_split_path} and paths in {array_path}")
            X_test, Y_test = X_all[test_indices], Y_all[test_indices]
            logger.info(f"Loaded test set: {len(X_test)} samples based on {test_split_path}")
        except Exception as e:
            logger.error(f"Error loading test data based on {test_split_path}: {e}")
            raise

        # Create Dataset
        test_dataset = EHR(X_test, Y_test, n_tokens, t_hours, dt, dynamic, logger)

        # Create DataLoader
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False, # No shuffle for test
                                     pin_memory=pin_memory,
                                     num_workers=4 if pin_memory else 0)

        return test_dataloader, None # Return None for the second value (no valid_loader)

def get_dataloaders_from_pkl(array_path, token_map_path,
                             train_split_path=None, valid_split_path=None, test_split_path=None,
                             validation=True, t_hours=48, dt=1.0, dynamic=True,
                             shuffle=True, pin_memory=True, batch_size=128,
                             logger=logging.getLogger(__name__)):
    """
    Creates PyTorch DataLoaders from .pkl array files and pre-defined data splits.
    This is an adaptation of get_dataloaders for use with pickled data.
    """
    pin_memory = pin_memory and torch.cuda.is_available()

    # --- Load Full Data Arrays and Token Map ---
    try:
        logger.info(f"Loading full dataset arrays from pickle file: {array_path}")
        # Load from .pkl file
        with open(array_path, 'rb') as f:
            arrs = pickle.load(f)
        
        X_all = arrs['X']
        Y_all = arrs['Y']
        Paths_all = arrs['paths']
        logger.info(f" Loaded data shapes: X={X_all.shape}, Y={Y_all.shape}, Paths={Paths_all.shape}")

        logger.info(f"Loading token map from: {token_map_path}")
        token2index = np.load(token_map_path, allow_pickle=True).item()
        n_tokens = len(token2index)
        logger.info(f" Token map loaded. Vocabulary size (n_tokens): {n_tokens}")

    except FileNotFoundError as e:
        logger.error(f"Error: Required file not found - {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data arrays or token map: {e}")
        raise

    # Create a mapping from relative path to index in the full arrays
    path_to_idx = {path: i for i, path in enumerate(Paths_all)}

    # --- Load Data Based on Splits ---
    if validation:
        if not train_split_path or not valid_split_path:
            raise ValueError("train_split_path and valid_split_path are required when validation=True")

        train_df = pd.read_csv(train_split_path)
        train_paths = train_df['subject_id'].astype(str).tolist() # Assuming subject_id maps to paths
        train_indices = [path_to_idx[p] for p in train_paths if p in path_to_idx]
        X_train, Y_train = X_all[train_indices], Y_all[train_indices]
        logger.info(f"Loaded training set: {len(X_train)} samples from {train_split_path}")

        valid_df = pd.read_csv(valid_split_path)
        valid_paths = valid_df['subject_id'].astype(str).tolist()
        valid_indices = [path_to_idx[p] for p in valid_paths if p in path_to_idx]
        X_valid, Y_valid = X_all[valid_indices], Y_all[valid_indices]
        logger.info(f"Loaded validation set: {len(X_valid)} samples from {valid_split_path}")

        train_dataset = EHR(X_train, Y_train, n_tokens, t_hours, dt, dynamic, logger)
        valid_dataset = EHR(X_valid, Y_valid, n_tokens, t_hours, dt, dynamic, logger)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=4 if pin_memory else 0)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=4 if pin_memory else 0)

        return train_dataloader, valid_dataloader

    else: # Load test set
        if not test_split_path:
            raise ValueError("test_split_path is required when validation=False")

        test_df = pd.read_csv(test_split_path)
        test_paths = test_df['subject_id'].astype(str).tolist()
        test_indices = [path_to_idx[p] for p in test_paths if p in path_to_idx]
        X_test, Y_test = X_all[test_indices], Y_all[test_indices]
        logger.info(f"Loaded test set: {len(X_test)} samples from {test_split_path}")

        test_dataset = EHR(X_test, Y_test, n_tokens, t_hours, dt, dynamic, logger)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=4 if pin_memory else 0)

        return test_dataloader, None