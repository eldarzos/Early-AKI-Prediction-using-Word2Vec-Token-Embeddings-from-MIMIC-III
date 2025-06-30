"""
Step 7: Quantize Events

Quantize continuous events using the value dictionary for a specific input window.

Applies tokenization to train, validation, and test sets based on rules:
- Unknown ITEMID_UOM -> '<UNK>'
- Continuous variable (sufficient unique values) -> 'ITEMID_UOM:bin_index'
- Discrete variable or continuous w/ few values -> 'ITEMID_UOM:value'

Saves the calculated bin boundaries and the token-to-index mapping.
Overwrites the truncated timeseries files with tokenized data.

Usage:
    python 07_quantize_events.py [--input_hours 24] [--n_bins 20] [--seed 0]
    
Make sure to run Steps 1-6 first.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import sys
import logging

# Add config to path
sys.path.append('../config')
from config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def quantize_events(root_dir, input_hours, n_bins, seed):
    """Quantizes events based on the value dictionary and data splits for input_hours."""

    # --- 1. Load Value Dictionary and Split Files ---
    dict_dir = os.path.join(root_dir, 'dictionaries')
    value_dict_path = os.path.join(dict_dir, f'{input_hours}_{seed}_{n_bins}-values.npy')
    
    # Try to find split files from improved preprocessing structure
    splits_improved_dir = os.path.join(root_dir, 'splits_improved')
    
    if not os.path.exists(value_dict_path):
        logger.error(f"Value dict not found: {value_dict_path}")
        logger.error("Please run Step 6 (generate value dictionary) first.")
        sys.exit(1)

    if not os.path.exists(splits_improved_dir):
        logger.error(f"Splits directory not found: {splits_improved_dir}")
        logger.error("Please run Step 5 (train/test split) first.")
        sys.exit(1)

    try:
        V_full = np.load(value_dict_path, allow_pickle=True).item()
        logger.info(f"Loaded value dictionary '{value_dict_path}' with {len(V_full)} keys.")
        
        # Combine all split files from all horizons
        possible_horizons = ['6h', '12h', '24h', '48h']
        all_train_files = []
        all_valid_files = []
        all_test_files = []
        horizons_found = []
        
        for horizon in possible_horizons:
            train_file = os.path.join(splits_improved_dir, f'{seed}-{input_hours}-{horizon}-train.csv')
            valid_file = os.path.join(splits_improved_dir, f'{seed}-{input_hours}-{horizon}-valid.csv')
            test_file = os.path.join(splits_improved_dir, f'{seed}-{input_hours}-{horizon}-test.csv')
            
            try:
                if os.path.exists(train_file):
                    df = pd.read_csv(train_file)
                    if 'Paths' in df.columns:
                        all_train_files.extend(df['Paths'].tolist())
                        if horizon not in horizons_found:
                            horizons_found.append(horizon)
                
                if os.path.exists(valid_file):
                    df = pd.read_csv(valid_file)
                    if 'Paths' in df.columns:
                        all_valid_files.extend(df['Paths'].tolist())
                
                if os.path.exists(test_file):
                    df = pd.read_csv(test_file)
                    if 'Paths' in df.columns:
                        all_test_files.extend(df['Paths'].tolist())
                        
            except Exception as e:
                logger.warning(f"Error reading files for horizon {horizon}: {e}")
        
        if all_train_files and all_valid_files and all_test_files:
            # Remove duplicates while preserving order
            train_paths = list(dict.fromkeys(all_train_files))
            valid_paths = list(dict.fromkeys(all_valid_files))
            test_paths = list(dict.fromkeys(all_test_files))
            
            logger.info(f"Combined files from horizons {horizons_found}: {len(train_paths)} train, {len(valid_paths)} valid, {len(test_paths)} test")
        else:
            logger.error("Could not find training, validation, and test split files")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error loading dictionary or split files: {e}")
        sys.exit(1)

    # --- 2. Determine Percentiles for Continuous Variables ---
    logger.info(f"Calculating percentiles for binning (n_bins = {n_bins})...")
    P = {}  # Dictionary storing percentile boundaries {key: np.array([...])}
    for key, subdict in V_full.items():
        cont_values = subdict.get('cont', np.array([]))
        if len(cont_values) > 0:
            unique_cont_values = np.unique(cont_values)
            if len(unique_cont_values) >= n_bins:
                try:
                    percentiles = np.linspace(0, 100, n_bins + 1)
                    boundaries = np.percentile(cont_values, percentiles)
                    unique_boundaries = np.unique(boundaries)
                    if len(unique_boundaries) < 2:
                         continue  # Skip binning for this key
                    P[key] = unique_boundaries
                except Exception as e:
                    logger.warning(f"Error calculating percentiles for {key}: {e}. Treating as discrete.")

    logger.info(f"Identified {len(P)} variables for percentile binning.")

    # --- Save Bin Boundaries ---
    bin_boundaries_filename = os.path.join(dict_dir, f'{input_hours}_{seed}_{n_bins}-bin_boundaries.npy')
    try:
        os.makedirs(os.path.dirname(bin_boundaries_filename), exist_ok=True)
        np.save(bin_boundaries_filename, P)
        logger.info(f"Saved bin boundaries to {bin_boundaries_filename}")
    except Exception as e:
        logger.error(f"Error saving bin boundaries: {e}")

    # --- 3. Define Tokenization Function ---
    token_column_name = f'TOKEN_{n_bins}'

    def tokenize_row(row):
        key = row['ITEMID_UOM']
        value = row['VALUE']
        if pd.isna(key): 
            return '<UNK>'
        key_str = str(key)
        if key_str not in V_full: 
            return '<UNK>'

        if key_str in P:  # Check if we calculated boundaries for this key
            try:
                float_value = float(value)
                if np.isnan(float_value): 
                    return f"{key_str}:nan"
                boundaries = P[key_str]
                bin_index = np.searchsorted(boundaries, float_value, side='right') - 1
                bin_index = max(0, min(bin_index, n_bins - 1))  # Clamp index
                return f"{key_str}:{bin_index}"
            except (ValueError, TypeError): 
                return f"{key_str}:{str(value)}"  # Fallback to discrete
        else: 
            return f"{key_str}:{str(value)}"  # Discrete or not binned continuous

    # --- 4. Check if files are already tokenized and handle accordingly ---
    sample_file_path = os.path.join(root_dir, train_paths[0]) if train_paths else None
    if sample_file_path and os.path.exists(sample_file_path):
        try:
            sample_header = pd.read_csv(sample_file_path, nrows=0)
            if token_column_name in sample_header.columns:
                logger.info(f"Files already contain {token_column_name} column. Skipping tokenization.")
                # Still need to create token-to-index mapping if it doesn't exist
                create_token_mapping_if_needed(root_dir, input_hours, n_bins, seed, train_paths, valid_paths, test_paths, token_column_name)
                return
        except Exception as e:
            logger.warning(f"Error checking sample file format: {e}")

    # --- 5. Process All Files ---
    all_file_paths = list(set(train_paths + valid_paths + test_paths))
    logger.info(f"Tokenizing {len(all_file_paths)} unique files...")

    all_tokens = set()
    
    for relative_path in tqdm(all_file_paths, desc="Tokenizing files"):
        full_path = os.path.join(root_dir, relative_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            continue
        
        try:
            # Read the file
            ts = pd.read_csv(full_path)
            
            if 'ITEMID_UOM' not in ts.columns or 'VALUE' not in ts.columns:
                logger.warning(f"Missing required columns in {full_path}")
                continue
            
            # Apply tokenization
            ts[token_column_name] = ts.apply(tokenize_row, axis=1)
            
            # Collect all tokens
            all_tokens.update(ts[token_column_name].unique())
            
            # Save back to file
            ts.to_csv(full_path, index=False)
            
        except Exception as e:
            logger.error(f"Error processing {full_path}: {e}")
            continue

    # --- 6. Create Token-to-Index Mapping ---
    logger.info(f"Creating token-to-index mapping for {len(all_tokens)} unique tokens...")
    
    # Sort tokens for consistent ordering
    sorted_tokens = sorted(list(all_tokens))
    
    # Reserve indices
    token2index = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    
    # Add all other tokens
    for i, token in enumerate(sorted_tokens):
        if token not in token2index:
            token2index[token] = len(token2index)
    
    # Save mapping
    token_map_path = os.path.join(dict_dir, f'{input_hours}_{seed}_{n_bins}-token2index.npy')
    try:
        np.save(token_map_path, token2index)
        logger.info(f"Saved token-to-index mapping to {token_map_path}")
        logger.info(f"Total vocabulary size: {len(token2index)}")
    except Exception as e:
        logger.error(f"Error saving token mapping: {e}")


def create_token_mapping_if_needed(root_dir, input_hours, n_bins, seed, train_paths, valid_paths, test_paths, token_column_name):
    """Create token mapping if files are already tokenized but mapping doesn't exist"""
    
    dict_dir = os.path.join(root_dir, 'dictionaries')
    token_map_path = os.path.join(dict_dir, f'{input_hours}_{seed}_{n_bins}-token2index.npy')
    
    if os.path.exists(token_map_path):
        logger.info("Token-to-index mapping already exists.")
        return
    
    logger.info("Creating token-to-index mapping from existing tokenized files...")
    
    all_file_paths = list(set(train_paths + valid_paths + test_paths))
    all_tokens = set()
    
    for relative_path in tqdm(all_file_paths, desc="Collecting tokens"):
        full_path = os.path.join(root_dir, relative_path)
        
        if not os.path.exists(full_path):
            continue
        
        try:
            ts = pd.read_csv(full_path, usecols=[token_column_name])
            all_tokens.update(ts[token_column_name].unique())
        except Exception as e:
            logger.error(f"Error reading {full_path}: {e}")
            continue
    
    # Create mapping
    sorted_tokens = sorted(list(all_tokens))
    token2index = {'<PAD>': 0, '<UNK>': 1}
    
    for token in sorted_tokens:
        if token not in token2index:
            token2index[token] = len(token2index)
    
    # Save mapping
    try:
        np.save(token_map_path, token2index)
        logger.info(f"Created token-to-index mapping with {len(token2index)} tokens")
    except Exception as e:
        logger.error(f"Error saving token mapping: {e}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Quantize and tokenize events using value dictionary')
    
    parser.add_argument('--data_path', type=str, default=DATA_ROOT_PATH,
                       help='Root path where subject data is stored')
    parser.add_argument('--input_hours', type=int, default=INPUT_HOURS,
                       help='Hours of input data used')
    parser.add_argument('--n_bins', type=int, default=N_BINS,
                       help='Number of bins for quantization')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed used for data splitting')
    
    args = parser.parse_args()
    
    print("Starting Step 7: Quantize Events")
    print("=" * 50)
    print(f"Data Path: {args.data_path}")
    print(f"Input Hours: {args.input_hours}")
    print(f"Number of Bins: {args.n_bins}")
    print(f"Random Seed: {args.seed}")
    print("=" * 50)
    
    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        print("Please run Steps 1-6 first.")
        return

    quantize_events(args.data_path, args.input_hours, args.n_bins, args.seed)
    logger.info("Event quantization complete.")
    
    print("✓ Event quantization completed successfully!")


if __name__ == '__main__':
    main() 