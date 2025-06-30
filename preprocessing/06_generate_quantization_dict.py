"""
Step 6: Generate Value Dictionary

Generate a dictionary characterizing variables based on the training data
for a specific input time window.

For each unique variable identifier (ITEMID_UOM), it collects all observed values
from the training set timeseries files (truncated to input_hours).
Saves the dictionary for later use in quantization.

Usage:
    python 06_generate_value_dict.py [--input_hours 24] [--n_bins 20] [--seed 0]
    
Make sure to run Steps 1-5 first.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import logging
import sys

# Add config to path
sys.path.append('../config')
from config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def try_to_float(v):
    """Attempt to convert value to float, return original if fails."""
    try: 
        return float(v)
    except (ValueError, TypeError): 
        return str(v)


def generate_value_dict(root_dir, input_hours, n_bins, seed):
    """Generates the value dictionary from training data for input_hours."""
    
    # Try to find training files from improved preprocessing structure first
    splits_improved_dir = os.path.join(root_dir, 'splits_improved')
    
    train_files_df = None
    
    # Try improved preprocessing structure first (horizon-specific files)
    if os.path.exists(splits_improved_dir):
        # Look for any horizon-specific training files and combine them
        possible_horizons = ['6h', '12h', '24h', '48h']
        all_train_files = []
        horizons_found = []
        
        for horizon in possible_horizons:
            potential_file = os.path.join(splits_improved_dir, f'{seed}-{input_hours}-{horizon}-train.csv')
            if os.path.exists(potential_file):
                try:
                    horizon_df = pd.read_csv(potential_file)
                    if 'Paths' in horizon_df.columns:
                        all_train_files.extend(horizon_df['Paths'].tolist())
                        horizons_found.append(horizon)
                except Exception as e:
                    logger.warning(f"Error reading {potential_file}: {e}")
        
        if all_train_files:
            # Remove duplicates while preserving order
            unique_train_files = list(dict.fromkeys(all_train_files))
            train_files_df = pd.DataFrame({'Paths': unique_train_files})
            logger.info(f"Combined training files from horizons: {horizons_found} ({len(unique_train_files)} unique files)")
    
    # Check if we found any training files
    if train_files_df is None:
        logger.error(f"Error: No training split files found in {splits_improved_dir}")
        logger.error("Please run Step 5 (train/test split) first.")
        return
    
    train_file_paths = train_files_df['Paths'].tolist()

    if not train_file_paths:
        logger.warning("Warning: No training files listed in the split file. Value dictionary will be empty.")
        value_dict = {}
    else:
        logger.info(f"Processing {len(train_file_paths)} training files for input_hours={input_hours}, seed={seed}...")
        value_dict = defaultdict(lambda: {'disc': set(), 'cont': []})

        for relative_path in tqdm(train_file_paths, desc="Generating value dictionary"):
            full_path = os.path.join(root_dir, relative_path)
            if not os.path.exists(full_path):
                logger.warning(f"Warning: Training file not found: {full_path}. Skipping.")
                continue

            try:
                # Check file format first - read header to determine if it's raw or tokenized
                header = pd.read_csv(full_path, nrows=0)
                
                if 'ITEMID_UOM' in header.columns and 'VALUE' in header.columns:
                    # Raw format - read ITEMID_UOM and VALUE from the truncated file
                    ts = pd.read_csv(full_path, usecols=['ITEMID_UOM', 'VALUE'])

                    for _, row in ts.iterrows():
                        key = row['ITEMID_UOM']
                        value = row['VALUE']
                        if pd.isna(key): 
                            continue

                        processed_value = try_to_float(value)
                        if isinstance(processed_value, float):
                            if not np.isnan(processed_value):
                                 value_dict[key]['cont'].append(processed_value)
                            else: 
                                value_dict[key]['disc'].add('nan')
                        else: 
                            value_dict[key]['disc'].add(str(processed_value))
                
                elif any(col.startswith('TOKEN_') for col in header.columns):
                    # File is already tokenized - skip processing for value dictionary
                    logger.debug(f"Skipping tokenized file {full_path} - value dictionary should use raw data")
                    continue
                    
                else:
                    # Unknown format
                    logger.warning(f"Unknown file format for {full_path}. Columns: {list(header.columns)}")
                    continue

            except pd.errors.EmptyDataError: 
                continue
            except Exception as e:
                logger.error(f"Error processing file {full_path}: {e}")
                continue

    # Post-processing the dictionary
    final_value_dict = {}
    logger.info("Finalizing dictionary...")
    for key, values in tqdm(value_dict.items(), desc="Converting lists"):
        final_value_dict[key] = {
            'disc': sorted(list(values['disc'])),
            'cont': np.array(values['cont'], dtype=np.float32) if values['cont'] else np.array([], dtype=np.float32)
        }

    # Define output directory for dictionaries
    dict_dir = os.path.join(root_dir, 'dictionaries')
    os.makedirs(dict_dir, exist_ok=True)
    # Define output filename including input_hours, seed, and n_bins to match later stages
    output_filename = os.path.join(dict_dir, f'{input_hours}_{seed}_{n_bins}-values.npy')

    # Save the dictionary
    try:
        np.save(output_filename, final_value_dict)
        logger.info(f"Value dictionary saved to {output_filename}")
        num_keys = len(final_value_dict)
        num_cont_keys = sum(1 for k in final_value_dict if len(final_value_dict[k]['cont']) > 0)
        logger.info(f" Dictionary contains {num_keys} unique ITEMID_UOM keys.")
        logger.info(f" {num_cont_keys} keys have continuous values.")
    except Exception as e:
        logger.error(f"Error saving value dictionary to {output_filename}: {e}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Generate value dictionary for quantization')
    
    parser.add_argument('--data_path', type=str, default=DATA_ROOT_PATH,
                       help='Root path where subject data is stored')
    parser.add_argument('--input_hours', type=int, default=INPUT_HOURS,
                       help='Hours of input data used')
    parser.add_argument('--n_bins', type=int, default=N_BINS,
                       help='Number of bins for quantization')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed used for data splitting')
    
    args = parser.parse_args()
    
    print("Starting Step 6: Generate Value Dictionary")
    print("=" * 50)
    print(f"Data Path: {args.data_path}")
    print(f"Input Hours: {args.input_hours}")
    print(f"Number of Bins: {args.n_bins}")
    print(f"Random Seed: {args.seed}")
    print("=" * 50)
    
    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        print("Please run Steps 1-5 first.")
        return

    generate_value_dict(args.data_path, args.input_hours, args.n_bins, args.seed)
    logger.info("Value dictionary generation complete.")
    
    print("✓ Value dictionary generation completed successfully!")


if __name__ == '__main__':
    main() 