"""
Step 8: Create Final Arrays

Create final NumPy arrays for multiple AKI prediction horizons with improved study design.

Handles:
- Multiple prediction horizons (6h, 12h, 24h, 48h)
- Improved train/test splits from the enhanced study design
- Separate arrays for each horizon
- Comprehensive validation and statistics

Usage:
    python 08_create_arrays_improved.py [--input_hours 24] [--n_bins 20] [--seed 0] [--max_len 10000]
    
Make sure to run Steps 1-7 first.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import sys
import logging

# Add config to path
sys.path.append('../config')
from config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ImprovedArrayCreator:
    """Enhanced array creator for multiple prediction horizons"""
    
    def __init__(self, root_dir, input_hours=INPUT_HOURS, n_bins=N_BINS, 
                 seed=RANDOM_SEED, max_len=MAX_SEQUENCE_LENGTH):
        self.root_dir = root_dir
        self.input_hours = input_hours
        self.n_bins = n_bins
        self.seed = seed
        self.max_len = max_len
        self.horizons = PREDICTION_HORIZONS
        
        # Load token mapping
        self.token2index = self._load_token_mapping()
        self.padding_idx = 0
        self.unk_index = self.token2index.get('<UNK>', 1)
        self.token_column_name = f'TOKEN_{n_bins}'
        
    def _load_token_mapping(self):
        """Load token to index mapping"""
        dict_dir = os.path.join(self.root_dir, 'dictionaries')
        token_map_path = os.path.join(dict_dir, f'{self.input_hours}_{self.seed}_{self.n_bins}-token2index.npy')
        
        if not os.path.exists(token_map_path):
            logger.error(f"Token map not found at {token_map_path}")
            logger.error("Please run Step 7 (quantize events) first.")
            sys.exit(1)
            
        try:
            token2index = np.load(token_map_path, allow_pickle=True).item()
            logger.info(f"Loaded token map with {len(token2index)} tokens")
            return token2index
        except Exception as e:
            logger.error(f"Error loading token map: {e}")
            sys.exit(1)
    
    def load_splits_for_horizon(self, horizon_key):
        """Load train/test/valid splits for a specific horizon"""
        split_dir = os.path.join(self.root_dir, 'splits_improved')
        
        splits = {}
        for split_name in ['train', 'valid', 'test']:
            split_file = os.path.join(split_dir, f'{self.seed}-{self.input_hours}-{horizon_key}-{split_name}.csv')
            
            if os.path.exists(split_file):
                try:
                    split_df = pd.read_csv(split_file)
                    splits[split_name] = split_df
                    logger.info(f"Loaded {horizon_key} {split_name}: {len(split_df)} samples")
                except Exception as e:
                    logger.warning(f"Error loading {split_file}: {e}")
                    splits[split_name] = pd.DataFrame()
            else:
                logger.warning(f"Split file not found: {split_file}")
                splits[split_name] = pd.DataFrame()
        
        return splits
    
    def process_timeseries_file(self, file_path, label):
        """Process a single timeseries file into array format"""
        full_path = os.path.join(self.root_dir, file_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            return None, None
        
        try:
            # Load tokenized timeseries
            ts = pd.read_csv(full_path, usecols=['Hours', self.token_column_name])
            if ts.empty:
                return None, None
            
            # Map tokens to indices
            ts['TokenIndex'] = ts[self.token_column_name].map(
                lambda x: self.token2index.get(x, self.unk_index)
            )
            
            # Create sequence array [Hours, TokenIndex]
            sequence = ts[['Hours', 'TokenIndex']].values.astype(np.float32)
            
            # Pad or truncate sequence
            seq_len = sequence.shape[0]
            if seq_len == 0:
                return None, None
            
            if seq_len < self.max_len:
                # Pad sequence
                pad_width = self.max_len - seq_len
                time_padding_value = self.input_hours  # Use input_hours as time padding
                
                padded_time = np.pad(
                    sequence[:, 0:1], 
                    ((0, pad_width), (0, 0)), 
                    mode='constant', 
                    constant_values=time_padding_value
                )
                padded_tokens = np.pad(
                    sequence[:, 1:2], 
                    ((0, pad_width), (0, 0)), 
                    mode='constant', 
                    constant_values=self.padding_idx
                )
                final_sequence = np.concatenate((padded_time, padded_tokens), axis=1)
                
            elif seq_len > self.max_len:
                # Truncate sequence (keep most recent events)
                final_sequence = sequence[-self.max_len:, :]
            else:
                final_sequence = sequence
            
            return final_sequence, int(label)
            
        except Exception as e:
            logger.error(f"Error processing {full_path}: {e}")
            return None, None
    
    def create_arrays_for_horizon(self, horizon_key):
        """Create arrays for a specific prediction horizon"""
        logger.info(f"Creating arrays for {horizon_key} horizon...")
        
        # Load splits for this horizon
        splits = self.load_splits_for_horizon(horizon_key)
        
        horizon_arrays = {}
        
        for split_name, split_df in splits.items():
            if split_df.empty:
                logger.warning(f"No data for {horizon_key} {split_name}")
                horizon_arrays[split_name] = {
                    'X': np.array([]),
                    'Y': np.array([]),
                    'paths': np.array([])
                }
                continue
            
            logger.info(f"Processing {horizon_key} {split_name} ({len(split_df)} samples)...")
            
            sequences = []
            labels = []
            paths = []
            
            for _, row in tqdm(split_df.iterrows(), 
                             total=len(split_df), 
                             desc=f"Processing {split_name}",
                             leave=False):
                
                file_path = row['Paths']
                label = row['Label']
                
                sequence, processed_label = self.process_timeseries_file(file_path, label)
                
                if sequence is not None:
                    sequences.append(sequence)
                    labels.append(processed_label)
                    paths.append(file_path)
            
            # Convert to numpy arrays
            if sequences:
                X = np.stack(sequences, axis=0).astype(np.float32)
                Y = np.array(labels).astype(np.int32)
                Paths = np.array(paths)
                
                logger.info(f"  {split_name}: X={X.shape}, Y={Y.shape}")
                logger.info(f"  {split_name} class distribution: {np.bincount(Y)}")
            else:
                X = np.array([])
                Y = np.array([])
                Paths = np.array([])
                logger.warning(f"  No valid sequences for {split_name}")
            
            horizon_arrays[split_name] = {
                'X': X,
                'Y': Y,
                'paths': Paths
            }
        
        return horizon_arrays
    
    def save_arrays_for_horizon(self, horizon_key, horizon_arrays):
        """Save arrays for a specific horizon"""
        arrays_dir = os.path.join(self.root_dir, 'arrays_improved')
        os.makedirs(arrays_dir, exist_ok=True)
        
        for split_name, arrays in horizon_arrays.items():
            if arrays['X'].size > 0:  # Only save non-empty arrays
                
                # Save features
                X_filename = os.path.join(arrays_dir, f'{self.seed}-{self.input_hours}-{horizon_key}-{split_name}-X.npy')
                np.save(X_filename, arrays['X'])
                
                # Save labels
                Y_filename = os.path.join(arrays_dir, f'{self.seed}-{self.input_hours}-{horizon_key}-{split_name}-Y.npy')
                np.save(Y_filename, arrays['Y'])
                
                # Save paths
                paths_filename = os.path.join(arrays_dir, f'{self.seed}-{self.input_hours}-{horizon_key}-{split_name}-paths.npy')
                np.save(paths_filename, arrays['paths'])
                
                logger.info(f"Saved {horizon_key} {split_name} arrays: {arrays['X'].shape}")
            else:
                logger.warning(f"Skipping empty {horizon_key} {split_name} arrays")
    
    def run_improved_array_creation(self):
        """Run the complete improved array creation pipeline"""
        
        logger.info("Starting improved array creation...")
        
        all_horizon_arrays = {}
        
        # Process each horizon
        for horizon in self.horizons:
            horizon_key = f'{horizon}h'
            
            try:
                # Create arrays for this horizon
                horizon_arrays = self.create_arrays_for_horizon(horizon_key)
                
                # Save arrays
                self.save_arrays_for_horizon(horizon_key, horizon_arrays)
                
                all_horizon_arrays[horizon_key] = horizon_arrays
                
            except Exception as e:
                logger.error(f"Error processing {horizon_key} horizon: {e}")
                continue
        
        if not all_horizon_arrays:
            logger.error("No arrays were created successfully!")
            return False
        
        logger.info("Improved array creation completed successfully!")
        return True


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Create final NumPy arrays for multiple AKI prediction horizons')
    
    parser.add_argument('--data_path', type=str, default=DATA_ROOT_PATH,
                       help='Root path where subject data is stored')
    parser.add_argument('--input_hours', type=int, default=INPUT_HOURS,
                       help='Hours of input data used')
    parser.add_argument('--n_bins', type=int, default=N_BINS,
                       help='Number of bins for quantization')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed used for data splitting')
    parser.add_argument('--max_len', type=int, default=MAX_SEQUENCE_LENGTH,
                       help='Maximum sequence length for arrays')
    
    args = parser.parse_args()
    
    print("Starting Step 8: Create Final Arrays")
    print("=" * 50)
    print(f"Data Path: {args.data_path}")
    print(f"Input Hours: {args.input_hours}")
    print(f"Number of Bins: {args.n_bins}")
    print(f"Random Seed: {args.seed}")
    print(f"Max Sequence Length: {args.max_len}")
    print(f"Prediction Horizons: {PREDICTION_HORIZONS}")
    print("=" * 50)
    
    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        print("Please run Steps 1-7 first.")
        return
    
    # Initialize array creator
    array_creator = ImprovedArrayCreator(
        root_dir=args.data_path,
        input_hours=args.input_hours,
        n_bins=args.n_bins,
        seed=args.seed,
        max_len=args.max_len
    )
    
    # Run array creation
    success = array_creator.run_improved_array_creation()
    
    if success:
        print("✓ Final array creation completed successfully!")
    else:
        print("✗ Final array creation failed!")


if __name__ == "__main__":
    main() 