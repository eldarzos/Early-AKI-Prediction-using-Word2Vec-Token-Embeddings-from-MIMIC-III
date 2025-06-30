"""
Step 5: Enhanced Train/Test Split

Improved train/test split for multiple AKI prediction horizons with smart exclusions.

Handles:
- Multiple prediction horizons (6h, 12h, 24h, 48h)
- Exclusion of problematic cases (baseline AKI, near-miss, etc.)
- Stratified splitting within each horizon
- Clean vs competing risk controls
- Comprehensive statistics and validation

Usage:
    python 05_split_train_test_improved.py [--input_hours 24] [--seed 0]
    
Make sure to run Steps 1-4 first.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import numpy as np
import sys
from collections import defaultdict

# Add config to path
sys.path.append('../config')
from config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ImprovedDataSplitter:
    """Enhanced data splitter for multiple horizons with exclusion handling"""
    
    def __init__(self, root_dir, input_hours=INPUT_HOURS, seed=RANDOM_SEED, 
                 test_size=TEST_SIZE, valid_size=VALIDATION_SIZE):
        self.root_dir = root_dir
        self.input_hours = input_hours
        self.seed = seed
        self.test_size = test_size
        self.valid_size = valid_size
        self.horizons = PREDICTION_HORIZONS
        
    def collect_episode_data(self):
        """Collect all episode data with multiple horizon labels"""
        
        subjects = [d for d in os.listdir(self.root_dir) 
                   if os.path.isdir(os.path.join(self.root_dir, d)) and d.isdigit()]
        
        logger.info(f"Scanning {len(subjects)} subject directories...")
        
        all_episodes = []
        
        for subject in tqdm(subjects, desc="Collecting episodes"):
            subject_dir = os.path.join(self.root_dir, subject)
            
            # Find episode files
            episode_files = [f for f in os.listdir(subject_dir) 
                           if f.startswith('episode') and f.endswith('.csv') and 'timeseries' not in f]
            
            for ep_file in episode_files:
                episode_num = ep_file.split('episode')[1].split('.csv')[0]
                
                # Check if corresponding timeseries exists
                ts_file = f'episode{episode_num}_timeseries_{self.input_hours}.csv'
                ts_path = os.path.join(subject_dir, ts_file)
                ep_path = os.path.join(subject_dir, ep_file)
                
                if not os.path.exists(ts_path) or not os.path.exists(ep_path):
                    continue
                
                try:
                    episode_data = pd.read_csv(ep_path)
                    if episode_data.empty:
                        continue
                    
                    # Add file paths
                    episode_data['timeseries_path'] = os.path.join(subject, ts_file)
                    episode_data['episode_path'] = os.path.join(subject, ep_file)
                    episode_data['subject_id'] = subject
                    episode_data['episode_num'] = episode_num
                    
                    all_episodes.append(episode_data)
                    
                except Exception as e:
                    logger.warning(f"Error reading {ep_path}: {e}")
                    continue
        
        if not all_episodes:
            logger.error("No valid episodes found!")
            return None
        
        combined_df = pd.concat(all_episodes, ignore_index=True)
        logger.info(f"Collected {len(combined_df)} total episodes")
        
        return combined_df
    
    def create_splits_per_horizon(self, episodes_df):
        """Create train/test/valid splits for each prediction horizon"""
        
        splits_results = {}
        
        for horizon in self.horizons:
            horizon_key = f'{horizon}h'
            aki_col = f'AKI_Stg3_{horizon_key}'
            
            if aki_col not in episodes_df.columns:
                logger.warning(f"Skipping {horizon_key}: missing column {aki_col}")
                continue
            
            logger.info(f"Creating splits for {horizon_key} horizon...")
            
            # Filter to usable episodes only (exclude -1 labels)
            usable_episodes = episodes_df[episodes_df[aki_col].isin([0, 1])].copy()
            
            if len(usable_episodes) == 0:
                logger.warning(f"No usable episodes for {horizon_key}")
                continue
            
            # Check class balance
            class_counts = usable_episodes[aki_col].value_counts()
            logger.info(f"  Class distribution: {dict(class_counts)}")
            
            if len(class_counts) < 2:
                logger.warning(f"Only one class present for {horizon_key}, skipping stratified split")
                continue
            
            # Prepare data for splitting
            split_data = usable_episodes[['timeseries_path', aki_col]].copy()
            split_data.columns = ['Paths', 'Label']
            
            try:
                # Test split
                train_val_df, test_df = train_test_split(
                    split_data,
                    test_size=self.test_size,
                    stratify=split_data['Label'],
                    random_state=self.seed
                )
                
                # Validation split
                actual_valid_size = min(self.valid_size, len(train_val_df))
                if actual_valid_size > 0 and len(train_val_df) > actual_valid_size:
                    valid_frac = actual_valid_size / len(train_val_df)
                    
                    if valid_frac < 1.0:
                        train_df, valid_df = train_test_split(
                            train_val_df,
                            test_size=valid_frac,
                            stratify=train_val_df['Label'],
                            random_state=self.seed
                        )
                    else:
                        train_df = train_val_df
                        valid_df = pd.DataFrame(columns=split_data.columns)
                else:
                    train_df = train_val_df
                    valid_df = pd.DataFrame(columns=split_data.columns)
                
                splits_results[horizon_key] = {
                    'train': train_df,
                    'valid': valid_df,
                    'test': test_df
                }
                
                logger.info(f"  {horizon_key} splits: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
                
            except Exception as e:
                logger.error(f"Error creating splits for {horizon_key}: {e}")
                continue
        
        return splits_results
    
    def save_splits(self, splits_results):
        """Save splits to files"""
        splits_dir = os.path.join(self.root_dir, 'splits_improved')
        os.makedirs(splits_dir, exist_ok=True)
        
        for horizon_key, splits in splits_results.items():
            for split_name, split_df in splits.items():
                filename = f'{self.seed}-{self.input_hours}-{horizon_key}-{split_name}.csv'
                filepath = os.path.join(splits_dir, filename)
                split_df.to_csv(filepath, index=False)
                logger.info(f"Saved {horizon_key} {split_name} split: {filepath} ({len(split_df)} samples)")
    
    def run_improved_splitting(self):
        """Run the complete improved splitting pipeline"""
        
        logger.info("Starting improved data splitting...")
        
        # Collect all episode data
        episodes_df = self.collect_episode_data()
        if episodes_df is None:
            return False
        
        # Create splits
        splits_results = self.create_splits_per_horizon(episodes_df)
        
        if not splits_results:
            logger.error("No splits were created successfully!")
            return False
        
        # Save splits
        self.save_splits(splits_results)
        
        logger.info("Improved splitting completed successfully!")
        return True


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Enhanced train/test split for multiple AKI prediction horizons')
    
    parser.add_argument('--data_path', type=str, default=DATA_ROOT_PATH,
                       help='Root path where subject data is stored')
    parser.add_argument('--input_hours', type=int, default=INPUT_HOURS,
                       help='Hours of input data used')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed for reproducible splits')
    parser.add_argument('--test_size', type=float, default=TEST_SIZE,
                       help='Fraction of data to use for testing')
    parser.add_argument('--valid_size', type=int, default=VALIDATION_SIZE,
                       help='Number of samples to use for validation')
    
    args = parser.parse_args()
    
    print("Starting Step 5: Enhanced Train/Test Split")
    print("=" * 50)
    print(f"Data Path: {args.data_path}")
    print(f"Input Hours: {args.input_hours}")
    print(f"Random Seed: {args.seed}")
    print(f"Test Size: {args.test_size}")
    print(f"Validation Size: {args.valid_size}")
    print(f"Prediction Horizons: {PREDICTION_HORIZONS}")
    print("=" * 50)
    
    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        print("Please run Steps 1-4 first.")
        return
    
    # Initialize splitter
    splitter = ImprovedDataSplitter(
        root_dir=args.data_path,
        input_hours=args.input_hours,
        seed=args.seed,
        test_size=args.test_size,
        valid_size=args.valid_size
    )
    
    # Run splitting
    success = splitter.run_improved_splitting()
    
    if success:
        print("✓ Enhanced train/test split completed successfully!")
    else:
        print("✗ Enhanced train/test split failed!")


if __name__ == "__main__":
    main() 