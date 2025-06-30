#!/usr/bin/env python3
"""
AKI Prediction LSTM - Step 9: Recreate Arrays with Word2Vec Vocabulary

This script recreates training arrays using a Word2Vec-based vocabulary mapping.
It converts the tokenized patient data to use Word2Vec token indices instead of 
the original quantization-based indices.

Author: Organized AKI Prediction Project
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import logging

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_patient_file(file_path, token2idx, max_len, n_bins):
    """
    Reads a patient's tokenized CSV and converts tokens to indices using Word2Vec vocab.
    
    Args:
        file_path (str): Path to patient's tokenized CSV file
        token2idx (dict): Word2Vec token to index mapping
        max_len (int): Maximum sequence length for padding/truncation
        n_bins (int): Number of bins used for quantization (for column naming)
    
    Returns:
        tuple: (patient_array, filename) or (None, None) if processing fails
    """
    token_column = f'TOKEN_{n_bins}'
    try:
        ts = pd.read_csv(file_path, usecols=['Hours', token_column])
        if ts.empty:
            logger.warning(f"Empty file: {file_path}")
            return None, None
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not read or parse file: {file_path} - {str(e)}")
        return None, None

    # Map tokens to their new indices using Word2Vec vocabulary
    # Unknown tokens are mapped to <PAD> index (0) which the model should learn to ignore
    unk_token_idx = token2idx.get('<PAD>', 0)
    ts['index'] = ts[token_column].map(lambda x: token2idx.get(x, unk_token_idx))

    # Create the array [time, index]
    patient_array = ts[['Hours', 'index']].to_numpy()

    # Pad or truncate the sequence
    if len(patient_array) > max_len:
        # Truncate to keep the latest events
        patient_array = patient_array[-max_len:]
    else:
        # Pad with [0, 0] which corresponds to (time=0, token=<PAD>)
        padding = np.zeros((max_len - len(patient_array), 2))
        patient_array = np.vstack([padding, patient_array])

    return patient_array, os.path.basename(file_path)


def recreate_arrays_with_word2vec(config):
    """
    Main function to recreate data arrays with Word2Vec vocabulary.
    
    Args:
        config: Configuration object containing all parameters
    """
    logger.info("=== Step 9: Recreating Arrays with Word2Vec Vocabulary ===")
    
    # Load Word2Vec vocabulary
    if not os.path.exists(config.word2vec_vocab_path):
        raise FileNotFoundError(f"Word2Vec vocabulary not found: {config.word2vec_vocab_path}")
    
    logger.info(f"Loading Word2Vec vocabulary from: {config.word2vec_vocab_path}")
    try:
        new_token2idx = np.load(config.word2vec_vocab_path, allow_pickle=True).item()
        logger.info(f"Loaded vocabulary with {len(new_token2idx)} tokens")
    except Exception as e:
        raise RuntimeError(f"Failed to load Word2Vec vocabulary: {str(e)}")

    # Create output directory
    output_dir = os.path.join(config.output_dir, 'arrays_word2vec')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Word2Vec arrays will be saved in: {output_dir}")

    # Process each prediction horizon
    total_processed = 0
    
    for horizon in config.prediction_horizons:
        logger.info(f"\n--- Processing {horizon}h prediction horizon ---")
        
        # Process each data split
        for split in ['train', 'valid', 'test']:
            split_file_path = os.path.join(
                config.output_dir, 'splits_improved',
                f'{config.random_seed}-{config.input_hours}-{horizon}h-{split}.csv'
            )
            
            if not os.path.exists(split_file_path):
                logger.warning(f"Split file not found, skipping: {split_file_path}")
                continue

            logger.info(f"Processing {split} split for {horizon}h horizon...")
            split_df = pd.read_csv(split_file_path)
            
            all_X, all_Y, all_paths = [], [], []

            # Process each patient file
            for _, row in tqdm(split_df.iterrows(), total=len(split_df), 
                             desc=f"Processing {split} files"):
                patient_file_path = row['Paths']
                label = row['Label']
                full_path = os.path.join(config.output_dir, patient_file_path)

                X, path = process_patient_file(
                    full_path, new_token2idx, config.max_sequence_length, config.n_bins
                )
                
                if X is not None:
                    all_X.append(X)
                    all_Y.append(label)
                    all_paths.append(path)

            if not all_X:
                logger.warning(f"No data processed for {split} split in {horizon}h horizon.")
                continue
                
            # Convert to numpy arrays
            # Data is already in correct format: (n_patients, max_len, 2)
            X_final = np.array(all_X)
            Y_final = np.array(all_Y)
            paths_final = np.array(all_paths)

            # Save arrays
            output_filename = f'{config.input_hours}_{config.random_seed}_{config.n_bins}-{horizon}h-{split}-arrays.npz'
            output_filepath = os.path.join(output_dir, output_filename)
            
            np.savez_compressed(
                output_filepath, 
                X=X_final, 
                Y=Y_final, 
                paths=paths_final
            )
            
            logger.info(f"Saved {split} data to {output_filename}")
            logger.info(f"  Shapes - X: {X_final.shape}, Y: {Y_final.shape}")
            total_processed += len(X_final)

    logger.info(f"\n=== Step 9 Complete ===")
    logger.info(f"Total patients processed: {total_processed}")
    logger.info(f"Word2Vec arrays saved in: {output_dir}")
    return True


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Step 9: Recreate data arrays with Word2Vec vocabulary mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python 09_recreate_arrays_word2vec.py

    # With custom Word2Vec vocabulary
    python 09_recreate_arrays_word2vec.py --word2vec_vocab_path /path/to/vocab.npy

    # With specific horizons
    python 09_recreate_arrays_word2vec.py --prediction_horizons 6 12 24
        """
    )
    
    parser.add_argument('--config_path', type=str, 
                       default='../config/config.py',
                       help='Path to configuration file')
    
    parser.add_argument('--word2vec_vocab_path', type=str,
                       help='Path to Word2Vec token2idx.npy file (overrides config)')
    
    parser.add_argument('--prediction_horizons', type=int, nargs='+',
                       help='Prediction horizons to process (overrides config)')
    
    parser.add_argument('--max_sequence_length', type=int,
                       help='Maximum sequence length for padding/truncating (overrides config)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        config = Config()
        
        # Override config with command line arguments if provided
        if args.word2vec_vocab_path:
            config.word2vec_vocab_path = args.word2vec_vocab_path
        
        if args.prediction_horizons:
            config.prediction_horizons = args.prediction_horizons
            
        if args.max_sequence_length:
            config.max_sequence_length = args.max_sequence_length

        # Validate required paths
        if not hasattr(config, 'word2vec_vocab_path') or not config.word2vec_vocab_path:
            raise ValueError("Word2Vec vocabulary path must be specified in config or via --word2vec_vocab_path")

        # Run the array recreation
        success = recreate_arrays_with_word2vec(config)
        
        if success:
            logger.info("Array recreation with Word2Vec vocabulary completed successfully!")
            sys.exit(0)
        else:
            logger.error("Array recreation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in array recreation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 