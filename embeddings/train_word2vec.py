"""
Train Word2Vec Embeddings for Medical Events

This script trains Word2Vec embeddings on hourly co-occurrence of medical events.
The embeddings can be used to initialize the embedding layer in the LSTM model.

Usage:
    python train_word2vec.py --split_file_path path/to/train_split.csv
    
Make sure to complete the preprocessing pipeline first to generate the required files.
"""

import argparse
import os
import sys
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec

# Add config to path
sys.path.append('../config')
from config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class HourlySentenceIterator:
    """
    Iterates over patient timeseries files and yields hourly "sentences".
    A sentence is a list of all tokens that occurred within a one-hour window.
    This is memory-efficient as it reads one file at a time.
    """
    def __init__(self, root_dir, file_paths, n_bins):
        self.root_dir = root_dir
        self.file_paths = file_paths
        self.token_column = f'TOKEN_{n_bins}'

    def __iter__(self):
        for relative_path in tqdm(self.file_paths, desc="Processing files"):
            full_path = os.path.join(self.root_dir, relative_path)
            if not os.path.exists(full_path):
                logger.warning(f"File not found, skipping: {full_path}")
                continue
            
            try:
                # Reading only necessary columns
                df = pd.read_csv(full_path, usecols=['Hours', self.token_column])
                if df.empty:
                    continue

                # Group by hour
                df['Hour_Block'] = df['Hours'].astype(int)
                
                # Yield each hour's tokens as a sentence
                for _, group in df.groupby('Hour_Block'):
                    sentence = group[self.token_column].astype(str).tolist()
                    if sentence:
                        yield sentence
            except Exception as e:
                logger.error(f"Error processing file {full_path}: {e}")


def train_word2vec_embeddings(data_root, split_file_path, n_bins, save_path, 
                            embedding_dim, window_size, min_count, workers, epochs):
    """
    Main function to train Word2Vec embeddings.
    """
    logger.info("Starting Word2Vec embedding training...")

    # --- 1. Load list of training files ---
    try:
        train_files_df = pd.read_csv(split_file_path)
        # Assuming the column is named 'Paths' based on preprocessing scripts
        training_files = train_files_df['Paths'].tolist()
        logger.info(f"Found {len(training_files)} training files from '{split_file_path}'.")
    except Exception as e:
        logger.error(f"Failed to read split file: {e}")
        return False

    # --- 2. Create sentence iterator ---
    sentences = HourlySentenceIterator(data_root, training_files, n_bins)

    # --- 3. Train Word2Vec model ---
    logger.info("Training Word2Vec model... (This may take a while)")
    model = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=window_size,
        min_count=min_count,
        workers=workers,
        sg=1,  # Using Skip-Gram
        epochs=epochs
    )
    logger.info("Word2Vec model training complete.")

    # --- 4. Save embeddings and vocabulary ---
    os.makedirs(save_path, exist_ok=True)
    
    # Get the vocabulary and vectors from the trained model
    word_vectors = model.wv
    
    # We need to add a <PAD> token at index 0, as the downstream model expects it.
    vocab_size = len(word_vectors.index_to_key)
    embedding_dim = word_vectors.vector_size
    
    # Create the final embedding matrix, leaving the first row for PAD
    # Note: The CustomEmbedding class expects the pretrained weights to match its size exactly
    # So we must create a vocab and weight matrix that includes the PAD token from the start
    final_embeddings = np.zeros((vocab_size + 1, embedding_dim), dtype=np.float32)
    final_embeddings[1:] = word_vectors.vectors

    # Create the corresponding token-to-index mapping
    final_token2idx = {token: i + 1 for i, token in enumerate(word_vectors.index_to_key)}
    final_token2idx['<PAD>'] = 0

    # Define file paths for the outputs
    embeddings_save_path = os.path.join(save_path, WORD2VEC_EMBEDDINGS_FILE)
    vocab_save_path = os.path.join(save_path, WORD2VEC_VOCAB_FILE)
    
    # Save the files
    np.save(embeddings_save_path, final_embeddings)
    np.save(vocab_save_path, final_token2idx)

    logger.info(f"Embeddings saved to: {embeddings_save_path}")
    logger.info(f"Vocabulary saved to: {vocab_save_path}")
    logger.info("---")
    logger.info("To use these embeddings, make sure to:")
    logger.info(f"1. Update your model to use '{vocab_save_path}' as the vocab file.")
    logger.info(f"2. Call the `load_pretrained_weights` method on your CustomEmbedding layer with '{embeddings_save_path}'.")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Train Word2Vec embeddings on hourly co-occurrence of medical events.")
    
    parser.add_argument('--data_root', type=str, default=DATA_ROOT_PATH, 
                       help="Root directory where patient data is stored.")
    parser.add_argument('--split_file_path', type=str, required=True, 
                       help="Path to the train split file (e.g., 'data/splits_improved/0-12-12h-train.csv').")
    parser.add_argument('--n_bins', type=int, default=N_BINS, 
                       help="Number of bins used during quantization to determine the token column name.")
    parser.add_argument('--save_path', type=str, default=get_models_path('word2vec_embeddings'), 
                       help='Directory to save the pre-trained embeddings and vocab.')
    
    # Word2Vec parameters
    parser.add_argument('--embedding_dim', type=int, default=EMBEDDING_DIM, 
                       help='Dimension of the embeddings.')
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE, 
                       help='Word2Vec context window size. Since our "sentences" are all tokens in an hour, a larger window might be beneficial.')
    parser.add_argument('--min_count', type=int, default=MIN_COUNT, 
                       help='Word2Vec min_count parameter.')
    parser.add_argument('--workers', type=int, default=WORKERS, 
                       help='Number of worker threads for training.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, 
                       help='Number of training epochs.')

    args = parser.parse_args()
    
    print("Starting Word2Vec Embedding Training")
    print("=" * 50)
    print(f"Data Root: {args.data_root}")
    print(f"Split File: {args.split_file_path}")
    print(f"Number of Bins: {args.n_bins}")
    print(f"Save Path: {args.save_path}")
    print(f"Embedding Dimension: {args.embedding_dim}")
    print(f"Window Size: {args.window_size}")
    print(f"Min Count: {args.min_count}")
    print(f"Workers: {args.workers}")
    print(f"Epochs: {args.epochs}")
    print("=" * 50)
    
    # Validate inputs
    if not os.path.exists(args.data_root):
        print(f"Error: Data root path does not exist: {args.data_root}")
        return
    
    if not os.path.exists(args.split_file_path):
        print(f"Error: Split file does not exist: {args.split_file_path}")
        print("Please run the preprocessing pipeline first to generate split files.")
        return
    
    # Run training
    success = train_word2vec_embeddings(
        data_root=args.data_root,
        split_file_path=args.split_file_path,
        n_bins=args.n_bins,
        save_path=args.save_path,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs
    )
    
    if success:
        print("✓ Word2Vec embedding training completed successfully!")
    else:
        print("✗ Word2Vec embedding training failed!")


if __name__ == '__main__':
    main() 