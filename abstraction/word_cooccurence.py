import os

os.environ["SCIPY_USE_PROPACK"] = "1"

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.linalg import svds
from transformers import GPT2Tokenizer
import sys
import gc


def fast_cooccurrence_matrix(words, window_size=2, offset=0):
    """
    Calculate word co-occurrence matrix using vectorized operations.
    
    Args:
        words: List of words
        window_size: Size of the context window
    
    Returns:
        DataFrame containing the co-occurrence matrix
    """
    # Get unique words and their indices
    #unique_words = list(set(words))
    #word_to_id = {word: idx for idx, word in enumerate(unique_words)}
    
    # Convert words to indices
    #word_indices = np.array([word_to_id[word] for word in words])
    word_indices = words
    print(word_indices.shape)
    vocab_size = 50257
    
    # Create sparse matrix for co-occurrences
    rows = []
    cols = []
    data = []
    interval_index = 1
    # Vectorized operation for each position in the window
    for i in tqdm(range(len(word_indices)), mininterval=5):
        # Get the context window boundaries

        start = max(0, i)
        end = min(len(word_indices), i + window_size + 1)
        
        # Add co-occurrences for all words in the window (except the word itself)
        for j in range(start, end):
            if i != j:  # Don't count co-occurrence with itself
                rows.append(word_indices[i])
                cols.append(word_indices[j])
                # Could add distance-based weighting here if desired
                data.append(1)

        if i % 100 == 0 and i != 0:
            co_matrix = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
            save_npz(f"/nlp/scr/jjian/datasets/openwebtext_filtered/ditransitive_motion_cooccurence_late/openwebtext_co_matrix_{i}.npz", co_matrix)
            rows = []
            cols = []
            data = []
            gc.collect()

    co_matrix = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
    save_npz(f"/nlp/scr/jjian/datasets/openwebtext_filtered/ditransitive_motion_cooccurence_late/openwebtext_co_matrix_{i}.npz", co_matrix)
    return co_matrix

def fast_cooccurrence_matrix_eot(words, window_size=2, offset=0, eos_token_id=50256, save_frequency=100):
    """
    Calculate word co-occurrence matrix using vectorized operations, respecting EOS token boundaries.
    Only considers right context (words that follow the current word).
    
    Args:
        words (list[int]): List of tokenized word IDs.
        window_size (int): Size of the right context window.
        offset (int): Offset for saving chunked matrices.
        eos_token_id (int): Token ID for <|endoftext|>, default is 50256 for GPT-2.
        save_frequency (int): Attempt to save after processing this many tokens (will save at next EOS).

    Returns:
        csr_matrix: Co-occurrence matrix in sparse format.
    """
    word_indices = words  # Assumes pre-tokenized input
    vocab_size = 50257
    
    # Initialize storage for co-occurrence
    rows = []
    cols = []
    data = []
    
    # Find all EOS token positions to identify text boundaries
    eos_positions = [-1] + [i for i, token in enumerate(word_indices) if token == eos_token_id] + [len(word_indices)]
    
    # Track when we last saved
    last_save = 0
    #next_save_number = save_frequency  # For filename numbering
    next_save_number = 50000000
    # Process each segment between EOS tokens
    for seg_idx in range(len(eos_positions) - 1):
        start_index = eos_positions[seg_idx] + 1  # Start after previous EOS
        end_index = eos_positions[seg_idx + 1]    # End at next EOS
        
        # Process each token in the current segment
        for i in tqdm(range(start_index, end_index), mininterval=5):
            # Only look at right context up to window_size tokens ahead
            window_end = min(end_index, i + window_size + 1)
            
            # Add co-occurrences for the right context only
            for j in range(i + 1, window_end):
                rows.append(word_indices[i])
                cols.append(word_indices[j])
                data.append(1)  # Optionally, use a weighting function
            
            # Check if we should save at this EOS token
            if i == end_index - 1 and i - last_save >= save_frequency:
                co_matrix = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
                save_npz(f"/nlp/scr/jjian/datasets/openwebtext_filtered/ditransitive_motion_cooccurence_late/openwebtext_co_matrix_{next_save_number}.npz", co_matrix)
                rows = []
                cols = []
                data = []
                last_save = i
                next_save_number += save_frequency  # Increment by save_frequency
                gc.collect()
    
    # Save final batch
    if rows:
        co_matrix = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
        save_npz(f"/nlp/scr/jjian/datasets/openwebtext_filtered/ditransitive_motion_cooccurence_late/openwebtext_co_matrix_{i}.npz", co_matrix)
        
    return co_matrix


def run_svd(file_prefix, total, k=300):
    """
    Collects matrices, adds them together, and runs SVD on the resulting matrix.
    """
    co_matrix_sum = csr_matrix((50257, 50257))
    for i in tqdm(range(1, total+1)):
        a = load_npz(f"/nlp/scr/jjian/datasets/openwebtext_filtered/cooccurence_earliest/{file_prefix}{i}.npz")
        co_matrix_sum += a
        del a
        gc.collect()
        if i % 2 == 0 and i != 0:
            co_matrix_log = co_matrix_sum.log1p()
            u, _, _ = svds(co_matrix_log, k=k, return_singular_vectors="u", solver="propack")
            np.save(f"/nlp/scr/jjian/datasets/openwebtext_filtered/word_vectors_earliest/{file_prefix}{i}.svd_l.propack.npy", u)
            # free up memory 
            del u, co_matrix_log
            gc.collect()
    return

if __name__ == "__main__":
    offset = 0
    # Define the window size for co-occurrence
    window_size = 10
    print("Loading token ids")
    words = np.load("/nlp/scr/jjian/datasets/openwebtext_filtered/ditransitive.motion.bash_filtered.combined.token_ids.npy")
    print("Loaded token ids")
    #words = words[0:(offset+1)*50000000]
    words = words[49900000:]
    
    co_matrix = fast_cooccurrence_matrix_eot(words, window_size, offset, save_frequency=100000)
    #run_svd("openwebtext_co_matrix_0", 20)

