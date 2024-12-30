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
    vocab_size = 50257
    
    # Create sparse matrix for co-occurrences
    rows = []
    cols = []
    data = []
    interval_index = 1
    # Vectorized operation for each position in the window
    for i in tqdm(range(len(word_indices)), mininterval=5):
        # Get the context window boundaries

        start = max(0, i - window_size)
        end = min(len(word_indices), i + window_size + 1)
        
        # Add co-occurrences for all words in the window (except the word itself)
        for j in range(start, end):
            if i != j:  # Don't count co-occurrence with itself
                rows.append(word_indices[i])
                cols.append(word_indices[j])
                # Could add distance-based weighting here if desired
                data.append(1)

        if i % 10000000 == 0 and i != 0:
            co_matrix = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
            save_npz(f"/nlp/scr/jjian/datasets/openwebtext_filtered/cooccurence/openwebtext_co_matrix_0{7 + (i // 10000000)}.npz", co_matrix)
            rows = []
            cols = []
            data = []

    co_matrix = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
    save_npz(f"/nlp/scr/jjian/datasets/openwebtext_filtered/cooccurence/openwebtext_co_matrix_0{7 + (i // 10000000) + 1}.npz", co_matrix)
    return co_matrix


def run_svd(file_prefix, total, k=300):
    """
    Collects matrices, adds them together, and runs SVD on the resulting matrix.
    """
    co_matrix_sum = csr_matrix((50257, 50257))
    for i in tqdm(range(1, total+1)):
        a = load_npz(f"/nlp/scr/jjian/datasets/openwebtext_filtered/cooccurence/{file_prefix}{i}.npz")
        co_matrix_sum += a
        del a
        if i % 2 == 0:
            u, _, _ = svds(co_matrix_sum.sqrt(), k=k, return_singular_vectors="u", solver="propack")
            np.save(f"/nlp/scr/jjian/datasets/openwebtext_filtered/word_vectors/{file_prefix}{i}.svd_s.propack.npy", u)
    return

if __name__ == "__main__":
    """offset = 0
    # Define the window size for co-occurrence
    window_size = 15
    print("Loading token ids")
    words = np.load("/nlp/scr/jjian/datasets/openwebtext_filtered/openwebtext_token_ids.42.1M.true.npy")
    print("Loaded token ids")
    words = words[70000000:(offset+1)*100000000]
    
    co_matrix = fast_cooccurrence_matrix(words, window_size, offset)"""
    run_svd("openwebtext_co_matrix_0", 110)

