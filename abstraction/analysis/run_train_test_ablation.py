import sys
import numpy as np
import pandas as pd
import h5py
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import random
import json
import os
from abstraction.analysis.metrics import pca_classifier, pca_classifier_train_test
from collections import Counter

def find_combination(items, total_sum, memo=None):
    if memo is None:
        memo = {}
    if total_sum <= 0:
        return []
    if (len(items), total_sum) in memo:
        return memo[(len(items), total_sum)]
    
    for i, (item, count) in enumerate(items):
        remaining = items[:i] + items[i+1:]
        result = find_combination(remaining, total_sum - count, memo)
        if result is not None:
            memo[(len(items), total_sum)] = result + [(item, count)]
            return memo[(len(items), total_sum)]
    
    memo[(len(items), total_sum)] = None
    return None

def get_random_items_to_sum(counter, total_sum):
    items = list(counter.items())
    random.shuffle(items)  # Randomize the order of items
    result = find_combination(items, total_sum)
    if result is not None:
        return dict(result)
    return None


def redistribution_ablation(sampled_dataset_a_path, sampled_dataset_b_path, splits=[]):
    random.seed(42)
    with open(sampled_dataset_a_path, 'r') as f:
        sampled_dataset_a = json.load(f)
    with open(sampled_dataset_b_path, 'r') as f:
        sampled_dataset_b = json.load(f)
    sample_size = len(sampled_dataset_a) // 2
    sampled_lemmas_a = [s['dependent_lemma'] for s in sampled_dataset_a]
    sampled_lemmas_b = [s['dependent_lemma'] for s in sampled_dataset_b]
    lemma_counts_a = Counter(sampled_lemmas_a)
    lemma_counts_b = Counter(sampled_lemmas_b)

    
    new_sampled_a_lemmas = get_random_items_to_sum(lemma_counts_a, sample_size)
    train_set_a = Counter(new_sampled_a_lemmas)
    print(train_set_a)
    test_set_a = lemma_counts_a - Counter(new_sampled_a_lemmas)
    print(test_set_a)
    new_sampled_b_lemmas = get_random_items_to_sum(lemma_counts_b, sample_size)
    train_set_b = Counter(new_sampled_b_lemmas)
    test_set_b = lemma_counts_b - Counter(new_sampled_b_lemmas)
    
    
    # return the indices of the new sampled datasets

    sampled_train_a_idx = []
    sampled_test_a_idx = []
    sampled_train_b_idx = []
    sampled_test_b_idx = []

    for i, s in enumerate(sampled_dataset_a):
        if s['dependent_lemma'] in train_set_a.keys():
            sampled_train_a_idx.append((splits[0], i))
        elif s['dependent_lemma'] in test_set_a.keys():
            sampled_test_a_idx.append((splits[0], i))

    for i, s in enumerate(sampled_dataset_b):
        if s['dependent_lemma'] in train_set_b.keys():
            sampled_train_b_idx.append((splits[1], i))
        elif s['dependent_lemma'] in test_set_b.keys():
            sampled_test_b_idx.append((splits[1], i))
    
    return [sampled_train_a_idx, sampled_train_b_idx], [sampled_test_a_idx, sampled_test_b_idx]


def get_ablation_splits(samples, splits, ablation_idx):
    """
    ablation_idx is a [[(str, int)]] list of tuples consisting of the split and the index drawn from that split.
    """

    new_sample_a = []
    new_sample_b = []

    for split, idx in ablation_idx[0]:
        if split == splits[0]:
            new_sample_a.append(samples[0][idx])
        elif split == splits[1]:
            new_sample_a.append(samples[1][idx])
        else:
            raise IndexError(f"Split {split} not found in splits {splits}")
    
    for split, idx in ablation_idx[1]:
        if split == splits[0]:
            new_sample_b.append(samples[0][idx])
        elif split == splits[1]:
            new_sample_b.append(samples[1][idx])
        else:
            raise IndexError(f"Split {split} not found in splits {splits}")
    
    return new_sample_a, new_sample_b


def get_checkpoint_results(model_name, metric, splits, train_idx, test_idx, rep="target", source="wikitext", pca_rank=4):
    all_results = []  # List to store all DataFrames
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    
    branches = sorted(branches)
    
    model_name_preprocessed = model_name.split("/")[-1]
    
    for checkpoint in tqdm(branches):
        try:
            split_a = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[0]}/{rep}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[0]}.embeddings.hdf5", 'r')
            split_b = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[1]}/{rep}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[1]}.embeddings.hdf5", 'r')
        except FileNotFoundError:
            continue
        
        sample_a = [np.array(split_a[str(idx)]) for idx in range(len(split_a.values()))]
        sample_b = [np.array(split_b[str(idx)]) for idx in range(len(split_b.values()))]
        
        # get the ablated train sets
        train_sample_a, train_sample_b = get_ablation_splits([sample_a, sample_b], splits, train_idx)
        # get the ablated test sets
        test_sample_a, test_sample_b = get_ablation_splits([sample_a, sample_b], splits, test_idx)
        
        try:
            results_df = metric([train_sample_a, train_sample_b], [test_sample_a, test_sample_b], layers=range(0, 12), labels=splits, checkpoint_n=checkpoint, pca_rank=pca_rank)
        except IndexError:
            print(f"IndexError at checkpoint {checkpoint}")
        # Append the DataFrame to the list
        all_results.append(results_df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(all_results, ignore_index=False, axis=1)
    
    return concatenated_df


if __name__ == "__main__":
    split_a_sample_path = sys.argv[1]
    split_b_sample_path = sys.argv[2]
    rep = sys.argv[3]
    pca_rank = int(sys.argv[4])
    
    # Define the splits and the metric
    splits = ["motion_balanced", "ditrans_balanced"]
    metric = pca_classifier_train_test

    # Define the model name
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"

    # iterate through 5 random seeds

    
    random.seed(seed)
    train_idx, test_idx = redistribution_ablation(split_a_sample_path, split_b_sample_path, splits=splits)  

    # Get the results
    results = get_checkpoint_results(model_name, metric, splits, train_idx, test_idx, rep=rep, source="wikitext", pca_rank=pca_rank)

    # Save the results
    outdir = f"results/ablation/wikitext/to/train_test_ablation/pca_{pca_rank}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    results.to_csv(f'{outdir}/battlestar_small.{splits[0]}.{splits[1]}.{rep}.tsv', sep='\t', index=False)