import json
import random
import sys
import h5py
import os
from collections import Counter
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import numpy as np

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


def redistribution_ablation(sampled_dataset_a_path, sampled_dataset_b_path):
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
    transfer_from_a_to_b = lemma_counts_a - Counter(new_sampled_a_lemmas)
    new_sampled_b_lemmas = get_random_items_to_sum(lemma_counts_b, sample_size)
    transfer_from_b_to_a = lemma_counts_b - Counter(new_sampled_b_lemmas)
    
    all_a_lemmas = set(new_sampled_a_lemmas.keys()).union(set(transfer_from_b_to_a.keys()))
    all_b_lemmas = set(new_sampled_b_lemmas.keys()).union(set(transfer_from_a_to_b.keys()))

    # return the indices of the new sampled datasets

    new_sampled_a = []
    new_sampled_b = []

    for i, s in enumerate(sampled_dataset_a):
        if s['dependent_lemma'] in all_a_lemmas:
            new_sampled_a.append(("a", i))
        else:
            new_sampled_b.append(("a", i))

    for i, s in enumerate(sampled_dataset_b):
        if s['dependent_lemma'] in all_b_lemmas:
            new_sampled_a.append(("b", i))
        else:
            new_sampled_b.append(("b", i))
    
    return new_sampled_a, new_sampled_b

def make_new_hdf5(directory, model_shorthand, split, sample_a_h5py, sample_b_h5py, new_sample_idx):
    """
    Loop over a dataset, get embeddings using get_embeddings function, and store them in an HDF5 file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    hdf5_path = os.path.join(directory, f"{model_shorthand}.{split}.embeddings.hdf5")
    hdf5_file = h5py.File(hdf5_path, 'w')
    sample_idx_from_a = [i for s, i in new_sample_idx if s == "a"]
    sample_idx_from_b = [i for s, i in new_sample_idx if s == "b"]

    sample_from_a = sample_a_h5py[sample_idx_from_a]
    sample_from_b = sample_b_h5py[sample_idx_from_b]

    # concatenate along the first dimension
    new_sample = np.concatenate([sample_from_a, sample_from_b], axis=0)

    for new_idx, sample_h5py in enumerate(new_sample):
        hdf5_file.create_dataset(f"{new_idx}", data=sample_h5py, compression='gzip', compression_opts=9)

    hdf5_file.close()
    return

def hdf5_to_numpy(file_path):
    with h5py.File(file_path, 'r') as hdf:
        datasets = []
        for key in hdf.keys():
            datasets.append(np.expand_dims(hdf[key][:], axis=0))  # Read each dataset and append to list, unsqueeze it along the first dimension
        big_array = np.concatenate(datasets, axis=0)  # Concatenate along the first axis
    return big_array

def loop_checkpoints_and_save(sample_a_split, sample_b_split, new_sample_a_idx, new_sample_b_idx, model_name, rep="target"):
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    
    for checkpoint in tqdm(branches):
        model_name_preprocessed = model_name.split("/")[-1]
        # get both splits
        sample_a_h5py = hdf5_to_numpy(f'/nlp/scr/jjian/data/wikitext/{sample_a_split}/{rep}/{model_name_preprocessed}.{checkpoint}.{sample_a_split}.embeddings.hdf5')
        sample_b_h5py = hdf5_to_numpy(f'/nlp/scr/jjian/data/wikitext/{sample_b_split}/{rep}/{model_name_preprocessed}.{checkpoint}.{sample_b_split}.embeddings.hdf5')

        # make a new hdf5 for ablation_split_a
        make_new_hdf5(f'/nlp/scr/jjian/data/ablation/wikitext/{sample_a_split}/{rep}/', f"{model_name_preprocessed}.{checkpoint}", sample_a_split, sample_a_h5py, sample_b_h5py, new_sample_a_idx)
        make_new_hdf5(f'/nlp/scr/jjian/data/ablation/wikitext/{sample_b_split}/{rep}/', f"{model_name_preprocessed}.{checkpoint}", sample_b_split, sample_a_h5py, sample_b_h5py, new_sample_b_idx)
    return


def main():
    # get the new indices
    sampled_dataset_a_path = sys.argv[1]
    sampled_dataset_b_path = sys.argv[2]
    rep = sys.argv[3]

    splits = ["motion_balanced", "ditrans_balanced"]

    new_sampled_a_idx, new_sampled_b_idx = redistribution_ablation(sampled_dataset_a_path, sampled_dataset_b_path)

    # loop over the checkpoints and save the new hdf5 files
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"

    loop_checkpoints_and_save(splits[0], splits[1], new_sampled_a_idx, new_sampled_b_idx, model_name, rep=rep)

if __name__ == "__main__":
    main()