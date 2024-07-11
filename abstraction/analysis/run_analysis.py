import sys
import numpy as np
import pandas as pd
import h5py
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import random
import json
import os
from abstraction.analysis.metrics import pca_classifier


def get_checkpoint_results(model_name, metric, splits, rep="target", sample_a_excl=[], sample_b_excl=[], source="wikitext", pca_rank=4):
    all_results = []  # List to store all DataFrames
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)
    
    model_name_preprocessed = model_name.split("/")[-1]
    
    for checkpoint in tqdm(branches):
        try:
            split_a = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[0]}/{rep}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[0]}.embeddings.hdf5", 'r')
            split_b = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[1]}/{rep}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[1]}.embeddings.hdf5", 'r')
        except FileNotFoundError:
            continue
        
        sample_a = [np.array(x) for x in split_a.values()]
        sample_b = [np.array(x) for x in split_b.values()]
        
        try:
            results_df = metric(sample_a, sample_b, layers=range(0, 12), labels=splits, checkpoint_n=checkpoint, pca_rank=pca_rank)
        except IndexError:
            print(f"IndexError at checkpoint {checkpoint}")
        # Append the DataFrame to the list
        all_results.append(results_df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(all_results, ignore_index=False, axis=1)
    
    return concatenated_df


if __name__ == "__main__":
    rep = sys.argv[1]
    # Define the splits and the metric
    splits = ["with.substance", "with.adjunct"]
    pca_rank = int(sys.argv[2])
    metric = pca_classifier
    #sample_a_remove = [40, 578, 678, 801, 1509, 1759, 1404, 1418, 1537]
    #sample_b_remove = [3, 930, 631, 741, 1109, 1925]

    # Define the model name
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"
    
    # Get the results
    results = get_checkpoint_results(model_name, metric, splits, rep=rep, source="wikitext", pca_rank=pca_rank)

    # Save the results
    outdir = f"results/wikitext/with/pca_{pca_rank}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    results.to_csv(f'{outdir}/battlestar_small.{splits[0]}.{splits[1]}.{rep}.tsv', sep='\t', index=False)