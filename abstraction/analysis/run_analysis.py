import sys
import numpy as np
import pandas as pd
import h5py
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import random
import json
import os
from abstraction.analysis.metrics import pca_classifier, pca_classifier_per_lemma, logistic_regression


def get_checkpoint_results(model_name, metric, splits, rep_a="target", rep_b="target", sample_a_excl=[], sample_b_excl=[], source="wikitext", pca_rank=4):
    all_results = []  # List to store all DataFrames
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)
    branches = branches[:150]
    
    model_name_preprocessed = model_name.split("/")[-1]
    
    for checkpoint in tqdm(branches):
        try:
            split_a = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[0]}/{rep_a}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[0]}.embeddings.hdf5", 'r')
            split_b = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[1]}/{rep_b}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[1]}.embeddings.hdf5", 'r')
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

def get_checkpoint_results_nominals(model_name, metric, splits, rep_a="target", rep_b="target", rep_c="target", sample_a_excl=[], sample_b_excl=[], source="wikitext", pca_rank=4):
    all_results = []  # List to store all DataFrames
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)[:150]
    
    model_name_preprocessed = model_name.split("/")[-1]
    
    for checkpoint in tqdm(branches):
        try:
            split_a = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[0]}/{rep_a}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[0]}.embeddings.hdf5", 'r')
            split_b = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[1]}/{rep_b}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[1]}.embeddings.hdf5", 'r')
            split_c = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[2]}/{rep_c}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[2]}.embeddings.hdf5", 'r')
        except FileNotFoundError:
            continue
        
        sample_b = [np.array(x) for x in split_c.values()]
        sample_b_n = len(sample_b)
        sample_a = random.sample([np.array(x) for x in split_a.values()], sample_b_n // 2) + random.sample([np.array(x) for x in split_b.values()], sample_b_n // 2)
        try:
            results_df = metric(sample_a, sample_b, layers=range(0, 12), labels=splits, checkpoint_n=checkpoint, pca_rank=pca_rank)
        except IndexError:
            print(f"IndexError at checkpoint {checkpoint}")
        # Append the DataFrame to the list
        all_results.append(results_df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(all_results, ignore_index=False, axis=1)
    
    return concatenated_df

def get_verb_split_mapping(dataset1_json, dataset2_json):
    
    with open(dataset1_json) as f:
        dataset1 = json.load(f)
    with open(dataset2_json) as f:
        dataset2 = json.load(f)
    
    mapping = {}

    # concatenate the two lists
    dataset_concat = dataset1 + dataset2

    for i, sentence in enumerate(dataset_concat):
        if sentence["dependent_lemma"] not in mapping:
            mapping[sentence["dependent_lemma"]] = []
        mapping[sentence["dependent_lemma"]].append(i)

    return mapping

def get_checkpoint_results_lemma(model_name, metric, splits, split_a_json, split_b_json, rep_a="target", rep_b="target", sample_a_excl=[], sample_b_excl=[], source="wikitext", pca_rank=4):
    all_results = []  # List to store all DataFrames
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)
    
    model_name_preprocessed = model_name.split("/")[-1]
    mapping = get_verb_split_mapping(split_a_json, split_b_json)
    for checkpoint in tqdm(branches[:150]):
        try:
            split_a = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[0]}/{rep_a}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[0]}.embeddings.hdf5", 'r')
            split_b = h5py.File(f"/nlp/scr/jjian/data/{source}/{splits[1]}/{rep_b}/{model_name_preprocessed}.checkpoint-{checkpoint}.{splits[1]}.embeddings.hdf5", 'r')
        except FileNotFoundError:
            continue
        
        # get the samples in numerical order of their indices the indices are strings
        
        sample_a = [np.array(split_a[str(i)]) for i in range(len(split_a))]
        sample_b = [np.array(split_b[str(i)]) for i in range(len(split_b))]
        
        try:
            results_df = metric(sample_a, sample_b, mapping=mapping, layers=range(0, 12), labels=splits, checkpoint_n=checkpoint, pca_rank=pca_rank)
        except IndexError:
            print(f"IndexError at checkpoint {checkpoint}")
        # Append the DataFrame to the list
        all_results.append(results_df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(all_results, ignore_index=False, axis=0)
    
    return concatenated_df


if __name__ == "__main__":
    rep_a = sys.argv[1]
    rep_b = sys.argv[2]
    rep_c = sys.argv[3]
    # Define the splits and the metric
    splits = ["ditrans_nominals", "motion_balanced", "ditrans_balanced"]
    pca_rank = int(sys.argv[4])
    metric = pca_classifier

    # Define the model name
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"
    
    # If needed the json files for the datasets

    split_a_json = "/nlp/scr/jjian/datasets/wikitext_parsed/by.passive.sampled.json"
    split_b_json = "/nlp/scr/jjian/datasets/wikitext_parsed/by.adjunct.constructed.json"

    # Get the results

    #results = get_checkpoint_results_nominals(model_name, metric, splits, rep_a=rep_a, rep_b=rep_b, rep_c=rep_c, source="wikitext", pca_rank=pca_rank)
    results = get_checkpoint_results_nominals(model_name, metric, splits, rep_a=rep_a, rep_b=rep_b, rep_c=rep_c, source="wikitext", pca_rank=pca_rank)

    # Save the results
    outdir = f"results/wikitext/ditrans_nominals/pca_{pca_rank}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    results.to_csv(f'{outdir}/battlestar_small.{splits[0]}.{splits[1]}.{splits[2]}.{rep_a}.tsv', sep='\t', index=True)