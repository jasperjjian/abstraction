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


def get_checkpoint_results(model_name, metric, splits, source="wikitext"):
    all_results = []  # List to store all DataFrames
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)
    branches = branches
    model_name_preprocessed = model_name.split("/")[-1]
    
    for checkpoint in tqdm(branches):
        try:
            split_a = json.load(open(f"/nlp/scr/jjian/data/{source}/ditransitive/{splits[0]}/{model_name_preprocessed}.checkpoint-{checkpoint}.entropy.json"))
        except FileNotFoundError:
            continue
        
        try:
            results_df = metric(split_a, checkpoint_n=checkpoint)
        except IndexError:
            print(f"IndexError at checkpoint {checkpoint}")
        # Append the DataFrame to the list
        all_results.append(results_df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(all_results, ignore_index=False, axis=1)
    
    return concatenated_df

def get_checkpoint_results_lemma(model_name, metric, splits, source="wikitext"):
    all_results = []  # List to store all DataFrames
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)
    branches = branches
    model_name_preprocessed = model_name.split("/")[-1]
    
    for checkpoint in tqdm(branches):
        try:
            split_a = json.load(open(f"/nlp/scr/jjian/data/{source}/ditransitive/{splits[0]}/{model_name_preprocessed}.checkpoint-{checkpoint}.entropy.json"))
        except FileNotFoundError:
            continue
        # get the list of unique verbs
        verbs = list(set([d["verb"] for d in split_a]))
        for verb in verbs:
            try:
                split_a_verb = [d for d in split_a if d["verb"] == verb]
                results_df = metric(split_a_verb, checkpoint_n=checkpoint, verb=verb)
            except IndexError:
                print(f"IndexError at checkpoint {checkpoint}")
            # Append the DataFrame to the list
            all_results.append(results_df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(all_results, ignore_index=False, axis=0)
    concatenated_df = concatenated_df.groupby(concatenated_df.index).sum()
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
    

if __name__ == "__main__":
    # Define the model name
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"
    # Define the splits and the metric
    splits = ["verb_fragment"]

    # lambda function to get the length of the token list and put that into the dataframe with the checkpoint name
    #metric = lambda x, checkpoint_n: pd.DataFrame({checkpoint_n : [sum([d["entropy"] for d in x]) / len(x)]})
    metric = lambda x, checkpoint_n, verb: pd.DataFrame({checkpoint_n: [sum([d["entropy"] for d in x]) / len(x)]}, index=[verb])

    #results = get_checkpoint_results_nominals(model_name, metric, splits, rep_a=rep_a, rep_b=rep_b, rep_c=rep_c, source="wikitext", pca_rank=pca_rank)
    results = get_checkpoint_results_lemma(model_name, metric, splits, source="wikitext")

    # Save the results
    outdir = f"results/wikitext/ditransitive/{splits[0]}/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    results.to_csv(f'{outdir}/battlestar_small.object_distribution.entropy.{splits[0]}.verb_wise.tsv', sep='\t', index=True)