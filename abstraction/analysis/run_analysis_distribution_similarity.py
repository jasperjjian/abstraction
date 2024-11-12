import numpy as np
import pandas as pd
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import json
import os
import h5py
from sklearn.metrics.pairwise import cosine_similarity

def get_checkpoint_results(model_name, metric, splits, verb_class, source="wikitext"):
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)[3:]
    model_name_preprocessed = model_name.split("/")[-1]
    
    # make an h5py file to store the results
    f = h5py.File(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[0]}/{model_name_preprocessed}.top75_cosine.ditrans_prep.ditrans_verb.hdf5", "w")

    for checkpoint in tqdm(branches):
        try:
            split_a = json.load(open(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[0]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json"))
            split_b = json.load(open(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[1]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json"))
            split_combined = split_a + split_b
        except FileNotFoundError:
            continue
        
        try:
            results_arr = metric(split_combined)
            
        except IndexError:
            print(f"IndexError at checkpoint {checkpoint}")
        
        # add to the h5py with the index equal to the checkpoint

        f.create_dataset(f"{checkpoint}", data=results_arr, compression="gzip", compression_opts=9)
    
    f.close()
    return 

def top_75_pairwise_cosine(x):
    list_of_words = set.union(*[set([y[0] for y in d["top_k_tokens"]]) for d in x])
    dict_of_words = {w : i for i, w in enumerate(list_of_words)}
    arr = np.zeros((len(x), len(dict_of_words)))

    for i, d in enumerate(x):
        for w, p in d["top_k_tokens"]:
            if p != 0:
                arr[i, dict_of_words[w]] = p
            if p == 0:
                arr[i, dict_of_words[w]] = 1e-4
    
    # give a list of lists of rows to sum together
    rows_to_sum = get_rows_to_sum(x)
    empty_array = np.zeros((len(rows_to_sum), len(dict_of_words)))
    for i, rows in enumerate(rows_to_sum):
        new_row = np.sum(arr[rows], axis=0)
        empty_array[i] = new_row
    
    sim_arr = cosine_similarity(empty_array)

    return sim_arr

def get_rows_to_sum(combined_splits):
    rows_to_sum = []
    prev_verb = ""
    for i, d in enumerate(combined_splits):
        if d["verb"] != prev_verb:
            rows_to_sum.append([i])
            prev_verb = d["verb"]
        else:
            rows_to_sum[-1].append(i)

    return rows_to_sum

if __name__ == "__main__":
    # Define the model name
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"
    # Define the splits and the metric
    splits = ["preposition_fragment", "verb_fragment"]
    verb_class = "ditransitive"
    results = get_checkpoint_results(model_name, top_75_pairwise_cosine, splits, verb_class, source="wikitext")