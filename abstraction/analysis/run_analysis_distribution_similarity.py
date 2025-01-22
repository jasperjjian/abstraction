import numpy as np
import pandas as pd
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import json
import os
import h5py
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import sys

def get_checkpoint_results(model_name, metric, splits, verb_class, source="wikitext"):
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)[3:]
    model_name_preprocessed = model_name.split("/")[-1]
    if not os.path.exists(f"/nlp/scr/jjian/data/{source}/final/{verb_class}/predictions/"):
        os.makedirs(f"/nlp/scr/jjian/data/{source}/final/{verb_class}/predictions/")
    # make an h5py file to store the results
    f = h5py.File(f"/nlp/scr/jjian/data/{source}/final/{verb_class}/predictions/{model_name_preprocessed}.top75_jsd.{splits[0]}.{splits[1]}.pairwise.hdf5", "w")
    # check if directory exists
    
        
    for checkpoint in tqdm(branches):
        try:
            split_a = json.load(open(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[0]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json"))
            split_b = json.load(open(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[1]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json"))
            verbs = [d["verb"] for d in split_a]
            set_a = set(verbs)
            split_b = [d for d in split_b if d["verb"] in set_a]
            assert len(split_a) == len(split_b)
            split_combined = split_a + split_b
        except FileNotFoundError:
            print(f"FileNotFoundError at checkpoint {checkpoint}")
        except json.decoder.JSONDecodeError:
            print(f"JSONDecodeError at checkpoint {checkpoint}")
            print(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[0]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json")
            print(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[1]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json")
            continue
        try:
            results_arr = metric(split_combined)
            
        except IndexError:
            print(f"IndexError at checkpoint {checkpoint}")
        
        # add to the h5py with the index equal to the checkpoint

        f.create_dataset(f"{checkpoint}", data=results_arr, compression="gzip", compression_opts=9)

    f.close()
    return 

def sentence_wise_jsd(x):
    list_of_words = set.union(*[set([y[0] for y in d["top_k_tokens"]]) for d in x])
    dict_of_words = {w : i for i, w in enumerate(list_of_words)}
    arr = np.zeros((len(x), len(dict_of_words)))

    for i, d in enumerate(x):
        for w, p in d["top_k_tokens"]:
            if p != 0:
                arr[i, dict_of_words[w]] = p
            if p == 0:
                arr[i, dict_of_words[w]] = 1e-5
    sim_list = []
    del x
    for i in range(len(arr) // 2):
        split_a_dist = arr[i]
        split_b_dist = arr[i + len(arr) // 2]
        sim_list.append(jensenshannon(split_a_dist, split_b_dist))
    sim_arr = np.array(sim_list)
    return sim_arr


def top_75_pairwise_cosine(x):
    list_of_words = set.union(*[set([y[0] for y in d["top_k_tokens"]]) for d in x])
    dict_of_words = {w : i for i, w in enumerate(list_of_words)}
    arr = np.zeros((len(x), len(dict_of_words)))

    for i, d in enumerate(x):
        for w, p in d["top_k_tokens"]:
            if p != 0:
                arr[i, dict_of_words[w]] = p
            if p == 0:
                arr[i, dict_of_words[w]] = 1e-5
    
    # give a list of lists of rows to sum together
    rows_to_sum = get_rows_to_sum(x)
    del x
    empty_array = np.zeros((len(rows_to_sum), len(dict_of_words)))
    for i, rows in enumerate(rows_to_sum):
        new_row = np.sum(arr[rows], axis=0)
        # normalize by sum
        #new_row = new_row / np.sum(new_row)
        empty_array[i] = new_row
    # individual sentence
    #empty_array = arr
    #sim_arr = cosine_similarity(empty_array)
    #pairwise jensen shannon
    sim_arr = np.zeros((len(empty_array), len(empty_array)))
    for i in range(len(empty_array)):
        for j in range(len(empty_array)):
            sim_arr[i, j] = jensenshannon(empty_array[i], empty_array[j])
    # round to 5 decimal places
    #sim_arr = np.round(sim_arr, 5)
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
    splits = ["preposition_fragment_bare_constructed_biclausal", "rel_clause_obj_biclausal"]
    verb_class = sys.argv[1]
    results = get_checkpoint_results(model_name, sentence_wise_jsd, splits, verb_class, source="wikitext")
