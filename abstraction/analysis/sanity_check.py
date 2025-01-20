import numpy as np
import pandas as pd
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import json
import os
import h5py
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

def get_checkpoint_results(model_name, metric, splits, verb_class, source="wikitext"):
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)[3:20]
    model_name_preprocessed = model_name.split("/")[-1]
    
    for checkpoint in tqdm(branches):
        try:
            split_a = json.load(open(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[0]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json"))
            split_b = json.load(open(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[1]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json"))
            split_combined = split_a + split_b
        except FileNotFoundError:
            continue
        except json.decoder.JSONDecodeError:
            print(f"JSONDecodeError at checkpoint {checkpoint}")
            print(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[0]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json")
            print(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[1]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json")
        
        try:
            results_arr = metric(split_combined)
            
        except IndexError:
            print(f"IndexError at checkpoint {checkpoint}")
        
        results_arr.to_csv(f"/nlp/scr/jjian/data/{source}/final/{verb_class}/predictions/sanity_check/{model_name_preprocessed}.{checkpoint}.top75_jsd.{splits[0]}.{splits[1]}.truncated10.csv")
        # add to the h5py with the index equal to the checkpoint

    return 

def top_75_pairwise_cosine(x):
    list_of_words = set.union(*[set([y[0] for y in d["top_k_tokens"]][:10]) for d in x])
    dict_of_words = {w : i for i, w in enumerate(list_of_words)}
    arr = np.zeros((len(x), len(dict_of_words)))

    for i, d in enumerate(x):
        for w, p in d["top_k_tokens"][:10]:
            if p != 0:
                arr[i, dict_of_words[w]] = p
            if p == 0:
                arr[i, dict_of_words[w]] = 1e-5
    
    # give a list of lists of rows to sum together
    rows_to_sum, verbs = get_rows_to_sum(x)
    assert len(verbs) % 2 == 0
    verbs = [f"{v}_matrix" for v in verbs[:len(verbs) // 2]] + [f"{v}_rc" for v in verbs[len(verbs) // 2:]]
    del x
    empty_array = np.zeros((len(rows_to_sum), len(dict_of_words)))
    for i, rows in enumerate(rows_to_sum):
        new_row = np.sum(arr[rows], axis=0)
        # normalize by sum
        #new_row = new_row / np.sum(new_row)
        empty_array[i] = new_row
    
    labeled_array = pd.DataFrame(empty_array, columns=dict_of_words.keys(), index=verbs)

    # sort alpha by index
    labeled_array = labeled_array.sort_index()

    return labeled_array

def get_rows_to_sum(combined_splits):
    verbs = []
    rows_to_sum = []
    prev_verb = ""
    for i, d in enumerate(combined_splits):
        if d["verb"] != prev_verb:
            rows_to_sum.append([i])
            prev_verb = d["verb"]
            verbs.append(d["verb"])
        else:
            rows_to_sum[-1].append(i)

    return rows_to_sum, verbs

if __name__ == "__main__":
    # Define the model name
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"
    # Define the splits and the metric
    splits = ["preposition_fragment_bare_constructed", "rel_clause_obj"]
    verb_class = "motion_annotated"
    results = get_checkpoint_results(model_name, top_75_pairwise_cosine, splits, verb_class, source="wikitext")
