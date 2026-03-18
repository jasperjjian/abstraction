import numpy as np
import pandas as pd
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import json
import sys
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

def configure_new_dataset(old_dataset, new_dataset):
    old = json.load(open(old_dataset, "r"))
    new = json.load(open(new_dataset, "r"))
    old = [d for d in old if d["decision"] == "y"]
    new_indexed = [(d["sent_id"], d["text"]) for d in new]
    reindex = []
    for i, d in enumerate(old):
        if (d["sent_id"], d["text"]) in new_indexed:
            reindex.append(i)
    assert len(reindex) == len(new)
    return reindex


def get_checkpoint_results(model_name, metric, splits, verb_class, noun, source="wikitext", reindex=None):
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)[:]
    model_name_preprocessed = model_name.split("/")[-1]
    # results array should have each row equal to a checkpoint
    results_arr = []
    verbs = []
    for checkpoint in tqdm(branches):
        try:
            split_a = json.load(open(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[0]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json", "r"))
            split_b = json.load(open(f"/nlp/scr/jjian/data/{source}/motion_annotated/predictions/{splits[1]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json", "r"))
            if reindex is not None:
                assert len(reindex[0]) < len(split_a)
                assert len(reindex[1]) < len(split_b)
                split_a = [split_a[i] for i in reindex[0]]
                split_b = [split_b[i] for i in reindex[1]]
        except FileNotFoundError:
            continue
        if checkpoint == branches[0]:
            verbs = list(dict.fromkeys([d["verb"] for d in split_a + split_b]).keys())
        try:
            results_df_at_checkpoint = metric(split_a, split_b, noun, verbs)
            # add the results to the results array where the row label is the checkpoint
            results_arr.append(results_df_at_checkpoint)
        except IndexError:
            continue    
    # join all the results into a single dataframe
    results_df = pd.concat(results_arr, axis=0)
    # make the indices the checkpoints
    results_df['Checkpoint'] = branches
    results_df = results_df.set_index('Checkpoint')
    return results_df

def heuristic_scoring_3(split_a, split_b, noun, verbs):
    # make a dataframe with the columns being "split_a_the" "split_a_is" "split_b_the" "split_b_is"       

    landlord_scores_a = 0
    for d in split_a:
        if f" {noun}" in [x[0] for x in d["top_k_tokens"]]:
            landlord_scores_a += 1
    landlord_scores_a = landlord_scores_a / len(split_a)
    
    landlord_scores_b = 0
    
    for d in split_b:
        if f" {noun}" in [x[0] for x in d["top_k_tokens"]]:
            landlord_scores_b += 1
    landlord_scores_b = landlord_scores_b / len(split_b)

    results_df = pd.DataFrame({'split_a_landlord': landlord_scores_a, 'split_b_landlord': landlord_scores_b}, index=[0])
    return results_df

def heuristic_scoring_4(split_a, split_b, noun, verbs):
    landlord_scores_dict = {}

    split_combined = split_a + split_b

    # split the split_combined into a list of dictionaries based on the verb
    split_combined_new = {v: [] for v in verbs} 
    for d in split_combined:
        split_combined_new[d["verb"]].append(d)
    
    for v in verbs:
        scores = []
        for d in split_combined_new[v]:
            scores_dict = {x[0]: x[1] for x in d["top_k_tokens"]}
            #scores_dict = [x[0] for x in d["top_k_tokens"]]
            if f" {noun}" in scores_dict:
                # append the rank based on the order of the top_k_tokens
                scores.append(scores_dict[f" {noun}"])
            else:
                scores.append(0)
            # if " landlord" in scores_dict:
            #     scores.append(scores_dict[" landlord"])
            # else:
            #     scores.append(0)
        
        landlord_scores_dict[v] = np.array(scores).mean()       

    results_df = pd.DataFrame(landlord_scores_dict, index=[0])
    return results_df

if __name__ == "__main__":
    TARGET_NOUN = sys.argv[1]
    model_name = "stanford-crfm/darkmatter-gpt2-small-x343"
    splits = ["preposition_fragment/prototype_nouns", "preposition_fragment/prototype_nouns"]
    verb_class = "ditrans_annotated"

    results = get_checkpoint_results(
        model_name, heuristic_scoring_4, splits, verb_class, TARGET_NOUN, source="wikitext"
    )
    results.to_csv(
        f"/afs/cs.stanford.edu/u/jjian/projects/abstraction/results/prototype_nouns/"
        f"ditrans_motion_{TARGET_NOUN}.darkmatter-343.verbwise.csv"
    )