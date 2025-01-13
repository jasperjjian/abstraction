import numpy as np
import pandas as pd
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import json
import sys

def get_checkpoint_results(model_name, metric, splits, verb_class, source="wikitext"):
    out = list_repo_refs(model_name)
    branches = [int(b.name.split('checkpoint-')[-1]) for b in out.tags]
    #branches = random.sample(branches, k=60)
    branches = sorted(branches)[3:]
    model_name_preprocessed = model_name.split("/")[-1]
    # results array should have each row equal to a checkpoint
    results_arr = []
    verbs = []
    for checkpoint in tqdm(branches):
        try:
            split_a = json.load(open(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[0]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json", "r"))
            split_b = json.load(open(f"/nlp/scr/jjian/data/{source}/{verb_class}/predictions/{splits[1]}/{model_name_preprocessed}.checkpoint-{checkpoint}.predictions.json", "r"))
        except FileNotFoundError:
            continue
        if checkpoint == branches[0]:
            verbs = list(dict.fromkeys([d["verb"] for d in split_a + split_b]).keys())
        try:
            results_df_at_checkpoint = metric(split_a, split_b, verbs)
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

def average(split_a, split_b, verbs):
    verbs_dict = {verb: {"the": [], "was": []} for verb in verbs}
    for d in split_a:
        if d["top_k_tokens"][0] == " the":
            verbs_dict[d["verb"]]["the"].append((d["top_k_tokens"][0], d["top_k_tokens"][1]))
        elif d["top_k_tokens"][0] == " was":
            verbs_dict[d["verb"]]["was"].append((d["top_k_tokens"][0], d["top_k_tokens"][1]))

    for d in split_b:
        if d["top_k_tokens"][0] == " the":
            verbs_dict[d["verb"]]["the"].append((d["top_k_tokens"][0], d["top_k_tokens"][1]))
        elif d["top_k_tokens"][0] == " was":
            verbs_dict[d["verb"]]["was"].append((d["top_k_tokens"][0], d["top_k_tokens"][1]))
    results_dict = {}
    for verb in verbs:
        the_a = np.mean([x[1] for x in verbs_dict[verb]["the"]])
        was_a = np.mean([x[1] for x in verbs_dict[verb]["was"]])
        results_dict[f"{verb}_the"] = the_a
        results_dict[f"{verb}_was"] = was_a
    
    results_df = pd.DataFrame(results_dict, index=[0])
    return results_df

def tse(split_a, split_b, verbs):
    verbs_dict = {verb : [] for verb in verbs}
    for d_a, d_b in zip(split_a, split_b):
        if d_a["input_prefix"] == d_b["input_prefix"]:
            was_prob = 0
            the_prob = 0
            if " the" == d_a["top_k_tokens"][0]:
                the_prob = d_a["top_k_tokens"][1]
            elif " was" == d_a["top_k_tokens"][0]:
                was_prob = d_a["top_k_tokens"][1]
            if " the" == d_b["top_k_tokens"][0]:
                the_prob = d_b["top_k_tokens"][1]
            elif " was" == d_b["top_k_tokens"][0]:
                was_prob = d_b["top_k_tokens"][1]
            if was_prob == 0 or the_prob == 0:
                continue
            verb = d_a["verb"]
            verbs_dict[verb].append(int(was_prob > the_prob))
    results_dict = {}
    for verb in verbs:
        results_dict[verb] = np.mean(verbs_dict[verb])
    results_df = pd.DataFrame(results_dict, index=[0])
    return results_df


if __name__ == "__main__":
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"
    verb_class = "reciprocal_annotated"
    splits = ["rel_clause_obj/the", "rel_clause_obj/was"]
    results_df = get_checkpoint_results(model_name, tse, splits, verb_class)

    results_df.to_csv(f"/afs/cs.stanford.edu/u/jjian/projects/abstraction/results/{verb_class}_rc_tse.csv")
