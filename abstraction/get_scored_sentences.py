from abstraction.minicons.minicons import cwe, scorer
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import torch
import os
import h5py
import json
import sys
import shutil
import random
import numpy as np
import jsonlines
import pandas as pd

def load_dataset(dataset):
    return list(jsonlines.open(dataset, "r"))

def save_results(directory, model_name, dataset, split, batch_size=32, model_shorthand=None, checkpoint="main", cache_dir=None, rep='target', tokenizer=None):
    model = scorer.IncrementalLMScorer(model_name, device='cuda:0', revision=checkpoint, cache_dir=cache_dir, tokenizer=tokenizer)
    model_name_preprocessed = model_name.split("/")[-1]
    results_dict = {}
    # batched run through the model
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        sentences_batch = []
        pair_ids = []

        for s in batch:
            sentences_batch.extend([s["sentence_good"], s["sentence_bad"]])
            pair_ids.append(s["pairID"])

        scores = model.sequence_score(sentences_batch, reduction = lambda x: x.sum(0))

        # Store results in dictionary
        for idx, pair_id in enumerate(pair_ids):
            results_dict[pair_id] = {
                "good_score": scores[idx * 2], 
                "bad_score": scores[idx * 2 + 1]
            }

    # turn this into a dataframe with columns good score bad score
    results_df = pd.DataFrame.from_dict(results_dict, orient="index", columns=["good_score", "bad_score"])
    results_df["compare"] = results_df["good_score"] > results_df["bad_score"]
    results_df.to_csv(f"/nlp/scr/jjian/data/blimp/{split}/logprob_eval/{model_name_preprocessed}.{checkpoint}.{split}.logprob.csv")
    return


def loop_checkpoints_and_save(model_name, split, instances, delete_from_cache=False, cache_dir=None, rep="target"):
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    # sort the branches by the checkpoint number
    branches = sorted(branches, key=lambda x: int(x.split('checkpoint-')[-1]))[:]
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    for checkpoint in tqdm(branches, mininterval=5):
        model_name_preprocessed = model_name.split("/")[-1]
        # check if the file already exists
        save_results(f'/nlp/scr/jjian/data/wikitext/{split}/{rep}/', model_name, instances, split, model_shorthand=f"{model_name_preprocessed}.{checkpoint}", checkpoint=checkpoint, cache_dir=cache_dir, rep=rep, tokenizer=tokenizer)
        if delete_from_cache:
            model_path = model_name.replace('/', '--')
            shutil.rmtree(f"/sailhome/jjian/.cache/huggingface/hub_1/models--{model_path}")
    return

if __name__ == "__main__":
    corpus_path = sys.argv[1]
    split = sys.argv[2]
    rep = sys.argv[3]
    
    model_list = ["stanford-crfm/battlestar-gpt2-small-x49"]
    cache_dir = "/nlp/scr/jjian/mistral-checkpoints/"
    sample_size = 2000

    dataset = load_dataset(corpus_path)

    for model in model_list:
        loop_checkpoints_and_save(model, split, dataset, cache_dir=cache_dir, delete_from_cache=False, rep=rep)