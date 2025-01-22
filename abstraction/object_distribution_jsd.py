import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import json
import os
from typing import List, Dict, Any
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import jensenshannon
import argparse

def get_next_token_distribution_batch(input_prefixes, model, tokenizer, model_name="gpt2"):
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set the model to evaluation mode
    model.eval()

    # Tokenize the batch of inputs save the original length of each sequence before padding to get the logits for the next token
    encoded = tokenizer(input_prefixes, return_tensors='pt', padding="longest")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device) 
    sequence_lengths = attention_mask.sum(dim=1) - 1  

    # Forward pass to get the logits for the next token
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get the logits for the last token in each input sequence
    batch_size = logits.shape[0]
    next_token_logits = torch.stack([
        logits[i, sequence_lengths[i], :] 
        for i in range(batch_size)
    ])

    # Apply softmax to get the probability distributions
    probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
    #print(probabilities.shape)

    return probabilities

def get_rows_to_sum(combined_splits):
    rows_to_sum = []
    prev_verb = ""
    for i, d in enumerate(combined_splits):
        if d["dependent_lemma"] != prev_verb:
            rows_to_sum.append([i])
            prev_verb = d["dependent_lemma"]
        else:
            rows_to_sum[-1].append(i)

    return rows_to_sum


def loop_checkpoints_and_save(model_name, split, instances_class_one, instances_class_two, cache_dir=None, rep="verb_fragment", batch_size=32, branch=None, comparison_setting="verbwise", source="wikitext", rep_two=None):
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    if branch == 10:
        branches = sorted(branches, key=lambda x: int(x.split('checkpoint-')[-1]))
    elif branch == 0:
        branches = sorted(branches, key=lambda x: int(x.split('checkpoint-')[-1]))[:20]
    elif branch == 1:
        branches = sorted(branches, key=lambda x: int(x.split('checkpoint-')[-1]))[:300]
    elif branch == 2:
        branches = sorted(branches, key=lambda x: int(x.split('checkpoint-')[-1]))[300:]
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|pad|>"]}
        )
        tokenizer.pad_token = "<|pad|>"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name_preprocessed = model_name.split("/")[-1]
    if rep_two is not None:
        output_path = f'/nlp/scr/jjian/data/{source}/final/{split}/predictions/{model_name_preprocessed}.full_jsd.{rep}.{rep_two}.{comparison_setting}.hdf5'
    else:
        output_path = f'/nlp/scr/jjian/data/{source}/final/{split}/predictions/{model_name_preprocessed}.full_jsd.{rep}.{rep}.{comparison_setting}.hdf5'
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    output_file = h5py.File(output_path, "w")
    

    for checkpoint in tqdm(branches):

        checkpoint_number = int(checkpoint.split("checkpoint-")[-1])

        model = AutoModelForCausalLM.from_pretrained(model_name, revision=checkpoint, cache_dir=cache_dir).to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

        # Process instances_class_one in batches
        results_class_one = []
        results_class_two = []
        # add tqdm to the loop

        for i in range(0, len(instances_class_one), batch_size):
            batch = instances_class_one[i:i + batch_size]
            input_prefixes = [data[rep].strip() for data in batch]

            # Get the next token distribution for the batch
            probabilities = get_next_token_distribution_batch(input_prefixes, model, tokenizer, model_name)

            results_class_one.extend(probabilities)
        
        for i in range(0, len(instances_class_two), batch_size):
            batch = instances_class_two[i:i + batch_size]
            if rep_two is not None:
                input_prefixes = [data[rep_two].strip() for data in batch]
            else:
                input_prefixes = [data[rep].strip() for data in batch]

            # Get the next token distribution for the batch
            probabilities = get_next_token_distribution_batch(input_prefixes, model, tokenizer, model_name)

            results_class_two.extend(probabilities)
        

        if comparison_setting == "verbwise":

            rows_to_sum_class_one = get_rows_to_sum(instances_class_one)
            rows_to_sum_class_two = get_rows_to_sum(instances_class_two)

            results_class_one = [torch.stack([results_class_one[i] for i in rows]).cpu() for rows in rows_to_sum_class_one]
            results_class_two = [torch.stack([results_class_two[i] for i in rows]).cpu() for rows in rows_to_sum_class_two]

            results_class_one = [torch.mean(r, dim=0) for r in results_class_one]
            results_class_two = [torch.mean(r, dim=0) for r in results_class_two]
            results = results_class_one + results_class_two

            sim_arr = np.zeros((len(results), len(results)))
            for i in range(len(results)):
                for j in range(len(results)):
                    sim_arr[i, j] = jensenshannon(results[i], results[j])

            output_file.create_dataset(f"{checkpoint_number}", data=sim_arr, compression="gzip", compression_opts=9)

        elif comparison_setting == "pairwise":
            results_class_one = [r.cpu().numpy() for r in results_class_one]
            results_class_two = [r.cpu().numpy() for r in results_class_two]
            assert len(results_class_one) == len(results_class_two)
            results = results_class_one + results_class_two
            
            sim_list = []

            for i in range(len(results) // 2):
                split_a_dist = results[i]
                split_b_dist = results[i + len(results) // 2]
                sim_list.append(jensenshannon(split_a_dist, split_b_dist))
            sim_arr = np.array(sim_list)
            
            output_file.create_dataset(f"{checkpoint_number}", data=sim_arr, compression="gzip", compression_opts=9)

    output_file.close()
    return


def main():
    parser = argparse.ArgumentParser(description="Process model checkpoints and dataset splits.")
    parser.add_argument("--rep", type=str, required=True, help="Repetition identifier")
    parser.add_argument("--rep_two", type=str, required=False, help="Repetition identifier")
    parser.add_argument("--branch", type=int, required=True, help="Branch identifier")
    parser.add_argument("--split", type=str, required=True, help="Dataset split type")
    parser.add_argument("--comparison_setting", type=str, required=True, help="Comparison setting")
    parser.add_argument("--class_one_file", type=str, required=True, help="Class one file")
    parser.add_argument("--class_two_file", type=str, required=True, help="Class two file")
    parser.add_argument("--source", type=str, required=True, help="Source")
    args = parser.parse_args()

    model_name = "stanford-crfm/battlestar-gpt2-small-x49"
    cache_dir = "/nlp/scr/jjian/mistral-checkpoints/"

    class_one_json = json.load(open(args.class_one_file, "r"))
    class_two_json = json.load(open(args.class_two_file, "r"))

    if args.comparison_setting == "pairwise":
        assert len(class_one_json) == len(class_two_json)
        assert args.rep_two is not None

    loop_checkpoints_and_save(
        model_name,
        args.split,
        class_one_json,
        class_two_json,
        cache_dir=cache_dir,
        rep=args.rep,
        batch_size=16,
        branch=args.branch,
        source=args.source,
        comparison_setting=args.comparison_setting,
        rep_two=args.rep_two,
    )

if __name__ == "__main__":
    main()
