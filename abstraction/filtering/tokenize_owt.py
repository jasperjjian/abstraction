import os

cache_dir = "/nlp/scr/jjian/datasets/openwebtext"
    
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2TokenizerFast
from tqdm import tqdm
import torch
import numpy as np
from itertools import islice

def batched(iterable, n, total=None):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(tqdm(iterable, total=total))
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

if __name__ == "__main__":
    model_name = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name, cache_dir="/sailhome/jjian/.cache/huggingface")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|pad|>"]}
        )
        tokenizer.pad_token = "<|pad|>"
    print("Loading dataset")
    dataset = load_dataset("openwebtext", cache_dir=cache_dir)
    #dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="/nlp/scr/jjian/datasets/wikitext-103-raw-v1")
    dataset_len = len(dataset["train"])
    print(f"Dataset length: {dataset_len}")
    print("Loaded dataset")
    dataset = dataset["train"].to_iterable_dataset(num_shards=128)
    shuffled_iterable_dataset = dataset.shuffle(seed=42, buffer_size=1000)
    count = 0
    token_ids = []
    # batch tokenize
    for batch in batched(shuffled_iterable_dataset, 16, total=dataset_len):
        batch = [d["text"] for d in batch]
        encoded = tokenizer(batch, padding="longest", return_tensors="pt")
        attention_mask = encoded["attention_mask"]
        # append token_ids to list without padding
        token_ids.extend(encoded["input_ids"][attention_mask == 1].tolist())
        count += 16
        if count > dataset_len // 8:
            break
    
    """for batch in tqdm(shuffled_iterable_dataset, total=dataset_len):
        batch = batch["text"]
        encoded = tokenizer(batch, padding="longest", return_tensors="pt")
        attention_mask = encoded["attention_mask"]
        # append token_ids to list without padding
        token_ids.extend(encoded["input_ids"][attention_mask == 1].tolist())"""
        
    token_ids = np.array(token_ids)
    # save as np unit16
    np.save(f"{cache_dir}_filtered/openwebtext_token_ids.42.1M.true.npy", token_ids)