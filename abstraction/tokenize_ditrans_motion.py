import os
import sys
import numpy as np
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from itertools import islice


def batched(iterable, n, total=None):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(tqdm(iterable, total=total, desc="Reading file"))
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


if __name__ == "__main__":
    file_path = sys.argv[1]
    output_path = sys.argv[2]
    cache_dir = os.environ.get("CACHE_DIR", "./cache")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        tokenizer.pad_token = "<|pad|>"

    eos_token = tokenizer.eos_token

    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    token_ids = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for batch in batched(f, 16, total=total_lines):
            batch = [line.strip() + eos_token for line in batch]
            encoded = tokenizer(batch, padding="longest", return_tensors="pt")
            attention_mask = encoded["attention_mask"]
            token_ids.extend(encoded["input_ids"][attention_mask == 1].tolist())

    token_ids = np.array(token_ids, dtype=np.uint16)
    np.save(output_path, token_ids)