import os
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
    model_name = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name, cache_dir="/sailhome/jjian/.cache/huggingface")

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        tokenizer.pad_token = "<|pad|>"

    eos_token = tokenizer.eos_token  # GPT-2 already has an EOS token

    file_path = "/nlp/scr/jjian/datasets/openwebtext_filtered/ditransitive.motion.bash_filtered.combined.txt"  # Change this to your actual file

    # Count total lines in the file for progress tracking
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    count = 0
    token_ids = []

    # Read file line by line and tokenize in batches with tqdm
    with open(file_path, 'r', encoding='utf-8') as f:
        for batch in batched(f, 16, total=total_lines):
            batch = [line.strip() + eos_token for line in batch]  # Append EOS token to each line
            
            encoded = tokenizer(batch, padding="longest", return_tensors="pt")
            attention_mask = encoded["attention_mask"]

            # Append token_ids to list without padding
            token_ids.extend(encoded["input_ids"][attention_mask == 1].tolist())

            count += len(batch)  # Update processed line count

    # Save as np.uint16
    token_ids = np.array(token_ids, dtype=np.uint16)
    np.save(f"/nlp/scr/jjian/datasets/openwebtext_filtered/ditransitive.motion.bash_filtered.combined.token_ids.npy", token_ids)
