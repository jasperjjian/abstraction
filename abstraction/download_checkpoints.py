import os
import sys

cache_dir = os.environ.get("CACHE_DIR", "./cache")

os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

from datasets import load_dataset
from transformers import AutoModel
from huggingface_hub import list_repo_refs
from tqdm import tqdm


if __name__ == "__main__":
    model_name = sys.argv[1]

    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    branches.sort(key=lambda x: int(x.split('-')[-1]) if x.startswith('checkpoint-') else float('inf'))

    for checkpoint in tqdm(branches):
        model = AutoModel.from_pretrained(
            model_name, return_dict=True, output_hidden_states=True,
            revision=checkpoint, cache_dir=cache_dir
        )
        del model