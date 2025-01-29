import os
cache_dir = "/nlp/scr/jjian/mistral-checkpoints"
    
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

from datasets import load_dataset
from datasets import load_dataset_builder
from transformers import AutoModel
from huggingface_hub import list_repo_refs, snapshot_download
from tqdm import tqdm

#dataset = load_dataset("Skylion007/openwebtext", cache_dir=cache_dir, trust_remote_code=True)
#dataset.save_to_disk(cache_dir)


if __name__ == "__main__":
    model_name = "stanford-crfm/expanse-gpt2-small-x777"
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    cache_dir = "/nlp/scr/jjian/mistral-checkpoints"
    for checkpoint in tqdm(branches):
        model = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True, revision=checkpoint, cache_dir=cache_dir)
        del model