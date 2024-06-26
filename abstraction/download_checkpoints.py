from transformers import AutoModel
from huggingface_hub import list_repo_refs
from tqdm import tqdm


if __name__ == "__main__":
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    cache_dir = "/nlp/scr/jjian/mistral-checkpoints"
    for checkpoint in tqdm(branches):
        model = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True, revision=checkpoint, cache_dir=cache_dir)
        del model