llama-2-7b="/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-hf"
python3 -m notebook --ip 0.0.0.0 --port 1234

from huggingface_hub import list_repo_refs
out = list_repo_refs("stanford-crfm/battlestar-gpt2-small-x49")
branches = [b.name for b in out.branches]
model = GPT2LMHeadModel.from_pretrained("stanford-crfm/battlestar-gpt2-small-x49", revision="checkpoint-10")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="/nlp/scr/jjian/datasets/wikitext-103-raw-v1")
