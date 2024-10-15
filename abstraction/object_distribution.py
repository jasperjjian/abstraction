import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import json
import os
from typing import List, Dict, Any
import sys

def get_next_token_distribution_batch(input_prefixes, model, tokenizer, model_name="gpt2"):
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set the model to evaluation mode
    model.eval()

    # Tokenize the batch of inputs
    encoded = tokenizer(input_prefixes, return_tensors='pt', padding="longest")
    input_ids = encoded["input_ids"].to(model.device)

    # Forward pass to get the logits for the next token
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Get the logits for the last token in each input sequence
    next_token_logits = logits[:, -1, :]

    # Apply softmax to get the probability distributions
    probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)

    return probabilities

def token_distribution_batch(input_prefixes: List[str], verbs: List[str], probabilities: torch.Tensor, tokenizer: AutoTokenizer, top_k: float = 0.9) -> List[Dict[str, Any]]:
    
    # Sort probabilities and get indices
    sorted_probs, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=1)
    
    # Create a mask for probabilities below top_k
    mask = cumulative_probs < top_k
    
    # Use the mask to get the relevant probabilities and indices
    relevant_probs = [probs[m] for probs, m in zip(sorted_probs, mask)]
    relevant_indices = [indices[m] for indices, m in zip(sorted_indices, mask)]
    
    # Convert indices to tokens
    token_lists = [tokenizer.convert_ids_to_tokens(indices) for indices in relevant_indices]
    token_strings = [[tokenizer.convert_tokens_to_string([t]) for t in tokens] for tokens in token_lists]
    
    # Create the final results
    batch_results = [
        {
            "input_prefix": prefix,
            "verb": verb,
            "top_k_tokens": tokens,
            #"top_k_tokens": list(zip(tokens, probs.tolist()))
        }
        for prefix, verb, tokens, probs in zip(input_prefixes, verbs, token_strings, relevant_probs)
    ]
    return batch_results

# do the same as above but save entropy of the distribution rather than the tokens

def token_entropy_batch(input_prefixes: List[str], verbs: List[str], probabilities: torch.Tensor) -> List[Dict[str, Any]]:
    
    # Compute entropy of the distribution
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)
    
    # Create the final results
    batch_results = [
        {
            "input_prefix": prefix,
            "verb": verb,
            "entropy": ent.item()
        }
        for prefix, verb, ent in zip(input_prefixes, verbs, entropy)
    ]
    return batch_results



def loop_checkpoints_and_save(model_name, split, instances, cache_dir=None, rep="verb_fragment", batch_size=32):
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    branches = sorted(branches, key=lambda x: int(x.split('checkpoint-')[-1]))[:150]
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|pad|>"]}
        )
        tokenizer.pad_token = "<|pad|>"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for checkpoint in tqdm(branches):
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=checkpoint, cache_dir=cache_dir).to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

        # Process instances in batches
        results = []
        # add tqdm to the loop

        for i in tqdm(range(0, len(instances), batch_size)):
            batch = instances[i:i + batch_size]
            input_prefixes = [data[rep].strip() for data in batch]
            verbs = [data["dependent_lemma"].strip() for data in batch]

            # Get the next token distribution for the batch
            probabilities = get_next_token_distribution_batch(input_prefixes, model, tokenizer, model_name)

            # Get token distribution for the batch
            #batch_results = token_distribution_batch(input_prefixes, verbs, probabilities, tokenizer, top_k=0.75)
            batch_results = token_entropy_batch(input_prefixes, verbs, probabilities)
            results.extend(batch_results)

        model_name_preprocessed = model_name.split("/")[-1]
        output_path = f'/nlp/scr/jjian/data/wikitext/{split}/{rep}/{model_name_preprocessed}.{checkpoint}.entropy.json'

        # make dir if it doesn't exist
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        if os.path.exists(output_path):
            continue

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

    return

if __name__ == "__main__":
    rep = sys.argv[1]
    model_name = "stanford-crfm/battlestar-gpt2-small-x49"
    cache_dir = "/nlp/scr/jjian/mistral-checkpoints/"
    
    split = "motion"
    ditrans_sampled = "/nlp/scr/jjian/datasets/wikitext_parsed/motion.fragments.json"
    ditrans_json = json.load(open(ditrans_sampled, "r"))
    
    loop_checkpoints_and_save(model_name, split, ditrans_json, cache_dir=cache_dir, rep=rep, batch_size=16)
