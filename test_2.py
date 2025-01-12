import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math
from huggingface_hub import list_repo_refs
from tqdm import tqdm


def get_top_k_predictions_with_entropy(sentence, model_name="gpt2", top_k=15):
    # Load model and tokenizer
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    branches = sorted(branches, key=lambda x: int(x.split('checkpoint-')[-1]))
    checkpoint = branches[600]
    print(f"Using checkpoint: {checkpoint}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir="/nlp/scr/jjian/mistral-checkpoints/")
    model = GPT2LMHeadModel.from_pretrained(model_name, revision=checkpoint, cache_dir="/nlp/scr/jjian/mistral-checkpoints/")

    # Encode input sentence and get the last token's logits
    inputs = tokenizer(sentence, return_tensors="pt")
    print(f"input: {inputs}")
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]  # Get the logits for the last token

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # get the value for token 373
    #print(f"was probability: {probs[0][373]}")

    # Calculate entropy
    entropy = -torch.sum(probs * torch.log(probs)).item()

    # Get the top-k predictions and their probabilities
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
    
    # Decode the tokens and pair them with probabilities
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices[0]]
    top_k_probs = top_k_probs[0].tolist()

    # Print entropy
    print(f"Entropy: {entropy:.4f}")

    # Return as list of (token, probability) pairs
    return list(zip(top_k_tokens, top_k_probs))



# Example usage
sentence =  "John asked the person who Mary hit"
#sentence = "The person that John told his story to"
top_predictions = get_top_k_predictions_with_entropy(sentence, model_name="stanford-crfm/battlestar-gpt2-small-x49", top_k=25)
for token, prob in top_predictions:
    print(f"Token: {token}, Probability: {prob:.4f}")
