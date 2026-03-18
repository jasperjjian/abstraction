import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import json
import os
from typing import List, Dict, Any, Optional
import sys


# Token IDs and string forms for the 30 prototype nouns of interest.
PROTOTYPE_TOKEN_IDS = [
    1171, 1751, 5386, 661, 1664, 1641, 968, 3517, 2137, 1181,
    1074, 649, 1266, 1644, 995, 1353, 2166, 4417, 1578, 4436,
    3715, 2323, 3420, 9003, 5093, 5461, 4220, 5366, 7421, 5417,
]
PROTOTYPE_TOKENS = [
    ' public', ' children', ' audience', ' people', ' company',
    ' family', ' New', ' British', ' player', ' state',
    ' team', ' new', ' best', ' police', ' world',
    ' top', ' front', ' surface', ' United', ' hospital',
    ' scene', ' ground', ' door', ' airport', ' north',
    ' finish', ' bottom', ' south', ' west', ' sea',
]


def get_next_token_distribution_batch(
    input_prefixes: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    """Run a batch of input prefixes through the model and return next-token
    probability distributions (one row per prefix)."""
    model.eval()

    encoded = tokenizer(input_prefixes, return_tensors='pt', padding="longest")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    sequence_lengths = attention_mask.sum(dim=1) - 1

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    next_token_logits = torch.stack([
        logits[i, sequence_lengths[i], :]
        for i in range(logits.shape[0])
    ])
    return torch.nn.functional.softmax(next_token_logits, dim=-1)


def token_distribution_batch(
    input_prefixes: List[str],
    verbs: List[str],
    probabilities: torch.Tensor,
    tokenizer: AutoTokenizer,
    token_ids: Optional[List[int]] = None,
    tokens: Optional[List[str]] = None,
    top_k: float = 0.9,
) -> List[Dict[str, Any]]:
    """Extract token probabilities from a batch of next-token distributions.

    If token_ids and tokens are provided, returns probabilities only for those
    specific tokens. Otherwise falls back to a top-k cutoff over the full
    vocabulary.

    Args:
        input_prefixes: The input strings fed to the model.
        verbs: The verb label for each input.
        probabilities: Softmax output from the model, shape (batch, vocab).
        token_ids: Optional list of specific token IDs to extract.
        tokens: Optional list of string labels corresponding to token_ids.
        top_k: Cumulative probability cutoff used when token_ids is not given.
    """
    if token_ids is not None and tokens is not None:
        # Specific token list mode
        probabilities_list = probabilities.tolist()
        return [
            {
                "input_prefix": prefix,
                "verb": verb,
                "top_k_tokens": [
                    [token, probs[tid]]
                    for tid, token in zip(token_ids, tokens)
                ],
            }
            for prefix, verb, probs in zip(input_prefixes, verbs, probabilities_list)
        ]
    else:
        # Top-k mode: include tokens until cumulative probability exceeds top_k
        sorted_probs, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=1)
        mask = cumulative_probs < top_k

        relevant_probs = [probs[m] for probs, m in zip(sorted_probs, mask)]
        relevant_indices = [indices[m] for indices, m in zip(sorted_indices, mask)]

        # Batch-convert all token ids to strings
        flattened_indices = torch.cat(relevant_indices).tolist()
        all_tokens = tokenizer.convert_ids_to_tokens(flattened_indices)
        split_sizes = [len(idx) for idx in relevant_indices]
        offsets = [0] + list(torch.cumsum(torch.tensor(split_sizes[:-1]), dim=0).tolist())
        token_lists = [
            all_tokens[offset:offset + size]
            for offset, size in zip(offsets, split_sizes)
        ]
        token_strings = [
            [tokenizer.convert_tokens_to_string([t]) for t in tokens_]
            for tokens_ in token_lists
        ]

        return [
            {
                "input_prefix": prefix,
                "verb": verb,
                "top_k_tokens": list(zip(tok_strs, [round(p.item(), 5) for p in probs])),
            }
            for prefix, verb, tok_strs, probs in zip(
                input_prefixes, verbs, token_strings, relevant_probs
            )
        ]


def loop_checkpoints_and_save(
    model_name: str,
    split: str,
    instances: List[Dict],
    cache_dir: str = None,
    rep: str = "verb_fragment",
    batch_size: int = 32,
    branch: int = 0,
    token_ids: Optional[List[int]] = None,
    tokens: Optional[List[str]] = None,
    top_k: float = 0.9,
    output_dir: str = ".",
) -> None:
    """Iterate over model checkpoints, run inference, and save results.

    Args:
        branch: Controls which checkpoints to run.
            0 = all checkpoints
            1 = first half
            2 = second half
        token_ids: If provided (with tokens), only record probabilities for
            these specific token IDs. Otherwise uses top-k.
        tokens: String labels corresponding to token_ids.
        top_k: Cumulative probability cutoff for top-k mode.

    Skips checkpoints whose output file already exists.
    """
    out = list_repo_refs(model_name)
    all_branches = sorted(
        [b.name for b in out.tags],
        key=lambda x: int(x.split('checkpoint-')[-1])
    )

    midpoint = len(all_branches) // 2
    if branch == 0:
        branches = all_branches
    elif branch == 1:
        branches = all_branches[:midpoint]
    elif branch == 2:
        branches = all_branches[midpoint:]
    else:
        raise ValueError(f"branch must be 0, 1, or 2 — got {branch}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        tokenizer.pad_token = "<|pad|>"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name_short = model_name.split("/")[-1]

    for checkpoint in tqdm(branches):
        output_path = os.path.join(
            output_dir, split, "predictions",
            rep, "prototype_nouns",
            f"{model_name_short}.{checkpoint}.predictions.json"
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            continue

        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=checkpoint, cache_dir=cache_dir
        ).to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

        results = []
        for i in tqdm(range(0, len(instances), batch_size)):
            batch = instances[i:i + batch_size]
            input_prefixes = [data[rep].strip() for data in batch]
            verbs = [data["dependent_lemma"].strip() for data in batch]

            probabilities = get_next_token_distribution_batch(
                input_prefixes, model, tokenizer
            )
            results.extend(token_distribution_batch(
                input_prefixes, verbs, probabilities, tokenizer,
                token_ids=token_ids, tokens=tokens, top_k=top_k
            ))

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    rep = sys.argv[1]
    branch = int(sys.argv[2])
    ditrans_sampled = sys.argv[3]
    split = sys.argv[4]

    model_name = "stanford-crfm/darkmatter-gpt2-small-x343"
    cache_dir = os.environ.get("CACHE_DIR", "./cache")
    output_dir = os.environ.get("OUTPUT_DIR", ".")

    ditrans_json = json.load(open(ditrans_sampled, "r"))

    # Use the hardcoded prototype noun list. Pass token_ids=None and tokens=None
    # instead to fall back to top-k over the full vocabulary.
    loop_checkpoints_and_save(
        model_name, split, ditrans_json,
        cache_dir=cache_dir, rep=rep, batch_size=16, branch=branch,
        token_ids=PROTOTYPE_TOKEN_IDS, tokens=PROTOTYPE_TOKENS,
        output_dir=output_dir,
    )