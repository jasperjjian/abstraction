import os
import gc
import sys
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz


def fast_cooccurrence_matrix_eot(
    words,
    window_size: int = 10,
    eos_token_id: int = 50256,
    save_frequency: int = 100000,
    output_dir: str = ".",
    file_offset: int = 0,
):
    """Calculate a word co-occurrence matrix, respecting EOS token boundaries.

    Only considers right context (words that follow the current word). Saves
    intermediate sparse matrices to disk at approximately every save_frequency
    tokens, always waiting for the next EOS boundary before saving.

    Args:
        words: Array of tokenized word IDs (already sliced to the desired range).
        window_size: Size of the right context window.
        eos_token_id: Token ID for <|endoftext|> (default 50256 for GPT-2).
        save_frequency: Minimum number of tokens between checkpoint saves.
        output_dir: Directory to write .npz matrix shards to.
        file_offset: Starting index for output file numbering, used to avoid
            filename collisions when running parallel jobs over different slices.
    """
    vocab_size = 50257
    rows, cols, data = [], [], []

    eos_positions = (
        [-1]
        + [i for i, token in enumerate(words) if token == eos_token_id]
        + [len(words)]
    )

    last_save = 0
    next_save_number = file_offset

    for seg_idx in range(len(eos_positions) - 1):
        start_index = eos_positions[seg_idx] + 1
        end_index = eos_positions[seg_idx + 1]

        for i in tqdm(range(start_index, end_index), mininterval=5):
            window_end = min(end_index, i + window_size + 1)
            for j in range(i + 1, window_end):
                rows.append(words[i])
                cols.append(words[j])
                data.append(1)

            # Save at EOS boundaries when enough tokens have accumulated
            if i == end_index - 1 and i - last_save >= save_frequency:
                co_matrix = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
                save_npz(os.path.join(output_dir, f"openwebtext_co_matrix_{next_save_number}.npz"), co_matrix)
                rows, cols, data = [], [], []
                last_save = i
                next_save_number += save_frequency
                gc.collect()

    # Save final batch
    if rows:
        co_matrix = csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
        save_npz(os.path.join(output_dir, f"openwebtext_co_matrix_{next_save_number}.npz"), co_matrix)

    return co_matrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute GPT-2 token co-occurrence matrices from a token ID array."
    )
    parser.add_argument("token_ids_path", type=str, help="Path to .npy file of token IDs.")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "."),
                        help="Directory to write output .npz shards to.")
    parser.add_argument("--window_size", type=int, default=10,
                        help="Right-context window size (default: 10).")
    parser.add_argument("--save_frequency", type=int, default=100000,
                        help="Minimum tokens between shard saves (default: 100000).")
    # Optional parallelism: slice the token array before processing
    parser.add_argument("--start", type=int, default=None,
                        help="Start index into the token array (for parallel jobs).")
    parser.add_argument("--end", type=int, default=None,
                        help="End index into the token array (for parallel jobs).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading token ids...")
    words = np.load(args.token_ids_path)

    start = args.start or 0
    end = args.end or len(words)
    words = words[start:end]
    print(f"Processing tokens [{start}:{end}] ({len(words)} tokens)")

    fast_cooccurrence_matrix_eot(
        words,
        window_size=args.window_size,
        save_frequency=args.save_frequency,
        output_dir=args.output_dir,
        file_offset=start,
    )