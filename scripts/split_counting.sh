#!/bin/bash

REPO_DIR=$(dirname "$(dirname "$(realpath "$0")")")
TOKEN_IDS_PATH="${1:?Usage: split_counting.sh <token_ids_path> [output_dir]}"
OUTPUT_DIR="${2:-./output}"
SHARD_SIZE=50000000  # tokens per parallel job

mkdir -p "$OUTPUT_DIR"

NUM_SHARDS=10
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    START=$((i * SHARD_SIZE))
    END=$(((i + 1) * SHARD_SIZE))
    nlprun -r 48G \
        "python3 $REPO_DIR/abstraction/word_cooccurence.py $TOKEN_IDS_PATH \
            --start $START --end $END --output_dir $OUTPUT_DIR"
done