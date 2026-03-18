#!/bin/bash

REPO_DIR=$(dirname "$(dirname "$(realpath "$0")")")
CACHE_DIR="${CACHE_DIR:-./cache}"
MODEL="${1:-stanford-crfm/darkmatter-gpt2-small-x343}"

CACHE_DIR=$CACHE_DIR python3 "$REPO_DIR/abstraction/download_checkpoints.py" "$MODEL"