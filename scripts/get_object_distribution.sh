#!/bin/bash

REPO_DIR=$(dirname "$(dirname "$(realpath "$0")")")
DATASETS_DIR="$REPO_DIR/datasets/final_datasets"
CLASS_TWO="$DATASETS_DIR/motion.fragments.augment.json"
CACHE_DIR="${CACHE_DIR:-./cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

python3 "$REPO_DIR/abstraction/object_distribution.py" \
    "preposition_fragment" 0 "$CLASS_TWO" "motion_annotated"