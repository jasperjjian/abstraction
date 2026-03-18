#!/bin/bash

REPO_DIR=$(dirname "$(dirname "$(realpath "$0")")")
DATASETS_DIR="$REPO_DIR/datasets/final_datasets"
CLASS_ONE="$DATASETS_DIR/ditransitive.fragments.augment.json"
CLASS_TWO="$DATASETS_DIR/motion.fragments.augment.json"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
CACHE_DIR="${CACHE_DIR:-./cache}"

declare -A models
models=(
  ["expanse-gpt2-small-x777"]="777"
  ["darkmatter-gpt2-small-x343"]="343"
  ["battlestar-gpt2-small-x49"]="49"
)

for MODEL_NAME in "${!models[@]}"; do
  MODEL_SEED="${models[$MODEL_NAME]}"

  nlprun -r 24G --job-name "ditransitive_motion_${MODEL_SEED}" \
    "python3 $REPO_DIR/abstraction/object_distribution_jsd.py \
        --rep preposition_fragment \
        --rep_two preposition_fragment \
        --split ditrans_annotated \
        --source wikitext \
        --comparison_setting verbwise \
        --branch 10 \
        --class_one_file $CLASS_ONE \
        --class_two_file $CLASS_TWO \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --cache_dir $CACHE_DIR"
done