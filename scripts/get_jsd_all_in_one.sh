#!/bin/bash

# --rep_two rel_clause_obj \
DATASETS_DIR="/nlp/scr/jjian/datasets/wikitext_parsed/"
CLASS_ONE=$DATASETS_DIR"ditransitive.fragments.augment.json"
CLASS_TWO=$DATASETS_DIR"substance.fragments.augment.json"

declare -A models
models=(
  ["expanse-gpt2-small-x777"]="777"
  ["darkmatter-gpt2-small-x343"]="343"
  ["battlestar-gpt2-small-x49"]="49"
)
# models=(
#   ["expanse-gpt2-small-x777"]="777"
# )

for MODEL_NAME in "${!models[@]}"; do
  MODEL_SEED="${models[$MODEL_NAME]}"
  
  nlprun -r 24G --job-name "ditransitive_substance_${MODEL_SEED}" \
    "python3 /sailhome/jjian/projects/abstraction/abstraction/object_distribution_jsd.py \
        --rep preposition_fragment \
        --rep_two preposition_fragment \
        --split substance_annotated \
        --source wikitext \
        --comparison_setting verbwise \
        --branch 10 \
        --class_one_file $CLASS_ONE \
        --class_two_file $CLASS_TWO \
        --model_name $MODEL_NAME"
done