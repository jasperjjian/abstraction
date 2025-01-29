#!/bin/bash

# --rep_two rel_clause_obj \
DATASETS_DIR="/nlp/scr/jjian/datasets/wikitext_parsed/"
CLASS_ONE=$DATASETS_DIR"reciprocal.rel_clause_obj.constructed_varied_verbs.biclausal.json"
CLASS_TWO=$DATASETS_DIR"reciprocal.rel_clause_obj.constructed_varied_verbs.biclausal.json"

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
  
  nlprun -r 24G --job-name "rel_clause_pairwise_reciprocal_varied_${MODEL_SEED}" \
    "python3 /sailhome/jjian/projects/abstraction/abstraction/object_distribution_jsd.py \
        --rep preposition_fragment_bare_varied_verb \
        --rep_two rel_clause_obj \
        --split reciprocal_annotated \
        --source wikitext \
        --comparison_setting pairwise \
        --branch 10 \
        --class_one_file $CLASS_ONE \
        --class_two_file $CLASS_TWO \
        --model_name $MODEL_NAME"
done