#!/bin/bash
DATASETS_DIR="/nlp/scr/jjian/datasets/wikitext_parsed/"
CLASS_ONE=$DATASETS_DIR"ditransitive.rel_clause_obj.refactored.jsonl"

declare -A models
models=(
  ["expanse-gpt2-small-x777"]="777"
  ["darkmatter-gpt2-small-x343"]="343"
  ["battlestar-gpt2-small-x49"]="49"
)

for MODEL_NAME in "${!models[@]}"; do
  MODEL_SEED="${models[$MODEL_NAME]}"
  
  nlprun -r 24G --job-name "ditrans_rc_two_pfx_${MODEL_SEED}" \
    "python3 /sailhome/jjian/projects/abstraction/abstraction/get_scored_sentences.py \
    --corpus_path $CLASS_ONE \
    --model_name $MODEL_NAME \
    --split ditrans_annotated \
    --rep two_pfx_rc \
    "
done