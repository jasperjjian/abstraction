#!/bin/bash

REPO_DIR=$(dirname "$(dirname "$(realpath "$0")")")
INPUT_DIR="${1:?Usage: split_parsing.sh <input_dir> [output_dir]}"
OUTPUT_DIR="${2:-./output/parsed}"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/ditransitive.*.txt; do
    filename=$(basename -- "$file")
    file_prefix="${filename%.*}"
    OUTPUT="$OUTPUT_DIR/${file_prefix}.parsed.conllu"
    nlprun -r 48G "python3 $REPO_DIR/abstraction/filtering/parse.py $file $OUTPUT"
done