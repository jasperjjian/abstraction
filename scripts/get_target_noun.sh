#!/bin/bash

REPO_DIR=$(dirname "$(dirname "$(realpath "$0")")")
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
CACHE_DIR="${CACHE_DIR:-./cache}"

NOUN_LIST=('public' 'children' 'audience' 'people' 'company' 'family' 'New' 'British' 'player' 'state' 'team' 'new' 'best' 'police' 'world' 'top' 'front' 'surface' 'United' 'hospital' 'scene' 'ground' 'door' 'airport' 'north' 'finish' 'bottom' 'south' 'west' 'sea')

for NOUN in "${NOUN_LIST[@]}"; do
    nlprun -q john -r 16G --job-name "$NOUN-343" \
        "OUTPUT_DIR=$OUTPUT_DIR CACHE_DIR=$CACHE_DIR python3 $REPO_DIR/abstraction/analysis/prototype_noun_analysis_verbwise.py $NOUN"
done