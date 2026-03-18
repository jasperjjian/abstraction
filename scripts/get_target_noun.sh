#!/bin/bash

# LOAD THE NOUN LIST

NOUN_LIST=('public' 'children' 'audience' 'people' 'company' 'family' 'New' 'British' 'player' 'state' 'team' 'new' 'best' 'police' 'world' 'top' 'front' 'surface' 'United' 'hospital' 'scene' 'ground' 'door' 'airport' 'north' 'finish' 'bottom' 'south' 'west' 'sea')
NOUN_LIST=('children' 'audience' 'people' 'company' 'player' 'state' 'team' 'police' 'world')

for NOUN in "${NOUN_LIST[@]}"
do
    nlprun -q john -r 16G --job-name "$NOUN-343" "python3 /afs/cs.stanford.edu/u/jjian/projects/abstraction/abstraction/analysis/prototype_noun_analysis_verbwise.py $NOUN"
done