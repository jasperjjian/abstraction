#!/bin/bash

# LOAD THE NOUN LIST

NOUN_LIST=('public' 'children' 'audience' 'people' 'company' 'family' 'New' 'British' 'player' 'state' 'team' 'new' 'best' 'police' 'world' 'top' 'front' 'surface' 'United' 'hospital' 'scene' 'ground' 'door' 'airport' 'north' 'finish' 'bottom' 'south' 'west' 'sea')

# here's the function to call while looping through the list
# nlprun -q john -r 16G "python3 /afs/cs.stanford.edu/u/jjian/projects/abstraction/abstraction/analysis/police_analysis_scratch.py $NOUN"

for NOUN in "${NOUN_LIST[@]}"
do
    nlprun -q john -r 16G "python3 /afs/cs.stanford.edu/u/jjian/projects/abstraction/abstraction/analysis/police_analysis_verbwise.py $NOUN"
done