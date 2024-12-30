#!/bin/bash

# loop from 1 to 10

for i in {1..10}; do
   nlprun -r 48G "python3 /afs/cs.stanford.edu/u/jjian/projects/abstraction/abstraction/analysis/word_cooccurence.py $i"
done