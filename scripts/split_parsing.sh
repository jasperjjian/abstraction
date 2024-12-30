#!/bin/bash

# loop through all files in the directory
mkdir -p /nlp/scr/jjian/datasets/openwebtext_filtered/parsed

for file in /nlp/scr/jjian/datasets/openwebtext_filtered/temp/ditransitive.*.txt; do
    # get the filename without the extension
    filename=$(basename -- "$file")
    file_prefix="${filename%.*}"
    # split the file into 1000 line chunks
    INPUT=$file
    OUTPUT="/nlp/scr/jjian/datasets/openwebtext_filtered/parsed/${file_prefix}.parsed.conllu"
    nlprun -r 48G "python3 /afs/cs.stanford.edu/u/jjian/projects/abstraction/abstraction/filtering/parse.py $INPUT $OUTPUT"
done