#!/bin/bash

INPUT="/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/with.substance.raw.filtered.txt"
OUTPUT="/nlp/scr/jjian/datasets/wikitext_parsed/with.substance.raw.filtered.parsed.conllu"

python3 /sailhome/jjian/projects/abstraction/abstraction/filtering/substance_adjunct.py $INPUT $OUTPUT