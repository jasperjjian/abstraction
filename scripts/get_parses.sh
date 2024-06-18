#!/bin/bash

INPUT="/sailhome/jjian/projects/abstraction/scraped_data/wikitext/by.raw.unfiltered.tail.txt"
OUTPUT="/nlp/scr/jjian/datasets/wikitext_parsed/by.raw.unfiltered.tail.parsed.conllu"

python3 /sailhome/jjian/projects/abstraction/abstraction/filtering/passive_adjunct.py $INPUT $OUTPUT