#!/bin/bash

python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/with.adjunct.parsed_filtered.sampled.json" "with.adjunct" "target"
python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/with.adjunct.parsed_filtered.sampled.json" "with.adjunct" "dependent"