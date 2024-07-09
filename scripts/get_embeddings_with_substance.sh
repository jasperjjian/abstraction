#!/bin/bash

python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/with.substance.parsed_filtered.manual.json" "with.substance" "target"
python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/with.substance.parsed_filtered.manual.json" "with.substance" "dependent"