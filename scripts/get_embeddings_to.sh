#!/bin/bash

python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed_filtered.json" "ditrans" "target"
python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed_filtered.json" "ditrans" "dependent"
