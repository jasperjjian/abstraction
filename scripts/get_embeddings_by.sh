#!/bin/bash

python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed.by.adjunct.parsed_filtered.json" "wikitext.by.adjunct" "target"
python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed.by.adjunct.parsed_filtered.json" "wikitext.by.adjunct" "dependent"
python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed.by.passive.parsed_filtered.json" "wikitext.by.passive" "target"
python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed.by.passive.parsed_filtered.json" "wikitext.by.passive" "dependent"
