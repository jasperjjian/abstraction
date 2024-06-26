#!/bin/bash

python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed_filtered.json" "wikitext.ditrans" "target"
python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed_filtered.json" "wikitext.ditrans" "dependent"
python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/motion.parsed_filtered.json" "wikitext.motion" "target"
python3 /sailhome/jjian/projects/abstraction/abstraction/get_embeddings.py "/nlp/scr/jjian/datasets/wikitext_parsed/motion.parsed_filtered.json" "wikitext.motion" "dependent"
