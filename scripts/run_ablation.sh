#!/bin/bash

sample_a="/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed.annotated.json"
sample_b="/nlp/scr/jjian/datasets/wikitext_parsed/motion.parsed.annotated.json"

python3 /sailhome/jjian/projects/abstraction/abstraction/analysis/run_ablation.py $sample_a $sample_b "target" 20