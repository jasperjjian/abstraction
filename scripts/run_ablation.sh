#!/bin/bash

sample_a="/nlp/scr/jjian/datasets/wikitext_parsed/motion.parsed_filtered.balanced_sampled.json"
sample_b="/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed_filtered.balanced_sampled.json"

python3 /sailhome/jjian/projects/abstraction/abstraction/analysis/run_ablation.py $sample_a $sample_b "target"