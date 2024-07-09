#!/bin/bash

sample_a="/nlp/scr/jjian/datasets/wikitext_parsed/motion.parsed_filtered.sampled.json"
sample_b="/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed_filtered.sampled.json"

python3 /sailhome/jjian/projects/abstraction/abstraction/filtering/ablation_sample.py $sample_a $sample_b "dependent"