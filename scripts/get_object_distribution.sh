#!/bin/bash

DATASETS_DIR="/nlp/scr/jjian/datasets/wikitext_parsed/"
CLASS_ONE=$DATASETS_DIR"ditransitive.fragments.augment.json"
CLASS_TWO=$DATASETS_DIR"motion.fragments.augment.json"

python3 /sailhome/jjian/projects/abstraction/abstraction/object_distribution.py "preposition_fragment" 0 $CLASS_TWO "motion_annotated"
