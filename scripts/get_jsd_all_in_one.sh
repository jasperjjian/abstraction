#!/bin/bash

# parser = argparse.ArgumentParser(description="Process model checkpoints and dataset splits.")
#     parser.add_argument("--rep", type=str, required=True, help="Repetition identifier")
#     parser.add_argument("--branch", type=int, required=True, help="Branch identifier")
#     parser.add_argument("--split", type=str, required=True, help="Dataset split type")
#     parser.add_argument("--comparison_setting", type=str, required=True, help="Comparison setting")
#     parser.add_argument("--class_one_file", type=str, required=True, help="Class one file")
#     parser.add_argument("--class_two_file", type=str, required=True, help="Class two file")
#     args = parser.parse_args()

#     model_name = "stanford-crfm/battlestar-gpt2-small-x49"
#     cache_dir = "/nlp/scr/jjian/mistral-checkpoints/"

#     class_one_json = json.load(open(args.class_one_file, "r"))
#     class_two_json = json.load(open(args.class_two_file, "r"))


#     loop_checkpoints_and_save(
#         model_name,
#         args.split,
#         class_one_json,
#         class_two_json,
#         cache_dir=cache_dir,
#         rep=args.rep,
#         batch_size=16,
#         branch=args.branch
#     )

# write a script to call the above python script

DATASETS_DIR="/nlp/scr/jjian/datasets/wikitext_parsed/"
CLASS_ONE=$DATASETS_DIR"reciprocal.rel_clause_obj.constructed.biclausal.json"
CLASS_TWO=$DATASETS_DIR"reciprocal.rel_clause_obj.constructed.biclausal.json"

python3 /sailhome/jjian/projects/abstraction/abstraction/all_in_one.py --rep "preposition_fragment_bare_constructed_biclausal" --rep_two "rel_clause_obj_biclausal" --branch 10 --split "reciprocal_annotated" --comparison_setting "pairwise" --class_one_file $CLASS_ONE --class_two_file $CLASS_TWO --source "wikitext"