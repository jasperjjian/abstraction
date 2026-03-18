#!/bin/bash
 
REPO_DIR=$(dirname "$(dirname "$(realpath "$0")")")
INPUT="${1:-$REPO_DIR/scraped_data/wikitext/with.substance.raw.filtered.txt}"
OUTPUT="${2:-./output/with.substance.raw.filtered.parsed.conllu}"
 
python3 "$REPO_DIR/abstraction/filtering/substance_adjunct.py" "$INPUT" "$OUTPUT"