# Abstraction

This is code for Jian & Manning (2026): `Humans and transformer LMs: Abstraction drives language learning'. This provides a pipeline for analysing linguistic behaviours of language models over the course of training using a distributional divergence-based metric.

## Environment Variables
 
Set these before running any script:
 
| Variable | Description |
|---|---|
| `DATA_DIR` | Root for HDF5 model outputs (`DATA_DIR/{wikitext,blimp}/final/...`) |
| `DATASETS_DIR` | Root for JSON dataset files |
| `COOC_DIR` | Directory containing cooccurrence HDF5 result files |
| `RESULTS_DIR` | Directory containing prototype noun verbwise CSV files |
| `CACHE_DIR` | HuggingFace model cache directory |
| `OUTPUT_DIR` | Output directory for inference scripts |
 
You can set them inline for a single command:
```bash
DATA_DIR=/your/data DATASETS_DIR=/your/datasets python abstraction_plots.py
```
 
Or export them for a whole session:
```bash
export DATA_DIR=/your/data
export DATASETS_DIR=/your/datasets
export COOC_DIR=/your/cooc_results
export RESULTS_DIR=/your/results
export CACHE_DIR=/your/hf_cache
export OUTPUT_DIR=/your/output
```
 
Or add those `export` lines to your `~/.bashrc` / `~/.zshrc` to set them permanently.

## Datasets

All the datasets can be found in `datasets/final_datasets/`. Datasets target next-token predictions given a structured prefix which contains a linguistic category of interest. See the paper for more details on the datasets. Several sentence prefixes are provided for each example.

## Pipeline

### 1. Run inference — `object_distribution_jsd.py`

The core of the method is found in `object_distribution_jsd.py`, which loads models from HuggingFace and runs the datasets through them. The code supports comparison between two classes of interest (e.g., two verb classes, two syntactic conditions, etc.,). Analyses can be performed at the sentence level (pairwise) or at a more abstract level (verbwise).

```bash
scripts/get_jsd.sh
```

This script calls `object_distribution_jsd.py` for the chosen models. The `CLASS_ONE`, `CLASS_TWO`, and `--split` arguments set what dataset to run on. The script is currently set up to compare to-datives and to-motion verb classes.

### 2. Prototype noun inference (optional) — `object_distribution.py`

To compute change in the probability of specific tokens over the course of training (here, prototype nouns in a given verb frame).

```bash
scripts/get_object_distribution.sh   # runs inference
scripts/get_target_noun.sh           # computes verbwise Spearman correlations
```

### 3. Cooccurrence baseline (optional)

To build the word cooccurrence baseline used in Figure 5.

```bash
scripts/split_parsing.sh <input_dir>   # parse raw text
scripts/split_counting.sh <token_ids>  # compute cooccurrence matrices
```

### 4. Analysis and plots

`abstraction/analysis/plots.ipynb` provides the additional analyses used to generate the plots in the paper.