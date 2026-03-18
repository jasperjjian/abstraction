import os
import re
import sys
import argparse
from tqdm import tqdm
 
from datasets import load_dataset
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from nltk.tokenize import sent_tokenize
 
from abstraction.filtering import utils
 
 
# =============================================================================
# STEP 1 — STRING-BASED FILTERING
# Scans raw text for sentences containing candidate verb patterns and writes
# them to disk for subsequent parsing and structural filtering.
# =============================================================================
 
def string_based_filtering(
    texts,
    target: str,
    motion_regex: str,
    ditransitive_regex: str,
    ditrans_path: str,
    motion_path: str,
    batch_size: int = 1000,
):
    """Filter sentences by regex and write matching ones to separate files.
 
    Args:
        texts: Iterable of raw text strings (one document per item).
        target: A string that must appear in a sentence (lowercased) before
            the more expensive regex checks are applied.
        motion_regex: Compiled-ready regex pattern for motion verbs.
        ditransitive_regex: Compiled-ready regex pattern for ditransitive verbs.
        ditrans_path: Output path for ditransitive sentences.
        motion_path: Output path for motion sentences.
        batch_size: Number of sentences to buffer before flushing to disk.
    """
    motion_pattern = re.compile(motion_regex)
    ditransitive_pattern = re.compile(ditransitive_regex)
 
    motion_buffer = []
    ditrans_buffer = []
 
    with open(ditrans_path, "w") as f_ditrans, open(motion_path, "w") as f_motion:
        for text in tqdm(texts, mininterval=5):
            if not text:
                continue
            for sentence in sent_tokenize(text):
                sentence_lower = sentence.lower()
                if target not in sentence_lower:
                    continue
                if motion_pattern.search(sentence_lower):
                    motion_buffer.append(sentence + "\n")
                if ditransitive_pattern.search(sentence_lower):
                    ditrans_buffer.append(sentence + "\n")
 
            if len(motion_buffer) >= batch_size:
                f_motion.writelines(motion_buffer)
                motion_buffer.clear()
            if len(ditrans_buffer) >= batch_size:
                f_ditrans.writelines(ditrans_buffer)
                ditrans_buffer.clear()
 
        if motion_buffer:
            f_motion.writelines(motion_buffer)
        if ditrans_buffer:
            f_ditrans.writelines(ditrans_buffer)
 
 
# =============================================================================
# STEP 2 — STRUCTURAL FILTERING
# Walks parsed CoNLL-U sentences and applies dependency-based heuristics to
# identify genuine ditransitive and motion-to constructions.
# =============================================================================
 
def structure_filtering_ditransitives(
    doc_sentences,
    target_lemma: str = 'to',
    target_upos: str = "ADP",
    lemmas: list = [],
    path_include=None,
    path_exclude=None,
    path_reasons=None,
):
    """Filter parsed sentences for ditransitive constructions.
 
    Looks for a preposition matching target_lemma whose grandparent is a verb
    that also has a direct object or passive subject — the hallmark of a
    ditransitive.
 
    Returns:
        Tuple of (include_list, exclude_list, reasons).
    """
    include_list, exclude_list, reasons = [], [], []
 
    for i, sentence in enumerate(doc_sentences):
        if "@-@" in sentence.text:
            continue
 
        char_start = char_end = 0
        for w in sentence.words:
            char_end += len(w.text)
 
            if w.lemma == target_lemma and w.upos == target_upos:
                if w.head == 0:
                    reasons.append("parent is root")
                    char_start += len(w.text) + 1
                    char_end += 1
                    continue
 
                parent_word = next(
                    (p for p in sentence.words if p.id == w.head), None
                )
                if parent_word is None or parent_word.upos != "NOUN":
                    reasons.append("parent not noun")
                    char_start += len(w.text) + 1
                    char_end += 1
                    continue
 
                # Walk up to the grandparent (the governing verb)
                gp_lemma = gp_start = gp_end = 0
                for w_g in sentence.words:
                    gp_end += len(w_g.text)
                    if w_g.id == parent_word.head:
                        gp_lemma = w_g.lemma
                        gp_id = w_g.id
                        gp_word = w_g
                        break
                    gp_start += len(w_g.text) + 1
                    gp_end += 1
 
                if gp_word.upos != "VERB":
                    reasons.append("grandparent not verb")
                    char_start += len(w.text) + 1
                    char_end += 1
                    continue
 
                sibling_deps = [
                    s.deprel for s in sentence.words if s.head == gp_id
                ]
                if "obj" not in sibling_deps and "nsubj:pass" not in sibling_deps:
                    reasons.append("not ditransitive")
                    char_start += len(w.text) + 1
                    char_end += 1
                    continue
 
                output_text = " ".join(ww.text for ww in sentence.words).strip()
                entry = {
                    'sent_id': i,
                    'text': output_text,
                    'target_lemma': target_lemma,
                    'target_slice': (char_start, char_end),
                    'dependent_lemma': gp_lemma,
                    'dependent_slice': (gp_start, gp_end),
                }
                (include_list if gp_lemma in lemmas else exclude_list).append(entry)
 
            char_start += len(w.text) + 1
            char_end += 1
 
    if path_include:
        utils.dump_json(include_list, path_include)
    if path_exclude:
        utils.dump_json(exclude_list, path_exclude)
    if path_reasons:
        with open(path_reasons, "w") as f:
            f.write("\n".join(reasons) + "\n")
 
    return include_list, exclude_list, reasons
 
 
def structure_filtering_motion(
    doc_sentences,
    target_lemma: str = 'to',
    target_upos: str = "ADP",
    lemmas: list = [],
    path_include=None,
    path_exclude=None,
    path_reasons=None,
):
    """Filter parsed sentences for motion-to constructions.
 
    Like the ditransitive filter but additionally checks that no intervening
    noun phrase sits between the verb and the 'to' preposition, which would
    indicate a ditransitive rather than a motion construction.
 
    Returns:
        Tuple of (include_list, exclude_list, reasons).
    """
    include_list, exclude_list, reasons = [], [], []
 
    for i, sentence in enumerate(doc_sentences):
        if "@-@" in sentence.text:
            continue
 
        char_start = char_end = 0
        for w in sentence.words:
            char_end += len(w.text)
 
            if w.lemma == target_lemma and w.upos == target_upos:
                if w.head == 0:
                    reasons.append("parent is root")
                    char_start += len(w.text) + 1
                    char_end += 1
                    continue
 
                parent_word = next(
                    (p for p in sentence.words if p.id == w.head), None
                )
                if parent_word is None or parent_word.upos != "NOUN":
                    reasons.append("parent not noun")
                    char_start += len(w.text) + 1
                    char_end += 1
                    continue
 
                gp_lemma = gp_start = gp_end = 0
                interveners = []
                for w_g in sentence.words:
                    gp_end += len(w_g.text)
                    if w_g.id == parent_word.head:
                        gp_lemma = w_g.lemma
                        gp_id = w_g.id
                        gp_word = w_g
                        if gp_id < w.id:
                            interveners = [
                                wi.upos for wi in sentence.words
                                if gp_id < wi.id < w.id
                            ]
                        break
                    gp_start += len(w_g.text) + 1
                    gp_end += 1
 
                if gp_word.upos != "VERB":
                    reasons.append("grandparent not verb")
                    char_start += len(w.text) + 1
                    char_end += 1
                    continue
 
                if any(pos in interveners for pos in ("NOUN", "PROPN", "PRON")):
                    reasons.append("intervening noun")
                    char_start += len(w.text) + 1
                    char_end += 1
                    continue
 
                output_text = " ".join(ww.text for ww in sentence.words).strip()
                entry = {
                    'sent_id': i,
                    'text': output_text,
                    'target_lemma': target_lemma,
                    'target_slice': (char_start, char_end),
                    'dependent_lemma': gp_lemma,
                    'dependent_slice': (gp_start, gp_end),
                }
                (include_list if gp_lemma in lemmas else exclude_list).append(entry)
 
            char_start += len(w.text) + 1
            char_end += 1
 
    if path_include:
        utils.dump_json(include_list, path_include)
    if path_exclude:
        utils.dump_json(exclude_list, path_exclude)
    if path_reasons:
        with open(path_reasons, "w") as f:
            f.write("\n".join(reasons) + "\n")
 
    return include_list, exclude_list, reasons
 
 
# =============================================================================
# DATASET LOADING
# =============================================================================
 
def load_texts(dataset_name: str, cache_dir: str):
    """Load raw texts from a HuggingFace dataset.
 
    Args:
        dataset_name: Either 'openwebtext' or 'wikitext'.
        cache_dir: Directory for HuggingFace cache.
 
    Returns:
        An iterable of text strings.
    """
    if dataset_name == "openwebtext":
        dataset = load_dataset(
            "openwebtext", cache_dir=cache_dir, trust_remote_code=True
        )
        return dataset['train']['text']
    elif dataset_name == "wikitext":
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir
        )
        return dataset['train']['text']
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'openwebtext' or 'wikitext'.")
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def main():
    parser = argparse.ArgumentParser(
        description="Filter a text dataset for ditransitive and motion-to constructions."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["openwebtext", "wikitext"],
        help="Dataset to filter."
    )
    parser.add_argument(
        "--cache_dir", type=str, default="/nlp/scr/jjian/datasets",
        help="HuggingFace cache directory."
    )
    parser.add_argument(
        "--output_dir", type=str, default="/nlp/scr/jjian/datasets/filtered",
        help="Directory for all output files."
    )
    parser.add_argument(
        "--ditransitives_list", type=str,
        default="/afs/cs.stanford.edu/u/jjian/projects/abstraction/data/ditransitives.txt",
        help="Path to newline-separated list of ditransitive verbs."
    )
    parser.add_argument(
        "--motion_list", type=str,
        default="/afs/cs.stanford.edu/u/jjian/projects/abstraction/data/motion.txt",
        help="Path to newline-separated list of motion verbs."
    )
    args = parser.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['HF_DATASETS_CACHE'] = args.cache_dir
 
    # --- Load verb lists ---
    ditransitives = [w.strip() for w in open(args.ditransitives_list).readlines()]
    motion = [w.strip() for w in open(args.motion_list).readlines()]
    ditransitives_inflected = utils.get_all_inflections(ditransitives)
    motion_inflected = utils.get_all_inflections(motion)
 
    # --- Build regex patterns ---
    motion_regex = "|".join(
        rf"to(?:\s+\w+){{0,2}}\s+{v}|{v}(?:\s+\w+){{0,2}}\s+to"
        for v in motion_inflected
    )
    ditransitive_regex = "|".join(
        rf"to(?:\s+\w+){{0,5}}\s+{v}|{v}(?:\s+\w+){{0,5}}\s+to"
        for v in ditransitives_inflected
    )
 
    # --- Step 1: String-based filtering ---
    print(f"Step 1: String-based filtering on {args.dataset}...")
    ditrans_raw_path = os.path.join(args.output_dir, f"{args.dataset}.ditransitive.raw.txt")
    motion_raw_path = os.path.join(args.output_dir, f"{args.dataset}.motion.raw.txt")
 
    texts = load_texts(args.dataset, args.cache_dir)
    string_based_filtering(
        texts, "to", motion_regex, ditransitive_regex,
        ditrans_raw_path, motion_raw_path
    )
    print(f"  Ditransitive sentences → {ditrans_raw_path}")
    print(f"  Motion sentences       → {motion_raw_path}")
 
    # --- Step 2: Parse ---
    print("Step 2: Parsing...")
    ditrans_conllu = os.path.join(args.output_dir, f"{args.dataset}.ditransitive.parsed.conllu")
    motion_conllu = os.path.join(args.output_dir, f"{args.dataset}.motion.parsed.conllu")
 
    for raw_path, conllu_path in [
        (ditrans_raw_path, ditrans_conllu),
        (motion_raw_path, motion_conllu),
    ]:
        sentences = open(raw_path).readlines()
        nlp = utils.get_stanza_pipeline()
        parsed = utils.stanza_parsing_batched(sentences, nlp, batch_size=64)
        doc = Document([])
        doc.sentences = parsed
        CoNLL.write_doc2conll(doc, conllu_path)
        print(f"  Parsed → {conllu_path}")
 
    # --- Step 3: Structural filtering ---
    print("Step 3: Structural filtering...")
 
    doc_ditransitive = CoNLL.conll2doc(ditrans_conllu).sentences
    structure_filtering_ditransitives(
        doc_ditransitive,
        lemmas=ditransitives,
        path_include=os.path.join(args.output_dir, f"{args.dataset}.ditransitive.filtered.json"),
        path_exclude=os.path.join(args.output_dir, f"{args.dataset}.ditransitive.excluded.json"),
        path_reasons=os.path.join(args.output_dir, f"{args.dataset}.ditransitive.reasons.txt"),
    )
    print(f"  Ditransitive filtering done.")
 
    doc_motion = CoNLL.conll2doc(motion_conllu).sentences
    structure_filtering_motion(
        doc_motion,
        lemmas=motion,
        path_include=os.path.join(args.output_dir, f"{args.dataset}.motion.filtered.json"),
        path_exclude=os.path.join(args.output_dir, f"{args.dataset}.motion.excluded.json"),
        path_reasons=os.path.join(args.output_dir, f"{args.dataset}.motion.reasons.txt"),
    )
    print(f"  Motion filtering done.")
    print("Done.")
 
 
if __name__ == "__main__":
    main()
