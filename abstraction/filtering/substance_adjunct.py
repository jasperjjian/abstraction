import os
import re
import argparse
from tqdm import tqdm

from datasets import load_dataset
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from nltk.tokenize import sent_tokenize

from abstraction.filtering import utils


# =============================================================================
# STEP 1 — STRING-BASED FILTERING
# Scans raw text for sentences containing 'with' and splits them into
# candidate substance vs. adjunct sentences by regex.
# =============================================================================

def string_based_filtering(
    texts,
    substance_regex: str,
    substance_path: str,
    adjunct_path: str,
    batch_size: int = 1000,
):
    """Filter sentences containing 'with' into substance and adjunct candidates.

    Substance candidates match the substance verb regex. Everything else that
    contains 'with' is treated as a potential adjunct.

    Args:
        texts: Iterable of raw text strings (one document per item).
        substance_regex: Compiled-ready regex pattern for substance verbs.
        substance_path: Output path for substance candidate sentences.
        adjunct_path: Output path for adjunct candidate sentences.
        batch_size: Number of sentences to buffer before flushing to disk.
    """
    substance_pattern = re.compile(substance_regex)

    substance_buffer = []
    adjunct_buffer   = []

    with open(substance_path, "w") as f_sub, open(adjunct_path, "w") as f_adj:
        for text in tqdm(texts, mininterval=5):
            if not text:
                continue
            for sentence in sent_tokenize(text):
                sentence_lower = sentence.lower()
                if "with" not in sentence_lower:
                    continue
                if substance_pattern.search(sentence_lower):
                    substance_buffer.append(sentence + "\n")
                else:
                    adjunct_buffer.append(sentence + "\n")

            if len(substance_buffer) >= batch_size:
                f_sub.writelines(substance_buffer)
                substance_buffer.clear()
            if len(adjunct_buffer) >= batch_size:
                f_adj.writelines(adjunct_buffer)
                adjunct_buffer.clear()

        if substance_buffer:
            f_sub.writelines(substance_buffer)
        if adjunct_buffer:
            f_adj.writelines(adjunct_buffer)


# =============================================================================
# STEP 2 — STRUCTURAL FILTERING
# =============================================================================

def structure_filtering_substance(
    doc_sentences,
    target_lemma: str = 'with',
    target_upos: str = "ADP",
    lemmas: list = [],
    path_include=None,
    path_exclude=None,
    path_reasons=None,
):
    """Filter parsed sentences for spray-load (substance) constructions.

    Looks for 'with NP' where the governing verb has a direct object or
    passive subject.

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
                for w_g in sentence.words:
                    gp_end += len(w_g.text)
                    if w_g.id == parent_word.head:
                        gp_lemma = w_g.lemma
                        gp_id    = w_g.id
                        gp_word  = w_g
                        break
                    gp_start += len(w_g.text) + 1
                    gp_end   += 1

                if gp_word.upos != "VERB":
                    reasons.append("grandparent not verb")
                    char_start += len(w.text) + 1
                    char_end   += 1
                    continue

                sibling_deps = [
                    s.deprel for s in sentence.words if s.head == gp_id
                ]
                if "obj" not in sibling_deps and "nsubj:pass" not in sibling_deps:
                    reasons.append("not ditransitive")
                    char_start += len(w.text) + 1
                    char_end   += 1
                    continue
                if "obj" in sibling_deps:
                    reasons.append("ditransitive")
                    char_start += len(w.text) + 1
                    char_end   += 1
                    continue
                if "nsubj:pass" in sibling_deps:
                    reasons.append("passive")
                    char_start += len(w.text) + 1
                    char_end   += 1
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
            char_end   += 1

    if path_include:
        utils.dump_json(include_list, path_include)
    if path_exclude:
        utils.dump_json(exclude_list, path_exclude)
    if path_reasons:
        with open(path_reasons, "w") as f:
            f.write("\n".join(reasons) + "\n")

    return include_list, exclude_list, reasons


def structure_filtering_adjunct(
    doc_sentences,
    target_lemma: str = 'with',
    target_upos: str = "ADP",
    lemmas: list = [],
    path_include=None,
    path_exclude=None,
    path_reasons=None,
):
    """Filter parsed sentences for with-adjunct constructions.

    Includes any 'with NP' governed by a verb, EXCLUDING those whose verb
    appears in the substance/reciprocal lemma list (i.e. argument uses of
    'with' are excluded, leaving only adjunct uses).

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
                    char_end   += 1
                    continue

                parent_word = next(
                    (p for p in sentence.words if p.id == w.head), None
                )
                if parent_word is None or parent_word.upos != "NOUN":
                    reasons.append("parent not noun")
                    char_start += len(w.text) + 1
                    char_end   += 1
                    continue

                gp_lemma = gp_start = gp_end = 0
                for w_g in sentence.words:
                    gp_end += len(w_g.text)
                    if w_g.id == parent_word.head:
                        gp_lemma = w_g.lemma
                        gp_word  = w_g
                        break
                    gp_start += len(w_g.text) + 1
                    gp_end   += 1

                if gp_word.upos != "VERB":
                    reasons.append("grandparent not verb")
                    char_start += len(w.text) + 1
                    char_end   += 1
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
                # Argument verbs (substance + reciprocal) go to exclude;
                # everything else is a genuine adjunct.
                (exclude_list if gp_lemma in lemmas else include_list).append(entry)

            char_start += len(w.text) + 1
            char_end   += 1

    if path_include:
        utils.dump_json(include_list, path_include)
    if path_exclude:
        utils.dump_json(exclude_list, path_exclude)
    if path_reasons:
        with open(path_reasons, "w") as f:
            f.write("\n".join(reasons) + "\n")

    return include_list, exclude_list, reasons


# =============================================================================
# SHARED UTILITIES
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
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Use 'openwebtext' or 'wikitext'."
        )


def parse_and_save(raw_path: str, conllu_path: str, batch_size: int = 64):
    """Parse a raw text file with Stanza and write CoNLL-U output."""
    sentences = open(raw_path).readlines()
    nlp = utils.get_stanza_pipeline()
    parsed = utils.stanza_parsing_batched(sentences, nlp, batch_size=batch_size)
    doc = Document([])
    doc.sentences = parsed
    CoNLL.write_doc2conll(doc, conllu_path)
    print(f"  Parsed → {conllu_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter a text dataset for spray-load (substance) and adjunct "
            "'with' constructions."
        )
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
        "--substance_list", type=str,
        default="/afs/cs.stanford.edu/u/jjian/projects/abstraction/data/substance.txt",
        help="Path to newline-separated list of substance (spray-load) verbs."
    )
    parser.add_argument(
        "--reciprocals_list", type=str,
        default="/afs/cs.stanford.edu/u/jjian/projects/abstraction/data/reciprocals.txt",
        help="Path to newline-separated list of reciprocal verbs."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['HF_DATASETS_CACHE'] = args.cache_dir
    d = args.dataset

    # --- Load and inflect verb lists ---
    substance   = [w.strip() for w in open(args.substance_list).readlines()]
    reciprocals = [w.strip() for w in open(args.reciprocals_list).readlines()]
    substance_inf = utils.get_all_inflections(substance)

    # Adjunct filtering excludes both substance and reciprocal verbs
    argument_with = substance + reciprocals

    # --- Build regex pattern for substance verbs ---
    substance_regex = "|".join(
        rf"with(?:\s+\w+){{0,10}}\s+{v}|{v}(?:\s+\w+){{0,10}}\s+with"
        for v in substance_inf
    )

    # --- Step 1: String-based filtering ---
    print(f"Step 1: String-based filtering on {d}...")
    substance_raw = os.path.join(args.output_dir, f"{d}.substance.raw.txt")
    adjunct_raw   = os.path.join(args.output_dir, f"{d}.adjunct.raw.txt")

    string_based_filtering(
        load_texts(args.dataset, args.cache_dir),
        substance_regex=substance_regex,
        substance_path=substance_raw,
        adjunct_path=adjunct_raw,
    )
    print(f"  Substance sentences → {substance_raw}")
    print(f"  Adjunct sentences   → {adjunct_raw}")

    # --- Step 2: Parse ---
    print("Step 2: Parsing...")
    substance_conllu = os.path.join(args.output_dir, f"{d}.substance.parsed.conllu")
    adjunct_conllu   = os.path.join(args.output_dir, f"{d}.adjunct.parsed.conllu")
    parse_and_save(substance_raw, substance_conllu)
    parse_and_save(adjunct_raw, adjunct_conllu)

    # --- Step 3: Structural filtering ---
    print("Step 3: Structural filtering...")

    structure_filtering_substance(
        CoNLL.conll2doc(substance_conllu).sentences,
        lemmas=substance,
        path_include=os.path.join(args.output_dir, f"{d}.substance.filtered.json"),
        path_exclude=os.path.join(args.output_dir, f"{d}.substance.excluded.json"),
        path_reasons=os.path.join(args.output_dir, f"{d}.substance.reasons.txt"),
    )
    print("  Substance filtering done.")

    structure_filtering_adjunct(
        CoNLL.conll2doc(adjunct_conllu).sentences,
        lemmas=argument_with,
        path_include=os.path.join(args.output_dir, f"{d}.adjunct.filtered.json"),
        path_exclude=os.path.join(args.output_dir, f"{d}.adjunct.excluded.json"),
        path_reasons=os.path.join(args.output_dir, f"{d}.adjunct.reasons.txt"),
    )
    print("  Adjunct filtering done.")
    print("Done.")


if __name__ == "__main__":
    main()