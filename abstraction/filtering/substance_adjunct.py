import os
import re
from tqdm import tqdm
from datasets import load_dataset
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from nltk.tokenize import sent_tokenize
import sys
from abstraction.filtering import utils

def string_based_filtering(dataset, target):
    by_list = []
    for data in tqdm(dataset, mininterval=5):
        if data != "":
            sentences = sent_tokenize(data["text"])
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if target in sentence_lower.split():
                    by_list.append(sentence)

    return by_list

def regex_splitting(sentences, target, substance_regex):
    substance_list = []
    adjunct_list = []
    for sentence in tqdm(sentences, mininterval=5):
        sentence_lower = sentence.lower()
        if target in sentence_lower:
            if re.search(rf"""{substance_regex}""", sentence_lower):
                substance_list.append(sentence)
            else:
                adjunct_list.append(sentence)

    return substance_list, adjunct_list

def string_filtering_tokenized(dataset, target_verbs):
    final = []

    for data in tqdm(dataset, mininterval=5):
        sentence_split = data.split()
        if any([word in sentence_split for word in target_verbs]):
            final.append(data)
    
    return final

def structure_filtering_substance(doc_sentences, target_lemma='with', target_upos="ADP", lemmas=[], path_1=None, path_2=None, path_3=None):
    include_list = []
    exclude_list = []
    reasons = []
    for i, sentence in enumerate(doc_sentences):
        if "@-@" in sentence.text:
            continue
        # get character indices to extract representations
        start = 0
        end = 0
        for w in sentence.words:
            end = end + len(w.text)
            lem = w.lemma
            pos = w.upos
            if lem == target_lemma and pos == target_upos:
                parent = w.head
                if parent == 0:
                    reasons.append("parent is root")
                    start = start + len(w.text) + 1
                    end = end + 1
                    continue
                parent_word = None
                for w_p in sentence.words:
                    if w_p.id == parent:
                        parent_word = w_p
                        break
                # these should be nouns
                if parent_word.upos != "NOUN":
                    #print(parent_word.upos)
                    reasons.append("parent not noun")
                    start = start + len(w.text) + 1
                    end = end + 1
                    continue
                # this is to get the identity of the verb so that we can use it to get representations
                grandparent_lemma = ""
                grandparent_start = 0
                grandparent_end = 0
                for w_g in sentence.words:
                    # add the end of the current word
                    grandparent_end = grandparent_end + len(w_g.text)
                    if w_g.id == parent_word.head:
                        grandparent_lemma = w_g.lemma
                        grandparent_id = w_g.id
                        grandparent_word = w_g
                        break
                    # add the spaces
                    grandparent_start = grandparent_start + len(w_g.text) + 1
                    grandparent_end = grandparent_end + 1
                # these should be verbs
                if grandparent_word.upos != "VERB":
                    reasons.append("grandparent not verb")
                    start = start + len(w.text) + 1
                    end = end + 1
                    continue
                # this is to ensure we're getting a ditransitive
                second_children_deps = [w_second.deprel for w_second in sentence.words if w_second.head == grandparent_id]
                if "obj" not in second_children_deps and "nsubj:pass" not in second_children_deps:
                    reasons.append("not ditransitive")
                    start = start + len(w.text) + 1
                    end = end + 1
                    continue
                output_text = " ".join([w.text for w in sentence.words]).strip()
                if grandparent_lemma in lemmas:
                    sentence_d = {'sent_id' : i, 'text' : output_text, 'target_lemma' : target_lemma, 'target_slice' : (start, end), 
                                'dependent_lemma' : grandparent_lemma, 'dependent_lemma' : grandparent_lemma, 'dependent_slice' : (grandparent_start, grandparent_end)}
                    include_list.append(sentence_d)
                else: 
                    sentence_d = {'sent_id' : i, 'text' : output_text, 'target_lemma' : target_lemma, 'target_slice' : (start, end), 
                                'dependent_lemma' : grandparent_lemma, 'dependent_lemma' : grandparent_lemma, 'dependent_slice' : (grandparent_start, grandparent_end)}
                    exclude_list.append(sentence_d)
            start = start + len(w.text) + 1
            end = end + 1
    if path_1 != None:
        utils.dump_json(include_list, path_1)
    if path_2 != None:
        utils.dump_json(exclude_list, path_2)
    if path_3 != None:
        with open(path_3, "w") as f:
            f.write("\n".join(reasons) + "\n")
    return include_list, exclude_list, reasons


def main():    
    # load the dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="/nlp/scr/jjian/datasets/wikitext-103-raw-v1")
    dataset = dataset['train']

    # filter the dataset based on the regex patterns
    with_list = string_based_filtering(dataset, "with")
    
    substance_path = "/afs/cs.stanford.edu/u/jjian/projects/abstraction/data/substance.txt"
    with open(substance_path, "r") as f:
        substances = f.readlines()
    substances = [str(w).strip() for w in substances]
    substances = utils.get_all_inflections(substances)
    substance_regex_patterns = [rf"with(?:\s+\w+){{0,10}}\s+{verb}|{verb}(?:\s+\w+){{0,10}}\s+with" for verb in list(substances)]
    substance_regex = "|".join(substance_regex_patterns)

    substance_list, adjunct_list = regex_splitting(with_list, "with", substance_regex)
    # serialize the lists to raw txt files
    
    """with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/with.adjunct.raw.unfiltered.txt", "w") as f:
        f.write("\n".join(adjunct_list))"""
    with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/with.substance.raw.unfiltered.txt", "w") as f:
        f.write("\n".join(substance_list))

if __name__ == "__main__":
    #main()
    
    # TODO: this is not integrated with the stuff above.
    filepath = sys.argv[1]
    output_path = sys.argv[2]

    with open(filepath, "r") as f:
        sentences = f.readlines()
    
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    parsed = utils.stanza_parsing_batched(sentences, nlp, 64)

    # serialize this 
    new_doc = Document([])
    new_doc.sentences = parsed
    CoNLL.write_doc2conll(new_doc, output_path)
    
    # load the CoNLLs
    print("Getting Substance")
    doc_substance = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/with.substance.raw.filtered.parsed.conllu")
    doc_substance = doc_substance.sentences
    substance = [w.strip() for w in open("/sailhome/jjian/projects/abstraction/data/substance.txt", 'r').readlines()]

    print("Filtering")
    include_list, exclude_list, reasons = structure_filtering_substance(doc_substance, target_lemma="with", lemmas=substance, path_1="/nlp/scr/jjian/datasets/wikitext_parsed/with.substance.parsed_filtered.json", path_2="/nlp/scr/jjian/datasets/wikitext_parsed/with.substance.excluded.json", path_3="/nlp/scr/jjian/datasets/wikitext_parsed/with.substance.reasons.txt")
    del doc_substance

    