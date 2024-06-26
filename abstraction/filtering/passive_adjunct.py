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

def string_filtering_tokenized(dataset, target_verbs):
    final = []

    for data in tqdm(dataset, mininterval=5):
        sentence_split = data.split()
        if any([word in sentence_split for word in target_verbs]):
            final.append(data)
    
    return final

def structure_filtering_by(doc_sentences, target_lemma='by', target_upos="ADP", path_1=None, path_2=None):
    adjunct_list = []
    passive_list = []
    reasons = []
    for i, sentence in enumerate(doc_sentences):
        # get character indices to extract representations
        start = 0
        end = 0
        for w in sentence.words:
            end = end + len(w.text)
            lem = w.lemma
            lem_id = w.id
            pos = w.upos
            deprel = w.deprel
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
                if parent_word.upos != "NOUN" and parent_word.upos != "NUM":
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
                second_children_deps = [w_second.deprel for w_second in sentence.words if w_second.head == grandparent_id]
                output_text = " ".join([w.text for w in sentence.words]).strip()
                if w_p.deprel == "obl:agent" and grandparent_word.xpos == "VBN" and "aux:pass" in second_children_deps:
                    sentence_d = {'sent_id' : i, 'text' : output_text, 'target_lemma' : target_lemma, 'target_slice' : (start, end), 
                                'dependent_lemma' : grandparent_lemma, 'dependent_lemma' : grandparent_lemma, 'dependent_slice' : (grandparent_start, grandparent_end), 'dependent_inflection' : grandparent_word.xpos}
                    passive_list.append(sentence_d)
                elif w_p.deprel == "obl" and grandparent_word.xpos != "VBN":
                    sentence_d = {'sent_id' : i, 'text' : output_text, 'target_lemma' : target_lemma, 'target_slice' : (start, end), 
                                'dependent_lemma' : grandparent_lemma, 'dependent_lemma' : grandparent_lemma, 'dependent_slice' : (grandparent_start, grandparent_end), 'dependent_inflection' : grandparent_word.xpos}
                    adjunct_list.append(sentence_d)
            start = start + len(w.text) + 1
            end = end + 1
    if path_1 != None:
        utils.dump_json(passive_list, path_1)
    if path_2 != None:
        utils.dump_json(adjunct_list, path_2)
    return passive_list, adjunct_list, reasons


def main():    
    # load the dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="/nlp/scr/jjian/datasets/wikitext-103-raw-v1")
    dataset = dataset['train']

    # filter the dataset based on the regex patterns
    by_list = string_based_filtering(dataset, "by")
    
    # serialize the lists to raw txt files
    
    with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/by.raw.unfiltered.txt", "w") as f:
        f.write("\n".join(by_list))
    
    #TODO: Not integrated
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

if __name__ == "__main__":
    #main()

    # load the CoNLLs
    by_head = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/by.raw.unfiltered.head.parsed.conllu")
    by_tail = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/by.raw.unfiltered.tail.parsed.conllu")
    by_combined = by_head.sentences + by_tail.sentences

    # filter the CoNLLs
    
    passive_list, adjunct_list, reasons = structure_filtering_by(by_combined, path_1="/nlp/scr/jjian/datasets/wikitext_parsed/by.passive.parsed_filtered.json", path_2="/nlp/scr/jjian/datasets/wikitext_parsed/by.adjunct.parsed_filtered.json")