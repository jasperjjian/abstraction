from abstraction.minicons.minicons import cwe
import torch
from transformers import BertModel, AutoConfig, AutoModel
from huggingface_hub import list_repo_refs
import os
import h5py
from tqdm import tqdm
import stanza
from stanza.utils.conll import CoNLL
import json
import sys
import shutil


def get_instances(doc, target_lemma='to', target_upos="ADP"):
    final_list = []
    for i in range(len(doc.sentences)):
        sentence = doc.sentences[i]
        for w in sentence.words:
            lem = w.lemma
            pos = w.upos
            if lem == target_lemma and pos == target_upos:
                final_list.append(tuple([i, sentence.text]))
    return final_list

def check_lexical_lemma(doc, instances, target_lemma='to', target_upos="ADP", lemmas=[], grandparent_upos="VERB", path_1=None, path_2=None):
    inclusion_list = []
    exclusion_list = []
    for i,_ in instances:
        sentence = doc.sentences[i]
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
                parent_word = None
                for w_p in sentence.words:
                    if w_p.id == parent:
                        parent_word = w_p
                        break
                grandparent_lemma = ""
                for w_g in sentence.words:
                    if w_g.id == parent_word.head:
                        grandparent_lemma = w_g.lemma
                        grandparent_id = w_g.id
                        grandparent = w_g
                        break
                output_text = " ".join([w.text for w in sentence.words]).strip()
                if grandparent_lemma in lemmas:
                    sentence_d = {'sent_id' : i, 'text' : output_text, 'target_lemma' : target_lemma, 'target_id' : lem_id, 'target_slice' : (start, end), 'lexical_lemma' : grandparent_lemma, 'lexical_id' : grandparent_id}
                    inclusion_list.append(sentence_d)
                elif parent_word.deprel == grandparent_upos:
                    sentence_d = {'sent_id' : i, 'text' : output_text, 'target_lemma' : target_lemma, 'target_id' : lem_id, 'target_slice' : (start, end), 'lexical_lemma' : grandparent_lemma, 'lexical_id' : grandparent_id}
                    exclusion_list.append(sentence_d)
                """elif grandparent.upos == grandparent_upos:
                    sentence_d = {'sent_id' : i, 'text' : output_text, 'target_lemma' : target_lemma, 'target_id' : lem_id, 'target_slice' : (start, end), 'lexical_lemma' : grandparent_lemma, 'lexical_id' : grandparent_id}
                    exclusion_list.append(sentence_d)"""
            start = start + len(w.text) + 1
            end = end + 1
    if path_1 != None:
        dump_json(inclusion_list, path_1)
    if path_2 != None:
        dump_json(exclusion_list, path_2)
    return inclusion_list, exclusion_list

def get_lemmas(txt_path):
    f = open(txt_path, 'r')
    return [str(w).strip() for w in f.readlines()]

def dump_json(list_of_dicts, path):
    json_data = json.dumps(list_of_dicts, indent=4)
    with open(path, "w") as json_file:
        json_file.write(json_data)
    return

def get_children(sentence):
    child_list = []

    """for w in sentence:
        w_id = w.id
        for """

    return child_list