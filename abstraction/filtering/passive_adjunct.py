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


def main():    
    # load the dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="/nlp/scr/jjian/datasets/wikitext-103-raw-v1")
    dataset = dataset['train']

    # filter the dataset based on the regex patterns
    by_list = string_based_filtering(dataset, "by")
    
    # serialize the lists to raw txt files
    
    with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/by.raw.unfiltered.txt", "w") as f:
        f.write("\n".join(by_list))

if __name__ == "__main__":
    #main()
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