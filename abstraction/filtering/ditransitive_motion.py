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

def string_based_filtering(dataset, target, motion_regex, ditransitive_regex):
    motion_list = []
    ditransitive_list = []
    for data in tqdm(dataset, mininterval=5):
        if data != "":
            sentences = sent_tokenize(data["text"])
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if target in sentence_lower:
                    if re.search(rf"""{motion_regex}""", sentence_lower):
                        motion_list.append(sentence)
                    if re.search(rf"""{ditransitive_regex}""", sentence_lower):
                        ditransitive_list.append(sentence)

    return motion_list, ditransitive_list        

def string_filtering_tokenized(dataset, target_verbs):
    final = []

    for data in tqdm(dataset, mininterval=5):
        sentence_split = data.split()
        if any([word in sentence_split for word in target_verbs]):
            final.append(data)
    
    return final


def main():
    # load lists of ditransitive and motion verbs
    ditranstive_path = "/afs/cs.stanford.edu/u/jjian/projects/abstraction/data/ditransitives.txt"
    with open(ditranstive_path, "r") as f:
        ditransitives = f.readlines()
    ditransitives = [str(w).strip() for w in ditransitives]

    motion_path = "/afs/cs.stanford.edu/u/jjian/projects/abstraction/data/motion.txt"
    with open(motion_path, "r") as f:
        motion = f.readlines()
    motion = [str(w).strip() for w in motion]

    # create a set of each list with all the possible inflections
    ditransitives = utils.get_all_inflections(ditransitives)
    motion = utils.get_all_inflections(motion)
    
    # create regex patterns for each list
    motion_regex_patterns = [rf"to(?:\s+\w+){{0,2}}\s+{verb}|{verb}(?:\s+\w+){{0,2}}\s+to" for verb in list(motion)]
    motion_regex = "|".join(motion_regex_patterns)

    ditransitive_regex_patterns = [rf"to(?:\s+\w+){{0,5}}\s+{verb}|{verb}(?:\s+\w+){{0,5}}\s+to" for verb in list(ditransitives)]
    ditransitive_regex = "|".join(ditransitive_regex_patterns)

    # load the dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="/nlp/scr/jjian/datasets/wikitext-103-raw-v1")
    dataset = dataset['train']

    # filter the dataset based on the regex patterns
    motion_list, ditransitive_list = string_based_filtering(dataset, "to", motion_regex, ditransitive_regex)
    
    # serialize the lists to raw txt files
    
    with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/motion.raw.unfiltered.txt", "w") as f:
        f.write("\n".join(motion_list))
    with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/ditransitive.raw.unfiltered.txt", "w") as f:
        f.write("\n".join(ditransitive_list))
    
    # load the raw txt files
    with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/motion.raw.unfiltered.txt", "r") as f:
        motion_list = f.readlines()
    with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/ditransitive.raw.unfiltered.txt", "r") as f:
        ditransitive_list = f.readlines()
    
    # filter the lists based on the tokenized dataset
    motion_list = string_filtering_tokenized(motion_list, motion)
    ditransitive_list = string_filtering_tokenized(ditransitive_list, ditransitives)

    # serialize the filtered lists to raw txt files
    with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/motion.raw.filtered.txt", "w") as f:
        f.write("".join(motion_list))
    with open("/afs/cs.stanford.edu/u/jjian/projects/abstraction/scraped_data/wikitext/ditransitive.raw.filtered.txt", "w") as f:
        f.write("".join(ditransitive_list))

if __name__ == "__main__":
    #main()
    # get from OS the path to a raw txt file of sentences

    filepath = sys.argv[1]
    output_path = sys.argv[2]

    with open(filepath, "r") as f:
        sentences = f.readlines()
    
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    parsed = utils.stanza_parsing(sentences, nlp)

    # serialize this 
    new_doc = Document([])
    new_doc.sentences = parsed
    CoNLL.write_doc2conll(new_doc, output_path)