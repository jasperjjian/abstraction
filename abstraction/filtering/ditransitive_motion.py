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
    
    for data in tqdm(dataset, mininterval=5, total=8000000):
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

def structure_filtering_ditransitives(doc_sentences, target_lemma='to', target_upos="ADP", lemmas=[], path_1=None, path_2=None, path_3=None):
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

def structure_filtering_motion(doc_sentences, target_lemma='to', target_upos="ADP", lemmas=[], grandparent_upos="VERB", path_1=None, path_2=None, path_3=None):
    include_list = []
    exclude_list = []
    reasons = []
    for i, sentence in enumerate(doc_sentences):
        # get character indices to extract representations
        if "@-@" in sentence.text:
            continue
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
                    interveners = []
                    # add the end of the current word
                    grandparent_end = grandparent_end + len(w_g.text)
                    if w_g.id == parent_word.head:
                        grandparent_lemma = w_g.lemma
                        grandparent_id = w_g.id
                        grandparent_word = w_g
                        if grandparent_id > lem_id:
                            break
                        else:
                            for w_i in sentence.words:
                                if w_i.id > grandparent_id and w_i.id < lem_id:
                                    interveners.append(w_i.upos)
                                elif w_i.id == lem_id:
                                    break
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
                if "NOUN" in interveners or "PROPN" in interveners or "PRON" in interveners:
                    reasons.append("intervening noun")
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
    #dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="/nlp/scr/jjian/datasets/wikitext-103-raw-v1")
    dataset = load_dataset("openwebtext", cache_dir="/nlp/scr/jjian/datasets/openwebtext", streaming=True)
    dataset = dataset['train']

    # filter the dataset based on the regex patterns
    motion_list, ditransitive_list = string_based_filtering(dataset, "to", motion_regex, ditransitive_regex)
    del dataset
    # serialize the lists to raw txt files
    
    with open("/nlp/scr/jjian/datasets/openwebtext/motion.raw.unfiltered.txt", "w") as f:
        f.write("\n".join(motion_list))
    with open("/nlp/scr/jjian/datasets/openwebtext/ditransitive.raw.unfiltered.txt", "w") as f:
        f.write("\n".join(ditransitive_list))
    """
    # open the above files
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
    CoNLL.write_doc2conll(new_doc, output_path)"""

if __name__ == "__main__":
    main()
    """
    print("Getting Ditransitives")
    doc_ditransitive = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.raw.filtered.parsed.conllu")
    doc_ditransitive = doc_ditransitive.sentences
    ditransitives = [w.strip() for w in open("/sailhome/jjian/projects/abstraction/data/ditransitives.txt", 'r').readlines()]

    print("Filtering")
    include_list, exclude_list, reasons = structure_filtering_ditransitives(doc_ditransitive, lemmas=ditransitives, path_1="/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed_filtered.json", path_2="/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.excluded.json", path_3="/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.reasons.txt")
    del doc_ditransitive
    
    print("Getting Motion")
    doc_motion = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/motion.raw.filtered.parsed.conllu")
    doc_motion = doc_motion.sentences

    # remove doc_motion_head and tail
    print("Filtering")
    motion = [w.strip() for w in open("/sailhome/jjian/projects/abstraction/data/motion.txt", 'r').readlines()]
    include_list, exclude_list, reasons = structure_filtering_motion(doc_motion, lemmas=motion, path_1="/nlp/scr/jjian/datasets/wikitext_parsed/motion.parsed_filtered.json", path_2="/nlp/scr/jjian/datasets/wikitext_parsed/motion.excluded.json", path_3="/nlp/scr/jjian/datasets/wikitext_parsed/motion.reasons.txt")
    """