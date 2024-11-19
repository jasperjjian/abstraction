import stanza
import json
import random
import pickle
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from lemminflect import getInflection
from abstraction.filtering import utils

def sample_ambiguous_sentences(passive_dataset, n=2000):
    random.seed(42)
    final_sample = []
    candidate_sentences = random.sample(passive_dataset, n * 3)

    for sentence in candidate_sentences:
        dependent_lemma = sentence["dependent_lemma"]
        if getInflection(dependent_lemma, tag='VBN') == getInflection(dependent_lemma, tag='VBD'):
            final_sample.append(sentence)
            if len(final_sample) == n:
                break

    return final_sample

def get_syntactic_subject(sentence, parse, predicate_id, ensure_contiguity=True):
    subject_dependent_id = -1
    for w in parse.words:
        if w.head == predicate_id:
            if w.deprel == "nsubj" or w.deprel == "nsubj:pass":
                subject_dependent_id = w.id
    
    subject_constituent_ids = [subject_dependent_id]
    size = 0

    while len(subject_constituent_ids) > size:
        size = len(subject_constituent_ids)
        for w in parse.words:
            if w.head in subject_constituent_ids and w.id not in subject_constituent_ids:
                subject_constituent_ids.append(w.id)

    if ensure_contiguity:
        subject_constituent_ids.sort()
        if subject_constituent_ids != list(range(subject_constituent_ids[0], subject_constituent_ids[-1] + 1)):
            return ""
    
    subject = " ".join([w.text for w in parse.words if w.id in subject_constituent_ids])

    return subject

"""def get_syntactic_object(sentence, parse, predicate_id, ensure_contiguity=True):
    object_dependent_id = -1
    for w in parse.words:
        if w.head == predicate_id:
            if w.deprel == "obl" or w.deprel == "obl:agent":
                object_dependent_id = w.id
    
    object_constituent_ids = [object_dependent_id]
    size = 0

    while len(object_constituent_ids) > size:
        size = len(object_constituent_ids)
        for w in parse.words:
            if w.head in object_constituent_ids and w.id not in object_constituent_ids and w.text.lower() != "by":
                object_constituent_ids.append(w.id)

    if ensure_contiguity:
        object_constituent_ids.sort()
        if object_constituent_ids != list(range(object_constituent_ids[0], object_constituent_ids[-1] + 1)):
            return ""
    
    obj = " ".join([w.text for w in parse.words if w.id in object_constituent_ids])

    return obj"""
def get_syntactic_object(sentence, parse, predicate_id, ensure_contiguity=True):
    object_dependent_id = -1
    for w in parse.words:
        if w.head == predicate_id:
            if w.deprel == "obj":
                for w1 in parse.words:
                    if w1.head == w.id and w1.deprel == "case" and w1.text.lower() == "to":
                        continue
                object_dependent_id = w.id
                break

    object_constituent_ids = [object_dependent_id]
    size = 0

    while len(object_constituent_ids) > size:
        size = len(object_constituent_ids)
        for w in parse.words:
            if w.head in object_constituent_ids and w.id not in object_constituent_ids and w.text.lower() != "by":
                object_constituent_ids.append(w.id)

    if ensure_contiguity:
        object_constituent_ids.sort()
        if object_constituent_ids != list(range(object_constituent_ids[0], object_constituent_ids[-1] + 1)):
            return ""
    
    obj = " ".join([w.text for w in parse.words if w.id in object_constituent_ids])

    return obj

"""def identify_predicate(sentence, parse):
    start = sentence["dependent_slice"][0]
    lemma = sentence["dependent_lemma"]
    count = 0
    for w in parse.words:
        if count == start and w.lemma == lemma:
            id = w.id
            for w1 in parse.words:
                if w1.head == id and w1.deprel == "aux:pass" and w1.text.lower() != "be":
                    return id
        count += len(w.text) + 1
    return -1"""

def identify_predicate(sentence, parse):
    start = sentence["dependent_slice"][0]
    lemma = sentence["dependent_lemma"]
    count = 0
    for w in parse.words:
        if count == start and w.lemma == lemma:
            id = w.id
            return id
            """for w1 in parse.words:
                if w1.head == id and w1.deprel == "obj":
                    return id"""
                
        count += len(w.text) + 1
    return -1

def inflect_pronoun(pronoun):
    if pronoun.lower() == "i":
        return "me"
    elif pronoun.lower() == "he":
        return "him"
    elif pronoun.lower() == "she":
        return "her"
    elif pronoun.lower() == "we":
        return "us"
    elif pronoun.lower() == "they":
        return "them"

def get_new_sentence(sample, parses):
    constructed_samples = []
    subject_problems = 0
    object_problems = 0

    for i, sentence in enumerate(sample):
        if sentence["decision"] != "y":
            continue
        parse = parses[i]
        predicate_id = identify_predicate(sentence, parse)
        if predicate_id == -1:
            continue
        logical_object = get_syntactic_subject(sentence, parse, predicate_id, ensure_contiguity=True)
        #logical_object = logical_object.lower()
        if logical_object == "":
            object_problems += 1
            continue
        if "who" in logical_object.lower() or "which" in logical_object.lower():
            continue
        if logical_object.split()[0].lower() in ["the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "you", "he", "she", "we", "they"]:
            logical_object = logical_object[0].lower() + logical_object[1:]
        """if logical_object.lower() in ["i", "he", "she", "we", "they"]:
            logical_object = inflect_pronoun(logical_object)"""
        logical_subject = get_syntactic_object(sentence, parse, predicate_id, ensure_contiguity=True)
        if logical_subject == "":
            subject_problems += 1
            continue
        
        """if logical_subject[0].isalpha():
            logical_subject = logical_subject[0].upper() + logical_subject[1:]"""
        predicate = getInflection(sentence["dependent_lemma"], tag='VBD')[0]
        #predicate = sentence["dependent_lemma"]
        
        new_data_instance = sentence

        """text = f"{logical_subject} {predicate} {logical_object} by"
        target_slice = [len(f"{logical_subject} {predicate} {logical_object} "), len(f"{logical_subject} {predicate} {logical_object} by")]
        dependent_slice = [len(f"{logical_subject} "), len(f"{logical_subject} {predicate}")]
        """
        text = f"The person that {logical_object} {predicate} {logical_subject} to"
        target_slice = [len(f"The person that {logical_object} {predicate} {logical_subject} "), len(f"The person that {logical_object} {predicate} {logical_subject} to")]
        dependent_slice = [len(f"The person that {logical_object} "), len(f"The person that {logical_object} {predicate}")]

        new_data_instance["text"] = text
        new_data_instance["target_slice"] = target_slice
        new_data_instance["dependent_slice"] = dependent_slice
        new_data_instance["dependent_inflection"] = "VBD"
        constructed_samples.append(new_data_instance)
    
    print(f"Subject problems: {subject_problems}")
    print(f"Object problems: {object_problems}")
    return constructed_samples

def main():
    #passive_path = "/nlp/scr/jjian/datasets/wikitext_parsed/by.passive.parsed_filtered.json"
    passive_path = "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed.annotated.json"
    # load the passive json dataset 
    passive_json = json.load(open(passive_path, "r"))
    passive_sample = passive_json
    #passive_sample = sample_ambiguous_sentences(passive_json, n=2000)
    #utils.dump_json(passive_sample, "/nlp/scr/jjian/datasets/wikitext_parsed/by.passive.sampled.json")
    """print(passive_sample[:10])

    print("Loading head")
    by_head = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/by.raw.unfiltered.head.parsed.conllu")
    print("Loading tail")
    by_tail = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/by.raw.unfiltered.tail.parsed.conllu")
    by_combined = by_head.sentences + by_tail.sentences
    del by_head
    del by_tail
    

    # get the parses for each sentence by indexing into the combined document
    passive_sample_parses = [by_combined[sentence["sent_id"]] for sentence in passive_sample]
    
    # pickle the thing above
    pickle.dump(passive_sample_parses, open("/nlp/scr/jjian/datasets/wikitext_parsed/by.passive.interim.pkl", "wb"))
    """
    #passive_sample_parses = pickle.load(open("/nlp/scr/jjian/datasets/wikitext_parsed/by.passive.interim.pkl", "rb"))
    
    # write it to a conll with CoNLL.write_doc2conll
    
    """passive_sample_docs = Document([s.to_dict() for s in passive_sample_parses])
    CoNLL.write_doc2conll(passive_sample_docs, "/nlp/scr/jjian/datasets/wikitext_parsed/by.passive.interim.conllu")"""
    passive_sample_parses = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.raw.filtered.parsed.conllu").sentences
    passive_sample_parses = [passive_sample_parses[sentence["sent_id"]] for sentence in passive_sample]
    # get the new sentences
    new_samples = get_new_sentence(passive_sample, passive_sample_parses)

    # dump the new samples
    #utils.dump_json(new_samples, "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.rel_clause_obj.constructed.json")

if __name__ == "__main__":
    main()