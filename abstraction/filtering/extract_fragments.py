from stanza.utils.conll import CoNLL
from abstraction.filtering import utils
import json 
import ast

def identify_predicate(sentence, parse):
    start = sentence["dependent_slice"][0]
    lemma = sentence["dependent_lemma"]
    count = 0
    for w in parse.words:
        if count == start and w.lemma == lemma:
            """if w.feats != None and "Voice=Pass" in w.feats:
                break"""
            id = w.id
            return id
        count += len(w.text) + 1
    return -1

def identify_preposition(sentence, parse):
    start = sentence["dependent_slice"][0]
    lemma = sentence["dependent_lemma"]
    count = 0
    for w in parse.words:
        if count == start and w.lemma == lemma:
            if w.feats != None and "Voice=Pass" in w.feats:
                return -1
    start = sentence["target_slice"][0]
    lemma = sentence["target_lemma"]
    count = 0
    for w in parse.words:
        if count == start and w.lemma == lemma:
            id = w.id
            return id
        count += len(w.text) + 1
    return -1

def get_syntactic_object(sentence, parse, predicate_id, ensure_contiguity=True):
    beginning = 0
    end = 0
    subject_text = ""
    nsubj_count = 0
    pass_count = 0
    for w in parse.words:
        end = beginning + len(w.text)
        if w.head == predicate_id:
            if w.deprel == "obj":
                if w.deprel == "nsubj":
                    nsubj_count += 1
                else:
                    pass_count += 1
                subject_text = w.text
                break
        beginning = end + 1
    
    return subject_text, [beginning, end], nsubj_count, pass_count

def get_fragments(sample, parses):
    constructed_samples = []
    for i, sentence in enumerate(sample):
        parse = parses[i]
        predicate_id = identify_predicate(sentence, parse)
        if predicate_id == -1:
            continue
        # extract all the words up to and including the verb
        fragment_text = ""
        for w in parse.words:
            if w.id <= predicate_id:
                fragment_text += w.text + " "
        fragment_text += "the"
        new_data_instance = sentence
        new_data_instance["verb_fragment"] = fragment_text

        preposition_id = identify_preposition(sentence, parse)
        if preposition_id == -1:
            continue
        fragment_text = ""
        for w in parse.words:
            if w.id <= preposition_id:
                fragment_text += w.text + " "
        new_data_instance["preposition_fragment"] = fragment_text + "the"
        new_data_instance["preposition_fragment_bare"] = fragment_text[:-1]
        #fragment_text += "the"
        #new_data_instance["preposition_fragment"] = fragment_text

        constructed_samples.append(new_data_instance)
    return constructed_samples

"""def get_fragments(sample):

    constructed_samples = []
    for i, sentence in enumerate(sample):
        if sentence["decision"] != "y":
            continue
        new_data_instance = sentence
        dep_slice = sentence["dependent_slice"]
        target_slice = sentence["target_slice"]
        new_data_instance["verb_fragment"] = sentence["text"][:dep_slice[1]] + " the"
        new_data_instance["preposition_fragment"] = sentence["text"][:target_slice[1]] + " the"
        constructed_samples.append(new_data_instance)
    return constructed_samples"""


if __name__ == "__main__":
    #main()
    print("Loading data...")
    ditrans_sampled = "/nlp/scr/jjian/datasets/wikitext_parsed/substance.parsed.annotated.json"
    ditrans_json = json.load(open(ditrans_sampled, "r"))
    print("Loading parses...")
    ditrans_parses = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/substance.raw.filtered.parsed.conllu")
    ditrans_parses = ditrans_parses.sentences
    ditrans_sample_parses = [ditrans_parses[sentence["sent_id"]] for sentence in ditrans_json]
    del ditrans_parses

    sentence_fragments = get_fragments(ditrans_json, ditrans_sample_parses)
    #sentence_fragments = get_fragments(ditrans_json)
    utils.dump_json(sentence_fragments, "/nlp/scr/jjian/datasets/wikitext_parsed/substance.fragments.json")