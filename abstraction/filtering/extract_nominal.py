from stanza.utils.conll import CoNLL
from abstraction.filtering import utils
import json 

def identify_predicate(sentence, parse):
    start = sentence["dependent_slice"][0]
    lemma = sentence["dependent_lemma"]
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
    for w in parse.words:
        end = beginning + len(w.text)
        if w.head == predicate_id:
            if w.deprel == "obj":
                subject_text = w.text
                break
        beginning = end + 1
    
    return subject_text, [beginning, end]

def get_nominals(sample, parses):
    constructed_samples = []
    for i, sentence in enumerate(sample):
        parse = parses[i]
        predicate_id = identify_predicate(sentence, parse)
        if predicate_id == -1:
            continue
        object_text, object_indices = get_syntactic_object(sentence, parse, predicate_id)
        if object_text == "":
            continue
        new_data_instance = sentence
        new_data_instance["object"] = object_text
        new_data_instance["object_slice"] = object_indices
        constructed_samples.append(new_data_instance)
    return constructed_samples

if __name__ == "__main__":
    #main()
    print("Loading data...")
    ditrans_sampled = "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.parsed_filtered.balanced_sampled.json"
    ditrans_json = json.load(open(ditrans_sampled, "r"))
    print("Loading parses...")
    ditrans_parses = CoNLL.conll2doc("/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.raw.filtered.parsed.conllu")
    ditrans_parses = ditrans_parses.sentences
    ditrans_sample_parses = [ditrans_parses[sentence["sent_id"]] for sentence in ditrans_json]
    del ditrans_parses

    nominals = get_nominals(ditrans_json, ditrans_sample_parses)
    utils.dump_json(nominals, "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.nominals.json")