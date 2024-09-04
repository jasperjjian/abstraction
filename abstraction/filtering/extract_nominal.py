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
    nsubj_count = 0
    pass_count = 0
    for w in parse.words:
        end = beginning + len(w.text)
        if w.head == predicate_id:
            if w.deprel == "nsubj" or w.deprel == "nsubj:pass":
                if w.deprel == "nsubj":
                    nsubj_count += 1
                else:
                    pass_count += 1
                subject_text = w.text
                break
        beginning = end + 1
    
    return subject_text, [beginning, end], nsubj_count, pass_count

def get_nominals(sample, parses):
    constructed_samples = []
    nsubj_total = 0
    pass_total = 0
    for i, sentence in enumerate(sample):
        parse = parses[i]
        predicate_id = identify_predicate(sentence, parse)
        if predicate_id == -1:
            continue
        object_text, object_indices, nsubj, passive = get_syntactic_object(sentence, parse, predicate_id)
        nsubj_total += nsubj
        pass_total += passive
        if object_text == "":
            continue
        new_data_instance = sentence
        new_data_instance["subject"] = object_text
        new_data_instance["subject_slice"] = object_indices
        constructed_samples.append(new_data_instance)
    print(f"nsubj count: {nsubj_total}, pass count: {pass_total}")
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
    #utils.dump_json(nominals, "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.subjects.json")