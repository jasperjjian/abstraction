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

def parse_conll(filename):
    doc = CoNLL.conll2doc(filename)
    return doc


def get_num_layers(model_name_or_path):
    # Load the model configuration
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Get the number of layers
    num_layers = getattr(config, "num_hidden_layers", None)

    if num_layers is not None:
        return num_layers
    else:
        raise ValueError(f"Unable to retrieve the number of layers for {model_name_or_path}")


def save_embeddings_to_hdf5(directory, model_name, dataset, split, model_shorthand=None, checkpoint="main"):
    """
    Loop over a dataset, get embeddings using get_embeddings function, and store them in an HDF5 file.
    """
    """model_shorthand = model_name
    if 'Llama-2' in model_shorthand:
        model_shorthand = model_name[11:-3]""" # pulls out, e.g., 'Llama-2-7b' from 'meta-llama/Llama-2-7b-hf'
    if not os.path.exists(directory):
        os.makedirs(directory)
    hdf5_path = os.path.join(directory, f"{model_shorthand}_{split}_embeddings.hdf5")
    hdf5_file = h5py.File(hdf5_path, 'w')
    #model = cwe.CWE(model_name, device='cuda:0', token="AUTH_TOKEN", cache_dir="CACHE_DIR")
    model = cwe.CWE(model_name, device='cuda:0', revision=checkpoint, cache_dir="/sailhome/jjian/.cache/huggingface/hub_1/")
    num_layers = get_num_layers(model_name)
    #print(f"Processing {len(dataset['sentence'])} sentences. This may take a while.")

    # so this is unbatched right now, but only needs to be run once per dataset which is not AWFUL?
    for idx, sentence in enumerate(tqdm(dataset)):
        indices = sentence['target_slice']
        embeddings = model.extract_representation([(sentence['text'], indices)], layer=list(range(num_layers)))  # I think this does averaging subword toks by default, how to change idk?
        embeddings = torch.stack(embeddings)
        # Create a dataset for each sentence index in the HDF5 file
        dset = hdf5_file.create_dataset(str(idx), data=embeddings)

    # Close the HDF5 file
    hdf5_file.close()

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

def main():
    f = open("/sailhome/jjian/projects/abstraction/ditrans_to.train.json", "r")
    ditrans = json.load(f)
    for x in ditrans:
        x['target_slice'] = tuple(x['target_slice'])

    f = open("/sailhome/jjian/projects/abstraction/other_to.train.json", "r")
    others = json.load(f)
    for x in others:
        x['target_slice'] = tuple(x['target_slice'])

    save_embeddings_to_hdf5('/nlp/scr/jjian/data/', 'bert-base-uncased', ditrans, 'ditransitive')
    save_embeddings_to_hdf5('/nlp/scr/jjian/data/', 'bert-base-uncased', others, 'others')

def get_from_checkpoint(model_name, split, instances=None, instances_path=None, delete_from_cache=True, predone=[]):
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    if instances == None and instances_path != None:
        f = open(instances_path, "r")
        instances = json.load(f)
        for x in instances:
            x['target_slice'] = tuple(x['target_slice'])
    else:
        raise FileNotFoundError
    branches = set(branches) - set(predone)
    for checkpoint in tqdm(branches):
        model_name_preprocessed = model_name.split("/")[-1]
        save_embeddings_to_hdf5(f'/nlp/scr/jjian/data/{split}/', model_name, instances, split, model_shorthand=f"{model_name_preprocessed}.{checkpoint}", checkpoint=checkpoint)
        if delete_from_cache:
            model_path = model_name.replace('/', '--')
            shutil.rmtree(f"/sailhome/jjian/.cache/huggingface/hub_1/models--{model_path}")
    return

if __name__ == "__main__":
    """lemma_path = sys.argv[1]
    corpus_path = sys.argv[2]
    save_path = sys.argv[3]
    split = sys.argv[4]
    model_list = ['gpt2', 'bert-base-uncased']

    for model in model_list:
        corpus = parse_conll(corpus_path)
        instances = get_instances(corpus)
        lemma_list = get_lemmas(lemma_path)
        incl, excl = check_lexical_lemma(corpus, [x[0] for x in instances], lemmas=lemma_list, path_1=save_path)
        save_embeddings_to_hdf5(f'/nlp/scr/jjian/data/{split}/', model, incl, split)"""
    corpus_path = sys.argv[1]
    split = sys.argv[2]
    model_list = ["stanford-crfm/battlestar-gpt2-small-x49"]
    preloaded = get_lemmas("/nlp/scr/jjian/data/inf/checkpoints.txt")
    preloaded = [f'checkpoint-{checkpoint}' for checkpoint in preloaded]
    
    for model in model_list:
        get_from_checkpoint(model, split, instances_path=corpus_path, predone=preloaded)