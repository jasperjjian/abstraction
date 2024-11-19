from abstraction.minicons.minicons import cwe
from transformers import AutoConfig
from huggingface_hub import list_repo_refs
from tqdm import tqdm
import torch
import os
import h5py
import json
import sys
import shutil
import random
import numpy as np

def subsample(dataset, num_samples):
    """
    Subsample a dataset.
    """
    random.seed(42)
    return random.sample(dataset, num_samples)

def get_num_layers(model_name_or_path, cache_dir=None):
    # Load the model configuration
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    # Get the number of layers
    num_layers = getattr(config, "num_hidden_layers", None)

    if num_layers is not None:
        return num_layers
    else:
        raise ValueError(f"Unable to retrieve the number of layers for {model_name_or_path}")

def batch_sentences(dataset, batch_size, rep='target'):
    """
    Batch sentences into chunks of size batch_size.
    """
    for i in range(0, len(dataset), batch_size):
        slice_name = f"{rep}_slice"
        if i + batch_size > len(dataset):
            batch = [(sentence['text'], sentence[slice_name]) for sentence in dataset[i:]]
        else:
            batch = [(sentence['text'], sentence[slice_name]) for sentence in dataset[i:i+batch_size]]
        yield batch
    
def save_embeddings_to_hdf5(directory, model_name, dataset, split, batch_size=32, model_shorthand=None, checkpoint="main", cache_dir=None, rep='target'):
    """
    Loop over a dataset, get embeddings using get_embeddings function, and store them in an HDF5 file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    hdf5_path = os.path.join(directory, f"{model_shorthand}.{split}.embeddings.hdf5")
    hdf5_file = h5py.File(hdf5_path, 'w')
    model = cwe.CWE(model_name, device='cuda:0', revision=checkpoint, cache_dir=cache_dir)
    num_layers = get_num_layers(model_name, cache_dir=cache_dir)
    
    # when batched minicons returns a list [tensor, tensor, ...] of shape (num_layers, batch_size, hidden_size)
    # we need it to be [tensor, tensor, ...] of shape (batch_size, num_layers, hidden_size) because it's easier to reason about across multiple batches
    for idx, batched_sentences in enumerate(tqdm(batch_sentences(dataset, batch_size, rep=rep), total=len(dataset)//batch_size + 1, mininterval=5)):
        embeddings = model.extract_representation(batched_sentences, layer=list(range(num_layers)))  # I think this does averaging subword toks by default, how to change idk?
        embeddings = torch.stack(embeddings)
        embeddings = embeddings.permute(1, 0, 2)
        
        # Create a dataset for each sentence index in the HDF5 file
        for idx_sentence in range(embeddings.shape[0]):
            dset = hdf5_file.create_dataset(str(idx * batch_size + idx_sentence), data=np.array(embeddings[idx_sentence]), compression='gzip', compression_opts=9)
        
        del embeddings
    # Close the HDF5 file
    hdf5_file.close()

def load_dataset(instances_path, subsample=True, sample_size=2000):
    with open(instances_path, "r") as f:
        instances = json.load(f)
    if subsample:
        instances = subsample(instances, sample_size)
    for x in instances:
        x['target_slice'] = tuple(x['target_slice'])
        x['dependent_slice'] = tuple(x['dependent_slice'])
    return instances

def loop_checkpoints_and_save(model_name, split, instances, delete_from_cache=False, cache_dir=None, rep="target"):
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.tags]
    # sort the branches by the checkpoint number
    branches = sorted(branches, key=lambda x: int(x.split('checkpoint-')[-1]))
    
    for checkpoint in tqdm(branches, mininterval=5):
        model_name_preprocessed = model_name.split("/")[-1]
        # check if the file already exists
        if os.path.exists(f'/nlp/scr/jjian/data/wikitext/{split}/{rep}/{model_name_preprocessed}.{checkpoint}.{split}.embeddings.hdf5'):
            continue
        save_embeddings_to_hdf5(f'/nlp/scr/jjian/data/wikitext/{split}/{rep}/', model_name, instances, split, model_shorthand=f"{model_name_preprocessed}.{checkpoint}", checkpoint=checkpoint, cache_dir=cache_dir, rep=rep)
        if delete_from_cache:
            model_path = model_name.replace('/', '--')
            shutil.rmtree(f"/sailhome/jjian/.cache/huggingface/hub_1/models--{model_path}")
    return

if __name__ == "__main__":
    corpus_path = sys.argv[1]
    split = sys.argv[2]
    rep = sys.argv[3]
    
    model_list = ["stanford-crfm/battlestar-gpt2-small-x49"]
    cache_dir = "/nlp/scr/jjian/mistral-checkpoints/"
    sample_size = 2000

    dataset = load_dataset(corpus_path, subsample=False, sample_size=sample_size)

    for model in model_list:
        loop_checkpoints_and_save(model, split, dataset, cache_dir=cache_dir, delete_from_cache=False, rep=rep)