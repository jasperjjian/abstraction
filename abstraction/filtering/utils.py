from lemminflect import getAllInflections, getAllInflectionsOOV
from tqdm import tqdm
import json

def get_all_inflections(word_list : list[str]):
    inflections = set()
    for word in word_list:
        word_inflections = getAllInflections(word, upos="VERB")
        if word_inflections == {}:
            word_inflections = getAllInflectionsOOV(word, upos="VERB")
        word_inflections = set([inflect[0] for inflect in word_inflections.values()])
        inflections = inflections.union(word_inflections)
    
    return inflections
    

def stanza_parsing(sentences, pipeline):
    parsed = []

    for sentence in tqdm(sentences, mininterval=5):
        doc = pipeline(sentence)
        parsed += doc.sentences
    
    return parsed

def stanza_parsing_batched(sentences, pipeline, batch_size):
    parsed = []

    # except the last batch
    for i in tqdm(range(0, (len(sentences) // batch_size) * batch_size, batch_size), mininterval=5):
        batch = sentences[i:i+batch_size]
        batch_doc = "".join(batch)
        docs = pipeline(batch_doc)
        parsed += docs.sentences
    
    # last batch
    if len(sentences) % batch_size != 0:
        batch = sentences[(len(sentences) // batch_size) * batch_size:]
        batch_doc = "".join(batch)
        docs = pipeline(batch_doc)
        parsed += docs.sentences
    
    return parsed

def dump_json(list_of_dicts, path):
    json_data = json.dumps(list_of_dicts, indent=4)
    with open(path, "w") as json_file:
        json_file.write(json_data)
    return
