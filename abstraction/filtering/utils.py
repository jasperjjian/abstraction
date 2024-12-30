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
    # sort sentences by length from longest to shortest
    sentences.sort(key=lambda x: len(x), reverse=True)
    avg_length = sum([len(sentence) for sentence in sentences]) / len(sentences)
    # partition the sentences into above and below average length
    long_sentence_end = 0
    for i, sentence in enumerate(sentences):
        if len(sentence) < 50 * avg_length:
            long_sentence_end = i
            break
    long_sentences = sentences[:long_sentence_end]
    short_sentences = sentences[long_sentence_end:]
    parsed = []

    # except the last batch
    for i in tqdm(range(0, (len(short_sentences) // batch_size) * batch_size, batch_size), mininterval=5):
        try:
            batch = short_sentences[i:i+batch_size]
            batch_doc = "".join(batch)
            docs = pipeline(batch_doc)
            parsed += docs.sentences
        except RuntimeError:
            print(f"Error parsing batch {i}")
    
    # parse long sentences next, no batching
    
    for i in tqdm(range(0, len(long_sentences)), mininterval=5):
        try:
            doc = pipeline(long_sentences[i])
            parsed += doc.sentences
        except RuntimeError:
            print(f"Error parsing sentence {i}")

    # last batch
    if len(short_sentences) % batch_size != 0:
        batch = short_sentences[(len(short_sentences) // batch_size) * batch_size:]
        batch_doc = "".join(batch)
        docs = pipeline(batch_doc)
        parsed += docs.sentences
    
    return parsed

def dump_json(list_of_dicts, path):
    json_data = json.dumps(list_of_dicts, indent=4)
    with open(path, "w") as json_file:
        json_file.write(json_data)
    return
