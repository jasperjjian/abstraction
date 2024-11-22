import json
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def json_to_list_of_counters(json_file):
    json_dicts = json.load(open(json_file, "r"))
    verb_list = set([d["dependent_lemma"] for d in json_dicts])

    dict_of_counters = {}
    for verb in verb_list:
        list_of_dicts = [d for d in json_dicts if d["dependent_lemma"] == verb]
        # split the "text" value for each dictionary and join into a list
        list_of_texts = [d["text"].split() for d in list_of_dicts]
        # flatten the list of lists
        flat_list_of_texts = [item for sublist in list_of_texts for item in sublist if item not in ["the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "you", "he", "she", "we", "they"]]
        # count the words
        counter = Counter(flat_list_of_texts)
        dict_of_counters[verb] = counter

    # normalize the counts by the total number of words
    for verb in dict_of_counters.keys():
        total_words = sum(dict_of_counters[verb].values())
        for word in dict_of_counters[verb].keys():
            dict_of_counters[verb][word] = dict_of_counters[verb][word] / total_words

    return dict_of_counters
        



def pairwise_cosine(x1, x2):
    list_of_words1 = set.union(*[set([w for w in d.keys()]) for d in x1.values()])
    list_of_words2 = set.union(*[set([w for w in d.keys()]) for d in x2.values()])
    dict_of_words = {w : i for i, w in enumerate(list_of_words1.union(list_of_words2))}
    arr = np.zeros((len(x1) + len(x2), len(dict_of_words)))

    for i, d in enumerate(x1.values()):
        for w, p in d.items():
            arr[i, dict_of_words[w]] = p
    
    for i, d in enumerate(x2.values()):
        for w, p in d.items():
            arr[i + len(x1), dict_of_words[w]] = p
    
    sim_arr = cosine_similarity(arr)

    return sim_arr

if __name__ == "__main__":
    json_motion = "/nlp/scr/jjian/datasets/wikitext_parsed/motion.objects.json"
    json_ditrans = "/nlp/scr/jjian/datasets/wikitext_parsed/ditransitive.objects.json"
    motion_counters = json_to_list_of_counters(json_motion)
    print(motion_counters["trek"])
    ditrans_counters = json_to_list_of_counters(json_ditrans)

    cosine_dist = pairwise_cosine(motion_counters, ditrans_counters)

    # save this as a pandas df

    all_verbs = list(motion_counters.keys()) + list(ditrans_counters.keys())
    df = pd.DataFrame(cosine_dist, index=all_verbs, columns=all_verbs)
    df.to_csv("/afs/cs.stanford.edu/u/jjian/projects/abstraction/results/motion_ditransitive_cosine_similarity.raw_counts.csv")
    print("done")