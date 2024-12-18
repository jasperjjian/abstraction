import json
import random
import sys
from collections import Counter

def balanced_sample(dataset, sample_size, threshold=None):
    random.seed(42)

    # split the dataset by the unique values of dependent_lemma
    split = {}
    for s in dataset:
        dependent_lemma = s['dependent_lemma']
        if dependent_lemma not in split:
            split[dependent_lemma] = []
        split[dependent_lemma].append(s)
    
    # get the count for each list
    count = {}
    for key in split:
        count[key] = len(split[key])
    count = Counter(count)
    count = count.most_common()

    if threshold != None:
        count = [(key, counts) for key, counts in count if counts >= threshold]

    minimum_sample_size = sample_size // len(count)

    # get the final sample
    sample = []
    sampled_lemmas = []
    # iterate reversed
    for key, counts in reversed(count):
        if counts < minimum_sample_size:
            sample.extend(split[key])
            sampled_lemmas.append(key)
        minimum_sample_size = (sample_size - len(sample)) // (len(count) - len(sampled_lemmas))
    
    for key, counts in reversed(count):
        if counts >= minimum_sample_size and key not in sampled_lemmas:
            sample.extend(random.sample(split[key], minimum_sample_size))
            sampled_lemmas.append(key)
            if (len(count) - len(sampled_lemmas)) == 0:
                break
        minimum_sample_size = (sample_size - len(sample)) // (len(count) - len(sampled_lemmas))
    return sample

def get_lemma_counts(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    lemmas = [s['dependent_lemma'] for s in dataset]
    return Counter(lemmas)

def main():
    data_path = sys.argv[1]
    sample_path = sys.argv[2]

    with open(data_path, 'r') as f:
        data = json.load(f)
    
    verbs = ["cooperate", "correspond", "correlate", "conflict", "team", "consult", "intersperse", "talk", "mate", "affiliate"]
    data = [s for s in data if s['dependent_lemma'] in verbs]
    
    sample = balanced_sample(data, 500)

    with open(sample_path, 'w') as f:
        json.dump(sample, f, indent=4)

    
if __name__ == '__main__':
    main()