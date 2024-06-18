from lemminflect import getAllInflections, getAllInflectionsOOV

def get_all_inflections(word_list : list[str]):
    inflections = set()
    for word in word_list:
        word_inflections = getAllInflections(word, upos="VERB")
        if word_inflections == {}:
            word_inflections = getAllInflectionsOOV(word, upos="VERB")
        word_inflections = set([inflect[0] for inflect in word_inflections.values()])
        inflections = inflections.union(word_inflections)
    
    return inflections
    