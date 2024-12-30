import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
import sys
import utils

if __name__ == "__main__":
    filepath = sys.argv[1]
    output_path = sys.argv[2]

    with open(filepath, "r") as f:
        sentences = f.readlines()
    
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    parsed = utils.stanza_parsing_batched(sentences, nlp, 128)

    # serialize this 
    new_doc = Document([])
    new_doc.sentences = parsed
    CoNLL.write_doc2conll(new_doc, output_path)