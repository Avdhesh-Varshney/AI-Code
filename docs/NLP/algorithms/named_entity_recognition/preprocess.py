import spacy
from spacy.tokens import DocBin
import os

def create_training_data(file_path):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    
    with open(file_path, 'r') as file:
        lines = file.read().split('\n')
    
    words = []
    tags = []
    for line in lines:
        if line:
            parts = line.split()
            words.append(parts[0])
            tags.append(parts[1])
        else:
            if words:
                doc = nlp.make_doc(' '.join(words))
                ents = []
                start = 0
                for i, word in enumerate(words):
                    end = start + len(word)
                    tag = tags[i]
                    if tag != 'O':
                        label = tag.split('-')[1]
                        span = doc.char_span(start, end, label=label)
                        if span:
                            ents.append(span)
                    start = end + 1
                doc.ents = ents
                doc_bin.add(doc)
                words = []
                tags = []
    
    doc_bin.to_disk("data/training_data.spacy")

if __name__ == "__main__":
    create_training_data("data/ner_data.txt")