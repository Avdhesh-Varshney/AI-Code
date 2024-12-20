import spacy
from spacy.util import minibatch, compounding
import random

def train_ner_model():
    # Load blank English model
    nlp = spacy.blank("en")
    
    # Create the NER pipeline component and add it to the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    
    # Add labels to the NER component
    labels = ["LANG", "SKILL", "TECH", "TASK"]
    for label in labels:
        ner.add_label(label)
    
    # Load training data
    doc_bin = DocBin().from_disk("data/training_data.spacy")
    training_data = list(doc_bin.get_docs(nlp.vocab))
    
    # Train the model
    optimizer = nlp.begin_training()
    for itn in range(20):
        random.shuffle(training_data)
        losses = {}
        batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.5))
        for batch in batches:
            texts, annotations = zip(*[(doc.text, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in batch])
            nlp.update(texts, annotations, drop=0.5, losses=losses)
        print(f"Iteration {itn}, Losses: {losses}")
    
    # Save the trained model
    output_dir = "models/ner_model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_ner_model()