import spacy

def recognize_entities(text):
    # Load the trained model
    nlp = spacy.load("models/ner_model")
    
    # Process the text
    doc = nlp(text)
    
    # Print the entities
    for ent in doc.ents:
        print(f"Text: {ent.text}, Label: {ent.label_}")

# Example usage
if __name__ == "__main__":
    text = "I love programming in Python. Machine learning is fascinating. Spacy is a useful library."
    recognize_entities(text)
