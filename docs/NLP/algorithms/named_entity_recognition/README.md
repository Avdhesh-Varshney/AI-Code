# Named Entity Recognition (NER) Project

This project demonstrates a basic Named Entity Recognition (NER) algorithm using Python and the `spacy` library. The goal is to identify named entities in text and classify them into predefined categories.

## Directory Structure

```
ner_project/
├── data/
│   └── ner_data.txt
├── models/
│   └── ner_model
├── preprocess.py
├── train.py
└── recognize.py
└── README.md
```

- **data/ner_data.txt**: Contains the dataset used for training the NER model.
- **models/ner_model**: Stores the trained NER model.
- **preprocess.py**: Contains the code for preprocessing the text data.
- **train.py**: Script for training the NER model.
- **recognize.py**: Script for recognizing named entities in new text using the trained model.
- **README.md**: Project documentation.

## Dataset

The dataset (`ner_data.txt`) contains sentences and their corresponding entity labels in the IOB format. Each line contains a word and its label, separated by a space. Sentences are separated by blank lines.

## Preprocessing

The `preprocess.py` file contains functions to preprocess the text data. It reads the dataset and converts it into a format suitable for training with `spacy`.

## Training the Model

The `train.py` script is used to train the NER model. It performs the following steps:

1. Load a blank English model.
2. Create the NER pipeline component and add it to the pipeline.
3. Add labels to the NER component.
4. Load the training data.
5. Train the model using the training data.
6. Save the trained model to `models/ner_model`.

### Running the Training Script

To train the model, run:
```bash
python train.py
```

## Recognizing Named Entities

The `recognize.py` script is used to recognize named entities in new text using the trained model. It performs the following steps:

1. Load the trained model.
2. Process the input text.
3. Print the recognized entities and their labels.

### Running the Recognition Script

To recognize named entities in new text, run:
```bash
python recognize.py
```

## Dependencies

The project requires the following Python libraries:

- spacy

You can install the dependencies using:
```bash
pip install spacy
```

## Example Usage

```python
# Example usage of the recognize.py script
if __name__ == "__main__":
    text = "I love programming in Python. Machine learning is fascinating. Spacy is a useful library."
    recognize_entities(text)
```

This project provides a basic implementation of Named Entity Recognition using the `spacy` library. You can expand it by using more advanced models or preprocessing techniques based on your requirements.