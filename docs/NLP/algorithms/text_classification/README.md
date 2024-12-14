# Text Classification Project

This project demonstrates a basic text classification algorithm using Python and common NLP libraries. The goal is to classify text into different categories, specifically positive and negative sentiments.

## Directory Structure

```
text_classification/
├── data/
│   └── text_data.csv
├── models/
│   └── text_classifier.pkl
├── preprocess.py
├── train.py
└── classify.py
```

- **data/text_data.csv**: Contains the dataset used for training and testing the model.
- **models/text_classifier.pkl**: Stores the trained text classification model.
- **preprocess.py**: Contains the code for preprocessing the text data.
- **train.py**: Script for training the text classification model.
- **classify.py**: Script for classifying new text using the trained model.

## Dataset

The dataset (`text_data.csv`) contains 200 entries with two columns: `text` and `label`. The `text` column contains the text to be classified, and the `label` column contains the corresponding category (positive or negative).

## Preprocessing

The `preprocess.py` file contains functions to preprocess the text data. It includes removing non-alphabetic characters, converting text to lowercase, removing stopwords, and lemmatizing the words.

## Training the Model

The `train.py` script is used to train the text classification model. It performs the following steps:

1. Load the dataset.
2. Preprocess the text data.
3. Split the dataset into training and testing sets.
4. Create a pipeline with `TfidfVectorizer` and `MultinomialNB`.
5. Train the model.
6. Evaluate the model and print the accuracy and classification report.
7. Save the trained model to `models/text_classifier.pkl`.

### Running the Training Script

To train the model, run:
```bash
python train.py
```

## Classifying New Text

The `classify.py` script is used to classify new text using the trained model. It performs the following steps:

1. Load the trained model.
2. Preprocess the input text.
3. Predict the class of the input text.

### Running the Classification Script

To classify new text, run:
```bash
python classify.py
```

## Dependencies

The project requires the following Python libraries:

- pandas
- scikit-learn
- nltk
- joblib

You can install the dependencies using:
```bash
pip install pandas scikit-learn nltk joblib
```

## Example Usage

```python
# Example usage of the classify.py script
if __name__ == "__main__":
    new_text = "I enjoy working on machine learning projects."
    print(f'Text: "{new_text}"')
    print(f'Predicted Class: {classify_text(new_text)}')
```

This project provides a simple text classification model using Naive Bayes with TF-IDF vectorization. You can expand it by using more advanced models or preprocessing techniques based on your requirements.