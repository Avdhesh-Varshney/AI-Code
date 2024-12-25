# Tf-Idf 

## Introduction

The `TFIDF` class converts a collection of documents into their respective TF-IDF (Term Frequency-Inverse Document Frequency) representations. TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus).

## Attributes

The `TFIDF` class is initialized with two main attributes:

- **`self.vocabulary`**: A dictionary that maps words to their indices in the TF-IDF matrix.
- **`self.idf_values`**: A dictionary that stores the IDF (Inverse Document Frequency) values for each word.

## Methods

### fit Method

#### Input

- **`documents`** (list of str): List of documents where each document is a string.

#### Purpose

Calculate the IDF values for all unique words in the corpus.

#### Steps

1. **Count Document Occurrences**: Determine how many documents contain each word.
2. **Compute IDF**: Calculate the importance of each word across all documents. Higher values indicate the word is more unique to fewer documents.
3. **Build Vocabulary**: Create a mapping of words to unique indexes.

### transform Method

#### Input

- **`documents`** (list of str): A list where each entry is a document in the form of a string.

#### Purpose

Convert each document into a numerical representation that shows the importance of each word.

#### Steps

1. **Compute Term Frequency (TF)**: Determine how often each word appears in a document relative to the total number of words in that document.
2. **Compute TF-IDF**: Multiply the term frequency of each word by its IDF to get a measure of its relevance in each document.
3. **Store Values**: Save these numerical values in a matrix where each row represents a document.

### fit_transform Method

#### Purpose

Perform both fitting (computing IDF values) and transforming (converting documents to TF-IDF representation) in one step.

## Explanation of the Code

The `TFIDF` class includes methods for fitting the model to the data, transforming new data into the TF-IDF representation, and combining these steps. Here's a breakdown of the primary methods:

1. **`fit` Method**: Calculates IDF values for all unique words in the corpus. It counts the number of documents containing each word and computes the IDF. The vocabulary is built with a word-to-index mapping.

2. **`transform` Method**: Converts each document into a TF-IDF representation. It computes Term Frequency (TF) for each word in the document, calculates TF-IDF by multiplying TF with IDF, and stores these values in a matrix where each row corresponds to a document.

3. **`fit_transform` Method**: Combines the fitting and transforming steps into a single method for efficient processing of documents.

## Code 

- main.py file 

```py
import math
from collections import Counter

class TFIDF:
    def __init__(self):
        self.vocabulary = {}  # Vocabulary to store word indices
        self.idf_values = {}  # IDF values for words

    def fit(self, documents):
        """
        Compute IDF values based on the provided documents.
        
        Args:
            documents (list of str): List of documents where each document is a string.
        """
        doc_count = len(documents)
        term_doc_count = Counter()  # To count the number of documents containing each word

        # Count occurrences of words in documents
        for doc in documents:
            words = set(doc.split())  # Unique words in the current document
            for word in words:
                term_doc_count[word] += 1

        # Compute IDF values
        self.idf_values = {
            word: math.log(doc_count / (count + 1))  # +1 to avoid division by zero
            for word, count in term_doc_count.items()
        }

        # Build vocabulary
        self.vocabulary = {word: idx for idx, word in enumerate(self.idf_values.keys())}

    def transform(self, documents):
        """
        Transform documents into TF-IDF representation.

        Args:
            documents (list of str): List of documents where each document is a string.
        
        Returns:
            list of list of float: TF-IDF matrix where each row corresponds to a document.
        """
        rows = []
        for doc in documents:
            words = doc.split()
            word_count = Counter(words)
            doc_length = len(words)
            row = [0] * len(self.vocabulary)

            for word, count in word_count.items():
                if word in self.vocabulary:
                    tf = count / doc_length
                    idf = self.idf_values[word]
                    index = self.vocabulary[word]
                    row[index] = tf * idf
            rows.append(row)
        return rows

    def fit_transform(self, documents):
        """
        Compute IDF values and transform documents into TF-IDF representation.

        Args:
            documents (list of str): List of documents where each document is a string.

        Returns:
            list of list of float: TF-IDF matrix where each row corresponds to a document.
        """
        self.fit(documents)
        return self.transform(documents)
# Example usage
if __name__ == "__main__":
    documents = [
        "the cat sat on the mat",
        "the dog ate my homework",
        "the cat ate the dog food",
        "I love programming in Python",
        "Machine learning is fun",
        "Python is a versatile language",
        "Learning new skills is always beneficial"
    ]

    # Initialize the TF-IDF model
    tfidf = TFIDF()
    
    # Fit the model and transform the documents
    tfidf_matrix = tfidf.fit_transform(documents)
    
    # Print the vocabulary
    print("Vocabulary:", tfidf.vocabulary)
    
    # Print the TF-IDF representation
    print("TF-IDF Representation:")
    for i, vector in enumerate(tfidf_matrix):
        print(f"Document {i + 1}: {vector}")

    # More example documents with mixed content
    more_documents = [
        "the quick brown fox jumps over the lazy dog",
        "a journey of a thousand miles begins with a single step",
        "to be or not to be that is the question",
        "the rain in Spain stays mainly in the plain",
        "all human beings are born free and equal in dignity and rights"
    ]

    # Fit the model and transform the new set of documents
    tfidf_more = TFIDF()
    tfidf_matrix_more = tfidf_more.fit_transform(more_documents)
    
    # Print the vocabulary for the new documents
    print("\nVocabulary for new documents:", tfidf_more.vocabulary)
    
    # Print the TF-IDF representation for the new documents
    print("TF-IDF Representation for new documents:")
    for i, vector in enumerate(tfidf_matrix_more):
        print(f"Document {i + 1}: {vector}")
```

## References

1. [TF-IDF - Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
2. [Understanding TF-IDF](https://towardsdatascience.com/understanding-tf-idf-a-traditional-approach-to-feature-extraction-in-nlp-a5bfbe04723f)
3. [Scikit-learn: TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
