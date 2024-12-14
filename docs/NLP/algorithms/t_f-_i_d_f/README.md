# TF-IDF Implementation

## Introduction

The `TFIDF` class converts a collection of documents into their respective TF-IDF (Term Frequency-Inverse Document Frequency) representations. TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus).

## Table of Contents

1. [Attributes](#attributes)
2. [Methods](#methods)
   - [fit Method](#fit-method)
   - [transform Method](#transform-method)
   - [fit_transform Method](#fit_transform-method)
3. [Explanation of the Code](#explanation-of-the-code)
4. [References](#references)

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

## References

1. [TF-IDF - Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
2. [Understanding TF-IDF](https://towardsdatascience.com/understanding-tf-idf-a-traditional-approach-to-feature-extraction-in-nlp-a5bfbe04723f)
3. [Scikit-learn: TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

---

This document provides a clear and structured explanation of the TF-IDF algorithm, including its attributes, methods, and overall functionality.