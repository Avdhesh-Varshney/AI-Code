# Word2Vec Skip-gram Implementation

## Introduction

Word2Vec is a technique to learn word embeddings using neural networks. The primary goal is to represent words in a continuous vector space where semantically similar words are mapped to nearby points. Word2Vec can be implemented using two main architectures:

1. **Continuous Bag of Words (CBOW)**: Predicts the target word based on the context words (surrounding words).
2. **Skip-gram**: Predicts the context words based on a given target word.

In this example, we focus on the Skip-gram approach, which is more commonly used in practice. The Skip-gram model tries to maximize the probability of context words given a target word.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Initialization](#initialization)
   - [Tokenization](#tokenization)
   - [Generate Training Data](#generate-training-data)
   - [Training](#training)
   - [Retrieve Word Vector](#retrieve-word-vector)
3. [Explanation of the Code](#explanation-of-the-code)
4. [References](#references)

## Installation

Ensure you have Python installed. You can install the necessary dependencies using pip:

```sh
pip install numpy
```

## Usage

### Initialization

Define the parameters for the Word2Vec model:

- `window_size`: Defines the size of the context window around the target word.
- `embedding_dim`: Dimension of the word vectors (embedding space).
- `learning_rate`: Rate at which weights are updated.

### Tokenization

The `tokenize` method creates a vocabulary from the documents and builds mappings between words and their indices.

### Generate Training Data

The `generate_training_data` method creates pairs of target words and context words based on the window size.

### Training

The `train` method initializes the weight matrices and updates them using gradient descent.

For each word-context pair, it computes the hidden layer representation, predicts context probabilities, calculates the error, and updates the weights.

### Retrieve Word Vector

The `get_word_vector` method retrieves the embedding of a specific word.

## Explanation of the Code

### Initialization

- **Parameters**:
  - `window_size`: Size of the context window around the target word.
  - `embedding_dim`: Dimension of the word vectors (embedding space).
  - `learning_rate`: Rate at which weights are updated.

### Tokenization

- The `tokenize` method creates a vocabulary from the documents.
- Builds mappings between words and their indices.

### Generate Training Data

- The `generate_training_data` method creates pairs of target words and context words based on the window size.

### Training

- The `train` method initializes the weight matrices.
- Updates the weights using gradient descent.
- For each word-context pair:
  - Computes the hidden layer representation.
  - Predicts context probabilities.
  - Calculates the error.
  - Updates the weights.

### Softmax Function

- The `softmax` function converts the output layer scores into probabilities.
- Used to compute the error and update the weights.

### Retrieve Word Vector

- The `get_word_vector` method retrieves the embedding of a specific word.

## References

1. [Word2Vec - Google](https://code.google.com/archive/p/word2vec/)
2. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
3. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)

---

This README file provides a comprehensive overview of the Word2Vec Skip-gram implementation, including installation instructions, usage details, and an explanation of the code.