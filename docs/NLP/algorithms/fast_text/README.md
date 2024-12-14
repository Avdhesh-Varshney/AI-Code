# FastText Implementation

## Introduction

The `FastText` class implements a word representation and classification tool developed by Facebook's AI Research (FAIR) lab. FastText extends the Word2Vec model by representing each word as a bag of character n-grams. This approach helps capture subword information and improves the handling of rare words.

## Table of Contents

1. [Explanation](#explanation)
2. [Advantages](#advantages)
3. [Applications](#applications)
4. [Implementation](#implementation)
5. [References](#references)

## Explanation

### Initialization

- **`vocab_size`**: Size of the vocabulary.
- **`embedding_dim`**: Dimension of the word embeddings.
- **`n_gram_size`**: Size of character n-grams.
- **`learning_rate`**: Learning rate for updating embeddings.
- **`epochs`**: Number of training epochs.

### Building Vocabulary

- **`build_vocab()`**: Constructs the vocabulary from the input sentences and creates a reverse mapping of words to indices.

### Generating N-grams

- **`get_ngrams()`**: Generates character n-grams for a given word. It pads the word with `<` and `>` symbols to handle edge cases effectively.

### Training

- **`train()`**: Updates word and context embeddings using a simple Stochastic Gradient Descent (SGD) approach. The loss is computed as the squared error between the predicted and actual values.

### Prediction

- **`predict()`**: Calculates the dot product between the target word and context embeddings to predict word vectors.

### Getting Word Vectors

- **`get_word_vector()`**: Retrieves the embedding for a specific word from the trained model.

### Normalization

- **`get_embedding_matrix()`**: Returns the normalized embedding matrix for better performance and stability.

## Advantages

- **Subword Information**: FastText captures morphological details by using character n-grams, improving handling of rare and out-of-vocabulary words.
- **Improved Representations**: The use of subwords allows for better word representations, especially for languages with rich morphology.
- **Efficiency**: FastText is designed to handle large-scale datasets efficiently, with optimizations for both training and inference.

## Applications

- **Natural Language Processing (NLP)**: FastText embeddings are used in tasks like text classification, sentiment analysis, and named entity recognition.
- **Information Retrieval**: Enhances search engines by providing more nuanced semantic matching between queries and documents.
- **Machine Translation**: Improves translation models by leveraging subword information for better handling of rare words and phrases.

## Implementation

### Preprocessing

1. **Initialization**: Set up parameters such as vocabulary size, embedding dimension, n-gram size, learning rate, and number of epochs.

### Building Vocabulary

2. **Build Vocabulary**: Construct the vocabulary from the input sentences and create a mapping for words.

### Generating N-grams

3. **Generate N-grams**: Create character n-grams for each word in the vocabulary, handling edge cases with padding.

### Training

4. **Train the Model**: Use SGD to update word and context embeddings based on the training data.

### Prediction

5. **Predict Word Vectors**: Calculate the dot product between target and context embeddings to predict word vectors.

### Getting Word Vectors

6. **Retrieve Word Vectors**: Extract the embedding for a specific word from the trained model.

### Normalization

7. **Normalize Embeddings**: Return the normalized embedding matrix for stability and improved performance.

For more advanced implementations, consider using optimized libraries like the FastText library by Facebook or other frameworks that offer additional features and efficiency improvements.

## References

1. [FastText - Facebook AI Research](https://fasttext.cc/)
2. [Understanding FastText](https://arxiv.org/pdf/1702.05531)
3. [FastText on GitHub](https://github.com/facebookresearch/fastText)

---

This README provides a clear overview of FastText, including its key concepts, advantages, applications, and implementation steps.