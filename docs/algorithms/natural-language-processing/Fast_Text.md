# Fast Text 

## Introduction

The `FastText` class implements a word representation and classification tool developed by Facebook's AI Research (FAIR) lab. FastText extends the Word2Vec model by representing each word as a bag of character n-grams. This approach helps capture subword information and improves the handling of rare words.

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

## Code 

- main.py 

```py
from fasttext import FastText

# Example sentences
sentences = [
    "fast text is a library for efficient text classification",
    "word embeddings are useful for NLP tasks",
    "fasttext models can handle out-of-vocabulary words"
]

# Initialize and train FastText model
fasttext_model = FastText(vocab_size=100, embedding_dim=50)
fasttext_model.build_vocab(sentences)
fasttext_model.train(sentences)

# Get the vector for a word
vector = fasttext_model.get_word_vector("fast")
print(f"Vector for 'fast': {vector}")
```

- fast_test.py

```py
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize

class FastText:
    def __init__(self, vocab_size, embedding_dim, n_gram_size=3, learning_rate=0.01, epochs=10):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_gram_size = n_gram_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.word_embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
        self.context_embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
        self.vocab = {}
        self.rev_vocab = {}

    def build_vocab(self, sentences):
        """
        Build vocabulary from sentences.
        
        Args:
        sentences (list): List of sentences (strings).
        """
        word_count = defaultdict(int)
        for sentence in sentences:
            words = sentence.split()
            for word in words:
                word_count[word] += 1
        self.vocab = {word: idx for idx, (word, _) in enumerate(word_count.items())}
        self.rev_vocab = {idx: word for word, idx in self.vocab.items()}
    
    def get_ngrams(self, word):
        """
        Get n-grams for a given word.
        
        Args:
        word (str): Input word.
        
        Returns:
        set: Set of n-grams.
        """
        ngrams = set()
        word = '<' * (self.n_gram_size - 1) + word + '>' * (self.n_gram_size - 1)
        for i in range(len(word) - self.n_gram_size + 1):
            ngrams.add(word[i:i + self.n_gram_size])
        return ngrams

    def train(self, sentences):
        """
        Train the FastText model using the given sentences.
        
        Args:
        sentences (list): List of sentences (strings).
        """
        for epoch in range(self.epochs):
            loss = 0
            for sentence in sentences:
                words = sentence.split()
                for i, word in enumerate(words):
                    if word not in self.vocab:
                        continue
                    word_idx = self.vocab[word]
                    target_ngrams = self.get_ngrams(word)
                    for j in range(max(0, i - 1), min(len(words), i + 2)):
                        if i != j and words[j] in self.vocab:
                            context_idx = self.vocab[words[j]]
                            prediction = self.predict(word_idx, context_idx)
                            error = prediction - 1 if j == i + 1 else prediction
                            loss += error**2
                            self.word_embeddings[word_idx] -= self.learning_rate * error * self.context_embeddings[context_idx]
                            self.context_embeddings[context_idx] -= self.learning_rate * error * self.word_embeddings[word_idx]
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss}')

    def predict(self, word_idx, context_idx):
        """
        Predict the dot product of the word and context embeddings.
        
        Args:
        word_idx (int): Index of the word.
        context_idx (int): Index of the context word.
        
        Returns:
        float: Dot product.
        """
        return np.dot(self.word_embeddings[word_idx], self.context_embeddings[context_idx])

    def get_word_vector(self, word):
        """
        Get the word vector for the specified word.
        
        Args:
        word (str): Input word.
        
        Returns:
        np.ndarray: Word vector.
        """
        if word in self.vocab:
            return self.word_embeddings[self.vocab[word]]
        else:
            raise ValueError(f"Word '{word}' not found in vocabulary")

    def get_embedding_matrix(self):
        """
        Get the normalized embedding matrix.
        
        Returns:
        np.ndarray: Normalized word embeddings.
        """
        return normalize(self.word_embeddings, axis=1)
```

## References

1. [FastText - Facebook AI Research](https://fasttext.cc/)
2. [Understanding FastText](https://arxiv.org/pdf/1702.05531)
3. [FastText on GitHub](https://github.com/facebookresearch/fastText)

