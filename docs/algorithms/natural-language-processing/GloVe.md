# GloVe 

## Introduction

The `GloVe` class implements the Global Vectors for Word Representation algorithm, developed by Stanford researchers. GloVe generates dense vector representations of words, capturing semantic relationships between them. Unlike traditional one-hot encoding, GloVe produces low-dimensional, continuous vectors that convey meaningful information about words and their contexts.

## Key Concepts

- **Co-occurrence Matrix**: GloVe starts by creating a co-occurrence matrix from a large corpus of text. This matrix counts how often words appear together within a given context window, capturing the frequency of word pairs.

- **Weighted Least Squares**: The main idea behind GloVe is to factorize this co-occurrence matrix to find word vectors that capture the relationships between words. It aims to represent words that frequently appear together in similar contexts with similar vectors.

- **Weighting Function**: To ensure that the optimization process doesn't get overwhelmed by very frequent co-occurrences, GloVe uses a weighting function. This function reduces the influence of extremely common word pairs.

- **Training Objective**: The goal is to adjust the word vectors so that their dot products align with the observed co-occurrence counts. This helps in capturing the similarity between words based on their contexts.

## GloVe Training Objective

GloVe’s training involves adjusting word vectors so that their interactions match the observed co-occurrence data. It focuses on ensuring that words appearing together often have similar vector representations.

## Advantages

- **Efficient Training**: By using a global co-occurrence matrix, GloVe captures semantic relationships effectively, including long-range dependencies.
- **Meaningful Vectors**: The resulting vectors can represent complex relationships between words, such as analogies (e.g., "king" - "man" + "woman" ≈ "queen").
- **Flexibility**: GloVe vectors are versatile and can be used in various NLP tasks, including sentiment analysis and machine translation.

## Applications

- **Natural Language Processing (NLP)**: GloVe vectors are used as features in NLP tasks like sentiment analysis, named entity recognition, and question answering.
- **Information Retrieval**: Enhance search engines by providing better semantic matching between queries and documents.
- **Machine Translation**: Improve translation models by capturing semantic similarities between words in different languages.

## Implementation

### Preprocessing

1. **Clean and Tokenize**: Prepare the text data by cleaning and tokenizing it into words.

### Building Vocabulary

2. **Create Vocabulary**: Construct a vocabulary and map words to unique indices.

### Co-occurrence Matrix

3. **Build Co-occurrence Matrix**: Create a matrix that captures how often each word pair appears together within a specified context.

### GloVe Model

4. **Initialization**: Set up the model parameters and hyperparameters.
5. **Weighting Function**: Define how to balance the importance of different co-occurrence counts.
6. **Training**: Use optimization techniques to adjust the word vectors based on the co-occurrence data.
7. **Get Word Vector**: Extract the vector representation for each word from the trained model.

For more advanced implementations, consider using libraries like TensorFlow or PyTorch, which offer enhanced functionalities and optimizations.

## Code 

- main.py file 

```py
import numpy as np
from preprocess import preprocess
from vocab_and_matrix import build_vocab, build_cooccurrence_matrix
from glove_model import GloVe

# Example text corpus
corpus = ["I love NLP", "NLP is a fascinating field", "Natural language processing with GloVe"]

# Preprocess the corpus
tokens = [token for sentence in corpus for token in preprocess(sentence)]

# Build vocabulary and co-occurrence matrix
word_to_index = build_vocab(tokens)
cooccurrence_matrix = build_cooccurrence_matrix(tokens, word_to_index)

# Initialize and train the GloVe model
glove = GloVe(vocab_size=len(word_to_index), embedding_dim=50)
glove.train(cooccurrence_matrix, epochs=100)

# Get the word vector for 'nlp'
word_vector = glove.get_word_vector('nlp', word_to_index)
print(word_vector)
```

- glove_model.py file 

```py
import numpy as np

class GloVe:
    def __init__(self, vocab_size, embedding_dim=50, x_max=100, alpha=0.75):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha
        self.W = np.random.rand(vocab_size, embedding_dim)
        self.W_tilde = np.random.rand(vocab_size, embedding_dim)
        self.b = np.random.rand(vocab_size)
        self.b_tilde = np.random.rand(vocab_size)
        self.gradsq_W = np.ones((vocab_size, embedding_dim))
        self.gradsq_W_tilde = np.ones((vocab_size, embedding_dim))
        self.gradsq_b = np.ones(vocab_size)
        self.gradsq_b_tilde = np.ones(vocab_size)
    
    def weighting_function(self, x):
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1.0
    
    def train(self, cooccurrence_matrix, epochs=100, learning_rate=0.05):
        for epoch in range(epochs):
            total_cost = 0
            for i in range(self.vocab_size):
                for j in range(self.vocab_size):
                    if cooccurrence_matrix[i, j] == 0:
                        continue
                    X_ij = cooccurrence_matrix[i, j]
                    weight = self.weighting_function(X_ij)
                    cost = weight * (np.dot(self.W[i], self.W_tilde[j]) + self.b[i] + self.b_tilde[j] - np.log(X_ij)) ** 2
                    total_cost += cost
                    
                    grad_common = weight * (np.dot(self.W[i], self.W_tilde[j]) + self.b[i] + self.b_tilde[j] - np.log(X_ij))
                    grad_W = grad_common * self.W_tilde[j]
                    grad_W_tilde = grad_common * self.W[i]
                    grad_b = grad_common
                    grad_b_tilde = grad_common
                    
                    self.W[i] -= learning_rate * grad_W / np.sqrt(self.gradsq_W[i])
                    self.W_tilde[j] -= learning_rate * grad_W_tilde / np.sqrt(self.gradsq_W_tilde[j])
                    self.b[i] -= learning_rate * grad_b / np.sqrt(self.gradsq_b[i])
                    self.b_tilde[j] -= learning_rate * grad_b_tilde / np.sqrt(self.gradsq_b_tilde[j])
                    
                    self.gradsq_W[i] += grad_W ** 2
                    self.gradsq_W_tilde[j] += grad_W_tilde ** 2
                    self.gradsq_b[i] += grad_b ** 2
                    self.gradsq_b_tilde[j] += grad_b_tilde ** 2
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Cost: {total_cost}')
    
    def get_word_vector(self, word, word_to_index):
        if word in word_to_index:
            word_index = word_to_index[word]
            return self.W[word_index]
        return None
```

- preprocess.py file 

```py
import string

def preprocess(text):
    """
    Preprocess the text by removing punctuation, converting to lowercase, and splitting into words.
    
    Args:
    text (str): Input text string.
    
    Returns:
    list: List of words (tokens).
    """
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = text.split()
    return tokens
```

- vocab_and_matrix.py file 

```py
import numpy as np
from collections import Counter

def build_vocab(tokens):
    """
    Build vocabulary from tokens and create a word-to-index mapping.
    
    Args:
    tokens (list): List of words (tokens).
    
    Returns:
    dict: Word-to-index mapping.
    """
    vocab = Counter(tokens)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    return word_to_index

def build_cooccurrence_matrix(tokens, word_to_index, window_size=2):
    """
    Build the co-occurrence matrix from tokens using a specified window size.
    
    Args:
    tokens (list): List of words (tokens).
    word_to_index (dict): Word-to-index mapping.
    window_size (int): Context window size.
    
    Returns:
    np.ndarray: Co-occurrence matrix.
    """
    vocab_size = len(word_to_index)
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
    
    for i, word in enumerate(tokens):
        word_index = word_to_index[word]
        context_start = max(0, i - window_size)
        context_end = min(len(tokens), i + window_size + 1)
        
        for j in range(context_start, context_end):
            if i != j:
                context_word = tokens[j]
                context_word_index = word_to_index[context_word]
                cooccurrence_matrix[word_index, context_word_index] += 1
    
    return cooccurrence_matrix
```


## References

1. [GloVe - Stanford NLP](https://nlp.stanford.edu/projects/glove/)
2. [Understanding GloVe](https://nlp.stanford.edu/pubs/glove.pdf)
3. [GloVe: Global Vectors for Word Representation - Wikipedia](https://en.wikipedia.org/wiki/GloVe)

