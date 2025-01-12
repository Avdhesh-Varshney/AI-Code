# Word 2 Vec 

## Introduction

Word2Vec is a technique to learn word embeddings using neural networks. The primary goal is to represent words in a continuous vector space where semantically similar words are mapped to nearby points. Word2Vec can be implemented using two main architectures:

1. **Continuous Bag of Words (CBOW)**: Predicts the target word based on the context words (surrounding words).
2. **Skip-gram**: Predicts the context words based on a given target word.

In this example, we focus on the Skip-gram approach, which is more commonly used in practice. The Skip-gram model tries to maximize the probability of context words given a target word.

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

## Code 

- word2vec.py file 

```py
import numpy as np

class Word2Vec:
    def __init__(self, window_size=2, embedding_dim=10, learning_rate=0.01):
        # Initialize parameters
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.vocabulary = {}
        self.word_index = {}
        self.index_word = {}
        self.W1 = None
        self.W2 = None

    def tokenize(self, documents):
        # Tokenize documents and build vocabulary
        vocabulary = set()
        for doc in documents:
            words = doc.split()
            vocabulary.update(words)
        
        self.vocabulary = list(vocabulary)
        self.word_index = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.index_word = {idx: word for idx, word in enumerate(self.vocabulary)}

    def generate_training_data(self, documents):
        # Generate training data for the Skip-gram model
        training_data = []
        for doc in documents:
            words = doc.split()
            for idx, word in enumerate(words):
                target_word = self.word_index[word]
                context = [self.word_index[words[i]] for i in range(max(0, idx - self.window_size), min(len(words), idx + self.window_size + 1)) if i != idx]
                for context_word in context:
                    training_data.append((target_word, context_word))
        return training_data

    def train(self, documents, epochs=1000):
        # Tokenize the documents and generate training data
        self.tokenize(documents)
        training_data = self.generate_training_data(documents)
        
        # Initialize weight matrices with random values
        vocab_size = len(self.vocabulary)
        self.W1 = np.random.uniform(-1, 1, (vocab_size, self.embedding_dim))
        self.W2 = np.random.uniform(-1, 1, (self.embedding_dim, vocab_size))
        
        for epoch in range(epochs):
            loss = 0
            for target_word, context_word in training_data:
                # Forward pass
                h = self.W1[target_word]  # Hidden layer representation of the target word
                u = np.dot(h, self.W2)    # Output layer scores
                y_pred = self.softmax(u) # Predicted probabilities
                
                # Calculate error
                e = np.zeros(vocab_size)
                e[context_word] = 1
                error = y_pred - e
                
                # Backpropagation
                self.W1[target_word] -= self.learning_rate * np.dot(self.W2, error)
                self.W2 -= self.learning_rate * np.outer(h, error)
                
                # Calculate loss (cross-entropy)
                loss -= np.log(y_pred[context_word])
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss}')

    def softmax(self, x):
        # Softmax function to convert scores into probabilities
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_word_vector(self, word):
        # Retrieve the vector representation of a word
        return self.W1[self.word_index[word]]

    def get_vocabulary(self):
        # Retrieve the vocabulary
        return self.vocabulary
# Example usage
if __name__ == "__main__":
    # Basic example usage    
    documents = [
        "the cat sat on the mat",
        "the dog ate my homework",
        "the cat ate the dog food",
        "I love programming in Python",
        "Machine learning is fun",
        "Python is a versatile language",
        "Learning new skills is always beneficial"
    ]

    # Initialize and train the Word2Vec model
    word2vec = Word2Vec()
    word2vec.train(documents)

    # Print the vocabulary
    print("Vocabulary:", word2vec.get_vocabulary())

    # Print the word vectors for each word in the vocabulary
    print("Word Vectors:")
    for word in word2vec.get_vocabulary():
        vector = word2vec.get_word_vector(word)
        print(f"Vector for '{word}':", vector)

    # More example documents with mixed content
    more_documents = [
        "the quick brown fox jumps over the lazy dog",
        "a journey of a thousand miles begins with a single step",
        "to be or not to be that is the question",
        "the rain in Spain stays mainly in the plain",
        "all human beings are born free and equal in dignity and rights"
    ]

    # Initialize and train the Word2Vec model on new documents
    word2vec_more = Word2Vec()
    word2vec_more.train(more_documents)

    # Print the word vectors for selected words
    print("\nWord Vectors for new documents:")
    for word in ['quick', 'journey', 'be', 'rain', 'human']:
        vector = word2vec_more.get_word_vector(word)
        print(f"Vector for '{word}':", vector)
```

## References

1. [Word2Vec - Google](https://code.google.com/archive/p/word2vec/)
2. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
3. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
