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