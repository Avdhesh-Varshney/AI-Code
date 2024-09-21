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