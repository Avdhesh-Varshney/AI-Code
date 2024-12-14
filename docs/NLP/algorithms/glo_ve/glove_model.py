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