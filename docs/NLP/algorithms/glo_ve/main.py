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