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