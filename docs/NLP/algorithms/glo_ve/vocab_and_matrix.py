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