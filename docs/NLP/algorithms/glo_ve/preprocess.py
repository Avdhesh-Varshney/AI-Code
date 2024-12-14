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