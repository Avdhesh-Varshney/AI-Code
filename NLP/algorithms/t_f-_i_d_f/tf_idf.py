import math
from collections import Counter

class TFIDF:
    def __init__(self):
        self.vocabulary = {}  # Vocabulary to store word indices
        self.idf_values = {}  # IDF values for words

    def fit(self, documents):
        """
        Compute IDF values based on the provided documents.
        
        Args:
            documents (list of str): List of documents where each document is a string.
        """
        doc_count = len(documents)
        term_doc_count = Counter()  # To count the number of documents containing each word

        # Count occurrences of words in documents
        for doc in documents:
            words = set(doc.split())  # Unique words in the current document
            for word in words:
                term_doc_count[word] += 1

        # Compute IDF values
        self.idf_values = {
            word: math.log(doc_count / (count + 1))  # +1 to avoid division by zero
            for word, count in term_doc_count.items()
        }

        # Build vocabulary
        self.vocabulary = {word: idx for idx, word in enumerate(self.idf_values.keys())}

    def transform(self, documents):
        """
        Transform documents into TF-IDF representation.

        Args:
            documents (list of str): List of documents where each document is a string.
        
        Returns:
            list of list of float: TF-IDF matrix where each row corresponds to a document.
        """
        rows = []
        for doc in documents:
            words = doc.split()
            word_count = Counter(words)
            doc_length = len(words)
            row = [0] * len(self.vocabulary)

            for word, count in word_count.items():
                if word in self.vocabulary:
                    tf = count / doc_length
                    idf = self.idf_values[word]
                    index = self.vocabulary[word]
                    row[index] = tf * idf
            rows.append(row)
        return rows

    def fit_transform(self, documents):
        """
        Compute IDF values and transform documents into TF-IDF representation.

        Args:
            documents (list of str): List of documents where each document is a string.

        Returns:
            list of list of float: TF-IDF matrix where each row corresponds to a document.
        """
        self.fit(documents)
        return self.transform(documents)
# Example usage
if __name__ == "__main__":
    documents = [
        "the cat sat on the mat",
        "the dog ate my homework",
        "the cat ate the dog food",
        "I love programming in Python",
        "Machine learning is fun",
        "Python is a versatile language",
        "Learning new skills is always beneficial"
    ]

    # Initialize the TF-IDF model
    tfidf = TFIDF()
    
    # Fit the model and transform the documents
    tfidf_matrix = tfidf.fit_transform(documents)
    
    # Print the vocabulary
    print("Vocabulary:", tfidf.vocabulary)
    
    # Print the TF-IDF representation
    print("TF-IDF Representation:")
    for i, vector in enumerate(tfidf_matrix):
        print(f"Document {i + 1}: {vector}")

    # More example documents with mixed content
    more_documents = [
        "the quick brown fox jumps over the lazy dog",
        "a journey of a thousand miles begins with a single step",
        "to be or not to be that is the question",
        "the rain in Spain stays mainly in the plain",
        "all human beings are born free and equal in dignity and rights"
    ]

    # Fit the model and transform the new set of documents
    tfidf_more = TFIDF()
    tfidf_matrix_more = tfidf_more.fit_transform(more_documents)
    
    # Print the vocabulary for the new documents
    print("\nVocabulary for new documents:", tfidf_more.vocabulary)
    
    # Print the TF-IDF representation for the new documents
    print("TF-IDF Representation for new documents:")
    for i, vector in enumerate(tfidf_matrix_more):
        print(f"Document {i + 1}: {vector}")