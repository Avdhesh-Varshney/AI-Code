# Text Preprocessing Techniques 

#### Welcome to this comprehensive guide on text preprocessing with NLTK (Natural Language Toolkit)! This notebook will walk you through various essential text preprocessing techniques, all explained in simple terms with easy-to-follow code examples. Whether you're just starting out in NLP (Natural Language Processing) or looking to brush up on your skills, you're in the right place! 🚀

#### NLTK provides a comprehensive suite of tools for processing and analyzing unstructured text data.

## 1. Tokenization
Tokenization is the process of splitting text into individual words or sentences.

#### Sentence Tokenization
```
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

text = "Hello World. This is NLTK. It is great for text processing."
sentences = sent_tokenize(text)
print(sentences)
```
#### Word Tokenization
```
from nltk.tokenize import word_tokenize

words = word_tokenize(text)
print(words)
```

## 2. Removing Stop Words

Stop words are common words that may not be useful for text analysis (e.g., "is", "the", "and").

```
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print(filtered_words)
```

## 3. Stemming

Stemming reduces words to their root form by chopping off the ends.
```
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print(stemmed_words)
```

## 4. Lemmatization

Lemmatization reduces words to their base form (lemma), taking into account the meaning of the word.
```
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print(lemmatized_words)
```

## 5. Part-of-Speech Tagging

Tagging words with their parts of speech (POS) helps understand the grammatical structure.

The complete POS tag list can be accessed from the Installation and set-up notebook.
```
nltk.download('averaged_perceptron_tagger')

pos_tags = nltk.pos_tag(lemmatized_words)
print(pos_tags)
```

## 6. Named Entity Recognition

Identify named entities such as names of people, organizations, locations, etc.

```
# Numpy is required to run this
%pip install numpy

nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.chunk import ne_chunk

named_entities = ne_chunk(pos_tags)
print(named_entities)
```

## 7. Word Frequency Distribution

Count the frequency of each word in the text.

```
from nltk.probability import FreqDist

freq_dist = FreqDist(lemmatized_words)
print(freq_dist.most_common(5))
```

## 8. Removing Punctuation

Remove punctuation from the text.

```
import string

no_punct = [word for word in lemmatized_words if word not in string.punctuation]
print(no_punct)
```

## 9. Lowercasing

Convert all words to lowercase.

```
lowercased = [word.lower() for word in no_punct]
print(lowercased)
```

## 10. Spelling Correction

Correct the spelling of words.

```
%pip install pyspellchecker

from nltk.corpus import wordnet
from spellchecker import SpellChecker

spell = SpellChecker()

def correct_spelling(word):
    if not wordnet.synsets(word):
        return spell.correction(word)
    return word

lemmatized_words = ['hello', 'world', '.', 'klown', 'taxt', 'procass', '.']
words_with_corrected_spelling = [correct_spelling(word) for word in lemmatized_words]
print(words_with_corrected_spelling)
```

## 11. Removing Numbers

Remove numerical values from the text.

```
lemmatized_words = ['hello', 'world', '88', 'text', 'process', '.']

no_numbers = [word for word in lemmatized_words if not word.isdigit()]
print(no_numbers)
```

## 12. Word Replacement

Replace specific words with other words (e.g., replacing slang with formal words).

```
lemmatized_words = ['hello', 'world', 'gr8', 'text', 'NLTK', '.']
replacements = {'NLTK': 'Natural Language Toolkit', 'gr8' : 'great'}

replaced_words = [replacements.get(word, word) for word in lemmatized_words]
print(replaced_words)
```

## 13. Synonym Replacement

Replace words with their synonyms.

```
from nltk.corpus import wordnet
lemmatized_words = ['hello', 'world', 'awesome', 'text', 'great', '.']

def get_synonym(word):
    synonyms = wordnet.synsets(word)
    if synonyms:
        return synonyms[0].lemmas()[0].name()
    return word

synonym_replaced = [get_synonym(word) for word in lemmatized_words]
print(synonym_replaced)
```

## 14. Extracting Bigrams and Trigrams

Extract bigrams (pairs of consecutive words) and trigrams (triplets of consecutive words).

```
from nltk import bigrams

bigrams_list = list(bigrams(lemmatized_words))
print(bigrams_list)

from nltk import trigrams

trigrams_list = list(trigrams(lemmatized_words))
print(trigrams_list)
```

## 15. Sentence Segmentation

Split text into sentences while considering abbreviations and other punctuation complexities.

```
import nltk.data

text = 'Hello World. This is NLTK. It is great for text preprocessing.'

# Load the sentence tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Tokenize the text into sentences
sentences = tokenizer.tokenize(text)

# Print the tokenized sentences
print(sentences)
```

## 16. Identifying Word Frequencies

Identify and display the frequency of words in a text.

```
from nltk.probability import FreqDist

lemmatized_words = ['hello', 'hello', 'awesome', 'text', 'great', '.', '.', '.']


word_freq = FreqDist(lemmatized_words)
for word, freq in word_freq.items():
    print(f"{word}: {freq}")
```

## 17. Removing HTML tags

Remove HTML tags from the text.
```
%pip install bs4

from bs4 import BeautifulSoup

html_text = "<p>Hello World. This is NLTK.</p>"
soup = BeautifulSoup(html_text, "html.parser")
cleaned_text = soup.get_text()
print(cleaned_text)
```

## 18. Detecting Language

Detect the language of the text.
```
%pip install langdetect

from langdetect import detect

language = detect(text)
print(language) #`en` (for English)
```

## 19. Tokenizing by Regular Expressions

Use Regular Expressions to tokenize text.
```
text = 'Hello World. This is NLTK. It is great for text preprocessing.'

from nltk.tokenize import regexp_tokenize

pattern = r'\w+'
regex_tokens = regexp_tokenize(text, pattern)
print(regex_tokens)
```

## 20. Remove Frequent Words

Removes frequent words (also known as “high-frequency words”) from a list of tokens using NLTK, you can use the nltk.FreqDist() function to calculate the frequency of each word and filter out the most common ones.
```
import nltk

# input text
text = "Natural language processing is a field of AI. I love AI."

# tokenize the text
tokens = nltk.word_tokenize(text)

# calculate the frequency of each word
fdist = nltk.FreqDist(tokens)

# remove the most common words (e.g., the top 10% of words by frequency)
filtered_tokens = [token for token in tokens if fdist[token] < fdist.N() * 0.1]

print("Tokens without frequent words:", filtered_tokens)
```

## 21. Remove extra whitespace

Tokenizes the input string into individual sentences and remove any leading or trailing whitespace from each sentence.
```
import nltk.data

# Text data
text = 'Hello World. This is NLTK. It is great for text preprocessing.'

# Load the sentence tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Tokenize the text into sentences
sentences = tokenizer.tokenize(text)

# Remove extra whitespace from each sentence
sentences = [sentence.strip() for sentence in sentences]

# Print the tokenized sentences
print(sentences)
```

# Conclusion 🎉

#### Text preprocessing is a crucial step in natural language processing (NLP) and can significantly impact the performance of your models and applications. With NLTK, we have a powerful toolset that simplifies and streamlines these tasks.
#### I hope this guide has provided you with a solid foundation for text preprocessing with NLTK. As you continue your journey in NLP, remember that preprocessing is just the beginning. There are many more exciting and advanced techniques to explore and apply in your projects.
