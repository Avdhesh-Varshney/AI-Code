import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

#data collection
data = [
    'Fashion is an art form and expression.',
    'Style is a way to say who you are without having to speak.',
    'Fashion is what you buy, style is what you do with it.',
    'With fashion, you convey a message about yourself without uttering a single word'
]

#text processing

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zs]',' ',text)
    return text

preprocessed_data = [preprocess_text(doc) for doc in data]

for i, doc in enumerate(preprocessed_data, 1):
    print(f'Data-{i} {doc}')



# removing words like the, is, are, and as they usually do not carry much useful information for the analysis.
vectorizer = CountVectorizer(stop_words='english')
X=vectorizer.fit_transform(preprocessed_data)
Word=vectorizer.get_feature_names_out()

bow_df = pd.DataFrame(X.toarray(),columns=Word)
bow_df.index =[f'Data {i}' for i in range(1, len(data) + 1)]

tfidf_transformer = TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X)
tfidf_df=pd.DataFrame(X_tfidf.toarray(), columns=Word)
tfidf_df.index=[f'Data {i}' for i in range(1, len(data) + 1)]


print()
print("--------------------------------BoW Represention----------------------------")
print(bow_df)

print()
print("--------------------------------TF-IDF Value----------------------------")
print(tfidf_df)
