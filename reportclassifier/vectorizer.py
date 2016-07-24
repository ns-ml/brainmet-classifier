from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem.porter import PorterStemmer
import re


porter = PorterStemmer()


def tokenizer_porter(text):
    text = re.sub('[\W]+', ' ', text.lower())
    tokenized = [porter.stem(word) for word in text]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer_porter)
