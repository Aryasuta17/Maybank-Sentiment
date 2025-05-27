# preprocessing.py

import os
import re
import emoji
import requests
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK resources (once)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer & stopwords
STOP_WORDS = set(stopwords.words('indonesian'))
STEMMER    = StemmerFactory().create_stemmer()

# Emoticon map → words
EMOTIKON_MAP = {
    ':/':   ' bingung ',
    '8)':   ' keren ',
    ':(':   ' sedih ',
    ':)':   ' senang ',
    '8/':   ' bingung ringan ',
    '8:':   ' kagum ',
    '=P':   ' julurkan bahasa ',
    "8')":  ' terharu ',
}
ESCAPED_MAP = {re.escape(k): v for k, v in EMOTIKON_MAP.items()}

def demojize_to_words(text: str) -> str:
    """
    Convert emojis in text to their word descriptions,
    remove skin-tone modifiers.
    """
    w = emoji.demojize(text, language='en')
    w = w.replace(':', ' ').replace('_', ' ')
    return re.sub(r"\s*(light|medium|dark)\s+skin\s+tone", "", w)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer that:
      - lowercases
      - removes URLs/hashtags/newlines
      - replaces emoticons
      - demojizes
      - strips non-alphanumerics
      - tokenizes, removes stopwords, stems
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = []
        for doc in X:
            txt = str(doc).lower()
            # remove URLs, hashtags, newlines
            txt = re.sub(r'http\S+|www\.\S+|#\w+|\n', ' ', txt)
            # replace emoticons
            txt = re.sub(
                '|'.join(ESCAPED_MAP.keys()),
                lambda m: ESCAPED_MAP[re.escape(m.group(0))],
                txt
            )
            # demojize
            txt = demojize_to_words(txt)
            # remove non-alphanumeric
            txt = re.sub(r'[^a-z0-9\s]', ' ', txt)
            txt = re.sub(r'\s+', ' ', txt).strip()
            # tokenize, stopword removal, stemming
            tokens = nltk.word_tokenize(txt)
            tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS]
            processed.append(" ".join(tokens))
        return processed

class LexiconCount(TransformerMixin):
    """
    Transformer that counts negative/positive words per document
    using local files or fallback to GitHub lexica.
    """
    def __init__(self,
                 neg_path: str = "negative.txt",
                 pos_path: str = "positive.txt"):
        self.neg_path = neg_path
        self.pos_path = pos_path
        self.neg = None
        self.pos = None

    def _load_lexicons(self):
        # Try local
        if os.path.isfile(self.neg_path) and os.path.isfile(self.pos_path):
            with open(self.neg_path, encoding='utf-8') as f:
                self.neg = set(line.strip() for line in f)
            with open(self.pos_path, encoding='utf-8') as f:
                self.pos = set(line.strip() for line in f)
        else:
            # Fallback fetch from GitHub
            try:
                neg_url = "https://raw.githubusercontent.com/masdevid/ID-OpinionWords/master/negative.txt"
                pos_url = "https://raw.githubusercontent.com/masdevid/ID-OpinionWords/master/positive.txt"
                self.neg = set(requests.get(neg_url, timeout=5).text.splitlines())
                self.pos = set(requests.get(pos_url, timeout=5).text.splitlines())
            except Exception:
                self.neg = set()
                self.pos = set()

    def fit(self, X, y=None):
        if self.neg is None or self.pos is None:
            self._load_lexicons()
        return self

    def transform(self, X):
        # Ensure lexica loaded
        if self.neg is None or self.pos is None:
            self._load_lexicons()

        counts = []
        for doc in X:
            toks = str(doc).split()
            neg_count = sum(1 for w in toks if w in self.neg)
            pos_count = sum(1 for w in toks if w in self.pos)
            counts.append([neg_count, pos_count])
        return counts

def full_preprocess(texts: list[str]) -> list[str]:
    """
    Quick helper to apply TextPreprocessor to a list of strings.
    """
    tp = TextPreprocessor()
    return tp.transform(texts)
