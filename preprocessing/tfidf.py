import pandas as pd
from typing import Tuple
from string import punctuation

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.stemmer = SnowballStemmer("english")
        self._prepare_stopwords()

    def _prepare_stopwords(self) -> str:
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update(punctuation)
        self.stop_words.add("...")

    def tokenize(self, text: str) -> str:
        return word_tokenize(text)

    def _tokenize(self, df, colname: str) -> str:
        return df[colname].apply(self.tokenize)

    def stem(self, words: str) -> str:
        return [self.stemmer.stem(word) for word in words]

    def _stem(self, df, colname: str) -> str:
        return df[colname].apply(lambda x: self.stem(x))

    def remove_stopwords(self, words: str) -> str:
        return [
            word for word in words if word not in self.stop_words and word.isalnum()
        ]

    def _remove_stopwords(self, df, colname: str) -> str:
        return df[colname].apply(lambda x: self.remove_stopwords(x))

    def preprocess(self, df: pd.DataFrame, colname: str) -> pd.DataFrame:
        df["tokens"] = self._tokenize(df, colname)
        df["tokens"] = self._remove_stopwords(df, "tokens")
        df["tokens"] = self._stem(df, "tokens")
        return df

    def fit_transform(self, df: pd.DataFrame, colname: str) -> pd.DataFrame:
        X = self.vectorizer.fit_transform(df[colname].apply(" ".join))
        return X

    def get_vectorizer(self) -> TfidfVectorizer:
        return self.vectorizer
