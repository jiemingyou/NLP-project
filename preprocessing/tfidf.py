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
        self.stop_words = set(stopwords.words("english")).update(punctuation).add("...")
        self.stemmer = SnowballStemmer("english")

    def _tokenize(self, df, colname: str) -> str:
        return df[colname].apply(word_tokenize)

    def _stem(self, df, colname: str) -> str:
        return df[colname].apply(lambda x: [self.stemmer.stem(word) for word in x])

    def _remove_stopwords(self, df, colname: str) -> str:
        return df[colname].apply(
            lambda x: [
                word
                for word in x
                if word.lower() not in self.stop_words and word.isalnum()
            ]
        )

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
