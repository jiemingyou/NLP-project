from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from string import punctuation
from typing import Tuple


def calculate_tf_idf(
    df: pd.DataFrame, colname: str
) -> Tuple[pd.DataFrame, TfidfVectorizer]:

    # Step  1: Tokenization
    df["tokens"] = df[colname].apply(word_tokenize)

    # Steps 2 and 3: remove stop words and punctuation
    stop_words = set(stopwords.words("english"))
    stop_words.update(punctuation)
    stop_words.add("...")
    df["tokens"] = df["tokens"].apply(
        lambda x: [
            word for word in x if word.lower() not in stop_words and word.isalnum()
        ]
    )

    # Step 4: Stemming
    stemmer = SnowballStemmer("english")
    df["tokens"] = df["tokens"].apply(lambda x: [stemmer.stem(word) for word in x])

    # Step 5. Present as tf-idf
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["tokens"].apply(" ".join))

    return (X, vectorizer)
