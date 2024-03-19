from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
from translator import Translator

from preprocessing_utils import concat_course_info
from tfidf import calculate_tf_idf


def preprocess(vectorize=False) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    # Load the data
    df = pd.read_csv("../scraper/data/course_data.csv")

    # Apply the translation pipeline
    df["course_description"] = df.apply(concat_course_info, axis=1)

    # Create a dataset
    dataset = Dataset.from_pandas(df)

    # Apply the translation pipeline
    translator = Translator(model_name="Helsinki-NLP/opus-tatoeba-fi-en")
    dataset = translator.translate_dataset(dataset, "course_description")

    # Remove __index_level_0__ column if it exists
    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns("__index_level_0__")

    # Save the translated dataset
    dataset.to_csv("translated_course_data_V2.csv")
    print("Saved translated dataset to translated_course_data.csv")

    if not vectorize:
        return dataset, None

    # Calculate the tf-idf
    dataset = dataset.to_pandas()
    X, tfidf_vectorizer = calculate_tf_idf(dataset, "course_description_en")

    # Add the tf-idf to the dataset
    dataset["tf_idf"] = list(X.toarray())

    return dataset, tfidf_vectorizer


if __name__ == "__main__":
    dataset = preprocess()
