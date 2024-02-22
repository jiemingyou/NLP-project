import pandas as pd
from datasets import Dataset
from preprocessing_utils import concat_course_info
from translator import Translator


def preprocess():
    # Load the data
    df = pd.read_csv("../scraper/data/course_data.csv")

    # Apply the translation pipeline
    df["course_description"] = df.apply(concat_course_info, axis=1)

    # Create a dataset
    dataset = Dataset.from_pandas(df)

    # Apply the translation pipeline
    translator = Translator(model_name="Helsinki-NLP/opus-tatoeba-fi-en")
    dataset = translator.translate_dataset(dataset, "course_description")

    return dataset


if __name__ == "__main__":
    dataset = preprocess()
    dataset.to_parquet("translated_course_data.parquet")
