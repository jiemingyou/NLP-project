import spacy
import logging

from datasets import Dataset
from langdetect import detect
from preprocessing_utils import split_text
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Translator:
    """
    Translation model instance for translating a dataset using a Huggingface model
    """

    def __init__(self, model_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_length = max_length
        # Load the English language model for sentence splitting
        self.sentencizer = spacy.blank("en")
        self.sentencizer.add_pipe("sentencizer")

    def translation_pipeline(self, row: dict, colname: str) -> dict:
        """
        Translation pipeline to be mapped onto a Huggingface dataset.
        Translates Finnish text to English using the translation model.
        Ignores empty and English text.
        """
        text = row[colname]

        # If the text is empty, return an empty string
        if text == "":
            row[f"{colname}_en"] = ""
            return row

        # If the text is already in English, return the original text
        try:
            if detect(text) == "en":
                row[f"{colname}_en"] = text
                return row
        except Exception as e:
            logging.error(f"Error detecting language: {e}")
            row[f"{colname}_en"] = text
            return row

        try:
            # Split the text into parts where each part is under n characters long
            parts = split_text(text, self.sentencizer, n=100)
            output_text = []

            # Translate each part
            for chunk in parts:
                input_ids = self.tokenizer.encode(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                output_ids = self.model.generate(input_ids, max_length=self.max_length)
                out_text = self.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                output_text.append(out_text)

            # Concatenate and save the translated text
            row[f"{colname}_en"] = " ".join(output_text)

        except Exception as e:
            logging.error(f"Error translating text: {e}")
            row[f"{colname}_en"] = None

        return row

    def translate_dataset(self, dataset: Dataset, colname: str) -> Dataset:
        """Translate the dataset using the translation pipeline"""

        if colname not in dataset.column_names:
            raise ValueError(f"The dataset must contain a column named '{colname}'")

        # Apply the translation pipeline to the dataset
        dataset = dataset.map(self.translation_pipeline, fn_kwargs={"colname": colname})
        return dataset
