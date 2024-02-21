from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import logging


class Translator:
    """
    Translation model instance for translating a dataset using a Huggingface model
    """

    def __init__(self, model_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_length = max_length

    def translation_pipeline(self, row):
        """Translation pipeline to be mapped onto a Huggingface dataset"""
        text = row["course_description"]

        # If the text is empty, return an empty string
        if text == "":
            row["course_description_en"] = ""
            return row

        try:
            input_ids = self.tokenizer.encode(
                row["course_description"],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            output_ids = self.model.generate(input_ids, max_length=self.max_length)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            row["course_description_en"] = output_text
        except Exception as e:
            logging.error(f"Error translating text: {e}")
            row["course_description_en"] = None

        return row

    def translate_dataset(self, dataset: Dataset) -> Dataset:
        """Translate the dataset using the translation pipeline"""

        if "course_description" not in dataset.column_names:
            raise ValueError(
                "The dataset must contain a column named 'course_description'"
            )

        # Apply the translation pipeline to the dataset
        dataset = dataset.map(self.translation_pipeline)
        return dataset
