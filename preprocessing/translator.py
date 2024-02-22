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

    def translation_pipeline(self, row, colname):
        """Translation pipeline to be mapped onto a Huggingface dataset"""
        text = row[colname]

        # If the text is empty, return an empty string
        if text == "":
            row[f"{colname}_en"] = ""
            return row

        try:
            input_ids = self.tokenizer.encode(
                row[colname],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            output_ids = self.model.generate(input_ids, max_length=self.max_length)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            row[f"{colname}_en"] = output_text
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
