# Data preprocessing

This directory contains the code for data preprocessing.

## Preprocessing steps

1. Chunk the course description to sentences and translate them from Finnish to
   English using the
   [`Helsinki-NLP/opus-tatoeba-fi-en`](https://huggingface.co/Helsinki-NLP/opus-tatoeba-fi-en)
   model from Hugging Face.
2. Tokenizing, lemmatizing and removing stop words from the course descriptions.
3. Calculating the $\text{tf-idf}$ matrix for the translated course
   descriptions.

## Usage

Copy the course data to the `../scraper/data` directory. Then run

```bash
cd preprocessing
python preprocess.py
```
