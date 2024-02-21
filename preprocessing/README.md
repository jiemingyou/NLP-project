# Data preprocessing

This directory contains the code for data preprocessing.

## Preprocessing steps

1. Course description translation (FI $\rightarrow$ EN) using
   `Helsinki-NLP/opus-tatoeba-fi-en` model from Hugging Face.

## Usage

Copy the course data to the `../scraper/data` directory. Then run

```bash
python preprocess.py
```
