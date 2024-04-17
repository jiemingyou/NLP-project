# NLP-project

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://aaltonlp.streamlit.app)

This is the working repository for the course project of the course ELEC-E5550
Statistical Natural Language Processing (SNLP) taught at Aalto University.

We build a course recommendation system based on the course descriptions using
retrieval-based methods.

- [x] Course description scraping using Selenium
- [x] Course description translation using Helsinki-NLP's Opus-MT
- [x] Evaluating different embedding models
- [x] Retrieval evaluation using NDGC
- [x] LLM component for user interaction
- [x] Application using Streamlit, Supabase, and pgvector

## Project structure

```
.
├── README.md
├── app                    # Web app and DB connection
├── bert                   # BERT model
├── embedding              # Embedding models
├── evaluation             # IR evaluation
├── preprocessing          # Preprocessing the course descriptions
├── scraper                # Scraping the course description
├── reader                 # Template for the LLM component
└── README.md
```
