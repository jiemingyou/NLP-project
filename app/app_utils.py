import time
import pandas as pd
import streamlit as st
from openai import OpenAI
from db_connection import DBConnection

MODEL = "gpt-3.5-turbo-0125"


@st.cache_resource
def init_connection_sqlalchemy():
    db_user = st.secrets["DB_USER"]
    db_password = st.secrets["DB_PASSWORD"]
    db_host = st.secrets["DB_HOST"]
    db_port = st.secrets["DB_PORT"]
    db_name = st.secrets["DB_NAME"]
    return DBConnection(db_user, db_password, db_host, db_port, db_name)


@st.cache_resource
def init_connection_openai():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


supabase = init_connection_sqlalchemy()
client = init_connection_openai()


@st.cache_data(ttl=600)
def get_embedding(text, model="text-embedding-3-small") -> list:
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


@st.cache_data(ttl=600)
def get_top_n_courses(query: str, n: int = 5):
    """
    Get top n courses based on the cosine similarity.
    """
    assert isinstance(n, int), "n should be an integer."
    embedding = get_embedding(query)
    query = f"""
        SELECT *
        FROM course_embeddings
        ORDER BY embedding_openai <=> '{embedding}' asc
        LIMIT {n}
        """
    return supabase.query(query)


def create_query(client, prompt):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                    You are part of an information retrieval system and
                    your task is to extract relevant parts of user query
                    that are then used for semantic search.
                """,
            },
            {
                "role": "user",
                "content": "I want to learn linear algebra and programming.",
            },
            {
                "role": "assistant",
                "content": "linear algebra and programming",
            },
            {
                "role": "user",
                "content": "financial modeling and strategy.",
            },
            {
                "role": "assistant",
                "content": "financial modeling and strategy",
            },
            {
                "role": "user",
                "content": "show me courses on linear algebra and also programming.",
            },
            {
                "role": "assistant",
                "content": "linear algebra and programming.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def infer_best_course(client, query, retriever_courses):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                    You are part of an information retrieval system and your task is to
                    infer what courses would be the best for the user based on their query.
                    Format the reponse using Markdown syntax as follows:
                    [Course name](URL) - Course description.
                """,
            },
            {
                "role": "user",
                "content": """
                Query: I want to learn about natural language processing.
                Retrieved courses: [
                    {
                        "code": "ELEC-E5550",
                        "name": "Statistical Natural Language Processing D",
                        "credits": "5",
                        "description": "After attending the course, the student knows how statistical and deep learning methods ...",
                        "url": "URL"
                    },
                ]
                """,
            },
            {
                "role": "assistant",
                "content": """
                Based on your query about natural language processing, here are some courses that you might be interested in:
                1. [Statistical Natural Language Processing D](URL) - Covers statistical and deep learning methods used in various NLP applications like machine translation, sentiment analysis, and more.
                """,
            },
            {
                "role": "user",
                "content": f"Query: {query} Retrieved courses: {retriever_courses}",
            },
        ],
    )
    return completion.choices[0].message.content


def response_streamer(response: str):
    """
    Stream responses to the user.
    """
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.03)


def format_response(df: pd.DataFrame):
    """
    Format the response.
    """
    response = [f"{idx+1}: [{row['name']}]({row['url']})" for idx, row in df.iterrows()]
    return "  \n".join(response)


def format_reponses_json(df: pd.DataFrame):
    """
    Format the response in json for feeding into the LLM.
    """
    response = [
        {
            "code": row["code"],
            "name": row["name"],
            "credits": row["credits"],
            "description": row["description"],
            "url": row["url"],
        }
        for _, row in df.iterrows()
    ]
    return response
