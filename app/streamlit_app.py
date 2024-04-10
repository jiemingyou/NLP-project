import time
import pandas as pd
import streamlit as st
from openai import OpenAI
from db_connection import DBConnection


# Initialize connections
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


def response_streamer(response: str):
    """
    Stream responses to the user.
    """
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)


def format_response(df: pd.DataFrame):
    """
    Format the response.
    """
    response = [f"{idx+1}: [{row['name']}]({row['url']})" for idx, row in df.iterrows()]
    return "  \n".join(response)


supabase = init_connection_sqlalchemy()
client = init_connection_openai()

# Set page title and caption
st.title("📚 OpenA!")
st.caption("A course recommender tool powered by NLP. Work in progress, stay tuned!")

# Reset chat history
if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

# Insert a chat message container.
with st.chat_message("assistant"):
    st.write(
        "Hello! I'm your course recommender bot. Feel free to ask me for course recommendations."
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("I want to learn about natural language processing."):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Response
    with st.spinner("Thinking..."):
        courses = get_top_n_courses(prompt)
        courses_string = format_response(courses)
        response = f"Here are some course recommendations: \n\n {courses_string}."
        with st.chat_message("assistant"):
            response = st.write_stream(response_streamer(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
