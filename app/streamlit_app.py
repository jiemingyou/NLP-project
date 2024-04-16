import streamlit as st
from openai import OpenAI

from app_utils import (
    init_connection_sqlalchemy,
    init_connection_openai,
    get_top_n_courses,
    create_query,
    infer_best_course,
    response_streamer,
    format_response,
    format_reponses_json,
)

supabase = init_connection_sqlalchemy()
client = init_connection_openai()
openai_api_key = ""

# Set page title and caption
st.title("📚 OpenA!")
st.caption(
    """
    A course recommender tool powered by NLP. 
    Course project for Statistical Natural Language Processing Course.
    """
)


# Reset chat history
if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

# Toggle LLM
with st.sidebar:
    llm = st.toggle("Activate LLM feature (optional)", value=False)
    if llm:
        openai_api_key = st.text_input("OpenAI API key")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


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

        if llm and openai_api_key == "":
            response = "Please provide an OpenAI API key to activate the LLM feature."

        elif llm and openai_api_key != "":
            # LLM: Format the response
            llm_client = OpenAI(api_key=openai_api_key)
            improved_query = create_query(llm_client, prompt)
            courses = get_top_n_courses(improved_query)
            courses_json = format_reponses_json(courses)
            courses_string = infer_best_course(llm_client, prompt, courses_json)
            response = f"{courses_string}"

        else:
            # NO-LLM: Format the response
            courses = get_top_n_courses(prompt)
            courses_string = format_response(courses)
            response = f"Here are some course recommendations: \n\n {courses_string}"

        with st.chat_message("assistant"):
            response = st.write_stream(response_streamer(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
