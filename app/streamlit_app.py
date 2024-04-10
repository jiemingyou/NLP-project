import streamlit as st
from db_connection import DBConnection


# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection_sqlalchemy():
    db_user = st.secrets["DB_USER"]
    db_password = st.secrets["DB_PASSWORD"]
    db_host = st.secrets["DB_HOST"]
    db_port = st.secrets["DB_PORT"]
    db_name = st.secrets["DB_NAME"]
    return DBConnection(db_user, db_password, db_host, db_port, db_name)


# supabase = init_connection()
supabase = init_connection_sqlalchemy()


@st.cache_data(ttl=600)
def query_courses(course_name: str = None):
    """
    Query courses from the database.
    """
    assert course_name is not None, "Please provide a course name"
    SQL_QUERY_EXAMPLE = f"""
    SELECT *
    FROM course_embeddings
    WHERE LOWER(name) LIKE '%{course_name.lower()}%'
    LIMIT 5;
    """
    return supabase.query(SQL_QUERY_EXAMPLE)


st.title("📚 OpenA!")
st.caption("A course recommender tool powered by NLP. Work in progress, stay tuned!")
st.text_input("🔎 Course name", key="course_name")
course_name = st.session_state.get("course_name")

if course_name:
    st.dataframe(
        query_courses(course_name)[["name", "description", "url"]],
        hide_index=True,
    )
else:
    st.info("Please enter a course name to search.", icon="ℹ️")
