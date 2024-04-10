import os
import pandas as pd
from supabase import create_client, Client


class VectorDB:

    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL")
        self.key = os.environ.get("SUPABASE_KEY")
        self.supabase = create_client(self.url, self.key)

    def _load_courses_embeddings(
        self,
        courses: pd.DataFrame,
        embeddings: list[pd.DataFrame],
    ):
        """
        Returns a generator of dictionaries from a pandas DataFrame
        """
        # Only specific columns are needed
        cols = ["course_code", "course_name", "credits", "course_description_en", "url"]
        courses = courses[cols]

        # Join the embeddings to the courses
        for name, emb in embeddings.items():
            # Rename the embedding for identification
            emb = emb.rename(columns={"embedding": name})
            courses = courses.merge(emb, how="inner", on="course_code")

        # Rename the columns to match the database schema
        courses = courses.rename(
            columns={
                "course_code": "code",
                "course_name": "name",
                "course_description_en": "description",
            }
        )

        # The rows are fed to the database one by one
        for row in courses.to_dict(orient="records"):
            yield row

    def insert_courses(
        self,
        courses: pd.DataFrame,
        embeddings: dict[str, pd.DataFrame],
        table_name: str,
    ):
        """
        Insert the courses into the database
        """
        data = self._load_courses_embeddings(courses, embeddings)
        for record in data:
            self.supabase.table(table_name).insert(record).execute()


if __name__ == "__main__":

    # Load the data
    courses = pd.read_csv("translated_course_data_no_duplicates.csv")
    emb_bert = pd.read_pickle("embeddings_pretranslated_all-distilroberta-v1.pkl")
    emb_oai = pd.read_pickle("embeddings_pretranslated_openai-small.pkl")
    emb_oai_mixed = pd.read_pickle("embeddings_notranslated_openai-small.pkl")
    emb_bert.embedding = emb_bert.embedding.apply(lambda x: x.tolist())
    emb_oai.embedding = emb_oai.embedding.apply(lambda x: x.tolist())

    embeddings = {
        "embedding_bert": emb_bert,
        "embedding_openai": emb_oai,
        "embedding_openai_mixed": emb_oai_mixed,
    }

    # Connect to the database
    vectordb = VectorDB()

    # Insert the courses into the database
    vectordb.insert_courses(courses, embeddings, "course_embeddings")
