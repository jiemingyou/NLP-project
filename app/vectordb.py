import os
import pandas as pd
from supabase import create_client, Client


class VectorDB:

    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL")
        self.key = os.environ.get("SUPABASE_KEY")
        self.supabase = create_client(self.url, self.key)

    def _load_courses_embeddings(self, courses, embeddings):
        """Returns a generator of dictionaries from a pandas DataFrame"""
        cols = ["course_code", "course_name", "credits", "course_description_en", "url"]
        courses = courses[cols]
        courses = courses.merge(embeddings, how="inner", on="course_code")
        courses = courses.rename(
            columns={
                "course_code": "code",
                "course_name": "name",
                "course_description_en": "description",
            }
        )
        courses["embedding"] = courses["embedding"].apply(lambda x: x.tolist())
        for row in courses.to_dict(orient="records"):
            yield row

    def insert_courses(self, courses, embeddings):
        data = self._load_courses_embeddings(courses, embeddings)
        for record in data:
            self.supabase.table("courses").insert(record).execute()


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("translated_course_data_no_duplicates.csv")
    emb = pd.read_pickle("embeddings_pretranslated_all-distilroberta-v1.pkl")

    # Connect to the database
    vectordb = VectorDB()

    # Insert the courses into the database
    vectordb.insert_courses(df, emb)
