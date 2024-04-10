import os
import pandas as pd
from sqlalchemy import create_engine, text


class DBConnection:
    def __init__(self, db_user, db_password, db_host, db_port, db_name):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_url = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.engine = create_engine(self.db_url)

    def test_connection(self):
        try:
            with self.engine.connect() as connection:
                print("Connection successful!")
                return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def query(self, query):
        assert isinstance(query, str), "Query must be a string"
        with self.engine.connect() as connection:
            result = connection.execute(text(query.strip()))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df


if __name__ == "__main__":

    # Replace the placeholders with your actual database credentials
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_PORT")
    db_name = os.environ.get("DB_NAME")

    # Create an instance of the DBConnection class
    db_connection = DBConnection(db_user, db_password, db_host, db_port, db_name)

    # Test the connection
    SQL_QUERY_EXAMPLE = """
    SELECT *
    FROM course_embeddings
    LIMIT 5;
    """
    df = db_connection.query(SQL_QUERY_EXAMPLE)

    print(df.iloc[:5, :3])
