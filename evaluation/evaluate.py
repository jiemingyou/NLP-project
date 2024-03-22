import json
import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class IREvaluator:
    """Evaluator class for Information Retrieval (IR) tasks. The class provides methods
    to evaluate the performance of course embeddings on an evaluation set using the Normalized
    Discounted Cumulative Gain (NDCG) metric. The class supports evaluation of both OpenAI
    and SentenceTransformer embeddings.
    """
    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.eval_set = self._load_evaluation_data()
        self.top_n_courses_openai = {}
        self.top_n_courses_transformers = {}

    def _load_evaluation_data(self) -> dict:
        """Loads the evaluation set from the given evaluation set path.

        Returns:
            dict: The evaluation set as a dictionary.
        """
        with open(self.eval_path, "r") as f:
            return json.load(f)

    def evaluate_transformers_embeddings_ir(self, model: str, embeddings_filepath: str) -> dict[str, float]:
        """Evaluates the SentenceTransformer course embeddings on the evaluation set using as metrics
        the average of Normalized Discounted Cumulative Gain (NDCG) of the evaluation queries.

        Args:
            model (str): The name of the SentenceTransformer model to use.
            embeddings_filepath (str): The path to the pickle file containing the course embeddings.

        Returns:
            dict[str, float]: A dictionary containing the metric scores for the evaluation set.
        """
        model = SentenceTransformer(model)
        embeddings, course_codes = self._load_course_embeddings(embeddings_filepath)

        ndcgs = []
        for query in self.eval_set["queries"]:
            query_embedding = model.encode(query["query"])
            top_n_idx = self._vector_search(query_embedding, embeddings)
            top_n_courses = course_codes[top_n_idx]
            self.top_n_courses_transformers[query["query"]] = top_n_courses
            ndcg = self.calculate_ndcg(ground_truths=query["answers"], predictions=top_n_courses)
            ndcgs.append(ndcg)
        
        return {"ndcg": np.mean(ndcgs)}

    def evaluate_openai_embeddings_ir(self, embeddings_filepath: str) -> dict[str, float]:
        """Evaluates the OpenAI course embeddings on the evaluation set using as metrics
        the average of Normalized Discounted Cumulative Gain (NDCG) of the evaluation queries.

        Args:
            embeddings_filepath (str): The path to the pickle file containing the course embeddings.

        Returns:
            dict[str, float]: A dictionary containing the metric scores for the evaluation set.
        """
        client = OpenAI()
        embeddings, course_codes = self._load_course_embeddings(embeddings_filepath)

        ndcgs = []
        for query in self.eval_set["queries"]:
            query_embedding = self._get_openai_embedding(query["query"], client=client)
            top_n_idx = self._vector_search(query_embedding, embeddings)
            top_n_courses = course_codes[top_n_idx]
            self.top_n_courses_openai[query["query"]] = top_n_courses
            ndcg = self.calculate_ndcg(ground_truths=query["answers"], predictions=top_n_courses)
            ndcgs.append(ndcg)

        return {"ndcg": np.mean(ndcgs)}

    def _load_course_embeddings(self, filepath: str) -> tuple[np.ndarray, pd.Series]:
        """Loads the course embeddings and course codes from a given file. File
        must be a pickle file of the format where the first column is the course 
        code and the second column is the embedding of the corresponding course as 
        numpy array.

        Args:
            filepath (str): The path to the pickle file containing the course embeddings.

        Returns:
            tuple[np.ndarray, pd.Series]: A tuple containing the course embeddings as a 
            numpy array and the course codes as Pandas Series.
        """
        embeddings_df = pd.read_pickle(filepath)
        embeddings = embeddings_df["embedding"].to_list()
        embeddings = np.vstack(embeddings)
        course_codes = embeddings_df["course_code"]
        return embeddings, course_codes

    def _get_openai_embedding(self, text: str, client: OpenAI, model="text-embedding-3-small") -> np.ndarray:
        """Get the OpenAI embedding for a given text using the OpenAI API. You need
        to have set up the OpenAI API key in your environment variables to use this.
        Check the OpenAI API documentation for more information on how to set up the
        API key.

        Args:
            text (str): Text to be encoded.
            client (openai.OpenAI): OpenAI client object.
            model (str, optional): Name of the model used for embeddings. 
            Defaults to "text-embedding-3-small".

        Returns:
            np.ndarray: The embedding of the text.
        """
        text = text.replace("\n", " ")
        embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
        return embedding
    
    def _vector_search(self, query: np.ndarray, embeddings: np.ndarray, top_n=5) -> np.ndarray:
        """Searches for the top_n most similar vectors to the query vector in the
        embeddings matrix using the dot product as similarity measure.

        Args:
            query (np.ndarray): The query embedding vector.
            embeddings (np.ndarray): The embeddings matrix.
            top_n (int, optional): Number of most similar instances to search. 
            Defaults to 5.

        Returns:
            np.ndarray: The indices of the top_n most similar vectors.
        """
        scores = np.dot(embeddings, query)
        top_n_idx = scores.argsort()[::-1][:top_n]
        return top_n_idx

    @staticmethod
    def calculate_ndcg(ground_truths: list[str], predictions: list[str], k=5) -> float:
        """Calculates the Normalized Discounted Cumulative Gain (NDCG) for a given query on
        the evaluation set. NDCG is a measure of ranking quality. It is calculated as the
        ratio of the DCG score to the IDCG score, where the DCG score is the sum of the
        relevance scores of the predicted courses, and the IDCG score is the sum of the
        relevance scores of the ground truth courses. The measure takes values between 0
        and 1, where 1 indicates the best possible ranking matching the ground truth.
        Our implementation uses a graded relevance scale of 1 to k, where k is the number
        of predictions to consider with k indicating the highest relevance score.

        Args:
            ground_truths (list[str]): A list of the course codes that are relevant to the query.
            predictions (list[str]): A list of the course codes that are predicted to be relevant to the query.
            k (int): The number of predictions to consider.

        Returns:
            float: The NDCG score for the query.
        """
        assert len(predictions) == k, "The number of predictions must be equal to k."

        relevances = [i for i in range(k, 0, -1)]
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevances)])

        dcg = 0

        for i, code in enumerate(predictions):
            rel = 0
            for eval_code, eval_rel in zip(ground_truths[:k], relevances):
                if code == eval_code:
                    rel = eval_rel
                    break
            
            dcg += rel / np.log2(i + 2)

        ndcg = dcg / idcg
        return ndcg