import json
import numpy as np
import pandas as pd
from openai import OpenAI


class IREvaluator:
    def __init__(self):
        self.eval_set = self.load_evaluation_data()
        self.top_n_courses_openai = {}

    def _load_evaluation_data(self):
        with open("eval_set.json", "r") as f:
            return json.load(f)

    def evaluate_openai_embeddings_ir(self) -> dict[str, float]:
        client = OpenAI()
        embeddings, course_codes = self.load_openai_course_embeddings()

        ndcgs = []
        for query in self.eval_set["queries"]:
            query_embedding = self.get_openai_embedding(query["query"], client=client)
            top_n_idx = self.vector_search(query_embedding, embeddings)
            top_n_courses = course_codes[top_n_idx]
            self.top_n_courses_openai[query["query"]] = top_n_courses
            ndcg = self.calculate_ndcg(ground_truths=query["answers"], predictions=top_n_courses)
            ndcgs.append(ndcg)

        return {"ndcg": np.mean(ndcgs)}

    def _load_openai_course_embeddings(self):
        openai_embeddings_df = pd.read_pickle("embeddings_pretranslated_openai-small.pkl")
        openai_embeddings = openai_embeddings_df["embedding"].to_list()
        openai_embeddings = np.vstack(openai_embeddings)
        course_codes = openai_embeddings_df["course_code"]
        return openai_embeddings, course_codes

    def _get_openai_embedding(self, text, client, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
        return embedding
    
    def _vector_search(self, query, embeddings, top_n=5):
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