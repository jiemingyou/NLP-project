from openai import OpenAI

api_key = "Insert your API key here."
client = OpenAI(api_key=api_key)


def create_query(prompt):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are part of an information retrieval system and your task is to extract relevant parts of user query that are then used for semantic search."},
            {"role": "user", "content": 'I want to learn linear algebra and programming.'},
            {"role": "assistant", "content": 'linear algebra and programming'},
            {"role": "user", "content": 'financial modeling and strategy.'},
            {"role": "assistant", "content": 'financial modeling and strategy'},
            {"role": "user", "content": 'show me courses on linear algebra and also programming.'},
            {"role": "assistant", "content": 'linear algebra and programming.'},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message

print(create_query("find me courses where I can learn operations management and lean manufacturing."))

def infer_best_course(query, retriever_courses):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are part of an information retrieval system and your task is to infer what course would be the best for the user based on their query."},
            {"role": "user", "content": f'Query: {query} Retrieved courses: {retriever_courses}'}
        ]
    )
    return completion.choices[0].message

print(infer_best_course("I want to learn operations management and foreccasting sales", ["TU-E2020 Advanced Operations Management The key contents of this course include operations strategy, visibility & forecasting, sales and operations planning & execution, master data management, lean manufacturing, and digitalization of manufacturing (additive manufacturing and internet of things).", "MS-E1142 Computational Algebraic Geometry The student will learn how to solve systems of polynomial equations and describe images of polynomial maps. The student will understand the correspondence between affine varieties (solution sets of polynomial equations) on the geometry side and ideals on the algebra side. The student will learn how to do computations using Groebner bases and the theory behind the computations. This course can be seen as a nonlinear extension of linear algebra. "]))