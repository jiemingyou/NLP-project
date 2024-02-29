import pandas as pd
from typing import List
from ast import literal_eval
from collections import deque
from nltk import sent_tokenize


def concat_course_info(row):
    """Concatenate the course name and course info into a single string"""

    info_dict = literal_eval(row["course_info"])
    out = row["course_name"] + ". "
    for key in info_dict.keys():
        if key == "LISÄTIEDOT":
            continue
        out += " ".join(info_dict[key])
    return out


def split_text(text: str, n=100) -> List[str]:
    """Split the text into n parts where each part is under n characters long"""
    sents = deque(sent_tokenize(text))
    all_sents = []
    part = ""
    while sents:
        sent = sents.popleft()
        if len(part) > n:
            all_sents.append(part.strip())
            part = sent
        else:
            part += " " + sent

    if part:
        all_sents.append(part.strip())

    return all_sents


if __name__ == "__main__":

    for idx, sent in enumerate(
        split_text(
            """Advanced Operations Management D. After completing the course, student is able to: Can evaluate, analyze and choose appropriate approaches for design, plan and control of advanced operations in global supply chain context. Can model different supply chain designs and planning solutions to address various trade-offs in operations management. Can apply analytical or simulation models with different OM models to solve challenging supply chain problems and use these to evaluate the suitability of solutions in different contexts: can identify the relevant drivers of performance and constraints in the environment and linked to the business strategy. Identifies and can analyze the changes and benefits from digitalization and advancement of production technologies. Evaluate sustainability implications of operations related decisions from strategy and design to planning. Identify with specific requirements of design and management of closed-loop supply chains.The course covers key operations topics from demand analysis to operations strategy. Moving from demand analytics and forecasting models, to planning processes and coordination of supply and demand through sales and operations planning. Based on the analysis of the demand side, we will design competitive operations with the use of process analysis and design, factory physics and supply chain design. We will evaluate solutions to operations problems using the fundamental models and relevant analytics and simulation tools, valuing operations flexibility, costs, network design, and optimizing capacity and decisions under uncertainty. We will also cover state-of-the art of planning and control of supply chains from practice and theory. Key focus area and viewpoint of the couse are sustainable operations and sustainability of the decisions by the operations managers. We will evaluate the options for local production and innovation for more resilient and sustainable operations that can be competitive and contribute to more sustainable businesses for future."""
        )
    ):
        print(idx, sent)
