import pandas as pd
from ast import literal_eval


def concat_course_info(row):
    """Concatenate the course name and course info into a single string"""

    info_dict = literal_eval(row["course_info"])
    out = row["course_name"] + ". "
    for key in info_dict.keys():
        if key == "LISÄTIEDOT":
            continue
        out += " ".join(info_dict[key])
    return out
