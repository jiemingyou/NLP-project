import os
import time
import random
import pandas as pd

from utils import scrape_course_data


def scrape_sisu() -> pd.DataFrame:

    # Get the URLs from the text files
    urls = []
    for file in os.listdir("urls/"):
        with open(f"urls/{file}", "r") as f:
            for url in f:
                urls.append(url.strip())

    # Store the course data into a dataframe
    course_data = pd.DataFrame(
        columns=["course_code", "course_name", "credits", "course_info", "url"]
    )

    for url in urls:
        print(f"({urls.index(url) + 1}/{len(urls)})")

        try:
            # Scrape the course data
            data = scrape_course_data(url)
            course_data = pd.concat(
                [course_data, pd.DataFrame([data])], ignore_index=True
            )

        except Exception as e:
            print(f"Failed to scrape {url} due to {e}")
            continue

        time.sleep(random.uniform(1, 3))  # avoid getting blocked ;-)

    return course_data


if __name__ == "__main__":
    df = scrape_sisu()
    df.to_csv("data/course_data.csv", index=False)
