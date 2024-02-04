import os
import time
import random
import pandas as pd

from utils import get_urls, scrape_course_data


def scrape_sisu() -> pd.DataFrame:

    # Store the course data into a dataframe
    course_data = pd.DataFrame(
        columns=["course_code", "course_name", "credits", "course_info", "url"]
    )

    # Get the URLs from the text files
    urls = get_urls("urls/")
    for idx, url in enumerate(urls):
        print(f"Scraping {idx+1}/{len(url)}")

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
