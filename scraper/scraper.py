import time
import random
import pandas as pd

from utils import (
    get_driver,
    get_course_code,
    get_course_name_credits,
    get_course_info,
)


def scrape_course_data(url):
    """Scrape course data from the given URL."""
    # 1. Init the driver
    driver = get_driver(url)

    # 2. Get the course code
    course_code = get_course_code(driver)

    # 3. Get the course name and credits
    course_name, credits = get_course_name_credits(driver)

    # 4. Get the course info
    course_info = get_course_info(driver)

    # 5. Return the course data
    return {
        "course_code": course_code,
        "course_name": course_name,
        "credits": credits,
        "course_info": course_info,
        "url": url,
    }


def main():

    # Get the URLs from the text files
    urls = []
    with open("urls/cs_urls.txt", "r") as cs, open("urls/ms_urls.txt", "r") as ms:
        for url in cs:
            urls.append(url.strip())
        for url in ms:
            urls.append(url.strip())

    # Store the course data into a dataframe
    course_data = pd.DataFrame(
        columns=[
            "course_code",
            "course_name",
            "credits",
            "course_info",
            "url",
        ]
    )

    for url in urls:

        # Log the progress
        print(f"({urls.index(url) + 1}/{len(urls)})")

        # Scrape the course data
        try:
            # Append the data to the dataframe
            data = scrape_course_data(url)

            course_data = pd.concat(
                [course_data, pd.DataFrame([data])], ignore_index=True
            )

        except Exception as e:
            print(f"Failed to scrape {url} due to {e}")
            continue

        # Sleep for a couple of seconds to avoid getting blocked :-)
        time.sleep(random.uniform(1, 3))

    # Save the dataframe to a CSV file
    course_data.to_csv("data/course_data.csv", index=False)

    return course_data


if __name__ == "__main__":
    df = main()
