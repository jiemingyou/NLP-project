import os
import re

from typing import Tuple, Dict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


def get_urls(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} does not exist.")

    urls = set()
    for file in os.listdir(path):
        with open(path + file, "r") as f:
            for url in f:
                urls.add(url.strip())
    return list(urls)


def get_driver(url):
    """
    Return a driver instance running headless
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-extensions")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    return driver


def get_course_code(driver):
    """
    Get course code from the course page.
    """
    return driver.find_element(
        By.CSS_SELECTOR,
        "div.course-unit-code.mb-3",
    ).text


def get_course_name_credits(driver) -> Tuple[str, str]:
    """
    Get course name and credits from the course page.
    """
    course_name = driver.find_element(
        By.CSS_SELECTOR,
        "h1[data-cy='course-unit-info-heading']",
    ).text.split("\n")[-1]

    # Regex to extract course name and credits
    name = re.search(r"(.+?) \(\d+(–\d+)? op\)", course_name).group(1)
    credits = re.search(r"\((\d+(–\d+)?) op\)", course_name).group(1)

    return (name, credits)


def get_course_info(driver) -> Dict[str, str]:
    """
    Get course description from the course page.
    """
    details = driver.find_element(
        By.CSS_SELECTOR,
        "app-course-unit-info-content-and-goals",
    ).text

    if details == "Tietoja ei ole annettu.":
        return {}

    lines = [x.strip() for x in details.split("\n")]
    content_dict = {
        "OSAAMISTAVOITTEET": [],
        "ASIASISÄLTÖ": [],
        "LISÄTIEDOT": [],
    }
    current_title = ""

    for line in lines:
        if line in ["OSAAMISTAVOITTEET", "ASIASISÄLTÖ", "LISÄTIEDOT"]:
            current_title = line
        else:
            content_dict[current_title].append(line)

    return content_dict


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

    driver.quit()

    # 5. Return the course data
    return {
        "course_code": course_code,
        "course_name": course_name,
        "credits": credits,
        "course_info": course_info,
        "url": url,
    }


if __name__ == "__main__":
    # url = "https://sisu.aalto.fi/student/courseunit/otm-c09c8242-5e9f-46e6-a38c-8855d9578cc1"
    # d#river = get_driver(url)
    # print(get_course_code(driver))
    # print(get_course_name_credits(driver))
    # print(get_course_info(driver))
    # driver.quit()
    with open("all_urls.txt", "w") as f:
        for url in get_urls("urls/"):
            f.write(url + "\n")
