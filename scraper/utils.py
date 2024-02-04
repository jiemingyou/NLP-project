import re
from typing import Tuple, Dict
from pprint import pprint

from selenium import webdriver
from selenium.webdriver.common.by import By


def get_driver(url):
    driver = webdriver.Chrome()
    driver.get(url)
    return driver


def get_course_code(driver):
    """Get course code from the course page."""

    return driver.find_element(By.CSS_SELECTOR, "div.course-unit-code.mb-3").text


def get_course_name_credits(driver) -> Tuple[str, str]:
    """Get course name and credits from the course page."""

    course_name = driver.find_element(
        By.CSS_SELECTOR, "h1[data-cy='course-unit-info-heading']"
    ).text.split("\n")[-1]

    # Regex to extract course name and credits
    name = re.search(r"(.+?) \(\d+ op\)", course_name).group(1)
    credits = re.search(r"\((\d+) op\)", course_name).group(1)

    return (name, credits)


def get_course_info(driver) -> Dict[str, str]:
    """Get course description from the course page."""
    details = driver.find_element(
        By.CSS_SELECTOR, "app-course-unit-info-content-and-goals"
    ).text

    if details == "Tietoja ei ole annettu.":
        return {}

    lines = [x.strip() for x in details.split("\n")]
    content_dict = {}
    current_title = ""

    for line in lines:
        if line in ["OSAAMISTAVOITTEET", "ASIASISÄLTÖ", "LISÄTIEDOT"]:
            current_title = line
            content_dict[current_title] = []
        elif current_title:
            content_dict[current_title].append(line)

    return content_dict


if __name__ == "__main__":
    url = "https://sisu.aalto.fi/student/courseunit/otm-c09c8242-5e9f-46e6-a38c-8855d9578cc1"
    driver = get_driver(url)
    print(get_course_code(driver))
    print(get_course_name_credits(driver))
    print(get_course_info(driver))
    driver.quit()
