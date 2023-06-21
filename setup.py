from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of required packages from requirement.txt
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements] #since when we read the text file \n will also get inserted to our list 

        if "-e ." in requirements:
            requirements.remove("-e .") # we also dont want "-e ." in our list of requirements

    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Vansh Joshi",
    author_email="vanshuwjoshi@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirement.txt"),
)
