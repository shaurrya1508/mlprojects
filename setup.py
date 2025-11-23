from setuptools import  find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:   
    """Read the requirements from a file and return as a list."""
    requirements=[]
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements
    
setup(
    name="mlprojects",               
    version="0.0.1",                  
    author="Shaurya Sharma",          
    author_email="shauryasharma4881@gmail.com",
    description="A machine learning projects package",
    packages=find_packages(),         
    install_requires=get_requirements('requirements.txt'),

    python_requires=">=3.8",
)
