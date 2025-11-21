from setuptools import setup, find_packages

setup(
    name="mlprojects",               
    version="0.0.1",                  
    author="Shaurya Sharma",          
    author_email="shauryasharma4881@gmail.com",
    description="A machine learning projects package",
    packages=find_packages(),         
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.8",
)
