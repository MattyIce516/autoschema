import os
from setuptools import setup, find_packages

with open("README.md", "rt", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", mode="r", encoding="utf-8") as file:
    requirements = [line.strip() for line in file]

setup(
    name="autoschema",
    version="1.0.0",
    author="MattHarris",
    author_email="mattlax516@gmail.com",
    description="Python library to enforce schemas on Pandas DataFrames",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MattyIce516/autoschema",
    package_dir={'': 'src'},  # Set the package directory to src
    packages=find_packages(where='src'),  # Find packages in src directory
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ]
)