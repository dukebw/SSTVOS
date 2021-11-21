"""SSTVOS setup.py"""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sstvos",
    version="0.0.1",
    author="Brendan Duke",
    author_email="brendanw.duke@gmail.com",
    description="SSTVOS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dukebw/sstvos",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
