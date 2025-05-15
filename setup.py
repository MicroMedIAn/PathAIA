# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="pathaia",
    version="0.2.5",
    description="procedures for wsi analysis",
    author="Arnaud Abreu",
    author_email="arnaud.abreu.p@gmail.com",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "fastcore>=1,<2",
        "numpy>=1,<2",
        "tqdm>=4",
        "openslide-python>=1.1",
        "opencv-python>=4",
        "scikit-image>=0.19,<1",
        "matplotlib>=3",
        "nptyping>=2,<3",
        "pandas>=1",
        "dataclasses",
        "sortedcontainers>=2,<3",
        "ordered-set>=4,<5",
        "shapely>=1,<2",
        "scikit-learn>=1,<2",
    ],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
