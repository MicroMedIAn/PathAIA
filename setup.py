from setuptools import setup, find_packages
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="pathaia",
    version="0.1.3",
    description="procedures for wsi analysis",
    author="Arnaud Abreu",
    author_email="arnaud.abreu.p@gmail.com",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "fastcore",
        "numpy",
        "tqdm",
        "openslide-python",
        "opencv-python",
        "scikit-image",
        "matplotlib",
        "nptyping",
        "pandas",
        "dataclasses"
    ],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
