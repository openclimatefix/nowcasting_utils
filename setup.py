""" Usual setup file for package """
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
install_requires = (this_directory / "requirements.txt").read_text().splitlines()
long_description = (this_directory / "README.md").read_text()

version = open("nowcasting_utils/version.py").readlines()[-1].split()[-1].strip("\"'")

setup(
    name="nowcasting_utils",
    packages=find_packages(),
    version=version,
    license="MIT",
    description="Nowcasting Utilities",
    author="Jacob Bieker, Jack Kelly, Peter Dudfield",
    author_email="jacob@openclimatefix.org",
    company="Open Climate Fix Ltd",
    url="https://github.com/openclimatefix/nowcasting_utils",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformer",
    ],
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
