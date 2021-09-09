from setuptools import setup, find_packages

exec(open("nowcasting_utils/version.py").read())
setup(
    name="nowcasting_utils",
    packages=find_packages(),
    version=__version__,
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
    install_requires=["huggingface_hub>=0.0.16",
                      "einops>=0.3.2",
                      "numpy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
