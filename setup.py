from setuptools import setup, find_packages

setup(
    name="echoverse",
    version="0.1.0",
    author="Buad",
    author_email="dwallin73@msn.com",
    description="GPU-accelerated semantic similarity and verse resonance explorer.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/echoverse",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pycuda",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
    ],
    python_requires=">=3.8",
)
