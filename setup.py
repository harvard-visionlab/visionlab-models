from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="visionlab-models",
    version="0.1.0",
    author="George Alvarez",
    author_email="alvarez@wjh.harvard.edu",
    description="PyTorch DNN models instrumented for research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harvard-visionlab/visionlab-models",
    packages=find_packages(),
    install_requires=[
        # 'git+https://github.com/harvard-visionlab/visionlab-models.git',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
