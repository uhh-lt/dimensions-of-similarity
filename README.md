# Dimensions of Similarity

This is the code-base for the paper "Dimensions of Similarity: Towards Interpretable Dimension-Based Text Similarity" published at ECAI-2023.
Detailed citation instructions and link to preprint are tbd.

The code in `main.py` mostly corresponds to individual experiments in the paper.
Most experiments in the paper roughly correspond to 1-2 subcommands in `main.py`, e.g. run `main.py main` for finetuning of individual models per dimension.

If you are looking to work with any of the datasets we experiment with in the paper, you may find the torch `Dataset` subclasses in the `dos` directory to be useful.

* SemEval News Similarity (2022 Task 8): `dos/dataset.py`
* SemEval Sentiment Dataset (2017 Task 4): `dos/sentiment.py`
* German Poetry Dataset: `dos/poetry.py`
* Amazon Reviews Dataset: `dos/reviews.py`
* CORE Text Register Dataset: `dos/core_dataset.py`


## Setup

First install the required dependencies into your Python installation:

- pytorch
- sentence-transformers
- pandas
- numpy
- spacy
- scikit-learn
- fasttext
- typer

Create a `data` directory and move the required datasets into it.
Run any of the subcommands output by `python main.py --help` to replicate experiments.

If you run into any issues, do not hesitate to open an issue.
