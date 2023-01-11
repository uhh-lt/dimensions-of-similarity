import dataclasses
from pathlib import Path
from typing import List
import pandas as pd
from itertools import count
import typer
from enum import Enum
from typing import Optional
from collections import defaultdict
import fasttext
import torch
import spacy
import numpy as np
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer, InputExample, losses
from dos.evaluator import CorrelationEvaluator
from dos.dataset import SemEvalDataset, ArticlePair
from torch.utils.data import DataLoader

from dos.multitask_evaluator import MultitaskCorrelationEvaluator

pd.set_option('display.precision', 2)

models = ["bert-base-multilingual-cased", "sentence-transformers/stsb-xlm-r-multilingual",
          "sentence-transformers/LaBSE", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]

app = typer.Typer()

nlp = spacy.load("en_core_web_lg")


def extract_embeddings(text, embedder, word_filter):
    doc = nlp(text)
    for token in doc:
        if word_filter(token):
            yield token.vector, embedder.get_word_vector(token.text)


KNOWN_NER_TAGS = 'ORDINAL', 'GPE', 'LOC', 'MONEY', 'CARDINAL', 'QUANTITY', 'LAW', 'DATE', 'NORP', 'EVENT', 'LANGUAGE', 'ORG', 'PERSON', 'TIME', 'PRODUCT', 'PERCENT', 'FAC', 'WORK_OF_ART'


class WordKind(Enum):
    ENTS = "entities"
    ALL_NER = "ALL_NER"
    VERB = "verb"
    ALL = "all"
    TIME = "time"
    GEO = "geo"

    def get_filter(self):
        if self == WordKind.ALL:
            return lambda token: True
        elif self == WordKind.VERB:
            return lambda token: token.pos_ == "VERB"
        elif self == WordKind.ENTS:
            return lambda token: token.ent_type_ in ["ORG", "PERSON", "GPE", "LOC", "LANGUAGE", "PRODUCT",
                                                     "WORK_OF_ART", "FAC"]
        elif self == WordKind.ALL_NER:
            return lambda token: token.ent_type_ != ""
        elif self == WordKind.TIME:
            return lambda token: token.ent_type_ == "TIME"
        elif self == WordKind.GEO:
            return lambda token: token.ent_type_ in ["LOC", "GPE"]
        else:
            raise ValueError("Invalid variant", self)


class DataSubset(Enum):
    EN = "en"
    ALL_TRANSLATED = "translated"

    def get_data_path(self, split):
        if self == DataSubset.EN:
            return Path(f"data/{split}_data")
        elif self == DataSubset.ALL_TRANSLATED:
            return Path(f"data/{split}_data_translated")


@app.command(name="fasttext")
def fasttext_similarity(limit: Optional[int] = None, split: str = "test", embeddings: str = "cc.en.300.bin",
                        kind: WordKind = WordKind.ALL, subset: DataSubset = "en"):
    current_dataset = None
    if split == "test":
        test = SemEvalDataset(Path("data/eval.csv"), subset.get_data_path("eval"),
                              langs=["en"] if subset == subset.EN else "all")
        current_dataset = test
    elif split == "train":
        train = SemEvalDataset(Path("data/train.csv"), subset.get_data_path("train"),
                               langs=["en"] if subset == subset.EN else "all")
        current_dataset = train
    else:
        raise ValueError("Invalid split")
    embedder = fasttext.load_model(embeddings)
    predicted_sims = []
    gold_sims = defaultdict(list)
    for pair, i in zip(current_dataset, range(limit) if limit is not None else count()):
        verbs_a = [fasttext_emb for spacy_emb, fasttext_emb in
                   extract_embeddings(pair.article_1.text, embedder, kind.get_filter())]
        verbs_b = [fasttext_emb for spacy_emb, fasttext_emb in
                   extract_embeddings(pair.article_2.text, embedder, kind.get_filter())]
        if len(verbs_a) == 0 or len(verbs_b) == 0:
            continue
        verbs_a_stacked = np.stack(verbs_a)
        verbs_b_stacked = np.stack(verbs_b)
        sim = cos_sim(verbs_a_stacked, verbs_b_stacked)
        sim.fill_diagonal_(0)
        predicted_sims.append(sim.max(0).values.mean() + sim.max(1).values.mean())
        for key, value in [("Geography", pair.geography), ("Entities", pair.entities), ("Time", pair.time),
                           ("Narrative", pair.narrative), ("Overall", pair.overall), ("Style", pair.style),
                           ("Tone", pair.tone)]:
            gold_sims[key].append(value)
    correlations = {}
    for key, gold_values in gold_sims.items():
        correlations[key] = torch.corrcoef(torch.stack([torch.tensor(predicted_sims), torch.tensor(gold_values)]))
    print("#### Only considering", kind, "on", subset)
    for key, corrs in correlations.items():
        print(f"{key} correlation {-corrs[0, 1].item():.2f}")


@app.command(name="multitask")
def multitask():
    dataset = SemEvalDataset(Path("data/train.csv"), Path("data/train_data"))
    train, dev = dataset.random_split(0.8)
    training_inputs = make_multitask_training_data(train)
    evaluator = MultitaskCorrelationEvaluator(dev)
    for model_name in models:
        try:
            model = SentenceTransformer(model_name)
            evaluator.model_name = model_name
            finetune_model(model, training_inputs, evaluator)
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)


def make_multitask_training_data(data: List[ArticlePair]) -> List[InputExample]:
    inputs: List[InputExample] = []
    for pair in data:
        pair_dict = dataclasses.asdict(pair)
        for dimension in ['geography', 'entities', 'time', 'narrative', 'overall', 'style', 'tone']:
            inputs.append(
                InputExample(
                    texts=[f"{dimension}: {pair.article_1.text}", f"{dimension}: {pair.article_2.text}"],
                    label=normalize_score(pair_dict[dimension])))
    return inputs


@app.command()
def main():
    dataset = SemEvalDataset(Path("data/train.csv"), Path("data/train_data"))
    test = SemEvalDataset(Path("data/eval.csv"), Path("data/eval_data"))
    train, dev = dataset.random_split(0.8)
    training_inputs = make_training_data(train)
    dev_evaluator = CorrelationEvaluator(dev)
    test_evaluator = CorrelationEvaluator(test)
    for model_name in models:
        try:
            model = SentenceTransformer(model_name)
            model.max_seq_length = 512
            dev_evaluator.model_name = model_name
            finetune_model(model, training_inputs, dev_evaluator)
            #test_evaluator(model)
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)


def normalize_score(one2four: float):
    return 1 - 2 * (one2four - 1) / 3

def normalize_score_01(one2four: float):
    return ((1 - 2 * (one2four - 1) / 3) + 1) / 2

def make_training_data(data: List[ArticlePair]) -> List[InputExample]:
    inputs: List[InputExample] = [InputExample(
        texts=[pair.article_1.text, pair.article_2.text], label=normalize_score_01(pair.overall)) for pair in data]
    return inputs


def finetune_model(model: SentenceTransformer, inputs: List[InputExample],
                   evaluator: CorrelationEvaluator | MultitaskCorrelationEvaluator):
    dataloader = DataLoader(inputs, shuffle=True, batch_size=16)
    loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(dataloader, loss)], epochs=3, warmup_steps=100, evaluator=evaluator, use_amp=True, output_path="models")


if __name__ == "__main__":
    app()
