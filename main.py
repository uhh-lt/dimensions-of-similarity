import dataclasses
import os
import random
from collections import Counter, defaultdict
from csv import DictWriter
from enum import Enum
from itertools import count
from pathlib import Path
from typing import List, Optional

import fasttext
import numpy as np
import pandas as pd
import sklearn.metrics
import spacy
import torch
import typer
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.util import cos_sim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dos.core_dataset import CoreDataset
from dos.cosine_loss_multiple_labels import CosineSimilarityLossForMultipleLabels
from dos.dataset import DIMENSIONS, ArticlePair, SemEvalDataset
from dos.evaluator import (
    CorrelationEvaluator,
    MultitaskHeadCorrelationEvaluator,
    MultitaskPromptCorrelationEvaluator,
)
from dos.input_example_multiple_labels import InputExampleWithMultipleLabels
from dos.poetry import PoetryDataset
from dos.reshape_normalize_layer import reshape_normalize_layer
from dos.reviews import ReviewDataset
from dos.sentiment import Sentiment, SentimentDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

pd.set_option("display.precision", 2)

train_models = [
    # "bert-base-multilingual-cased",
    # "sentence-transformers/stsb-xlm-r-multilingual",
    "sentence-transformers/LaBSE",
    # "all-MiniLM-L6-v2",
    # "all-mpnet-base-v2",
]

dimensions = [
    "geography",
    "entities",
    "time",
    "narrative",
    "overall",
    "style",
    "tone",
]

app = typer.Typer()

nlp = spacy.load("en_core_web_lg")


def extract_embeddings(text, embedder, word_filter):
    doc = nlp(text)
    for token in doc:
        if word_filter(token):
            yield token.vector, embedder.get_word_vector(token.text)


KNOWN_NER_TAGS = (
    "ORDINAL",
    "GPE",
    "LOC",
    "MONEY",
    "CARDINAL",
    "QUANTITY",
    "LAW",
    "DATE",
    "NORP",
    "EVENT",
    "LANGUAGE",
    "ORG",
    "PERSON",
    "TIME",
    "PRODUCT",
    "PERCENT",
    "FAC",
    "WORK_OF_ART",
)


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
            return lambda token: token.ent_type_ in [
                "ORG",
                "PERSON",
                "GPE",
                "LOC",
                "LANGUAGE",
                "PRODUCT",
                "WORK_OF_ART",
                "FAC",
            ]
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
    DE = "de"
    ALL_TRANSLATED = "translated"

    def get_data_path(self, split):
        if self == DataSubset.EN or self == DataSubset.DE:
            return Path(f"data/{split}_data")
        elif self == DataSubset.ALL_TRANSLATED:
            return Path(f"data/{split}_data_translated")


@app.command(name="fasttext")
def fasttext_similarity(
    limit: Optional[int] = None,
    split: str = "test",
    embeddings: str = "cc.en.300.bin",
    kind: WordKind = WordKind.ALL,
    subset: DataSubset = "en",
):
    """
    Code for section 4.2 "Basesline and Entity-Focus" of the paper.
    """
    current_dataset = None
    if split == "test":
        test = SemEvalDataset(
            Path("data/eval.csv"),
            subset.get_data_path("eval"),
            langs=["en"] if subset == subset.EN else "all",
        )
        current_dataset = test
    elif split == "train":
        train = SemEvalDataset(
            Path("data/train.csv"),
            subset.get_data_path("train"),
            langs=["en"] if subset == subset.EN else "all",
        )
        current_dataset = train
    else:
        raise ValueError("Invalid split")
    embedder = fasttext.load_model(embeddings)
    predicted_sims = []
    gold_sims = defaultdict(list)
    for pair, i in zip(current_dataset, range(limit) if limit is not None else count()):
        verbs_a = [
            fasttext_emb
            for spacy_emb, fasttext_emb in extract_embeddings(
                pair.article_1.text, embedder, kind.get_filter()
            )
        ]
        verbs_b = [
            fasttext_emb
            for spacy_emb, fasttext_emb in extract_embeddings(
                pair.article_2.text, embedder, kind.get_filter()
            )
        ]
        if len(verbs_a) == 0 or len(verbs_b) == 0:
            continue
        verbs_a_stacked = np.stack(verbs_a)
        verbs_b_stacked = np.stack(verbs_b)
        sim = cos_sim(verbs_a_stacked, verbs_b_stacked)
        sim.fill_diagonal_(0)
        predicted_sims.append(sim.max(0).values.mean() + sim.max(1).values.mean())
        for key, value in [
            ("Geography", pair.geography),
            ("Entities", pair.entities),
            ("Time", pair.time),
            ("Narrative", pair.narrative),
            ("Overall", pair.overall),
            ("Style", pair.style),
            ("Tone", pair.tone),
        ]:
            gold_sims[key].append(value)
    correlations = {}
    for key, gold_values in gold_sims.items():
        correlations[key] = torch.corrcoef(
            torch.stack([torch.tensor(predicted_sims), torch.tensor(gold_values)])
        )
    print("#### Only considering", kind, "on", subset)
    for key, corrs in correlations.items():
        print(f"{key} correlation {-corrs[0, 1].item():.2f}")


@app.command(name="multitask-prompt")
def multitask_prompt():
    """
    Unused code for finetuning.

    This prefixes the model input with the desired attention.
    In our experiments this process worked worse still than the MTL head variant below.
    """
    dataset = SemEvalDataset(Path("data/train.csv"), Path("data/train_data"))
    test = SemEvalDataset(Path("data/eval.csv"), Path("data/eval_data"))
    train, dev = dataset.random_split(0.8)
    training_inputs = make_multitask_prompt_training_data(train)
    dev_evaluator = MultitaskPromptCorrelationEvaluator(dev)
    test_evaluator = MultitaskPromptCorrelationEvaluator(test)
    for model_name in train_models:
        try:
            # init model
            model = SentenceTransformer(model_name)
            model.max_seq_length = 512
            dev_evaluator.model_name = model_name
            test_evaluator.model_name = model_name

            # eval untrained model on dev
            print("Dev set untrained:")
            dev_evaluator(model)

            # eval untrained model on test
            print("Test set untrained:")
            test_evaluator(model)

            # finetune model on train (& eval on dev)
            finetune_model(
                model,
                training_inputs,
                dev_evaluator,
                loss_fcn=losses.CosineSimilarityLoss,
            )

            # eval finetuned model on test
            print("Test set finetuned:")
            test_evaluator(model, epoch=1)

            # print results
            print("Dev set results:")
            dev_evaluator.print_results()
            print("Test set results:")
            test_evaluator.print_results()

            # write results
            model_name_ = model_name.split("/")[1] if "/" in model_name else model_name
            dev_evaluator.write_results(
                path=Path(f"multitask-prompt-{model_name_}-dev.csv")
            )
            test_evaluator.write_results(
                path=Path(f"multitask-prompt-{model_name_}-test.csv")
            )
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)


def make_multitask_prompt_training_data(data: List[ArticlePair]) -> List[InputExample]:
    inputs: List[InputExample] = []
    for pair in data:
        pair_dict = dataclasses.asdict(pair)
        for dimension in [
            "geography",
            "entities",
            "time",
            "narrative",
            "overall",
            "style",
            "tone",
        ]:
            inputs.append(
                InputExample(
                    texts=[
                        f"{dimension}: {pair.article_1.text}",
                        f"{dimension}: {pair.article_2.text}",
                    ],
                    label=normalize_score_01(pair_dict[dimension]),
                )
            )
    return inputs


@app.command(name="multitask-head")
def multitask_head():
    """
    Code for section 4.3 "Finetuning" of the paper, multitask variant.
    """
    dataset = SemEvalDataset(Path("data/train.csv"), Path("data/train_data"))
    test = SemEvalDataset(Path("data/eval.csv"), Path("data/eval_data"))
    train, dev = dataset.random_split(0.8)
    training_inputs = make_multitask_head_training_data(train)
    dev_evaluator = MultitaskHeadCorrelationEvaluator(dev)
    test_evaluator = MultitaskHeadCorrelationEvaluator(test)
    model_name = "sentence-transformers/LaBSE"
    for out_features in [16, 32, 64, 128, 256, 512, 768]:
        try:
            # init model
            model = SentenceTransformer(model_name)
            model.max_seq_length = 512
            model.add_module(
                "3",
                models.Dense(
                    in_features=768,
                    out_features=out_features * 7,
                    activation_function=nn.Tanh(),
                ),
            )
            model.add_module("4", reshape_normalize_layer(num_labels=7))
            dev_evaluator.model_name = model_name
            test_evaluator.model_name = model_name

            # eval untrained model on dev
            print("Dev set untrained:")
            dev_evaluator(model)

            # eval untrained model on test
            print("Test set untrained:")
            test_evaluator(model)

            # finetune model on train (& eval on dev)
            finetune_model(
                model,
                training_inputs,
                dev_evaluator,
                loss_fcn=CosineSimilarityLossForMultipleLabels,
            )

            # eval finetuned model on test
            print("Test set finetuned:")
            test_evaluator(model, epoch=3)

            # print results
            print("Dev set results:")
            dev_evaluator.print_results()
            print("Test set results:")
            test_evaluator.print_results()

            # write results
            dev_evaluator.write_results(
                path=Path(f"multitask-head-{out_features}-dev.csv")
            )
            test_evaluator.write_results(
                path=Path(f"multitask-head-{out_features}-test.csv")
            )
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)


def make_multitask_head_training_data(
    data: List[ArticlePair],
) -> List[InputExampleWithMultipleLabels]:
    inputs: List[InputExample] = [
        InputExampleWithMultipleLabels(
            texts=[pair.article_1.text, pair.article_2.text],
            label=[
                normalize_score_01(pair.geography),
                normalize_score_01(pair.entities),
                normalize_score_01(pair.time),
                normalize_score_01(pair.narrative),
                normalize_score_01(pair.overall),
                normalize_score_01(pair.style),
                normalize_score_01(pair.tone),
            ],
        )
        for pair in data
    ]
    return inputs


@app.command(name="compare-correlations")
def compare_correlations():
    """
    Code for section 4.4 "Comparing human judgments and machine judgments" of the paper.
    """
    dataset = SemEvalDataset(Path("data/eval.csv"), Path("data/eval_data"))

    geography = torch.tensor([pair.geography for pair in dataset])
    entities = torch.tensor([pair.entities for pair in dataset])
    time = torch.tensor([pair.time for pair in dataset])
    narrative = torch.tensor([pair.narrative for pair in dataset])
    overall = torch.tensor([pair.overall for pair in dataset])
    style = torch.tensor([pair.style for pair in dataset])
    tone = torch.tensor([pair.tone for pair in dataset])

    human_similiarities = torch.stack(
        (
            geography,
            entities,
            time,
            narrative,
            overall,
            style,
            tone,
        )
    )
    human_corrs = torch.corrcoef(human_similiarities)

    pd.set_option("display.precision", 2)

    human_df = pd.DataFrame(human_corrs, index=DIMENSIONS, columns=dimensions)
    print(human_df)

    model_dict = {
        "geography": "models/finetuned-LaBSE-geography",
        "entities": "models/finetuned-LaBSE-entities",
        "time": "models/finetuned-LaBSE-time",
        "narrative": "models/finetuned-LaBSE-narrative",
        "overall": "models/finetuned-LaBSE-overall",
        "style": "models/finetuned-LaBSE-style",
        "tone": "models/finetuned-LaBSE-tone",
    }

    model_similiarities = []
    for name, model_id in model_dict.items():
        # load model
        model = SentenceTransformer(model_id, device="cuda:0")

        article_1 = model.encode(
            [pair.article_1.text for pair in dataset],
            show_progress_bar=True,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        article_2 = model.encode(
            [pair.article_2.text for pair in dataset],
            show_progress_bar=True,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        sims = torch.sum(article_1 * article_2, dim=-1)
        model_similiarities.append(sims)

    model_similiarities = torch.stack(model_similiarities)
    model_corrs = torch.corrcoef(model_similiarities)
    model_df = pd.DataFrame(model_corrs.cpu(), index=DIMENSIONS, columns=dimensions)
    print(model_df)

    diff = model_corrs.cpu() - human_corrs
    diff_df = pd.DataFrame(diff.cpu(), index=DIMENSIONS, columns=dimensions)
    with pd.option_context("display.float_format", "{:0.2f}".format):
        print(diff_df)


@app.command(name="compare-correlations-mtl")
def compare_correlations_mtl():
    """
    Code for section 4.4 "Comparing human judgments and machine judgments" of the paper, mtl variant.
    """
    mtl_model = (
        "/ltstorage/home/tfischer/Development/dimensions-of-similarity/models/mtl-LaBSE"
    )
    model = SentenceTransformer(mtl_model, device="cuda:0")

    dataset = SemEvalDataset(Path("data/eval.csv"), Path("data/eval_data"))

    geography = torch.tensor([pair.geography for pair in dataset])
    entities = torch.tensor([pair.entities for pair in dataset])
    time = torch.tensor([pair.time for pair in dataset])
    narrative = torch.tensor([pair.narrative for pair in dataset])
    overall = torch.tensor([pair.overall for pair in dataset])
    style = torch.tensor([pair.style for pair in dataset])
    tone = torch.tensor([pair.tone for pair in dataset])

    human_similiarities = torch.stack(
        (
            geography,
            entities,
            time,
            narrative,
            overall,
            style,
            tone,
        )
    )
    human_corrs = torch.corrcoef(human_similiarities)

    pd.set_option("display.precision", 2)

    human_df = pd.DataFrame(human_corrs, index=DIMENSIONS, columns=dimensions)
    print(human_df)

    model_similiarities = []

    article_1 = model.encode(
        [pair.article_1.text for pair in dataset],
        show_progress_bar=True,
        batch_size=32,
        convert_to_tensor=True,
    )
    article_2 = model.encode(
        [pair.article_2.text for pair in dataset],
        show_progress_bar=True,
        batch_size=32,
        convert_to_tensor=True,
    )

    for dim_id, dimension in enumerate(DIMENSIONS):
        sims = F.cosine_similarity(
            article_1[:, dim_id, :], article_2[:, dim_id, :], dim=1
        )
        model_similiarities.append(sims)

    model_similiarities = torch.stack(model_similiarities)
    model_corrs = torch.corrcoef(model_similiarities)
    model_df = pd.DataFrame(model_corrs.cpu(), index=DIMENSIONS, columns=dimensions)
    print(model_df)

    diff = model_corrs.cpu() - human_corrs
    diff_df = pd.DataFrame(diff.cpu(), index=DIMENSIONS, columns=dimensions)
    with pd.option_context("display.float_format", "{:0.2f}".format):
        print(diff_df)


@app.command()
def main():
    """
    Code for section 4.3 "Finetuning" of the paper, individual models per dimension.
    """
    result_dir = "results"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    dataset = SemEvalDataset(Path("data/train.csv"), Path("data/train_data"))
    test = SemEvalDataset(Path("data/eval.csv"), Path("data/eval_data"))
    train, dev = dataset.random_split(0.8)
    # This is to safeguard against any rng shenanigans
    assert train[0].article_2.title == "The 20 Best Places to Travel in 2020"
    assert (
        train[1].article_2.title
        == "Shoplifters arrested for stealing beer and meat from Guelph grocery store"
    )
    training_inputs_per_dim = [make_training_data(train, dim) for dim in dimensions]
    dev_evaluator = CorrelationEvaluator(dev)
    test_evaluator = CorrelationEvaluator(test)
    for model_name in train_models:
        try:
            # init model
            model = SentenceTransformer(model_name)
            model.max_seq_length = 512
            dev_evaluator.model_name = model_name

            model_name_ = model_name.split("/")[1] if "/" in model_name else model_name

            # eval untrained model on dev
            print("Dev set untrained:")
            dev_evaluator(model)

            # eval untrained model on test
            print("Test set untrained:")
            test_evaluator(model)

            # iterate dimensions to fine-tune on
            for i, (dim, training_inputs) in enumerate(
                zip(dimensions, training_inputs_per_dim)
            ):
                dev_evaluator.score_column = i

                # finetune model on train (& eval on dev)
                print(f"Fine-tuning model {model_name_} for dimension {dim}")
                finetune_model(
                    model,
                    training_inputs,
                    dev_evaluator,
                    loss_fcn=losses.CosineSimilarityLoss,
                    name=f"finetuned-{model_name_}-{dim}",
                )

                # eval finetuned model on test
                print("Test set finetuned:")
                test_evaluator(model)

                # print results
                print(f"Dev set results fine-tuned on {dim}:")
                dev_evaluator.print_results()
                print(f"Test set results fine-tuned on {dim}:")
                test_evaluator.print_results()

                # write results
                dev_evaluator.write_results(
                    path=Path(result_dir, f"finetuned-{model_name_}-{dim}-dev.csv")
                )
                test_evaluator.write_results(
                    path=Path(result_dir, f"finetuned-{model_name_}-{dim}-test.csv")
                )
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)


@app.command()
def pretrained_eval_sweep():
    for datasplit in ["train", "eval"]:
        for languages in [None, ["en"]]:
            pretrained_eval(languages, datasplit)


@app.command()
def pretrained_eval(
    languages: Optional[List[str]] = None,
    datasplit: str = "train",
    translated: bool = False,
):
    dataset = SemEvalDataset(
        Path(f"data/{datasplit}.csv"),
        Path(f"data/{datasplit}_data{'_translated' if translated else ''}"),
        langs=languages or "all",
    )
    evaluator = CorrelationEvaluator(dataset)
    for model_name in train_models:
        try:
            print("### Using", model_name)
            # init model
            model = SentenceTransformer(model_name)
            evaluator.model_name = model_name
            for max_seq_len in [256, 512]:
                model.max_seq_length = max_seq_len

                # eval untrained model on dev
                print(f"{datasplit} set untrained:")
                evaluator(model)
                evaluator.print_results()

                # write results
                model_name_ = (
                    model_name.split("/")[1] if "/" in model_name else model_name
                )
                evaluator.write_results(
                    path=Path(
                        f"pretrained-{model_name_}-{datasplit}-{'_'.join(languages) if languages is not None else 'all'}{'-translated' if translated else ''}-seq_len={model.max_seq_length}.csv"
                    )
                )
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)


def make_training_data(
    data: List[ArticlePair], dim_name="overall"
) -> List[InputExample]:
    inputs: List[InputExample] = [
        InputExample(
            texts=[pair.article_1.text, pair.article_2.text],
            label=normalize_score_01(getattr(pair, dim_name)),
        )
        for pair in data
    ]
    return inputs


def normalize_score(one2four: float):
    return 1 - 2 * (one2four - 1) / 3


def normalize_score_01(one2four: float):
    return ((1 - 2 * (one2four - 1) / 3) + 1) / 2


def finetune_model(
    model: SentenceTransformer,
    inputs: List[InputExample],
    evaluator: CorrelationEvaluator
    | MultitaskPromptCorrelationEvaluator
    | MultitaskHeadCorrelationEvaluator,
    loss_fcn: nn.Module,
    name: str,
):
    dataloader = DataLoader(inputs, shuffle=True, batch_size=32)
    loss = loss_fcn(model)
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=3,
        warmup_steps=100,
        evaluator=evaluator,
        use_amp=True,
        output_path=f"models/{name}",
    )


class ReviewSimilarityDimensions(Enum):
    DIFFERENT_PRODUCT_SAME_RATING = "different_product_same_rating"
    SAME_PRODUCT_DIFFERENT_RATING = "same_product_different_rating"
    SAME_RATING = "same_rating"
    SAME_PRODUCT = "same_product"

    def review_comparison(self):
        return {
            ReviewSimilarityDimensions.DIFFERENT_PRODUCT_SAME_RATING: lambda a, b: a.product_id
            != b.product_id
            and round(a.rating) == round(b.rating),
            ReviewSimilarityDimensions.SAME_PRODUCT_DIFFERENT_RATING: lambda a, b: a.product_id
            == b.product_id
            and round(a.rating) != round(b.rating),
            ReviewSimilarityDimensions.SAME_RATING: lambda a, b: round(a.rating)
            == round(b.rating),
            ReviewSimilarityDimensions.SAME_PRODUCT: lambda a, b: a.product_id
            == b.product_id,
        }[self]


@app.command()
def reviews(
    dimension: ReviewSimilarityDimensions = "different_product_same_rating",
    limit: int = 1000,
):
    """
    Code for section 5.1 "Product Reviews Experiments" of the paper.

    prams:
        split: randomly only use `split` fraction of the dataset
    """
    path = Path("data/amazon_reviews/amazon_total.txt")
    dataset = ReviewDataset(path, limit=limit)
    print(dataset.description())
    print(f"Using first {limit} products of the data that is {len(dataset)} reviews.")
    embeddings_per_model = dict()
    gold_label_func = dimension.review_comparison()
    csv_writer = DictWriter(
        open(f"reviews-{limit}.csv", "w"),
        fieldnames=[
            "name",
            "map",
            "r-precision",
            "precision@5",
            "precision@10",
            "precision@100",
        ],
    )
    csv_writer.writeheader()
    for model_name in [
        "models/finetuned-LaBSE-tone",
        "models/finetuned-LaBSE-entities",
        "models/finetuned-LaBSE-overall",
        "sentence-transformers/LaBSE",
    ]:
        model = SentenceTransformer(model_name, device="cuda:0")
        os.makedirs("data/cache/", exist_ok=True)
        review_texts = [r.review or "" for r in dataset]
        cache_path = f"data/cache/{path.stem}-{model_name.split('/')[-1]}.pt"
        if not os.path.exists(cache_path):
            # We need to take the numpy version first to move the data of the gpu
            encoded = torch.from_numpy(
                model.encode(review_texts, show_progress_bar=True, batch_size=64)
            )
            torch.save(encoded, cache_path)
        else:
            print("Loading from", cache_path)
            encoded = torch.load(cache_path)
        dataset_sample = dataset
        encoded_sample = encoded[[r.embedding_index for r in dataset_sample]]
        embeddings_per_model[model_name] = encoded_sample
        precisions = get_precisions(
            [(1.0, encoded_sample)],
            dataset_sample,
            gold_label=gold_label_func,
            ks=[5, 10, 100],
        )
        print(f"MAP with {model_name}:", precisions["map"])
        csv_writer.writerow(
            {"name": model_name, **{k: f"{v:.3f}" for k, v in precisions.items()}}
        )
    for combination in [
        "tone - entities",
        "2 * tone - overall",
        # "overall + entities - tone",
        # "overall + entities",
        # "overall - tone",
        # "-overall",
        # "-entities",
        # "tone - overall - entites",
        # "random",
    ]:
        if combination == "tone - overall":
            encoded_sample = [
                (1.0, embeddings_per_model["models/finetuned-LaBSE-tone"]),
                (-1.0, embeddings_per_model["models/finetuned-LaBSE-overall"]),
            ]
        if combination == "2 * tone - overall":
            encoded_sample = [
                (2.0, embeddings_per_model["models/finetuned-LaBSE-tone"]),
                (-1.0, embeddings_per_model["models/finetuned-LaBSE-overall"]),
            ]
        elif combination == "tone - entities":
            encoded_sample = [
                (1.0, embeddings_per_model["models/finetuned-LaBSE-tone"]),
                (-1.0, embeddings_per_model["models/finetuned-LaBSE-entities"]),
            ]
        elif combination == "overall + entities - tone":
            encoded_sample = [
                (-1.0, embeddings_per_model["models/finetuned-LaBSE-tone"]),
                (1.0, embeddings_per_model["models/finetuned-LaBSE-overall"]),
                (1.0, embeddings_per_model["models/finetuned-LaBSE-entities"]),
            ]
        elif combination == "overall + entities":
            encoded_sample = [
                (1.0, embeddings_per_model["models/finetuned-LaBSE-entities"]),
                (1.0, embeddings_per_model["models/finetuned-LaBSE-overall"]),
            ]
        elif combination == "overall - tone":
            encoded_sample = [
                (-1.0, embeddings_per_model["models/finetuned-LaBSE-tone"]),
                (1.0, embeddings_per_model["models/finetuned-LaBSE-overall"]),
            ]
        elif combination == "tone - overall - entites":
            encoded_sample = [
                (1.0, embeddings_per_model["models/finetuned-LaBSE-tone"]),
                (-1.0, embeddings_per_model["models/finetuned-LaBSE-overall"]),
                (-1.0, embeddings_per_model["models/finetuned-LaBSE-entities"]),
            ]
        elif combination == "-overall":
            encoded_sample = [
                (-1.0, embeddings_per_model["models/finetuned-LaBSE-overall"]),
            ]
        elif combination == "-entities":
            encoded_sample = [
                (
                    -1.0,
                    embeddings_per_model["models/finetuned-LaBSE-entities"],
                ),
            ]
        elif combination == "random":
            encoded_sample = [
                (
                    1.0,
                    torch.rand_like(
                        embeddings_per_model["models/finetuned-LaBSE-entities"]
                    ),
                ),
            ]
        combined_precisions = get_precisions(
            encoded_sample,
            dataset_sample,
            gold_label=gold_label_func,
            ks=[5, 10, 100],
        )
        csv_writer.writerow(
            {
                "name": combination,
                **{k: f"{v:.3f}" for k, v in combined_precisions.items()},
            }
        )
        print(f"{combination} MAP", combined_precisions["map"])


def random_sample(dataset, frac, seed=42):
    dataset, _ = torch.utils.data.random_split(
        dataset, [frac, 1 - frac], generator=torch.Generator().manual_seed(seed)
    )
    return dataset


def weighted_sims(weighted_embeddings):
    final_embeddings = None
    for weight, embeddings in weighted_embeddings:
        if final_embeddings is None:
            final_embeddings = cos_sim(embeddings, embeddings) * weight
        else:
            final_embeddings += cos_sim(embeddings, embeddings) * weight
    return final_embeddings


def get_precisions(
    weighted_embeddings,
    dataset,
    ks=[5, 10],
    gold_label=lambda x, y: round(x.rating) == round(y.rating),
):
    sims = weighted_sims(weighted_embeddings)
    sims.fill_diagonal_(0)
    precisions_at_k = defaultdict(list)
    r_precisions = []
    average_precisions = []
    for i, sim_line in enumerate(sims):
        labels = torch.tensor(
            [
                gold_label(item, dataset[i]) if j != i else False
                for j, item in enumerate(dataset)
            ]
        )
        for k in ks:
            indices = sim_line.topk(k).indices
            average_precisions.append(
                sklearn.metrics.average_precision_score(labels, sim_line)
            )
            precisions_at_k[k].append(labels[indices].sum() / k)
        indices = sim_line.topk(labels.sum()).indices
        r_precisions.append(labels[indices].sum() / labels.sum())
    out = {
        "r-precision": (sum(r_precisions) / len(r_precisions)).item(),
        "map": sum(average_precisions) / len(average_precisions),
    }
    out.update(
        {
            f"precision@{k}": (sum(precisions) / len(precisions)).item()
            for k, precisions in precisions_at_k.items()
        }
    )
    return out


def select_all_other_embeddings(embeddings, indexes):
    mask = torch.ones(embeddings.shape[0])
    mask[indexes] = 0
    return embeddings.masked_select(
        mask.unsqueeze(-1).to(dtype=torch.bool, device=embeddings.device)
    ).reshape(-1, embeddings.shape[-1])


@app.command()
def poetry(all_combinations: bool = False, subtract_overall: bool = False):
    """
    Code for section 5.4 "German Poetry Experiment" of the paper.
    """
    dataset = PoetryDataset("data/jcls2022-poem-similarity")
    model_dict = {
        "content": "models/finetuned-LaBSE-narrative",
        "style": "models/finetuned-LaBSE-style",
        "emotion": "models/finetuned-LaBSE-tone",
        "overall": "models/finetuned-LaBSE-overall",
    }
    overall_model = SentenceTransformer(model_dict["overall"])
    overall_weight = -1
    for dimension in ["content", "form", "style", "emotion", "overall"]:
        overall_different = dataset.with_unambigious_dimension(dimension)
        model_name = model_dict.get(dimension, "sentence-transformers/LaBSE")
        for model_name in (
            [model_name] if not all_combinations else list(model_dict.values())
        ):
            print("Using", model_name)
            model = SentenceTransformer(model_name)
            *texts, labels = PoetryDataset.texts_and_labels(overall_different)
            anchor_embs, left_embs, right_embs = [
                model.encode(collection, convert_to_tensor=True) for collection in texts
            ]
            left_anchor_similarity = torch.nn.functional.cosine_similarity(
                left_embs, anchor_embs
            )
            right_anchor_similarity = torch.nn.functional.cosine_similarity(
                right_embs, anchor_embs
            )
            if subtract_overall:
                overall_anchor_embs, overall_left_embs, overall_right_embs = [
                    overall_model.encode(collection, convert_to_tensor=True)
                    for collection in texts
                ]
                overall_left_anchor_similarity = torch.nn.functional.cosine_similarity(
                    overall_left_embs, overall_anchor_embs
                )
                overall_right_anchor_similarity = torch.nn.functional.cosine_similarity(
                    overall_right_embs, overall_anchor_embs
                )
            else:
                overall_left_anchor_similarity = torch.zeros_like(
                    left_anchor_similarity
                )
                overall_right_anchor_similarity = torch.zeros_like(
                    right_anchor_similarity
                )
            predictions = (
                left_anchor_similarity
                + (overall_weight * overall_left_anchor_similarity)
                < right_anchor_similarity
                + (overall_weight * overall_right_anchor_similarity)
            ).cpu()
            label_tensor = torch.tensor(labels, dtype=torch.bool)
            balanced_accuracy = sklearn.metrics.balanced_accuracy_score(
                label_tensor, predictions
            )
            print("Accuracy on", dimension, f"{balanced_accuracy:.02f}")


@app.command()
def sentiment_svm(lang: str = "english", model_name: str = "tone"):
    """
    Code for section 5.3 "Sentiment Classification Experiment" in the paper.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    # train_dataset =
    if lang == "english":
        train_dir = (
            "data/SemEval2017-task4-test/train/2017_English_final/GOLD/Subtask_A"
        )
        train_datasets = []
        for file_name in os.listdir(train_dir):
            if file_name != "README.txt" and file_name.endswith(".txt"):
                train_datasets.append(
                    SentimentDataset(os.path.join(train_dir, file_name))
                )
        dataset = torch.utils.data.ConcatDataset(train_datasets)
        test_dataset = SentimentDataset(
            f"data/SemEval2017-task4-test/SemEval2017-task4-test.subtask-A.english.txt",
        )
    elif lang == "arabic":
        dataset = SentimentDataset(
            f"data/SemEval2017-task4-test/train/2017_Arabic_train_final/GOLD/SemEval2017-task4-train.subtask-A.arabic.txt",
        )
        test_dataset = SentimentDataset(
            f"data/SemEval2017-task4-test/SemEval2017-task4-test.subtask-A.arabic.txt",
        )
    else:
        raise ValueError("Invalid language", lang)
    model = SentenceTransformer(f"models/finetuned-LaBSE-{model_name}")
    labels_train, features_train = zip(*dataset)
    labels_test, features_test = zip(*test_dataset)
    labels_train = [str(label) for label in labels_train]
    labels_test = [str(label) for label in labels_test]
    embeddings_train = model.encode(
        features_train, convert_to_tensor=True, show_progress_bar=True
    )
    embeddings_test = model.encode(
        features_test, convert_to_tensor=True, show_progress_bar=True
    )
    print("=== KNN")
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=8)
    knn.fit(embeddings_train.cpu(), labels_train)
    outputs = knn.predict(embeddings_test.cpu())
    score = recall_score(labels_test, outputs, average="macro")
    f1 = f1_score(labels_test, outputs, average="macro")
    accuracy = accuracy_score(labels_test, outputs)
    print("Recall", score)
    print("Accuracy", accuracy)
    print("F1", f1)
    print("=== Linear SVM")
    svm = SVC(kernel="linear")
    svm.fit(embeddings_train.cpu(), labels_train)
    outputs = svm.predict(embeddings_test.cpu())
    score = recall_score(labels_test, outputs, average="macro")
    f1 = f1_score(labels_test, outputs, average="macro")
    accuracy = accuracy_score(labels_test, outputs)
    print("Recall", score)
    print("F1", f1)
    print("Accuracy", accuracy)
    print("=== K-Means")
    kmeans = KMeans(n_clusters=3).fit(embeddings_train.cpu())
    labels = kmeans.predict(embeddings_train.cpu())
    counts = defaultdict(Counter)
    for label, label_name in zip(labels, labels_train):
        counts[label].update({label_name: 1})
    translation = {k: v.most_common(1)[0][0] for k, v in counts.items()}
    print(translation)
    outputs = [translation[label] for label in kmeans.predict(embeddings_test.cpu())]
    score = recall_score(labels_test, outputs, average="macro")
    f1 = f1_score(labels_test, outputs, average="macro")
    accuracy = accuracy_score(labels_test, outputs)
    print("Recall", score)
    print("F1", f1)
    print("Accuracy", accuracy)


@app.command()
def sentiment(lang: str):
    """
    Code for section 5.3 "Sentiment Classification Experiment" in the paper, KNN variant.
    """
    dataset = SentimentDataset(
        f"data/SemEval2017-task4-test/SemEval2017-task4-test.subtask-A.{lang}.txt",
        skip_neutral=True,
    )
    model_dict = {
        "style": "models/finetuned-LaBSE-style",
        "tone": "models/finetuned-LaBSE-tone",
        "narrative": "models/finetuned-LaBSE-narrative",
        "entities": "models/finetuned-LaBSE-entities",
        "overall": "models/finetuned-LaBSE-overall",
        "unfinetuned": "sentence-transformers/LaBSE",
    }
    print("Dataset size", len(dataset))
    csv_writer = DictWriter(
        open(f"sentiments-{lang}.csv", "w"),
        fieldnames=[
            "name",
            "map",
            "r-precision",
            "precision@5",
            "precision@10",
            "precision@100",
            "silhouette_score",
        ],
    )
    csv_writer.writeheader()
    for name, model_id in model_dict.items():
        model = SentenceTransformer(model_id)
        encoded = model.encode(
            [text for _label, text in dataset],
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        dist_matrix = 1 - cos_sim(encoded, encoded).cpu()
        dist_matrix.fill_diagonal_(0)
        silhouette_score = sklearn.metrics.silhouette_score(
            dist_matrix, [label.value for label, _text in dataset], metric="precomputed"
        )
        precisions = get_precisions(
            [(1.0, encoded.cpu())],
            dataset,
            gold_label=lambda x, y: x[0] == y[0],
            ks=[5, 10, 100],
        )
        csv_writer.writerow(
            {
                "name": name,
                "silhouette_score": silhouette_score,
                **{k: f"{v:.3f}" for k, v in precisions.items()},
            }
        )


@app.command()
def core(tags1: List[str], tags2: List[str]):
    """
    Code for section 5.2 "Text Register Experiment" in the paper.
    """
    tag1 = "_".join(tags1)
    tag2 = "_".join(tags2)

    dataset = CoreDataset(
        train_path="./data/CORE-corpus/train.tsv.gz",
        test_path="./data/CORE-corpus/test.tsv.gz",
    )
    model_dict = {
        "style": "models/finetuned-LaBSE-style",
        "tone": "models/finetuned-LaBSE-tone",
        "narrative": "models/finetuned-LaBSE-narrative",
        "entities": "models/finetuned-LaBSE-entities",
        "overall": "models/finetuned-LaBSE-overall",
        "unfinetuned": "sentence-transformers/LaBSE",
    }
    print("Dataset size", len(dataset))

    csv_writer = DictWriter(
        open(f"core-{tag1}-{tag2}.csv", "w"),
        fieldnames=[
            "name",
            "map",
            "r-precision",
            "precision@5",
            "precision@10",
            "precision@100",
            "silhouette_score",
        ],
    )
    csv_writer.writeheader()
    for name, model_id in model_dict.items():
        # load model
        model = SentenceTransformer(model_id, device="cuda:0")

        # create cache dir
        os.makedirs("data/cache/", exist_ok=True)

        # load or embedd text
        cache_path_tag_1 = f"data/cache/core-{tag1}-{model_id.split('/')[-1]}.pt"
        if not os.path.exists(cache_path_tag_1):
            # We need to take the numpy version first to move the data of the gpu
            encoded_tag1 = torch.from_numpy(
                model.encode(
                    [document.text for document in dataset.documents_by_tags(tags1)],
                    show_progress_bar=True,
                    batch_size=32,
                )
            )
            torch.save(encoded_tag1, cache_path_tag_1)
        else:
            encoded_tag1 = torch.load(cache_path_tag_1)

        cache_path_tag_2 = f"data/cache/core-{tag2}-{model_id.split('/')[-1]}.pt"
        if not os.path.exists(cache_path_tag_2):
            # We need to take the numpy version first to move the data of the gpu
            encoded_tag2 = torch.from_numpy(
                model.encode(
                    [document.text for document in dataset.documents_by_tags(tags2)],
                    show_progress_bar=True,
                    batch_size=32,
                )
            )
            torch.save(encoded_tag2, cache_path_tag_2)
        else:
            encoded_tag2 = torch.load(cache_path_tag_2)
        encoded = torch.cat((encoded_tag1, encoded_tag2))
        dist_matrix = 2 - cos_sim(encoded, encoded).cpu()
        dist_matrix.fill_diagonal_(0)
        tags = [tag1 for _ in dataset.documents_by_tags(tags1)] + [
            tag2 for _ in dataset.documents_by_tags(tags2)
        ]
        print("Calculating silhouette score")
        silhouette_score = sklearn.metrics.silhouette_score(
            dist_matrix,
            tags,
            metric="precomputed",
        )
        print(silhouette_score)
        print("Calculating precisions")
        precisions = get_precisions(
            [(1.0, encoded)],
            tags,
            gold_label=lambda x, y: x == y,
            ks=[5, 10, 100],
        )
        csv_writer.writerow(
            {
                "name": name,
                "silhouette_score": silhouette_score,
                **{k: f"{v:.3f}" for k, v in precisions.items()},
            }
        )


if __name__ == "__main__":
    app()
