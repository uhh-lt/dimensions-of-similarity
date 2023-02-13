import dataclasses
from collections import defaultdict
from enum import Enum
from itertools import count
from pathlib import Path
import os
from typing import List, Optional

import fasttext
import numpy as np
import pandas as pd
import spacy
import sklearn.metrics
import torch
import typer
import random
from tqdm import tqdm
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from torch import nn

from dos.cosine_loss_multiple_labels import CosineSimilarityLossForMultipleLabels
from dos.dataset import ArticlePair, SemEvalDataset
from dos.evaluator import CorrelationEvaluator, MultitaskPromptCorrelationEvaluator, MultitaskHeadCorrelationEvaluator
from dos.input_example_multiple_labels import InputExampleWithMultipleLabels
from dos.reshape_normalize_layer import ReshapeAndNormalize
from dos.reviews import ReviewDataset
from dos.poetry import PoetryDataset
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
            for spacy_emb, fasttext_emb in extract_embeddings(pair.article_1.text, embedder, kind.get_filter())
        ]
        verbs_b = [
            fasttext_emb
            for spacy_emb, fasttext_emb in extract_embeddings(pair.article_2.text, embedder, kind.get_filter())
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
        correlations[key] = torch.corrcoef(torch.stack([torch.tensor(predicted_sims), torch.tensor(gold_values)]))
    print("#### Only considering", kind, "on", subset)
    for key, corrs in correlations.items():
        print(f"{key} correlation {-corrs[0, 1].item():.2f}")


@app.command(name="multitask-prompt")
def multitask_prompt():
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
            finetune_model(model, training_inputs, dev_evaluator, loss_fcn=losses.CosineSimilarityLoss)

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
            dev_evaluator.write_results(path=Path(f"multitask-prompt-{model_name_}-dev.csv"))
            test_evaluator.write_results(path=Path(f"multitask-prompt-{model_name_}-test.csv"))
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
                models.Dense(in_features=768, out_features=out_features * 7, activation_function=nn.Tanh()),
            )
            model.add_module("4", ReshapeAndNormalize(num_labels=7))
            dev_evaluator.model_name = model_name
            test_evaluator.model_name = model_name

            # eval untrained model on dev
            print("Dev set untrained:")
            dev_evaluator(model)

            # eval untrained model on test
            print("Test set untrained:")
            test_evaluator(model)

            # finetune model on train (& eval on dev)
            finetune_model(model, training_inputs, dev_evaluator, loss_fcn=CosineSimilarityLossForMultipleLabels)

            # eval finetuned model on test
            print("Test set finetuned:")
            test_evaluator(model, epoch=3)

            # print results
            print("Dev set results:")
            dev_evaluator.print_results()
            print("Test set results:")
            test_evaluator.print_results()

            # write results
            dev_evaluator.write_results(path=Path(f"multitask-head-{out_features}-dev.csv"))
            test_evaluator.write_results(path=Path(f"multitask-head-{out_features}-test.csv"))
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)


def make_multitask_head_training_data(data: List[ArticlePair]) -> List[InputExampleWithMultipleLabels]:
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


@app.command()
def main():
    result_dir = "results"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    dataset = SemEvalDataset(Path("data/train.csv"), Path("data/train_data"))
    test = SemEvalDataset(Path("data/eval.csv"), Path("data/eval_data"))
    train, dev = dataset.random_split(0.8)
    # This is to safeguard against any rng shenanigans
    assert train[0].article_2.title == "The 20 Best Places to Travel in 2020"
    assert train[1].article_2.title == "Shoplifters arrested for stealing beer and meat from Guelph grocery store"
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
            for i, (dim, training_inputs) in enumerate(zip(dimensions, training_inputs_per_dim)):
                dev_evaluator.score_column = i

                # finetune model on train (& eval on dev)
                print(f"Fine-tuning model {model_name_} for dimension {dim}")
                finetune_model(model, training_inputs, dev_evaluator, loss_fcn=losses.CosineSimilarityLoss, name=f"finetuned-{model_name_}-{dim}")

                # eval finetuned model on test
                print("Test set finetuned:")
                test_evaluator(model)

                # print results
                print(f"Dev set results fine-tuned on {dim}:")
                dev_evaluator.print_results()
                print(f"Test set results fine-tuned on {dim}:")
                test_evaluator.print_results()

                # write results
                dev_evaluator.write_results(path=Path(result_dir,f"finetuned-{model_name_}-{dim}-dev.csv"))
                test_evaluator.write_results(path=Path(result_dir, f"finetuned-{model_name_}-{dim}-test.csv"))
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)


@app.command()
def pretrained_eval_sweep():
    for datasplit in ["train", "eval"]:
        for languages in [None, ["en"]]:
            pretrained_eval(languages, datasplit)


@app.command()
def pretrained_eval(languages: Optional[List[str]] = None, datasplit: str = "train", translated: bool = False):
    dataset = SemEvalDataset(Path(f"data/{datasplit}.csv"), Path(f"data/{datasplit}_data{'_translated' if translated else ''}"), langs=languages or "all")
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
                model_name_ = model_name.split("/")[1] if "/" in model_name else model_name
                evaluator.write_results(path=Path(
                    f"pretrained-{model_name_}-{datasplit}-{'_'.join(languages) if languages is not None else 'all'}{'-translated' if translated else ''}-seq_len={model.max_seq_length}.csv")
                )
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)


def make_training_data(data: List[ArticlePair], dim_name="overall") -> List[InputExample]:
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
    evaluator: CorrelationEvaluator | MultitaskPromptCorrelationEvaluator | MultitaskHeadCorrelationEvaluator,
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
    SAME_RATING = "same_rating"
    SAME_PRODUCT = "same_product"

    def review_comparison(self):
        return {
            ReviewSimilarityDimensions.DIFFERENT_PRODUCT_SAME_RATING: lambda a, b: a.product_id != b.product_id and round(a.rating) == round(b.rating),
            ReviewSimilarityDimensions.SAME_RATING: lambda a, b: round(a.rating) == round(b.rating),
            ReviewSimilarityDimensions.SAME_PRODUCT: lambda a, b: a.product_id == b.product_id,
        }[self]


@app.command()
def reviews(dimension: ReviewSimilarityDimensions = "different_product_same_rating", split: float=1.0, sum_embeddings: bool = False):
    """
    prams:
        split: randomly only use `split` fraction of the dataset
    """
    path = Path("data/amazon_reviews/amazon_80k.txt")
    dataset = ReviewDataset(path, sample=split)
    print(f"Using {split * 100}% of the data that is {len(dataset)} reviews.")
    embeddings_per_model = dict()
    gold_label_func = dimension.review_comparison()
    for model_name in ["models/finetuned-LaBSE-tone", "models/finetuned-LaBSE-entities", "models/finetuned-LaBSE-overall", "sentence-transformers/LaBSE"]:
        model = SentenceTransformer(model_name, device="cuda:0")
        os.makedirs("data/cache/", exist_ok=True)
        review_texts = [r.review or "" for r in dataset]
        cache_path = f"data/cache/{path.stem}-{model_name.split('/')[-1]}{'-split=' + str(split) if split != 1.0 else ''}.pt"
        if not os.path.exists(cache_path):
            # We need to take the numpy version first to move the data of the gpu
            encoded = torch.from_numpy(model.encode(review_texts, show_progress_bar=True, batch_size=32))
            torch.save(encoded, cache_path)
        else:
            encoded = torch.load(cache_path, map_location="cpu")
        review_embs = {}
        map_sample = 0.1
        map_dataset = random_sample(dataset, map_sample)
        map_encoded = encoded[[r.embedding_index for r in map_dataset]]
        print(f"MAP with {model_name}:", mean_average_precision([(1.0, map_encoded)], map_dataset, gold_label=gold_label_func, sum_embeddings=sum_embeddings))
        embeddings_per_model[model_name] = encoded
    map_dataset = random_sample(dataset, map_sample)
    map_encoded = [
        (1.0, embeddings_per_model["models/finetuned-LaBSE-tone"][[r.embedding_index for r in map_dataset]]),
        (-1.0, embeddings_per_model["models/finetuned-LaBSE-overall"][[r.embedding_index for r in map_dataset]]),
    ]
    print(f"MAP with both", mean_average_precision(map_encoded, map_dataset, gold_label=gold_label_func, sum_embeddings=sum_embeddings))
    print(f"MAP random", mean_average_precision([(1.0, torch.rand_like(map_encoded[0][1]))], map_dataset, gold_label=gold_label_func, sum_embeddings=sum_embeddings))


def random_sample(dataset, frac, seed=42):
    dataset, _ = torch.utils.data.random_split(dataset, [frac, 1 - frac], generator=torch.Generator().manual_seed(seed))
    return dataset


def mean_average_precision(weighted_embeddings, dataset, gold_label=lambda x, y: round(x.rating) == round(y.rating), sum_embeddings=False):
    sims = None
    if not sum_embeddings:
        for weight, embeddings in weighted_embeddings:
            if sims is None:
                sims = cos_sim(embeddings, embeddings) * weight
            else:
                sims += cos_sim(embeddings, embeddings) * weight
    else:
        final_embeddings = torch.zeros_like(weighted_embeddings[0][1])
        for weight, embeddings in weighted_embeddings:
            final_embeddings += embeddings * weight
        sims = cos_sim(final_embeddings, final_embeddings)
    maps = []
    for i, sim_line in enumerate(sims):
        labels = [gold_label(r, dataset[i]) for r in dataset]
        ap = sklearn.metrics.average_precision_score(labels, sim_line)
        maps.append(ap)
    return sum(maps) / len(maps)


def select_all_other_embeddings(embeddings, indexes):
    mask = torch.ones(embeddings.shape[0])
    mask[indexes] = 0
    return embeddings.masked_select(
        mask.unsqueeze(-1).to(dtype=torch.bool, device=embeddings.device)
    ).reshape(-1, embeddings.shape[-1])


@app.command()
def poetry(all_combinations: bool = False, subtract_overall: bool = False):
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
        for model_name in [model_name] if not all_combinations else list(model_dict.values()):
            print("Using", model_name)
            model = SentenceTransformer(model_name)
            *texts, labels = PoetryDataset.texts_and_labels(overall_different)
            anchor_embs, left_embs, right_embs = [model.encode(collection, convert_to_tensor=True) for collection in texts]
            left_anchor_similarity = torch.nn.functional.cosine_similarity(left_embs, anchor_embs)
            right_anchor_similarity = torch.nn.functional.cosine_similarity(right_embs, anchor_embs)
            if subtract_overall:
                overall_anchor_embs, overall_left_embs, overall_right_embs = [overall_model.encode(collection, convert_to_tensor=True) for collection in texts]
                overall_left_anchor_similarity = torch.nn.functional.cosine_similarity(overall_left_embs, overall_anchor_embs)
                overall_right_anchor_similarity = torch.nn.functional.cosine_similarity(overall_right_embs, overall_anchor_embs)
            else:
                overall_left_anchor_similarity = torch.zeros_like(left_anchor_similarity)
                overall_right_anchor_similarity = torch.zeros_like(right_anchor_similarity)
            predictions = (
                left_anchor_similarity + (overall_weight * overall_left_anchor_similarity) <
                right_anchor_similarity + (overall_weight * overall_right_anchor_similarity)
            ).cpu()
            label_tensor = torch.tensor(labels, dtype=torch.bool)
            balanced_accuracy = sklearn.metrics.balanced_accuracy_score(label_tensor, predictions)
            print("Accuracy on", dimension, f"{balanced_accuracy:.02f}")


@app.command()
def sentiment():
    dataset = SentimentDataset("data/SemEval2017-task4-test/SemEval2017-task4-test.subtask-A.english.txt")
    model_dict = {
        "narrative": "models/finetuned-LaBSE-narrative",
        "style": "models/finetuned-LaBSE-style",
        "entities": "models/finetuned-LaBSE-entities",
        "tone": "models/finetuned-LaBSE-tone",
        "overall": "models/finetuned-LaBSE-overall",
        "unfinetuned": "sentence-transformers/LaBSE",
    }
    for name, model_id in model_dict.items():
        model = SentenceTransformer(model_id)
        encoded = model.encode([text for _label, text in dataset], convert_to_tensor=True, show_progress_bar=True)
        print("Using model", name)
        dist_matrix = 1 - cos_sim(encoded, encoded).cpu()
        dist_matrix.fill_diagonal_(0)
        silhouette_score = sklearn.metrics.silhouette_score(dist_matrix, [label.value for label, _text in dataset], metric="precomputed")
        print(f"Silhouette score {silhouette_score:.4f}")


if __name__ == "__main__":
    app()
