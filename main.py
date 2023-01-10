from pathlib import Path
from typing import List
import pandas as pd
from itertools import count
import typer
from typing import Optional
import fasttext
import torch
import spacy
import numpy as np
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer, InputExample, losses
from dos.evaluator import CorrelationEvaluator
from dos.dataset import SemEvalDataset, ArticlePair
from torch.utils.data import DataLoader

pd.set_option('display.precision', 2)

models = ["bert-base-multilingual-cased", "sentence-transformers/stsb-xlm-r-multilingual",
          "sentence-transformers/LaBSE", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]

app = typer.Typer()
nlp = spacy.load("en_core_web_lg")


def extract_embeddings(text, embedder, only_verbs=True):
    doc = nlp(text)
    for token in doc:
        if not only_verbs or token.pos_ == "VERB":
            yield token.vector, embedder.get_word_vector(token.text)


@app.command()
def verbs(limit: Optional[int] = None, split: str = "test"):
    current_dataset = None
    if split == "test":
        test = SemEvalDataset(Path("data/eval.csv"), Path("data/eval_data"))
        current_dataset = test
    elif split == "train":
        train = SemEvalDataset(Path("data/train.csv"), Path("data/train_data"))
        current_dataset = train
    else:
        raise ValueError("Invalid split")
    embedder = fasttext.load_model("cc.en.300.bin")
    predicted_sims = []
    gold_sims_nar = []
    gold_sims_ent = []
    gold_sims = []
    for only_verbs in [True, False]:
        for pair, i in zip(current_dataset, range(limit) if limit is not None else count()):
            verbs_a = [fasttext_emb for spacy_emb, fasttext_emb in extract_embeddings(pair.article_1.text, embedder, only_verbs=only_verbs)]
            verbs_b = [fasttext_emb for spacy_emb, fasttext_emb in extract_embeddings(pair.article_2.text, embedder, only_verbs=only_verbs)]
            if len(verbs_a) == 0 or len(verbs_b) == 0:
                continue
            verbs_a_stacked = np.stack(verbs_a)
            verbs_b_stacked = np.stack(verbs_b)
            sim = cos_sim(verbs_a_stacked, verbs_b_stacked)
            sim.fill_diagonal_(0)
            predicted_sims.append(sim.max(0).values.mean() + sim.max(1).values.mean())
            # predicted_sims.append(sim.mean())
            gold_sims_ent.append(pair.entities)
            gold_sims_nar.append(pair.narrative)
            gold_sims.append(pair.overall)
        corrs = torch.corrcoef(torch.stack([torch.tensor(predicted_sims), torch.tensor(gold_sims)]))
        corrs_narr = torch.corrcoef(torch.stack([torch.tensor(predicted_sims), torch.tensor(gold_sims_nar)]))
        corrs_ent = torch.corrcoef(torch.stack([torch.tensor(predicted_sims), torch.tensor(gold_sims_ent)]))
        print("#### ", "Only Verbs" if only_verbs else "All Words")
        print("Overall correlation", corrs[0,1])
        print("Narrative correlation", corrs_narr[0,1])
        print("Entity correlation", corrs_ent[0,1])

@app.command()
def main():
    dataset = SemEvalDataset(Path("data/train.csv"), Path("data/train_data"))
    train, dev = dataset.random_split(0.8)
    training_inputs = make_training_data(train)
    evaluator = CorrelationEvaluator(dev)
    for model_name in models:
        try:
            model = SentenceTransformer(model_name)
            evaluator.model_name = model_name
            finetune_model(model, training_inputs, evaluator)
        except Exception as e:
            print("Error evaluating", model_name)
            print(e)

def normalize_score(one2four: float):
    return 1 - 2 * (one2four - 1) / 3

def make_training_data(data: List[ArticlePair]) -> List[InputExample]:
    inputs: List[InputExample] = [InputExample(
        texts=[pair.article_1.text, pair.article_2.text], label=normalize_score(pair.overall)) for pair in data]
    return inputs

def finetune_model(model: SentenceTransformer, inputs: List[InputExample], evaluator: CorrelationEvaluator):
    dataloader = DataLoader(inputs, shuffle=True, batch_size=16)
    loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(dataloader, loss)], epochs=3, warmup_steps=100, evaluator=evaluator)


if __name__ == "__main__":
    app()
