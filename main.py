from pathlib import Path
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from dos.evaluator import CorrelationEvaluator
from dos.dataset import SemEvalDataset, ArticlePair
from torch.utils.data import DataLoader

pd.set_option('display.precision', 2)

models = ["bert-base-multilingual-cased", "sentence-transformers/stsb-xlm-r-multilingual",
          "sentence-transformers/LaBSE", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]


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
    main()
