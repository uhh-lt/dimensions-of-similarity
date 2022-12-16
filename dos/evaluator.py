from sentence_transformers.evaluation import SentenceEvaluator
from typing import List
from torch.nn.functional import cosine_similarity
import torch
from dos.dataset import ArticlePair, DIMENSIONS
import pandas as pd

class CorrelationEvaluator(SentenceEvaluator):
    def __init__(self, dataset: List[ArticlePair]):
        self.article1 = [pair.article_1.text for pair in dataset]
        self.article2 = [pair.article_2.text for pair in dataset]
        self.geography = torch.tensor([pair.geography for pair in dataset])
        self.entities = torch.tensor([pair.entities for pair in dataset])
        self.time = torch.tensor([pair.time for pair in dataset])
        self.narrative = torch.tensor([pair.narrative for pair in dataset])
        self.overall = torch.tensor([pair.overall for pair in dataset])
        self.style = torch.tensor([pair.style for pair in dataset])
        self.tone = torch.tensor([pair.tone for pair in dataset])
        self.model_name = ''

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        embeddings1 = model.encode(self.article1, convert_to_tensor=True)
        embeddings2 = model.encode(self.article2, convert_to_tensor=True)
        sims = cosine_similarity(embeddings1, embeddings2, dim=1)
        sims = (1 - sims).to(self.overall.device)
        similiarities = torch.stack((
            sims,
            self.geography,
            self.entities,
            self.time,
            self.narrative,
            self.overall,
            self.style,
            self.tone,
        ))
        corrs = torch.corrcoef(similiarities)

        df = pd.DataFrame(
        corrs, columns=["predict"] + DIMENSIONS, index=["predict"] + DIMENSIONS)
        df = df[["predict", "GEO", "ENT", "TIME", "NAR", "STYLE", "TONE", "Overall"]].reindex(
            ["predict"])
        print(f"Epoch {epoch} stats for {self.model_name}")
        print(df)
        score = corrs[0,5].item()
        print(f"Epoch {epoch} score: {score}")
        return score