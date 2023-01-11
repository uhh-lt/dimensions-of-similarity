from typing import List

import pandas as pd
import torch
from sentence_transformers.evaluation import SentenceEvaluator
from torch.nn.functional import cosine_similarity

from dos.dataset import ArticlePair, DIMENSIONS


class MultitaskCorrelationEvaluator(SentenceEvaluator):

    def __init__(self, dataset: List[ArticlePair]):
        self.dimensions = ['geography', 'entities', 'time', 'narrative', 'overall', 'style', 'tone']
        self.article1 = {dimension: [] for dimension in self.dimensions}
        self.article2 = {dimension: [] for dimension in self.dimensions}
        for pair in dataset:
            for dimension in self.dimensions:
                self.article1[dimension].append(f"{dimension}: {pair.article_1.text}")
                self.article2[dimension].append(f"{dimension}: {pair.article_2.text}")

        self.geography = torch.tensor([pair.geography for pair in dataset])
        self.entities = torch.tensor([pair.entities for pair in dataset])
        self.time = torch.tensor([pair.time for pair in dataset])
        self.narrative = torch.tensor([pair.narrative for pair in dataset])
        self.overall = torch.tensor([pair.overall for pair in dataset])
        self.style = torch.tensor([pair.style for pair in dataset])
        self.tone = torch.tensor([pair.tone for pair in dataset])
        self.model_name = ''

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        final_score = 0
        for dimension in self.dimensions:
            embeddings1 = model.encode(self.article1[dimension], convert_to_tensor=True)
            embeddings2 = model.encode(self.article2[dimension], convert_to_tensor=True)
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
            print(f"Epoch {epoch} stats for {self.model_name} dimension {dimension}")
            print(df)
            score = corrs[0, 5].item()
            final_score += score

        final_score = final_score / len(self.dimensions)
        print(f"Epoch {epoch} score: {final_score}")

        return final_score