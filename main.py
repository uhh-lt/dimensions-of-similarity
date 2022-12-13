from pathlib import Path
from typing import List
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from dos.dataset import SemEvalDataset, DIMENSIONS
from scipy.stats import pearsonr

from torch.nn.functional import cosine_similarity

pd.set_option('display.precision',2)

def main():
    model = SentenceTransformer("sentence-transformers/LaBSE") # 'all-MiniLM-L6-v2')
    sims = []
    gold_geography: List[float] = []
    gold_entities: List[float] = []
    gold_time: List[float] = []
    gold_narrative: List[float] = []
    gold_overall: List[float] = []
    gold_style: List[float] = []
    gold_tone: List[float] = []

    dataset = SemEvalDataset(Path("data/eval.csv"), Path("data/eval_data"))
    for pair in dataset:
        embeddings = model.encode([pair.article_1.text, pair.article_2.text], convert_to_tensor=True)
        sim = cosine_similarity(embeddings[0], embeddings[1], dim=0)
        sims.append(1 - sim.item())
        gold_geography.append(pair.geography)
        gold_entities.append(pair.entities)
        gold_time.append(pair.time)
        gold_narrative.append(pair.narrative)
        gold_overall.append(pair.overall)
        gold_style.append(pair.style)
        gold_tone.append(pair.tone)
    similiarities = torch.stack((
        torch.tensor(sims),
        torch.tensor(gold_geography),
        torch.tensor(gold_entities),
        torch.tensor(gold_time),
        torch.tensor(gold_narrative),
        torch.tensor(gold_overall),
        torch.tensor(gold_style),
        torch.tensor(gold_tone),
    ))
    print("Scipy stats", pearsonr(gold_time, gold_geography))
    print(similiarities.shape)
    corrs = torch.corrcoef(similiarities)
    df = pd.DataFrame(corrs, columns=["predict"] + DIMENSIONS, index=["predict"] + DIMENSIONS)
    df = df[["predict", "GEO", "ENT", "TIME", "NAR", "STYLE", "TONE", "Overall"]].reindex(["predict", "GEO", "ENT", "TIME", "NAR", "STYLE", "TONE", "Overall"])
    print(df)

        


if __name__ == "__main__":
    main()
