from torch.utils.data.dataset import Dataset
from dataclasses import dataclass
import json
import csv
from typing import Dict, Union, Optional
from pathlib import Path
import glob

DIMENSIONS = ["GEO", "ENT", "TIME", "NAR", "Overall", "STYLE", "TONE"]
DIMENSIONS_LONG = ["Geography", "Entities", "Time", "Narrative", "Overall", "Style", "Tone"]


@dataclass
class Article():
    text: str
    title: str
    publish_date: str


@dataclass
class ArticlePair():
    url1_lang: str
    url2_lang: str
    pair_id: str
    link_1: str
    link_2: str
    ia_link_1: str
    ia_link_2: str
    geography: float
    entities: float
    time: float
    narrative: float
    overall: float
    style: float
    tone: float
    article_1: Optional[Article]
    article_2: Optional[Article]

    @property
    def id_1(self) -> str:
        return self.pair_id.split("_")[0]

    @property
    def id_2(self) -> str:
        return self.pair_id.split("_")[1]


class SemEvalDataset(Dataset):
    def __init__(self, csv_path: Path, json_path: Path):
        self.json_path = json_path
        self.article_pairs = list(self.load_csv(csv_path))

    def load_csv(self, csv_path):
        reader = csv.DictReader(open(csv_path))
        for data in reader:
            converted: Dict[str, Union[float, str]] = {
                    k.lower() : float(v)
                    if k in DIMENSIONS_LONG or k in DIMENSIONS
                    else v
                    for k, v in data.items()
            }
            for old_name, new_name in [
                    ("link1", "link_1"),
                    ("link2", "link_2"),
                    ("ia_link1", "ia_link_1"),
                    ("ia_link2", "ia_link_2")
                    ]:
                converted[new_name] = converted[old_name]
                del converted[old_name]
            pair = ArticlePair(**converted, article_1=None, article_2=None)
            try:
                pair.article_1 = self.get_article_by_id(pair.id_1)
                pair.article_2 = self.get_article_by_id(pair.id_2)
                yield pair
            except ValueError as e:
                print("Error loading article", e)

    def get_article_by_id(self, article_id: str) -> Article:
        file_list = glob.glob(str(self.json_path / "*" / (article_id + ".json")))
        if len(file_list) > 1:
            raise ValueError(f"More than one article with id {article_id}")
        elif len(file_list) == 0:
            raise ValueError(f"No articles with id {article_id}")
        else:
            data = json.load(open(file_list[0]))
            return Article(text=data["text"], title=data["title"], publish_date=data["publish_date"])

    def __getitem__(self, i: ArticlePair):
        return self.article_pairs[i]


def main():
    total = 0
    nar_total_different = 0
    for pair in SemEvalDataset(Path("data/train.csv"), Path("data/train_data")):
        if abs(pair.narrative - pair.overall) >= 2:
            print(f"======= NAR {pair.narrative} - Overall {pair.overall} =========")
            print(pair.url1_lang)
            print(pair.url2_lang)
            print(pair.ia_link_1)
            print(pair.ia_link_2)
            nar_total_different += 1
        total += 1
    print(f"{(nar_total_different / total) * 100:.3f}% of pair have NAR/OVERALL Δ >= 2, that's", nar_total_different, "pairs")


if __name__ == "__main__":
    main()
