from torch.utils.data.dataset import Dataset
from dataclasses import dataclass
import json
import csv
from typing import Tuple, List, Dict, Union, Optional
from pathlib import Path
import glob
import random

DIMENSIONS = ["GEO", "ENT", "TIME", "NAR", "Overall", "STYLE", "TONE"]
DIMENSIONS_LONG = ["Geography", "Entities", "Time", "Narrative", "Overall", "Style", "Tone"]


@dataclass
class Article():
    body: str
    title: str
    publish_date: str

    @property
    def text(self) -> str:
        return ' '.join((self.title, self.body))


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
    def __init__(self, csv_path: Path, json_path: Path, langs: Union[List[str], str] = "all"):
        self.json_path = json_path
        self.langs = langs
        self.article_pairs = list(self.load_csv(csv_path))

    def load_csv(self, csv_path):
        reader = csv.DictReader(open(csv_path, encoding="utf-8"))
        i = 0
        for data in reader:
            # if i > 200:
            #     break
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
            if self.langs != "all" and (pair.url1_lang not in self.langs or pair.url2_lang not in self.langs):
                continue
            try:
                pair.article_1 = self.get_article_by_id(pair.id_1)
                pair.article_2 = self.get_article_by_id(pair.id_2)
                i += 1
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
            data = json.load(open(file_list[0], encoding="utf-8"))
            return Article(body=data["text"], title=data["title"], publish_date=data["publish_date"])

    def __getitem__(self, i: int) -> ArticlePair:
        return self.article_pairs[i]

    def __len__(self) -> int:
        return len(self.article_pairs)

    def random_split(self, percent: float) -> Tuple[List[ArticlePair], List[ArticlePair]]:
        pairs = self.article_pairs.copy()
        randomizer = random.Random(42)
        randomizer.shuffle(pairs)
        split = int(percent * len(pairs))
        return pairs[:split], pairs[split:]


def main():
    total = 0
    nar_total_different = 0
    for pair in SemEvalDataset(Path("data/all.csv"), Path("data/all_data")):
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
