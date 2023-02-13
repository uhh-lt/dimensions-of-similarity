from torch.utils.data.dataset import Dataset, Subset
from dataclasses import dataclass
import itertools
from collections import Counter
from typing import Tuple, List, Optional, Union, Dict
import more_itertools
from pathlib import Path
import enum
import csv


class Direction(enum.Enum):
    LEFT = 0
    RIGHT = 1
    SAME = 2

    @classmethod
    def from_str(cls, direction_str) -> "Direction":
        if direction_str == "left":
            return cls.LEFT
        elif direction_str == "right":
            return cls.RIGHT
        elif direction_str == "same":
            return cls.SAME
        else:
            raise ValueError("Invalid direction", direction_str)


@dataclass
class PoetryPair():
    annotator: str
    base_title: str
    base_id: str
    left_id: str
    right_id: str

    base_text: str
    right_text: str
    left_text: str

    form: Direction
    content: Direction
    emotion: Direction
    style: Direction
    overall: Direction

    def dimensions(self) -> Dict[str, Direction]:
        return {
            "form": self.form,
            "content": self.form,
            "emotion": self.emotion,
            "style": self.style,
            "overall": self.overall,
        }



class PoetryDataset(Dataset):
    dimensions = ["form", "content", "emotion", "style", "overall"]

    def __init__(self, path: str, num: Optional[int] = None):
        raw_pairs = []
        reader = csv.DictReader(open(Path(path) / "raw_annotations.tsv"), delimiter="\t")
        for i, entry in enumerate(reader):
            if num is not None and i >= num:
                break
            pair = PoetryPair(
                annotator=entry["annotator"],
                base_title=entry["title"],
                base_id=entry["base_ID"],
                left_id=entry["left_ID"],
                right_id=entry["right_ID"],
                base_text=PoetryDataset.get_text(path, entry["base_ID"]),
                right_text=PoetryDataset.get_text(path, entry["right_ID"]),
                left_text=PoetryDataset.get_text(path, entry["left_ID"]),
                form=Direction.from_str(entry["Form"]),
                content=Direction.from_str(entry["Content"]),
                emotion=Direction.from_str(entry["Emotion"]),
                style=Direction.from_str(entry["Style"]),
                overall=Direction.from_str(entry["Overall"]),
            )
            raw_pairs.append(pair)
        keyfunc = lambda pair: pair.base_id + pair.left_id + pair.right_id
        grouped = itertools.groupby(sorted(raw_pairs, key=keyfunc), key=keyfunc)
        self.pairs = []
        for triple_id, group in grouped:
            annos = list(group)
            new_anno = annos[0]
            new_anno.annotator = " ".join([a.annotator for a in annos])
            for dim in PoetryDataset.dimensions:
                directions = Counter([getattr(anno, dim) for anno in annos])
                common = directions.most_common(2)
                if len(common) == 1:
                    # easy all agree
                    setattr(new_anno, dim, common[0][0])
                elif common[0][1] > common[1][1]:
                    # still a majority
                    setattr(new_anno, dim, common[0][0])
                else:
                    # no majority: just say same
                    setattr(new_anno, dim, Direction.SAME)
            self.pairs.append(new_anno)

    @staticmethod
    def get_text(basepath, text_id):
        return "".join(open(Path(basepath) / "corpus" / (text_id + ".txt")).readlines())

    def __getitem__(self, i: int) -> PoetryPair:
        return self.pairs[i]

    def __len__(self) -> int:
        return len(self.pairs)

    def with_unambigious_dimension(self, dimension) -> Subset:
        return Subset(self, [i for i in range(len(self)) if getattr(self[i], dimension) != Direction.SAME])

    @classmethod
    def texts_and_labels(cls, instance):
        anchors = []
        lefts = []
        rights = []
        labels = []
        for doc in instance:
            rights.append(doc.right_text)
            lefts.append(doc.left_text)
            anchors.append(doc.base_text)
            labels.append(doc.overall.value)
        return anchors, lefts, rights, labels



if __name__ == "__main__":
    ds = PoetryDataset("data/jcls2022-poem-similarity/", 200)
    print(len(ds))
    print(ds[21])
