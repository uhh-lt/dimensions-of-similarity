import itertools
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import more_itertools
from torch.utils.data.dataset import Dataset


@dataclass
class Review:
    rating: float
    product_id: str
    helpfulness: Tuple[Union[int, None], int]
    id: str
    embedding_index: int
    review_by: str
    title: str
    review_time: str
    review: str

    @classmethod
    def from_lines(cls, lines: List[str], embedding_index: int) -> "Review":
        kwargs = {
            k.lower().strip(): v.strip() for k, v in (l.split(": ", 1) for l in lines)
        }
        return cls(
            rating=float(kwargs["rating"].split(" ", 1)[0]),
            product_id=kwargs["product_id"],
            helpfulness=tuple(
                int(x.replace(",", "")) if x != "out" else None
                for x in kwargs["helpfulness"].split("/", 1)
            ),
            id=kwargs["id"],
            embedding_index=embedding_index,
            review_by=kwargs["review_by"],
            title=kwargs["title"],
            review_time=kwargs["review_time"],
            review=kwargs["review"],
        )


class ReviewDataset(Dataset):
    def __init__(self, path: str, sample: float = 1.0, seed=42):
        num_fields = 8
        self.reviews: List[Review] = []
        in_file = open(path)
        iterator = more_itertools.windowed(
            (line for line in in_file if len(line.strip()) > 0),
            num_fields,
            step=num_fields,
        )
        i = 0
        randomizer = random.Random(seed)
        for entry in iterator:
            if randomizer.random() > sample:
                continue
            self.reviews.append(Review.from_lines(entry, i))
            i += 1

    def description(self):
        return "\n".join(
            (
                f"Number of reviews: {len(self)}",
                f"Number of unique products: {len(list(self.grouped_by_product()))}",
                f"Rating occurences:",
                "\n".join(
                    [f"\t{k}: {len(list(v))}" for k, v in self.grouped_by_rating()]
                ),
            )
        )

    def grouped_by_product(self):
        return self.grouped_by_attr("product_id")

    def grouped_by_rating(self):
        return self.grouped_by_attr("rating", lambda x: round(x))

    def grouped_by_attr(self, attr, round_function=lambda x: x):
        keyfunc = lambda r: round_function(getattr(r, attr))
        return itertools.groupby(sorted(self.reviews, key=keyfunc), key=keyfunc)

    def __getitem__(self, i: int) -> Review:
        return self.reviews[i]

    def __len__(self) -> int:
        return len(self.reviews)
