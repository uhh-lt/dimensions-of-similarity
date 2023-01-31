from torch.utils.data.dataset import Dataset
from dataclasses import dataclass
import itertools
from typing import Tuple, List, Optional, Union
import more_itertools


@dataclass
class Review():
    rating: float
    product_id: str
    helpfulness: Tuple[Union[int, None], int]
    id: str
    review_by: str
    title: str
    review_time: str
    review: str

    @classmethod
    def from_lines(cls, lines: List[str]) -> "Review":
        kwargs = {k.lower().strip(): v.strip() for k, v in (l.split(": ", 1) for l in lines)}
        return cls(
            rating=float(kwargs["rating"].split(" ", 1)[0]),
            product_id=kwargs["product_id"],
            helpfulness=tuple(int(x.replace(",", "")) if x != "out" else None for x in kwargs["helpfulness"].split("/", 1)),
            id=kwargs["id"],
            review_by=kwargs["review_by"],
            title=kwargs["title"],
            review_time=kwargs["review_time"],
            review=kwargs["review"],
        )


class ReviewDataset(Dataset):
    def __init__(self, path: str, num: Optional[int] = None):
        num_fields = 8
        self.reviews: List[Review] = []
        in_file = open(path)
        iterator = more_itertools.windowed((line for line in in_file if len(line.strip()) > 0), num_fields, step=num_fields)
        i = 0
        for i, entry in enumerate(iterator):
            if num is not None and i >= num:
                break
            self.reviews.append(Review.from_lines(entry))

    def grouped_by_product(self):
        keyfunc = lambda r: r.product_id
        return itertools.groupby(sorted(self.reviews, key=keyfunc), key=keyfunc)

    def __getitem__(self, i: int) -> Review:
        return self.reviews[i]

    def __len__(self) -> int:
        return len(self.reviews)