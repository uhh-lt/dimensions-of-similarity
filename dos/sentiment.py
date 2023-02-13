from pathlib import Path
from torch.utils.data.dataset import Dataset
from enum import Enum


class Sentiment(Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class SentimentDataset(Dataset):
    def __init__(self, path, *args, **kwargs):
        path = Path(path)
        self.sentences = []
        self.labels = []
        data_file = open(path)
        for line in data_file:
            _id, sentiment, tweet = line.split("\t")
            self.sentences.append(tweet)
            self.labels.append(Sentiment(sentiment))
        super(*args, **kwargs)

    def __getitem__(self, i):
        return self.labels[i], self.sentences[i]

    def __len__(self):
        return len(self.labels)


if __name__  == "__main__":
    for label, sent in SentimentDataset("data/SemEval2017-task4-test/SemEval2017-task4-test.subtask-A.english.txt"):
        print(label, sent)