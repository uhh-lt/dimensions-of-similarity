from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data.dataset import Dataset


@dataclass
class CoreDocument:
    tags: List[str]
    index: int
    text: str

    @classmethod
    def from_line(cls, line: str) -> Optional["CoreDocument"]:
        tokens = line.split("\t")
        if len(tokens) != 3:
            return None
        tags = tokens[0].split(" ")
        index = int(tokens[1])
        text = tokens[2]
        return cls(tags=tags, index=index, text=text)


class CoreDataset(Dataset):
    def __init__(self, train_path: str, test_path: str):
        self.documents = self.__create_documents__(train_path) + self.__create_documents__(test_path)
        self.id2document = {document.index: document for document in self.documents}
        self.tag2documents = {}
        for document in self.documents:
            for tag in document.tags:
                if tag not in self.tag2documents:
                    self.tag2documents[tag] = []
                self.tag2documents[tag].append(document)

    def __create_documents__(self, path: str) -> List[CoreDocument]:
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                document = CoreDocument.from_line(line)
                if document is None:
                    print(f"Invalid line: {line}")
                    continue
                documents.append(document)
        return documents

    def documents_by_tag(self, tag: str) -> List[CoreDocument]:
        if tag not in self.tag2documents:
            print(f"Tag {tag} not found")
            return []
        return self.tag2documents[tag]

    def __getitem__(self, i: int) -> CoreDocument:
        return self.documents[i]

    def __len__(self) -> int:
        return len(self.documents)
