import gzip
from collections import Counter
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
        self.documents = self.__create_documents__(
            train_path
        ) + self.__create_documents__(test_path)
        self.id2document = {document.index: document for document in self.documents}
        self.tag2documents = {}
        for document in self.documents:
            for tag in document.tags:
                if tag not in self.tag2documents:
                    self.tag2documents[tag] = []
                self.tag2documents[tag].append(document)

    def __create_documents__(self, path: str) -> List[CoreDocument]:
        documents = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
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

    def documents_by_tags(self, tags: List[str]) -> List[CoreDocument]:
        # make sure that tags exist
        for tag in tags:
            if tag not in self.tag2documents:
                print(f"Tag {tag} not found")
                return []

        # find all documents that have one of the tags
        document_ids_with_one_tag = []
        for tag in tags:
            document_ids_with_one_tag.extend(
                [doc.index for doc in self.tag2documents[tag]]
            )

        # a document that has all tags is n times in the list (n = len(tags))
        document_ids_with_all_tags = [
            item
            for item, count in Counter(document_ids_with_one_tag).items()
            if count == len(tags)
        ]

        # map ids to documents
        return [
            self.id2document[document_id] for document_id in document_ids_with_all_tags
        ]

    def __getitem__(self, i: int) -> CoreDocument:
        return self.documents[i]

    def __len__(self) -> int:
        return len(self.documents)
