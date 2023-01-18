from typing import List, Union


class InputExampleWithMultipleLabels:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(
        self,
        guid: str = "",
        texts: List[str] = None,
        label: List[Union[int, float]] = 0,
    ):
        """
        Creates one InputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> labels: {}, texts: {}".format(
            "; ".join(self.label), "; ".join(self.texts)
        )
