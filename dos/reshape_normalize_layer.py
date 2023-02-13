from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn


class ReshapeAndNormalize(nn.Module):
    """
    This layer normalizes embeddings to unit length
    """

    def __init__(self, num_labels: int):
        super(ReshapeAndNormalize, self).__init__()
        self.num_labels = num_labels

    def forward(self, features: Dict[str, Tensor]):
        batch_size, _ = features["sentence_embedding"].shape
        features.update(
            {
                "sentence_embedding": features["sentence_embedding"].reshape(
                    batch_size, self.num_labels, -1
                )
            }
        )
        features.update(
            {
                "sentence_embedding": F.normalize(
                    features["sentence_embedding"], p=2, dim=2
                )
            }
        )
        return features

    def save(self, output_path):
        pass

    @staticmethod
    def load(input_path):
        return ReshapeAndNormalize()
