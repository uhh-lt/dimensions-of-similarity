import torch
from typing import Dict, Iterable
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer


class CosineSimilarityLossForMultipleLabels(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss_fct=nn.MSELoss(),
        cos_score_transformation=nn.Identity(),
        num_labels=7,
    ):
        super(CosineSimilarityLossForMultipleLabels, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.num_labels = num_labels

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        output = torch.stack(
            [
                self.cos_score_transformation(
                    torch.cosine_similarity(
                        embeddings[0][:, dim, :], embeddings[1][:, dim, :]
                    )
                )
                for dim in range(self.num_labels)
            ]
        ).T
        return self.loss_fct(output, labels)
