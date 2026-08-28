# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.modules.attention import AdditiveAttention


def build_entity_embeddings() -> tuple[torch.Tensor, int, dict[str, int]]:
    """
    Load pretrained MIND entity embeddings and build the entity vocab from .vec file

    The embedding matrix reserves index `0` for padding and index `1` for
    unknown entities. Unknown entities are represented using the mean of
    all known entity embeddings.

    Returns:
        embedding_weights:
            pretrained entity embedding matrix with shape `(num_entities + 2, embedding_dim)`
        embedding_dim:
            dimension of each entity embedding vector
        entity_vocab:
            mapping from MIND entity IDs to their corresponding embedding
            indices. Includes `<PAD>` at index `0` and `<UNK>` at index `1`
    """
    vec_path = Path("./data/MIND/entity_embedding.vec")
    embedding_dim = 100

    entity_vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    vectors = [
        np.zeros(embedding_dim, dtype=np.float32),  # PAD
        np.zeros(embedding_dim, dtype=np.float32),  # UNK
    ]

    with vec_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip().split("\t")

            entity_id = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)

            if vector.size != embedding_dim:
                raise ValueError(
                    f"{entity_id} has dimension {vector.size}, "
                    f"expected {embedding_dim}"
                )

            entity_vocab[entity_id] = len(vectors)
            vectors.append(vector)

    embedding_matrix = np.stack(vectors)

    # UNK = mean of all known entity embeddings
    embedding_matrix[1] = embedding_matrix[2:].mean(axis=0)

    return (
        torch.from_numpy(embedding_matrix),
        embedding_dim,
        entity_vocab,
    )


class EntityEncoder(nn.Module):
    """
    Encode a sequence of news entities into a single dense representation.

    The encoder maps entity IDs to pretrained entity embeddings, projects them
    into `d_model` and aggregates the entity sequence using additive attention pooling.

    Args:
        d_model:
            dimension of the projected entity representations and final encoded
            entity representation
        pool_hidden_dim:
            hidden dimension used by the additive attention pooling network
            to compute entity importance scores
        dropout:
            dropout probability applied to the projected entity representations
        freeze_pretrained_embedding:
            whether to freeze the pretrained entity embedding weights during
            training
    """

    def __init__(
        self,
        embedding_dim: int,
        embedding_weights: torch.Tensor | None,
        d_model: int,
        pool_hidden_dim: int,
        dropout: float = 0.1,
        freeze_pretrained_embedding: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_weights,
            freeze=freeze_pretrained_embedding,
            padding_idx=0,
        )
        self.projection = nn.Linear(embedding_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pooling = AdditiveAttention(d_model, pool_hidden_dim)

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        # entity_ids: (batch_size, entity_len)

        valid_mask = entity_ids != 0

        assert valid_mask.any(dim=1).all(), "found news with no valid entity"

        x = self.embedding(entity_ids)
        # (batch_size, entity_len) -> (batch_size, entity_len, embedding_dim)

        x = self.projection(x)
        # (batch_size, entity_len, embedding_dim) -> (batch_size, entity_len, d_model)

        x = self.dropout(x)
        x = self.pooling(x, valid_mask)
        # (batch_size, entity_len, d_model) -> (batch_size, d_model)

        return x
