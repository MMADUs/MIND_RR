# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

from src.modules.attention import MultiHeadAttention, AdditiveAttention


def get_hf_tokenizer_embeddings(
    model_name: str = "bert-base-uncased",
) -> tuple[torch.Tensor, AutoTokenizer]:
    """
    Load a pretrained BERT tokenizer and token embedding matrix

    Args:
        model_name:
            hugging Face model name containing the pretrained tokenizer
            and embedding weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)

    embedding_weights = (
        model.get_input_embeddings()
        .weight
        .detach()
        .clone()
    )

    return embedding_weights, tokenizer


class TextEncoder(nn.Module):
    """
    Encode a token sequence into a single dense text representation

    The encoder projects token embeddings into `d_model`, contextualizes them
    with multi-head self-attention and aggregates the sequence using additive attention pooling

    Args:
        vocab_size:
            number of tokens in the vocabulary
        embedding_dim:
            dimension of the input token embeddings
        embedding_weights:
            optional pretrained embedding matrix. If `None`, embeddings are
            initialized randomly
        d_model:
            dimension of the internal token representations and final encoded
            text representation
        num_heads:
            number of attention heads used by multi-head self-attention
        pool_hidden_dim:
            hidden dimension used by the additive attention pooling network
            to compute token importance scores
        dropout:
            dropout probability applied to the self-attention output
        freeze_pretrained_embedding:
            whether to freeze pretrained embedding weights during training
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        embedding_weights: torch.Tensor | None,
        d_model: int,
        num_heads: int = 4,
        pool_hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_pretrained_embedding: bool = True,
    ):
        super().__init__()

        if embedding_weights is None:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=embedding_weights,
                freeze=freeze_pretrained_embedding,
                padding_idx=0,
            )

        self.input_projection = nn.Linear(embedding_dim, d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pooling = AdditiveAttention(d_model, pool_hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch_size, seq_len)

        valid_mask = token_ids != 0

        assert valid_mask.any(dim=1).all(), "found a fully padded sequence"

        x = self.embedding(token_ids)
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)

        x = self.input_projection(x)
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, d_model)

        attn_out = self.mha(x, valid_mask)

        x = self.norm(x + self.dropout(attn_out))

        x = self.pooling(x, valid_mask)
        # (batch_size, seq_len, d_model) -> (batch_size, d_model)

        return x
