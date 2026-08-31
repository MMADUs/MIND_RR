# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

from src.modules.attention import TransformerBlock, AdditiveAttention


def get_hf_tokenizer_embeddings(
    model_name: str = "bert-base-uncased",
) -> tuple[torch.Tensor, AutoTokenizer]:
    """
    Load a pretrained BERT tokenizer and token embedding matrix

    Args:
        model_name:
            hugging Face model name containing the pretrained tokenizer
            and embedding weights

    Returns:
        embedding_weights:
            pretrained token embedding matrix with shape `(vocab_size, embedding_dim)`
        embedding_dim:
            dimension of each token embedding vector
        tokenizer:
            tokenizer corresponding to the pretrained embedding vocabulary
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)

    embedding_weights = model.get_input_embeddings().weight.detach().clone()

    embedding_dim = embedding_weights.shape[1]

    return embedding_weights, embedding_dim, tokenizer


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
        num_layers:
            number of repeated transformer block
        d_ff:
            dimension of the feed-forward network followed after MHA
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
        num_heads: int,
        num_layers: int,
        d_ff: int,
        pool_hidden_dim: int,
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

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.pooling = AdditiveAttention(d_model, pool_hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch_size, seq_len)

        valid_mask = token_ids != 0
        # valid_mask: (batch_size, seq_len)

        x = self.embedding(token_ids)
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)

        x = self.input_projection(x)
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, d_model)

        attn_mask = valid_mask[:, None, None, :]
        # attn_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)

        for layer in self.layers:
            x = layer(x, attn_mask)

        x = self.norm(x)
        x = self.pooling(x, valid_mask)
        # (batch_size, seq_len, d_model) -> (batch_size, d_model)

        return x
