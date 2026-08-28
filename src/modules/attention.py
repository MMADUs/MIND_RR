# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention Block

    Args:
        d_model:
            dimension of the model (embedding size)
        h:
            number of attention heads
        dropout:
            dropout rate
    """

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()

        # make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_model = d_model  # embedding vector size
        self.h = h  # number of heads
        self.d_k = d_model // h  # dimension of vector seen by each head

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq (query)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk (key)
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv (value)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo (output)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout: nn.Dropout,
        mask=None,
    ):
        # qkv: (batch, h, seq_len, d_k)
        # mask: (batch_size, seq_len)
        d_k = q.shape[-1]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # attention score
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)

        if mask is not None:
            # expand key mask dim to match scores dim
            key_mask = mask[:, None, None, :]
            # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)

            # mark with -inf to the position where mask == False
            scores.masked_fill_(~key_mask, float("-inf"))

        attn_w = F.softmax(scores, dim=-1)  # compute proba with softmax
        # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attn_w = dropout(attn_w)

        # weighted sum of values
        out = attn_w @ v
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)

        if mask is not None:
            # expand query mask to match output dimensions
            query_mask = mask[:, None, :, None]
            # (batch, seq_len) -> (batch, 1, seq_len, 1)

            # zero output vectors for padded query positions
            out = out.masked_fill(~query_mask, 0.0)

        return out, attn_w

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.size()

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        # qkv: (batch, seq_len, d_model)

        q = q.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)
        # qkv: (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)

        out, _attn_w = self.attention(q, k, v, self.dropout, mask)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)

        return self.w_o(out)


class AdditiveAttention(nn.Module):
    """
    Attention Pooling Module

    Transform sequence of vector dim into single-representation vector

    Args:
        input_dim:
            input dimension, typically the model dimension `d_model`
        hidden_dim:
            hidden dimension to compute query scores, larger value
            increases the attention capacity
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.projection = nn.Linear(input_dim, hidden_dim)
        self.w_q = nn.Linear(hidden_dim, 1, bias=False)  # Wq (query)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (batch, seq_len, input_dim)

        hidden = self.projection(x)
        hidden = torch.tanh(hidden)
        # (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim)

        scores = self.w_q(hidden).squeeze(-1)
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, 1) -> (batch, seq_len)

        if mask is not None:
            # masking operation
            # mark with -inf to the position where mask == False
            scores.masked_fill_(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        # (batch, seq_len)

        weights = weights.unsqueeze(-1)
        # (batch, seq_len) -> (batch, seq_len, 1)

        output = torch.sum(x * weights, dim=1)
        # (batch, seq_len, input_dim) * (batch, seq_len, 1) = (batch, seq_len, input_dim)
        # sum dim=1 : (batch, seq_len, input_dim) -> (batch, input_dim)

        return output
