# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionBias(nn.Module):
    """
    learnable relative position bias for sequential history

    Args:
        max_distance:
            maximum relative distance represented by a distinct bias value
    """

    def __init__(self, max_distance: int):
        super().__init__()

        self.max_distance = max_distance

        self.bias = nn.Embedding(
            num_embeddings=max_distance + 1,
            embedding_dim=1,
        )
        nn.init.zeros_(self.bias.weight)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        position = torch.arange(seq_len, device=device)

        relative_position = position[:, None] - position[None, :]
        relative_position = relative_position.clamp(min=0, max=self.max_distance)

        bias = self.bias(relative_position).squeeze(-1)

        return bias[None, None, :, :]


class HSTUBlock(nn.Module):
    """
    Hierarchical Sequential Transduction Unit (HSTU) block for encoding
    sequential user history

    reference:
        HSTU paper: https://arxiv.org/abs/2402.17152
        meta-recsys GRs: https://github.com/meta-recsys/generative-recommenders

    Args:
        d_model:
            dimension of the input and output representations
        num_heads:
            number of attention heads
        qk_dim:
            dimension of the query and key vectors for each attention head
        value_dim:
            dimension of the value vector for each attention head
        max_distance:
            maximum relative position distance represented by the
            positional bias
        dropout:
            dropout probability applied to the block output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qk_dim: int,
        value_dim: int,
        max_distance: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.value_dim = value_dim

        self.uv_dim = num_heads * value_dim
        self.qk_total_dim = num_heads * qk_dim

        self.input_norm = nn.LayerNorm(d_model)

        # HSTU single projection from X to U, Q, K, V
        # U: learned gating representation
        # Q: query
        # K: key
        # V: value
        self.uvqk = nn.Linear(
            d_model,
            2 * self.uv_dim + 2 * self.qk_total_dim,
        )

        self.relative_bias = RelativePositionBias(max_distance=max_distance)
        self.attention_norm = nn.LayerNorm(self.uv_dim)

        self.output_projection = nn.Linear(
            self.uv_dim,
            d_model,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch, seq_len, d_model)

        batch_size, seq_len = x.shape[:2]

        residual = x

        x = self.input_norm(x)

        uvqk = self.uvqk(x)
        # (batch, seq_len, d_model) -> (batch, seq_len, 2 * head * Dv + 2 * head * Dqk)

        u, v, q, k = torch.split(
            uvqk,
            [self.uv_dim, self.uv_dim, self.qk_total_dim, self.qk_total_dim],
            dim=-1,
        )
        # u: (batch, seq_len, head * Dv)
        # v: (batch, seq_len, head * Dv)
        # q: (batch, seq_len, head * Dqk)
        # k: (batch, seq_len, head * Dqk)

        u = F.silu(u)  # U is the HSTU gate

        # split qkv into heads
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_dim).transpose(
            1, 2
        )  # (batch, seq_len, head * Dqk) -> (batch, seq_len, head, Dqk)
        k = k.view(batch_size, seq_len, self.num_heads, self.qk_dim).transpose(
            1, 2
        )  # (batch, seq_len, head * Dqk) -> (batch, seq_len, head, Dqk)
        v = v.view(batch_size, seq_len, self.num_heads, self.value_dim).transpose(
            1, 2
        )  # (batch, seq_len, head * Dqk) -> (batch, seq_len, head, Dqk)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.qk_dim)
        # (batch, head, seq_len, Dqk) @ (batch, head, Dqk, seq_len) -> (batch, head, seq_len, seq_len)

        scores += self.relative_bias(
            seq_len=seq_len,
            device=x.device,
        )
        # (batch, head, seq_len, seq_len) + (1, 1, seq_len, seq_len) = (batch, head, seq_len, seq_len)

        # uses SiLU instead of softmax, then normalize by sequence length
        attn_w = F.silu(scores)
        attn_w /= seq_len
        # (batch, head, seq_len, seq_len)

        # apply causal mask to attention weights
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )
        attn_w *= causal_mask
        # (batch, head, seq_len, seq_len)

        if valid_mask is not None:
            key_mask = valid_mask[:, None, None, :]
            # (batch, seq_len) -> (batch, 1, 1, seq_len)

            attn_w *= key_mask

        # weighted sum of values
        av = attn_w @ v
        # (batch, head, seq_len, seq_len) @ (batch, head, seq_len, Dv) -> (batch, head, seq_len, Dv)

        av = (
            av.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.value_dim)
        )
        # (batch, head, seq_len, Dv) -> (batch, seq_len, head, Dv) -> (batch, seq_len, head * Dv)

        # core HSTU operation: Norm(AV) * U
        norm_av = self.attention_norm(av)
        norm_av *= u
        # (batch, seq_len, head * Dv)

        output = self.output_projection(norm_av)
        output = self.dropout(output)
        # (batch, seq_len, head * Dv) -> (batch, seq_len, d_model)

        output = residual + output

        # zero out padded history positions
        if valid_mask is not None:
            output *= valid_mask.unsqueeze(-1)

        return output
