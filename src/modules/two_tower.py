# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.text_encoder import TextEncoder
from src.modules.entity_encoder import EntityEncoder
from src.modules.attention import AdditiveAttention
from src.modules.hstu_block import HSTUBlock


class NewsEncoder(nn.Module):
    """
    Encode multiple news features into a single news representation

    The encoder combines textual representations from the title and abstract
    with category, subcategory, and optionally entity representations

    The encoder needs to satisfy the args `entity_embedding_dim` and `entity_embedding_weights`
    in order to enable entity encoding

    Args:
        vocab_size:
            number of tokens in the text vocabulary
        category_vocab_size:
            number of categories in the category vocabulary
        category_embedding_dim:
            dimension of the category embedding.
        subcategory_vocab_size:
            number of subcategories in the subcategory vocabulary
        subcategory_embedding_dim:
            dimension of the subcategory embedding
        entity_embedding_dim:
            optional dimension of the entity embeddings
        entity_embedding_weights:
            optional pretrained entity embedding weights from .vec file
        text_embedding_dim:
            dimension of the input token embeddings
        text_embedding_weights:
            optional pretrained token embedding matrix
        d_model:
            dimension of the final news representation
        num_heads:
            number of attention heads used by the text encoder
        pool_hidden_dim:
            hidden dimension used by additive attention pooling
        dropout:
            dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        category_vocab_size: int,
        category_embedding_dim: int,
        subcategory_vocab_size: int,
        subcategory_embedding_dim: int,
        entity_embedding_dim: int | None,
        entity_embedding_weights: torch.Tensor | None,
        text_embedding_dim: int,
        text_embedding_weights: torch.Tensor | None,
        d_model: int = 256,
        num_heads: int = 4,
        pool_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            embedding_weights=text_embedding_weights,
            d_model=d_model,
            num_heads=num_heads,
            pool_hidden_dim=pool_hidden_dim,
            dropout=dropout,
        )

        self.cat_embedding = nn.Embedding(
            num_embeddings=category_vocab_size,
            embedding_dim=category_embedding_dim,
            padding_idx=0,
        )
        self.subcat_embedding = nn.Embedding(
            num_embeddings=subcategory_vocab_size,
            embedding_dim=subcategory_embedding_dim,
            padding_idx=0,
        )
        self.cat_projection = nn.Linear(category_embedding_dim, d_model)
        self.subcat_projection = nn.Linear(subcategory_embedding_dim, d_model)

        if entity_embedding_dim and entity_embedding_weights:
            self.entity_encoder = EntityEncoder(
                embedding_dim=entity_embedding_dim,
                embedding_weights=entity_embedding_weights,
                d_model=d_model,
                pool_hidden_dim=pool_hidden_dim,
                dropout=dropout,
            )
        else:
            self.entity_encoder = None

        self.fuse_attention = AdditiveAttention(
            input_dim=d_model,
            hidden_dim=pool_hidden_dim,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        title_ids: torch.Tensor,
        abstract_ids: torch.Tensor,
        category_ids: torch.Tensor,
        subcategory_ids: torch.Tensor,
        entity_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        title_out = self.text_encoder(title_ids)
        abs_out = self.text_encoder(abstract_ids)
        # title, abs: (batch, d_model)

        cat_out = self.cat_embedding(category_ids)
        cat_out = self.cat_projection(cat_out)
        # cat: (batch, cat_dim) -> (batch, d_model)

        subcat_out = self.subcat_embedding(subcategory_ids)
        subcat_out = self.subcat_projection(subcat_out)
        # subcat: (batch, subcat_dim) -> (batch, d_model)

        output_stack = [title_out, abs_out, cat_out, subcat_out]

        if self.entity_encoder is not None and entity_ids is not None:
            entity_out = self.entity_encoder(entity_ids)
            # entity: (batch, d_model)

            output_stack.append(entity_out)

        fused = torch.stack(output_stack, dim=1)
        # (batch, features, d_model)

        out = self.fuse_attention(fused)
        out = self.norm(out)
        out = self.dropout(out)
        # (batch, d_model)

        return out


class UserEncoder(nn.Module):
    """
    Encode a user's news consumption history into a single user representation

    The encoder applies stacked HSTU blocks over the sequence of historical
    news representations

    Args:
        d_model:
            dimension of each news representation and the resulting user
            representation
        num_layers:
            number of stacked HSTU blocks
        num_heads:
            number of attention heads in each HSTU block
        qk_dim:
            dimension of the query and key vectors for each attention head
        value_dim:
            dimension of the value vector for each attention head
        max_distance:
            maximum relative position distance represented by the HSTU
            positional bias
        dropout:
            dropout probability
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        qk_dim: int = 64,
        value_dim: int = 64,
        max_distance: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                HSTUBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    qk_dim=qk_dim,
                    value_dim=value_dim,
                    max_distance=max_distance,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, history: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = history
        # x: (batch, history_len, d_model)

        for layer in self.layers:
            x = layer(x, valid_mask)
        # (batch, history_len, d_model)

        x = self.norm(x)

        length = valid_mask.sum(dim=1)  # number of article from each user
        last_idx = (length - 1).clamp_min(0)  # clamp in case history is empty
        batch_idx = torch.arange(x.size(0), device=x.device)

        out = x[batch_idx, last_idx]  # pair each user with the latest vaid HSTU state

        # zero out embedding for user without history
        empty_indices = length == 0
        if empty_indices.any():
            out = out.clone()
            out[empty_indices] = 0.0

        return out


class TwoTowerModel(nn.Module):
    """
    Two-tower recommendation model

    The news tower independently encodes historical and candidate news
    articles into dense representations. Historical news representations
    are passed through the user tower to construct a user embedding

    Candidate relevance is computed using the dot product between the
    user representation and each candidate news representation

    Args:
        news_tower:
            news encoder used to generate dense news representations
        user_tower:
            user encoder used to aggregate historical news representations
            into a single user representation
        normalize_embeddings:
            whether to L2-normalize user and candidate embeddings before
            computing similarity scores (enabling this, is equivalent to cosine similarity)
    """

    def __init__(
        self,
        news_tower: NewsEncoder,
        user_tower: UserEncoder,
        normalize_embeddings: bool = False,
    ):
        super().__init__()

        self.news_tower = news_tower
        self.user_tower = user_tower
        self.normalize_embeddings = normalize_embeddings

    def _encode_news(
        self,
        title_ids: torch.Tensor,
        abstract_ids: torch.Tensor,
        category_ids: torch.Tensor,
        subcategory_ids: torch.Tensor,
        entity_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        # title_ids: (batch, num_news, token_ids)
        batch_size, num_news = title_ids.shape[:2]

        title_ids = title_ids.flatten(0, 1)
        abstract_ids = abstract_ids.flatten(0, 1)
        category_ids = category_ids.flatten(0, 1)
        subcategory_ids = subcategory_ids.flatten(0, 1)
        # title, abstract, cat, subcat: (batch, num_news, token_ids) -> (batch x num_news, token_ids)

        if entity_ids is not None:
            entity_ids = entity_ids.flatten(0, 1)
            # entity: (batch, num_news, token_ids) -> (batch x num_news, token_ids)

        news_out = self.news_tower(
            title_ids,
            abstract_ids,
            category_ids,
            subcategory_ids,
            entity_ids,
        )
        # (batch x num_news, token_ids) -> (batch x num_news, d_model)

        news_out = news_out.view(batch_size, num_news, -1)
        # (batch x num_news, d_model) -> (batch_size, num_news, d_model)

        return news_out

    def forward(
        self,
        # history news
        history_title_ids: torch.Tensor,
        history_abstract_ids: torch.Tensor,
        history_category_ids: torch.Tensor,
        history_subcategory_ids: torch.Tensor,
        history_entity_ids: torch.Tensor | None,
        history_mask: torch.Tensor,
        # candidate news
        candidate_title_ids: torch.Tensor,
        candidate_abstract_ids: torch.Tensor,
        candidate_category_ids: torch.Tensor,
        candidate_subcategory_ids: torch.Tensor,
        candidate_entity_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        history_out = self._encode_news(
            title_ids=history_title_ids,
            abstract_ids=history_abstract_ids,
            category_ids=history_category_ids,
            subcategory_ids=history_subcategory_ids,
            entity_ids=history_entity_ids,
        )

        user_out = self.user_tower(
            history_out,
            history_mask,
        )

        candidate_out = self._encode_news(
            title_ids=candidate_title_ids,
            abstract_ids=candidate_abstract_ids,
            category_ids=candidate_category_ids,
            subcategory_ids=candidate_subcategory_ids,
            entity_ids=candidate_entity_ids,
        )

        # normalize + dot product = consine similarity
        if self.normalize_embeddings:
            user_out = F.normalize(user_out, p=2, dim=-1)
            candidate_out = F.normalize(candidate_out, p=2, dim=-1)

        # final dot product score
        scores = torch.einsum("bd,bcd->bc", user_out, candidate_out)

        return scores
