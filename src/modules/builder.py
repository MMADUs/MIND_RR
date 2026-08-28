# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

from src.modules.two_tower import (
    NewsEncoder,
    UserEncoder,
    TwoTowerModel,
)
from src.modules.entity_encoder import build_entity_embeddings
from src.modules.text_encoder import get_hf_tokenizer_embeddings


def build_two_tower_model(
    d_model: int,
    pool_hidden_dim: int,
    text_num_heads: int,
    hstu_num_heads: int,
    num_hstu_layers: int,
    qk_dim: int,
    value_dim: int,
    max_distance: int,
    text_embedding_dim: int | None,
    use_pretrained_embedding: bool,
    use_entity_embedding: bool,
    category_vocab_size: int,
    category_embedding_dim: int,
    subcategory_vocab_size: int,
    subcategory_embedding_dim: int,
    dropout: float,
    normalize_embeddings: bool = False,
):
    if use_pretrained_embedding:
        text_embedding_weights, text_embedding_dim, tokenizer = (
            get_hf_tokenizer_embeddings()
        )
    else:
        text_embedding_weights = None
        _, _, tokenizer = get_hf_tokenizer_embeddings() # maybe train our own tokenizer, idk (ignore this for now)

    if use_entity_embedding:
        entity_embedding_weights, entity_embedding_dim, entity_vocab = (
            build_entity_embeddings()
        )
    else:
        entity_embedding_weights, entity_embedding_dim, entity_vocab = None, None, None

    news_encoder = NewsEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=text_num_heads,
        pool_hidden_dim=pool_hidden_dim,
        text_embedding_dim=text_embedding_dim,
        text_embedding_weights=text_embedding_weights,
        entity_embedding_dim=entity_embedding_dim,
        entity_embedding_weights=entity_embedding_weights,
        category_vocab_size=category_vocab_size,
        category_embedding_dim=category_embedding_dim,
        subcategory_vocab_size=subcategory_vocab_size,
        subcategory_embedding_dim=subcategory_embedding_dim,
        dropout=dropout,
    )
    user_encoder = UserEncoder(
        d_model=d_model,
        num_layers=num_hstu_layers,
        num_heads=hstu_num_heads,
        qk_dim=qk_dim,
        value_dim=value_dim,
        max_distance=max_distance,
        dropout=dropout,
    )
    model = TwoTowerModel(
        news_tower=news_encoder,
        user_tower=user_encoder,
        normalize_embeddings=normalize_embeddings,
    )
    return model, tokenizer, entity_vocab
