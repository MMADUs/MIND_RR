# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import torch
import yaml

T = TypeVar("T")


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    types = get_type_hints(cls)
    kwargs = {}

    for f in fields(cls):
        if f.name not in data:
            if f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                kwargs[f.name] = f.default_factory()
            else:
                raise ValueError(f"missing required config key: {f.name}")
            continue

        value = data[f.name]
        field_type = types[f.name]

        if is_dataclass(field_type):
            value = from_dict(field_type, value)

        kwargs[f.name] = value

    return cls(**kwargs)


def load_config(cls: type[T], filepath: str | Path) -> type[T]:
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    return from_dict(cls, data)


@dataclass
class DataConfig:
    num_workers: int
    pin_memory: bool
    non_blocking: bool
    max_title_len: int 
    max_abstract_len: int 
    max_entities: int 
    max_history: int 
    num_negatives: int


@dataclass
class NewsTowerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    pool_hidden_dim: int
    text_embedding_dim: int | None
    use_pretrained_embedding: bool
    use_entity_embedding: bool
    category_embedding_dim: int
    subcategory_embedding_dim: int
    dropout: float


@dataclass
class UserTowerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    dropout: float


@dataclass
class TwoTowerConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    early_stop_patience: int
    news_tower: NewsTowerConfig
    user_tower: UserTowerConfig
    normalize_embeddings: bool


@dataclass
class Config:
    output_dir: str
    ckpt_basename: str
    ckpt_format: str

    data: DataConfig
    two_tower: TwoTowerConfig

    random_seed: int = 42
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
