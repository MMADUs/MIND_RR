# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import argparse

from torch.utils.data import DataLoader, Subset

from src.modules import (
    build_entity_embeddings,
    get_hf_tokenizer_embeddings,
    build_two_tower_model,
)
from src.database import NewsDatabase
from src.dataset import build_ds
from src.utils import DeviceDataLoader, PROJECT_ROOT
from src.trainer import Trainer
from config import load_config, Config

N_TEST_SAMPLES = 500


class CustomSubset(Subset):
    def labels(self):
        labels = []

        for idx in self.indices:
            sample_idx, _ = self.dataset._index[idx]
            labels.append(self.dataset.samples[sample_idx]["label_path"])

        return labels


def main():
    config = load_config(Config, (PROJECT_ROOT / "config" / "yaml" / "baseline_retrieval.yaml"))

    print(f"{config}")

    parser = argparse.ArgumentParser(description="trainer script arg parser")

    parser.add_argument(
        "--test",
        action="store_true",
        help="run in test mode with a small subset of data",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["retrieval"],
        choices=["retrieval", "reranker"],
        help="model to train",
    )

    args = parser.parse_args()

    train_path = PROJECT_ROOT / ".dataset" / "train"
    val_path = PROJECT_ROOT / ".dataset" / "validation"

    if config.two_tower.news_tower.use_pretrained_embedding:
        text_embedding_weights, text_embedding_dim, tokenizer = (
            get_hf_tokenizer_embeddings()
        )
    else:
        # TODO: make own tokenizer
        text_embedding_weights = None
        text_embedding_dim = config.two_tower.news_tower.text_embedding_dim
        tokenizer = ...
        pass

    if config.two_tower.news_tower.use_entity_embedding:
        entity_embedding_weights, entity_embedding_dim, entity_vocab = (
            build_entity_embeddings(vec_path=(train_path / "entity_embedding.vec"))
        )
    else:
        entity_embedding_weights, entity_embedding_dim, entity_vocab = None, None, None

    news_db = NewsDatabase(
        news_paths=[(train_path / "news.tsv"), (val_path / "news.tsv")],
        tokenizer=tokenizer,
        entity_vocab=entity_vocab,
        max_title_len=config.data.max_title_len,
        max_abstract_len=config.data.max_abstract_len,
        max_entities=config.data.max_entities,
    )
    news_id_to_idx, news_data = news_db.build_news()

    category_vocab_size = news_db.category_vocab_size
    subcategory_vocab_size = news_db.subcategory_vocab_size

    max_history = config.data.max_history
    num_negatives = config.data.num_negatives

    train_ds = build_ds(
        news_id_to_idx,
        news_data,
        behaviors_path=(train_path / "behaviors.tsv"),
        max_history=max_history,
        num_negatives=num_negatives,
    )
    val_ds = build_ds(
        news_id_to_idx,
        news_data,
        behaviors_path=(val_path / "behaviors.tsv"),
        max_history=max_history,
        num_negatives=num_negatives,
    )

    if args.test:
        train_ds = CustomSubset(train_ds, list(range(N_TEST_SAMPLES)))
        val_ds = CustomSubset(val_ds, list(range(N_TEST_SAMPLES // 4)))

    batch_size = config.two_tower.batch_size
    val_batch_size = batch_size // 2

    num_workers = config.data.num_workers
    pin_memory = config.data.pin_memory

    train_dl = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        val_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    device = config.device
    non_blocking = config.data.non_blocking

    train_ddl = DeviceDataLoader(train_dl, device, non_blocking)
    val_ddl = DeviceDataLoader(val_dl, device, non_blocking)

    selected_models = args.model

    if "retrieval" in selected_models:
        two_tower = build_two_tower_model(
            model_config=config.two_tower,
            category_vocab_size=category_vocab_size,
            subcategory_vocab_size=subcategory_vocab_size,
            text_embedding_weights=text_embedding_weights,
            text_embedding_dim=text_embedding_dim,
            tokenizer=tokenizer,
            entity_embedding_weights=entity_embedding_weights,
            entity_embedding_dim=entity_embedding_dim,
        )
        two_tower_trainer = Trainer(
            model=two_tower,
            model_name="two_tower_retrieval",
            train_loader=train_ddl,
            val_loader=val_ddl,
            config=config,
        )
        two_tower_trainer.fit()

    if "reranker" in selected_models:
        pass  # coming soon


# python script.py --test --model retrieval reranker
if __name__ == "__main__":
    main()
