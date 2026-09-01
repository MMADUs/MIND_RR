# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import numpy as np

from torch.utils.data import DataLoader, Subset

from src.modules import (
    build_entity_embeddings,
    get_hf_tokenizer_embeddings,
    build_two_tower_model,
)
from src.database import NewsDatabase
from src.dataset import build_ds
from src.utils import DeviceDataLoader, PROJECT_ROOT, count_parameters
from src.trainer import Trainer
from config import load_config, Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

N_TEST_SAMPLES = 500


def main():
    logger.info("reading configuration file")

    config = load_config(
        Config, (PROJECT_ROOT / "config" / "yaml" / "baseline_retrieval.yaml")
    )

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

    hf_model_name = config.two_tower.news_tower.hf_pretrained_model_name

    if config.two_tower.news_tower.use_pretrained_embedding:
        logger.info("loading pretrained embedding and tokenizer from %s", hf_model_name)
        text_embedding_weights, text_embedding_dim, tokenizer = (
            get_hf_tokenizer_embeddings(model_name=hf_model_name)
        )
    else:
        # TODO: make own tokenizer
        text_embedding_weights = None
        text_embedding_dim = config.two_tower.news_tower.text_embedding_dim
        tokenizer = ...
        pass

    vec_path = train_path / "entity_embedding.vec"

    if config.two_tower.news_tower.use_entity_embedding:
        logger.info("loading pretrained entity embedding from %s", vec_path)
        entity_embedding_weights, entity_embedding_dim, entity_vocab = (
            build_entity_embeddings(vec_path=vec_path)
        )
    else:
        entity_embedding_weights, entity_embedding_dim, entity_vocab = None, None, None

    logger.info("building news data lookup table")

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

    logger.info("building behavior dataset (train + validation)")

    train_ds, len_train = build_ds(
        news_id_to_idx,
        news_data,
        behaviors_path=(train_path / "behaviors.tsv"),
        max_history=max_history,
        num_negatives=num_negatives,
    )
    val_ds, len_val = build_ds(
        news_id_to_idx,
        news_data,
        behaviors_path=(val_path / "behaviors.tsv"),
        max_history=max_history,
        num_negatives=num_negatives,
    )

    logger.info(
        "total trainng behavior: %d rows | total validation behavior: %d rows",
        len_train,
        len_val,
    )

    if args.test:
        # tiny deterministic smoke test
        train_ds = Subset(train_ds, range(min(N_TEST_SAMPLES, len(train_ds))))
        val_ds = Subset(val_ds, range(min(N_TEST_SAMPLES // 4, len(val_ds))))

        logger.info(
            "test mode: using %d training rows | %d validation rows",
            len(train_ds),
            len(val_ds),
        )
    else:
        # normal experiment: randomly sample according to config
        rng = np.random.default_rng(config.random_seed)

        max_train_rows = config.data.max_train_rows
        max_val_rows = config.data.max_val_rows

        # training subset
        if max_train_rows is not None and max_train_rows < len(train_ds):
            train_indices = rng.choice(
                len(train_ds), size=max_train_rows, replace=False
            )

            train_ds = Subset(train_ds, train_indices)

        # validation subset
        if max_val_rows is not None and max_val_rows < len(val_ds):
            val_indices = rng.choice(len(val_ds), size=max_val_rows, replace=False)

            val_ds = Subset(val_ds, val_indices)

        logger.info(
            "using %d training rows | %d validation rows",
            len(train_ds),
            len(val_ds),
        )

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

    logger.info("model training begin shortly")

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

        total, trainable = count_parameters(two_tower)
        logger.info(
            "Parameters: total=%s | trainable=%s",
            f"{total:,}",
            f"{trainable:,}",
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
