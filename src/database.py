# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OrdinalEncoder
from transformers import PreTrainedTokenizerBase


class NewsDatabase:
    NEWS_COLUMNS = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    def __init__(
        self,
        news_paths: str | Path | list[str | Path],
        tokenizer: PreTrainedTokenizerBase,
        entity_vocab: dict[str, int] | None = None,
        max_title_len: int = 32,
        max_abstract_len: int = 64,
        max_entities: int = 10,
    ):
        self.tokenizer = tokenizer
        self.entity_vocab = entity_vocab
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len
        self.max_entities = max_entities

        if isinstance(news_paths, (str, Path)):
            news_paths = [news_paths]

        # load news df + fit categorical encoder
        self._load_news_df(news_paths)
        self._build_encoders()

    def build_news(self) -> tuple[dict[str, int], dict[str, torch.Tensor]]:
        if self.tokenizer.pad_token_id != 0:
            raise ValueError("Current TextEncoder expects pad_token_id == 0")

        news_id_to_idx = {"<PAD>": 0}

        for news_id in self.news_df["news_id"]:
            news_id_to_idx[news_id] = len(news_id_to_idx)  # auto-increment index

        num_news = len(news_id_to_idx)

        tokenized_titles = self._tokenize(
            texts=self.news_df["title"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_title_len,
        )
        # (num_real_news, max_title_len)

        tokenized_abstracts = self._tokenize(
            texts=self.news_df["abstract"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_abstract_len,
        )
        # (num_real_news, max_abstract_len)

        encoded_categories = self._encode_ordinal(column="category")
        encoded_subcategories = self._encode_ordinal(column="subcategory")

        category_ids = torch.zeros(num_news, dtype=torch.long)
        subcategory_ids = torch.zeros(num_news, dtype=torch.long)
        title_ids = torch.zeros(num_news, self.max_title_len, dtype=torch.long)
        abstract_ids = torch.zeros(num_news, self.max_abstract_len, dtype=torch.long)

        # skip index 0 because it is PAD for news
        title_ids[1:] = tokenized_titles
        abstract_ids[1:] = tokenized_abstracts
        category_ids[1:] = torch.from_numpy(encoded_categories)
        subcategory_ids[1:] = torch.from_numpy(encoded_subcategories)

        # fill the unknown data with empty string
        unk_title = self.tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=self.max_title_len,
            return_tensors="pt",
        )["input_ids"][0]

        unk_abstract = self.tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=self.max_abstract_len,
            return_tensors="pt",
        )["input_ids"][0]

        title_ids[0] = unk_title
        abstract_ids[0] = unk_abstract

        # NOTE: the embedding explicitly put `padding_idx=0`, meaning we dont need to fill unk category/sub-category

        entity_ids = None

        if self.entity_vocab is not None:
            pad_entity_idx = self.entity_vocab["<PAD>"]
            unk_entity_idx = self.entity_vocab["<UNK>"]

            entity_ids = torch.full(
                size=(num_news, self.max_entities),
                fill_value=pad_entity_idx,
                dtype=torch.long,
            )

            entity_ids[0, 0] = unk_entity_idx

            for row_idx, row in enumerate(
                self.news_df.itertuples(index=False), start=1
            ):
                entities = self._parse_entities(
                    title_entities=row.title_entities,
                    abstract_entities=row.abstract_entities,
                )

                mapped_entities = [
                    self.entity_vocab.get(entity_id, unk_entity_idx)
                    for entity_id in entities[: self.max_entities]
                ]

                if not mapped_entities:
                    mapped_entities = [unk_entity_idx]

                entity_ids[row_idx, : len(mapped_entities)] = torch.tensor(
                    mapped_entities, dtype=torch.long
                )

        news_data = {
            "title_ids": title_ids,
            "abstract_ids": abstract_ids,
            "category_ids": category_ids,
            "subcategory_ids": subcategory_ids,
        }

        if entity_ids is not None:
            news_data["entity_ids"] = entity_ids

        return (news_id_to_idx, news_data)

    def _load_news_df(self, news_paths: list[str | Path]):
        dfs = []

        for news_path in news_paths:
            df = pd.read_csv(
                news_path,
                sep="\t",
                header=None,
                names=self.NEWS_COLUMNS,
                dtype=str,
            )
            dfs.append(df)

        self.news_df = pd.concat(
            dfs,
            axis=0,
            ignore_index=True,
        )

        # Verify duplicate news IDs
        duplicated = self.news_df["news_id"].duplicated(keep=False)

        if duplicated.any():
            duplicate_ids = self.news_df.loc[duplicated, "news_id"].unique()

            # Keep first occurrence
            self.news_df = self.news_df.drop_duplicates(
                subset="news_id",
                keep="first",
            ).reset_index(drop=True)

        # missing text
        self.news_df["title"] = self.news_df["title"].fillna("")
        self.news_df["abstract"] = self.news_df["abstract"].fillna("")

        # missing entity JSON
        self.news_df["title_entities"] = self.news_df["title_entities"].fillna("[]")

        self.news_df["abstract_entities"] = self.news_df["abstract_entities"].fillna(
            "[]"
        )

    def _build_encoders(self):
        """
        + 2 is added for extra cateogry (PAD & UNK)
        """
        self.category_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
        self.subcategory_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )

        self.category_encoder.fit(self.news_df[["category"]])
        self.subcategory_encoder.fit(self.news_df[["subcategory"]])

        self.category_vocab_size = len(self.category_encoder.categories_[0]) + 2
        self.subcategory_vocab_size = len(self.subcategory_encoder.categories_[0]) + 2

    def _encode_ordinal(self, column: Literal["category", "subcategory"]) -> np.ndarray:
        """
        + 2 is added for extra cateogry (PAD & UNK)
        """
        if column == "category":
            encoded = self.category_encoder.transform(
                self.news_df[["category"]]
            ).reshape(-1)
        elif column == "subcategory":
            encoded = self.subcategory_encoder.transform(
                self.news_df[["subcategory"]]
            ).reshape(-1)
        else:
            raise ValueError(f"column name: {column} is invalid")

        encoded += 2
        return encoded.astype(np.int64)

    def _get_entity_ids(self, raw_entities: str) -> list[str]:
        if not raw_entities:
            return []

        try:
            entities = json.loads(raw_entities)
        except (json.JSONDecodeError, TypeError):
            return []

        entity_ids = []

        for entity in entities:
            entity_id = entity.get("WikidataId")

            if entity_id:
                entity_ids.append(entity_id)

        return entity_ids

    def _parse_entities(self, title_entities: str, abstract_entities: str) -> list[str]:
        entities = self._get_entity_ids(title_entities) + self._get_entity_ids(
            abstract_entities
        )
        # remove duplicates while preserving order
        return list(dict.fromkeys(entities))

    def _tokenize(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        tokenizer_batch_size: int = 2048,
    ) -> torch.Tensor:
        batches = []

        for start in range(0, len(texts), tokenizer_batch_size):
            batch_texts = texts[start : start + tokenizer_batch_size]
            encoded = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batches.append(encoded["input_ids"])

        return torch.cat(batches, dim=0)
