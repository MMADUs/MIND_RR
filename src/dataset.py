# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class BehaviorDataset:
    BEHAVIOR_COLUMNS = [
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions",
    ]

    def __init__(self, news_id_to_idx: dict[str, int], behaviors_path: str | Path):
        self.news_id_to_idx = news_id_to_idx

        self._load_behaviors_df(behaviors_path)

    def build_data(self, max_history) -> pd.DataFrame:
        """
        build MIND behavior rows into training samples,
        one MIND impression remains one training sample

        if an impression contains multiple positive clicks, all positives
        are retained in `positive_ids`, MINDTrainDataset randomly chooses
        one positive when the sample is loaded
        """

        def parse_history(history: str) -> list[int]:
            history_ids = [
                self.news_id_to_idx[news_id]
                for news_id in history.split()
                if news_id in self.news_id_to_idx
            ]
            # MIND history is chronological:
            # oldest -> ... -> newest
            # keep the latest max_history articles
            return history_ids[-max_history:]

        behavior_df = pd.DataFrame()

        behavior_df["history_ids"] = self.behaviors_df["history"].map(parse_history)

        parsed_impressions = self.behaviors_df["impressions"].map(
            lambda impressions: self._parse_label(impressions)
        )
        behavior_df["positive_ids"] = parsed_impressions.map(lambda x: x[0])
        behavior_df["negative_ids"] = parsed_impressions.map(lambda x: x[1])

        # both classes are required for sampled-softmax training
        valid = (
            behavior_df["history_ids"].map(bool)
            & behavior_df["positive_ids"].map(bool)
            & behavior_df["negative_ids"].map(bool)
        )
        behavior_df = behavior_df.loc[valid].reset_index(drop=True)

        return behavior_df

    def _load_behaviors_df(self, behaviors_path: str | Path):
        self.behaviors_df = pd.read_csv(
            behaviors_path,
            sep="\t",
            header=None,
            names=self.BEHAVIOR_COLUMNS,
            dtype=str,
        )

        # some impressions contain no previous user history
        self.behaviors_df["history"] = self.behaviors_df["history"].fillna("")

    def _parse_label(self, impressions: str) -> tuple[list[int], list[int]]:
        """
        Parse a MIND impression into positive and negative news IDs

        from: N1-0 N2-1 N3-0 N4-1
        to: positives = [N2, N4], negatives = [N1, N3]
        """
        pos = []
        neg = []

        for item in impressions.split():
            news_id, label = item.rsplit(
                "-",
                maxsplit=1,
            )
            news_idx = self.news_id_to_idx.get(news_id)

            if news_idx is None:
                continue

            if label == "1":
                pos.append(news_idx)
            elif label == "0":
                neg.append(news_idx)
            else:
                raise ValueError(f"found {label} is not a valid label")

        return pos, neg


class MINDDataset(Dataset):
    """
    MIND training dataset for the two-tower recommendation model

    For each item:
        1. Randomly choose one clicked candidate.
        2. Randomly sample `num_negatives` non-clicked candidates.
        3. Place the positive candidate at index 0.
        4. Return class label 0 for CrossEntropyLoss.

    historical news is truncated to the latest `max_history` articles
    and right-padded: [N1, N2, N3, PAD, PAD]
    """

    def __init__(
        self,
        news_data: dict[str, torch.Tensor],
        behavior_df: pd.DataFrame,
        max_history: int = 50,
        num_negatives: int = 4,
    ):
        super().__init__()

        self.max_history = max_history
        self.num_negatives = num_negatives

        self.news_data = news_data

        self.histories = behavior_df["history_ids"].tolist()
        self.positives = behavior_df["positive_ids"].tolist()
        self.negatives = behavior_df["negative_ids"].tolist()

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        history = self.histories[index]

        history_len = len(history)

        history_indices = np.zeros(self.max_history, dtype=np.int64)
        history_mask = np.zeros(self.max_history, dtype=np.bool_)

        if history_len > 0:
            history_indices[:history_len] = history
            history_mask[:history_len] = True

        # positive sampling
        positive_pool = self.positives[index]
        positive = np.random.choice(positive_pool)

        # negative sampling
        negative_pool = self.negatives[index]
        sampled_negatives = np.random.choice(
            negative_pool,
            size=self.num_negatives,
            replace=(len(negative_pool) < self.num_negatives),
        )

        # candidate index 0 is always positive
        candidate_indices = np.concatenate(
            [
                np.array([positive], dtype=np.int64),
                sampled_negatives.astype(np.int64),
            ]
        )

        history_indices = torch.from_numpy(history_indices)
        history_mask = torch.from_numpy(history_mask)
        candidate_indices = torch.from_numpy(candidate_indices)

        output = {
            # history
            "history_title_ids": self.news_data["title_ids"][history_indices],
            "history_abstract_ids": self.news_data["abstract_ids"][history_indices],
            "history_category_ids": self.news_data["category_ids"][history_indices],
            "history_subcategory_ids": self.news_data["subcategory_ids"][
                history_indices
            ],
            "history_mask": history_mask,
            # candidates
            "candidate_title_ids": self.news_data["title_ids"][candidate_indices],
            "candidate_abstract_ids": self.news_data["abstract_ids"][candidate_indices],
            "candidate_category_ids": self.news_data["category_ids"][candidate_indices],
            "candidate_subcategory_ids": self.news_data["subcategory_ids"][
                candidate_indices
            ],
            # candidate[0] = positive
            "label": torch.tensor(0, dtype=torch.long),
        }

        # optional entities
        if "entity_ids" in self.news_data:
            output["history_entity_ids"] = self.news_data["entity_ids"][history_indices]
            output["candidate_entity_ids"] = self.news_data["entity_ids"][
                candidate_indices
            ]

        return output


def build_ds(
    news_id_to_idx: dict[str, int],
    news_data: dict[str, torch.Tensor],
    behaviors_path: str | Path,
    max_history: int,
    num_negatives: int,
) -> MINDDataset:
    """helper function to build dataset"""
    behavior_dataset = BehaviorDataset(news_id_to_idx, behaviors_path=behaviors_path)
    behavior_df = behavior_dataset.build_data(max_history)

    return MINDDataset(
        news_data,
        behavior_df,
        max_history,
        num_negatives,
    )
