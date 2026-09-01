# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import torch


def compute_auc(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute mean AUC for one positive candidate per sample.

    Args:
        scores:
            candidate scores with shape (batch, num_candidates)
        labels:
            positive candidate index with shape (batch,)
    """
    positive_scores = scores.gather(1, labels.unsqueeze(1))

    negative_scores = scores.clone()
    negative_scores.scatter_(1, labels.unsqueeze(1), float("-inf"))

    correct = (positive_scores > negative_scores).sum(dim=1)

    num_negatives = scores.size(1) - 1

    auc = correct.float() / num_negatives

    return auc.mean().item()


def compute_mrr(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    """
    positive_scores = scores.gather(1, labels.unsqueeze(1))

    rank = (scores > positive_scores).sum(dim=1) + 1

    reciprocal_rank = 1.0 / rank.float()

    return reciprocal_rank.mean().item()


def compute_ndcg(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """
    Compute nDCG@k for one positive candidate per sample.
    """
    positive_scores = scores.gather(1, labels.unsqueeze(1))

    rank = (scores > positive_scores).sum(dim=1) + 1

    ndcg = torch.where(
        rank <= k,
        1.0 / torch.log2(rank.float() + 1.0),
        torch.zeros_like(rank, dtype=torch.float),
    )

    return ndcg.mean().item()


def compute_ranking_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """
    Compute recommendation ranking metrics.
    """
    return {
        "auc": compute_auc(scores, labels),
        "mrr": compute_mrr(scores, labels),
        "ndcg@5": compute_ndcg(scores, labels, k=5),
    }