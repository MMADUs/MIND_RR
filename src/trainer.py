# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import pickle
import time
import logging

from pathlib import Path

import torch
import torch.nn as nn

from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from src.callbacks import TrainingCallback, TrainCheckpoint, EarlyStopping
from src.metrics import compute_ranking_metrics
from src.utils import time_formatter, DeviceDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def _get_amp_dtype(device):
    return (
        torch.float16
        if device.type == "cuda" and not torch.cuda.is_bf16_supported()
        else torch.bfloat16
    )


class Trainer:
    """
    Trainer class for the two-tower retrieval model

    Config keys:
        device, output_dir, epochs, lr, weight_decay,
        grad_clip, early_stop_patience,
        ckpt_basename, ckpt_format

    Args:
        model:
            two-tower retrieval model
        model_name:
            model alias name
        train_loader:
            training set torch dataloader
        val_loader:
            validation set torch dataloader
        config:
            configuration class
    """

    def __init__(
        self,
        model,
        model_name,
        train_loader: DeviceDataLoader,
        val_loader: DeviceDataLoader,
        config: Config,
    ):
        self.config = config
        self.device = config.device
        self.amp_dtype = _get_amp_dtype(self.device)
        self.scaler = GradScaler(
            device=self.device.type,
            enabled=(self.amp_dtype == torch.float16),
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        logger.info("preparing training: %s", model_name)

        self.model = model.to(self.device)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            raise ValueError("No trainable parameters found for optimizer setup.")

        self.optimizer = AdamW(
            trainable_params,
            lr=config.two_tower.lr,
            weight_decay=config.two_tower.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.two_tower.epochs,
            eta_min=config.two_tower.lr * 1e-2,
        )

        ckpt_filename = config.ckpt_basename.format(model_name) + config.ckpt_format

        ckpt_path = model_dir / ckpt_filename

        self.callbacks = TrainingCallback(
            checkpoint=TrainCheckpoint(
                filepath=ckpt_path,
                mode="max",
            ),
            early_stop=EarlyStopping(
                patience=config.two_tower.early_stop_patience,
                mode="max",
            ),
        )

        history_filename = f"{model_name}_history.pkl"
        self.history_path = model_dir / history_filename

        self.history = []

    def _forward(self, batch):
        scores = self.model(
            history_title_ids=batch["history_title_ids"],
            history_abstract_ids=batch["history_abstract_ids"],
            history_category_ids=batch["history_category_ids"],
            history_subcategory_ids=batch["history_subcategory_ids"],
            history_entity_ids=batch.get("history_entity_ids"),
            history_mask=batch["history_mask"],
            candidate_title_ids=batch["candidate_title_ids"],
            candidate_abstract_ids=batch["candidate_abstract_ids"],
            candidate_category_ids=batch["candidate_category_ids"],
            candidate_subcategory_ids=batch["candidate_subcategory_ids"],
            candidate_entity_ids=batch.get("candidate_entity_ids"),
        )
        labels = batch["label"]

        return scores, labels

    def fit(self):
        logger.info("training started with %d epochs", self.config.two_tower.epochs)

        start_time = time.time()

        for epoch in range(1, self.config.two_tower.epochs + 1):
            epoch_start = time.time()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            epoch_lr = self.optimizer.param_groups[0]["lr"]

            # train
            self.model.train()

            train_loss = 0.0

            train_scores = []
            train_labels = []

            batch_iter = tqdm(self.train_loader, desc=f"epoch {epoch}")

            for batch in batch_iter:
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    scores, labels = self._forward(batch)
                    loss = self.criterion(scores, labels)

                self.scaler.scale(loss).backward()

                if self.config.two_tower.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.two_tower.grad_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()

                train_scores.append(scores.detach())
                train_labels.append(labels.detach())

                batch_iter.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(self.train_loader)

            train_scores = torch.cat(train_scores, dim=0)
            train_labels = torch.cat(train_labels, dim=0)

            train_metrics = compute_ranking_metrics(train_scores, train_labels)

            del train_scores
            del train_labels

            self.scheduler.step()

            # eval
            self.model.eval()

            val_loss = 0.0

            val_scores = []
            val_labels = []

            with torch.no_grad():

                for batch in tqdm(self.val_loader, desc="validation"):
                    with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        scores, labels = self._forward(batch)
                        loss = self.criterion(scores, labels)

                    val_loss += loss.item()

                    val_scores.append(scores)
                    val_labels.append(labels)

            val_loss /= len(self.val_loader)

            val_scores = torch.cat(val_scores, dim=0)
            val_labels = torch.cat(val_labels, dim=0)

            val_metrics = compute_ranking_metrics(val_scores, val_labels)

            del val_scores
            del val_labels

            # logging
            epoch_time = time.time() - epoch_start

            logger.info(
                "epoch %d/%d - %s | lr=%.2e | "
                "train_loss=%.6f | val_loss=%.6f | "
                "train_AUC=%.4f | val_AUC=%.4f | "
                "train_MRR=%.4f | val_MRR=%.4f | "
                "train_nDCG@5=%.4f | val_nDCG@5=%.4f | ",
                epoch,
                self.config.two_tower.epochs,
                time_formatter(epoch_time),
                epoch_lr,
                train_loss,
                val_loss,
                train_metrics["auc"],
                val_metrics["auc"],
                train_metrics["mrr"],
                val_metrics["mrr"],
                train_metrics["ndcg@5"],
                val_metrics["ndcg@5"],
            )

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }
            )

            # callbacks
            model_dict = self.model.state_dict()
            optimizer_dict = {
                "adam": self.optimizer.state_dict(),
                "cosine_scheduler": self.scheduler.state_dict(),
            }
            metadata = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "train_mrr": train_metrics["mrr"],
                "val_mrr": val_metrics["mrr"],
                "train_ndcg@5": train_metrics["ndcg@5"],
                "val_ndcg@5": val_metrics["ndcg@5"],
            }

            is_stopping = self.callbacks.step(
                monitor_value=val_metrics["mrr"],
                model_dict=model_dict,
                metadata=metadata,
                optimizer_dict=optimizer_dict,
            )

            if is_stopping:
                break

            print("\n")

        end_time = time.time()

        logger.info("elapsed time: %s", time_formatter(end_time - start_time))

        with open(self.history_path, "wb") as f:
            pickle.dump(self.history, f)

        logger.info("training complete | history saved to %s", self.history_path)
