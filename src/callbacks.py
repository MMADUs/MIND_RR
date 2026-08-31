# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Literal

import torch


def is_improvement(value: float, best: float, mode: str, epsilon: float) -> bool:
    if mode == "min":
        return (best - value) > epsilon
    else:
        return (value - best) > epsilon


class TrainCheckpoint:
    """
    TrainCheckpoint saves the model checkpoint when there is an improvement in the monitored metric.

    Args:
        filepath:
            path to save the checkpoint
        mode:
            whether the monitored metric should be minimized or maximized
        epsilon:
            minimum improvement to consider as an actual improvement
    """

    def __init__(
        self,
        filepath,
        mode: Literal["min", "max"] = "min",
        epsilon: float = 0.0,
    ) -> None:
        assert mode in {"min", "max"}

        self.filepath = filepath
        self.mode = mode
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else -float("inf")

    def step(self, value: float, checkpoint_dict: Dict) -> None:
        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            self.best_value = value
            torch.save(checkpoint_dict, self.filepath)


class EarlyStopping:
    """
    EarlyStopping stops training when a monitored metric has not improved for a given number of epochs.

    Args:
        patience:
            number of epochs to wait for an improvement before stopping
        epsilon:
            minimum improvement to consider as an actual improvement
        mode:
            whether the monitored metric should be minimized or maximized
    """

    def __init__(
        self,
        patience: int,
        epsilon: float = 1e-4,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        assert mode in {"min", "max"}

        self.patience = patience
        self.epsilon = epsilon
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else -float("inf")
        self.wait = 0

    def step(self, value: float) -> bool:
        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            print(f"* metrics improved from {self.best_value:.6f} to {value:.6f}")
            self.best_value = value
            self.wait = 0
            return False
        else:
            print(f"metrics did not improve from {self.best_value:.6f}")

        self.wait += 1
        early_stop = self.wait >= self.patience

        if early_stop:
            print(f"Early stopping, no improvement in the last {self.patience} epochs")

        return early_stop


class TrainingCallback:
    """
    TrainingCallback is a class that orchestrate the provided callbacks.

    Args:
        checkpoint:
            `TrainCheckpoint` instance for saving model checkpoints
        early_stop:
            `EarlyStopping` instance for early stopping
    """

    def __init__(
        self,
        checkpoint: Optional[TrainCheckpoint] = None,
        early_stop: Optional[EarlyStopping] = None,
    ) -> None:
        self.stop_training = False
        self.checkpoint = checkpoint
        self.early_stop = early_stop

    def reset(self) -> None:
        self.stop_training = False

        if self.checkpoint:
            self.checkpoint.reset()
        if self.early_stop:
            self.early_stop.reset()

    def is_training_stopped(self) -> bool:
        return self.stop_training

    def step(
        self,
        monitor_value: float,
        model_dict: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        optimizer_dict: Optional[Dict] = None,
    ) -> bool:
        # training is stopped
        if self.stop_training:
            return True

        # model checkpoint
        if self.checkpoint and model_dict:
            checkpoint_dict = (
                {"model": model_dict, **metadata} if metadata else {"model": model_dict}
            )
            # only if optimizer is provided
            if optimizer_dict:

                checkpoint_dict["optimizer"] = {
                    k: v.state_dict() if hasattr(v, "state_dict") else v
                    for k, v in optimizer_dict.items()
                }
            # save checkpoint
            self.checkpoint.step(monitor_value, checkpoint_dict)

        # early stopping
        if self.early_stop:
            # step
            stop_training = self.early_stop.step(monitor_value)
            self.stop_training = stop_training
            return stop_training

        # default behavior is to continue training
        return False
