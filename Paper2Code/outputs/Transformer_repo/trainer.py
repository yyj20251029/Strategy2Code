"""trainer.py

This module implements the Trainer class for training the TransformerModel as described in
"Attention Is All You Need". It handles the training loop including the custom learning
rate schedule, optimizer integration, label smoothing loss computation, gradient updates,
and checkpoint saving.

Configuration is based on config.yaml and all hyperparameters (e.g. steps, optimizer parameters,
learning rate schedule, dropout, label smoothing) are read from the config dictionary.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import math
import time
import logging
from typing import Any, Dict, Iterator, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss. Instead of using one-hot targets, the target distribution is smoothed.
    
    Args:
        label_smoothing (float): Smoothing factor e.g., 0.1
        vocab_size (int): Size of the vocabulary.
        ignore_index (int): Padding index to ignore in the loss computation.
    """
    def __init__(self, label_smoothing: float, vocab_size: int, ignore_index: int = 0) -> None:
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 <= label_smoothing < 1.0, "label_smoothing must be in [0, 1)"
        self.label_smoothing: float = label_smoothing
        self.vocab_size: int = vocab_size
        self.ignore_index: int = ignore_index
        self.confidence: float = 1.0 - label_smoothing

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the label-smoothed cross entropy loss.
        
        Args:
            pred_logits (Tensor): Logits output from the model of shape (batch, seq_len, vocab_size).
            target (Tensor): Target token ids of shape (batch, seq_len).
        
        Returns:
            loss (Tensor): Scalar tensor representing the loss.
        """
        # Compute log probabilities
        pred_log_probs = F.log_softmax(pred_logits, dim=-1)  # shape: (batch, seq_len, vocab_size)
        
        # Create a tensor filled with smooth value
        smooth_value = self.label_smoothing / (self.vocab_size - 1)
        true_dist = torch.full_like(pred_log_probs, smooth_value)
        
        # For non-ignored targets, set the probability for the true label to confidence.
        # target.unsqueeze(2) shape: (batch, seq_len, 1)
        true_dist.scatter_(2, target.unsqueeze(2), self.confidence)
        
        # For positions with ignore_index, zero out the distribution.
        mask = (target == self.ignore_index).unsqueeze(2)
        true_dist.masked_fill_(mask, 0.0)
        
        # Compute KL divergence loss sum over all tokens
        loss = F.kl_div(pred_log_probs, true_dist, reduction='sum')
        # Count the non-ignored tokens.
        non_pad = torch.sum(target != self.ignore_index).item()
        loss = loss / non_pad if non_pad > 0 else loss
        return loss


class Trainer:
    """
    Trainer class for training a TransformerModel.
    
    Methods:
        __init__(model, train_data, config): Initializes trainer with optimizer, hyperparameters, etc.
        _update_learning_rate(current_step): Update optimizer learning rate based on schedule.
        train(): Runs the training loop for the specified number of steps.
        save_checkpoint(path): Saves a checkpoint containing model and optimizer states.
    """

    def __init__(self, model: nn.Module, train_data: DataLoader, config: Dict[str, Any]) -> None:
        """
        Initializes the Trainer instance.

        Args:
            model (nn.Module): An instance of TransformerModel.
            train_data (DataLoader): PyTorch DataLoader providing training batches.
            config (Dict[str, Any]): Configuration dictionary from config.yaml.
        """
        self.model: nn.Module = model
        self.train_data: DataLoader = train_data
        self.config: Dict[str, Any] = config

        # Retrieve training hyperparameters from config with defaults.
        training_config = config.get("training", {})
        self.total_steps: int = training_config.get("steps", 100000)
        optimizer_config: Dict[str, Any] = training_config.get("optimizer", {})
        self.beta1: float = optimizer_config.get("beta1", 0.9)
        self.beta2: float = optimizer_config.get("beta2", 0.98)
        self.epsilon: float = optimizer_config.get("epsilon", 1e-9)
        lr_schedule_config: Dict[str, Any] = training_config.get("learning_rate_schedule", {})
        # d_model is read from model config.
        model_config: Dict[str, Any] = config.get("model", {})
        self.d_model: int = model_config.get("d_model", 512)
        self.warmup_steps: int = lr_schedule_config.get("warmup_steps", 4000)
        self.label_smoothing: float = training_config.get("label_smoothing", 0.1)
        
        # Checkpoint configuration.
        self.checkpoint_interval: int = config.get("checkpoint_interval", 10000)
        self.checkpoint_dir: str = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # List to store saved checkpoint paths for optional averaging.
        self.checkpoints: List[str] = []

        # Set default pad token id.
        self.pad_id: int = 0  # Default pad token id

        # Initialize Adam optimizer with initial lr set to 0.
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon
        )

        # Optionally get gradient clipping norm if provided.
        self.max_grad_norm: Optional[float] = training_config.get("max_grad_norm", None)

        # Initialize label smoothing loss criterion.
        self.criterion = LabelSmoothingLoss(
            label_smoothing=self.label_smoothing,
            vocab_size=model_config.get("vocab_size", 37000),
            ignore_index=self.pad_id
        )

        # Internal step counter.
        self.current_step: int = 0

        # Set device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logging.info("Trainer initialized with total_steps=%d, d_model=%d, warmup_steps=%d, label_smoothing=%.3f",
                     self.total_steps, self.d_model, self.warmup_steps, self.label_smoothing)

    def _update_learning_rate(self, current_step: int) -> None:
        """
        Update the learning rate according to the schedule:
            lr = d_model^(-0.5) * min(current_step^(-0.5), current_step * warmup_steps^(-1.5))

        Args:
            current_step (int): The current training step (1-indexed).
        """
        # Ensure current_step is at least 1.
        step_val: float = float(max(current_step, 1))
        new_lr: float = (self.d_model ** -0.5) * min(step_val ** -0.5, step_val * (self.warmup_steps ** -1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        # Optionally log the updated learning rate.
        # logging.debug("Updated learning rate to %.6f at step %d", new_lr, current_step)

    def train(self) -> None:
        """
        Main training loop. Iterates over training batches for the total number of steps,
        updates model parameters, updates the learning rate, logs progress, and saves checkpoints.
        """
        self.model.train()
        start_time = time.time()
        train_iter: Iterator = iter(self.train_data)
        progress_bar = tqdm(total=self.total_steps, desc="Training", unit="step")

        while self.current_step < self.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_data)
                batch = next(train_iter)

            # Move batch tensors to device.
            batch = {key: value.to(self.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            
            # Forward pass with teacher forcing.
            # The model expects batch dict with keys "src" and "tgt".
            logits = self.model.forward(batch, mode="train")  # shape: (batch, tgt_len, vocab_size)
            
            # Typically, we shift the target sequence: predict token t+1 given token t.
            # Here we assume the input target includes BOS; we compute loss on predictions for positions 1..end.
            if "tgt" not in batch:
                raise ValueError("Batch missing 'tgt' key for training.")
            tgt: torch.Tensor = batch["tgt"]  # shape: (batch, tgt_len)
            # Shift logits and targets: predict t+1 (thus remove last logit, remove first target token)
            logits_for_loss = logits[:, :-1, :]  # shape: (batch, tgt_len-1, vocab_size)
            target_tokens = tgt[:, 1:]  # shape: (batch, tgt_len-1)

            loss: torch.Tensor = self.criterion(logits_for_loss, target_tokens)
            
            # Backpropagation.
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update learning rate based on current step (1-indexed)
            self._update_learning_rate(self.current_step + 1)
            self.current_step += 1

            # Update progress display.
            current_lr: float = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=loss.item(), lr=f"{current_lr:.6f}")
            progress_bar.update(1)

            # Save checkpoint at specified intervals.
            if self.current_step % self.checkpoint_interval == 0:
                ckpt_path: str = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.current_step}.pt")
                self.save_checkpoint(ckpt_path)
                self.checkpoints.append(ckpt_path)
                logging.info("Checkpoint saved at step %d to %s", self.current_step, ckpt_path)

        progress_bar.close()
        total_time: float = time.time() - start_time
        logging.info("Training complete in %.2f seconds over %d steps", total_time, self.current_step)

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the current model and optimizer states along with the current training step.
        
        Args:
            path (str): File path to save the checkpoint.
        """
        checkpoint: Dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "current_step": self.current_step
        }
        torch.save(checkpoint, path)
        logging.info("Checkpoint successfully saved at %s", path)
