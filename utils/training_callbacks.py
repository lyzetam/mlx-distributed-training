"""Training callbacks for distributed MLX training."""

import time
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.tuner.trainer import TrainingCallback


class DistributedTrainingCallback(TrainingCallback):
    """Callback for distributed training with gradient averaging."""
    
    def __init__(self, world_size: int = 1, rank: int = 0):
        """
        Initialize the distributed training callback.
        
        Args:
            world_size: Total number of processes
            rank: Current process rank
        """
        self.world_size = world_size
        self.rank = rank
        self.train_losses = []
        self.val_losses = []
        self.start_time = time.time()
        
    def on_after_backward(self, model, grads, step):
        """Average gradients across all processes after backward pass."""
        if self.world_size > 1:
            # Average gradients across all ranks
            return tree_map(
                lambda g: mx.distributed.all_sum(g) / self.world_size, 
                grads
            )
        return grads
    
    def on_train_loss_report(self, info: Dict[str, Any]):
        """Record training loss."""
        iteration = info.get("iteration")
        train_loss = info.get("train_loss")
        if iteration is not None and train_loss is not None:
            self.train_losses.append((iteration, train_loss))
            if self.rank == 0:  # Only print from rank 0
                elapsed = time.time() - self.start_time
                print(f"[Train] Iteration {iteration}: Loss = {train_loss:.4f}, "
                      f"Time = {elapsed:.1f}s")
    
    def on_val_loss_report(self, info: Dict[str, Any]):
        """Record validation loss."""
        iteration = info.get("iteration")
        val_loss = info.get("val_loss")
        if iteration is not None and val_loss is not None:
            self.val_losses.append((iteration, val_loss))
            if self.rank == 0:  # Only print from rank 0
                print(f"[Valid] Iteration {iteration}: Loss = {val_loss:.4f}")
    
    def plot_losses(self, save_path: Optional[str] = None, show: bool = True):
        """Plot training and validation losses."""
        if not self.train_losses and not self.val_losses:
            print("No losses to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        if self.train_losses:
            train_its, train_vals = zip(*self.train_losses)
            plt.plot(train_its, train_vals, '-o', label='Train Loss', alpha=0.7)
        
        if self.val_losses:
            val_its, val_vals = zip(*self.val_losses)
            plt.plot(val_its, val_vals, '-o', label='Validation Loss', alpha=0.7)
        
        plt.title("Training Progress")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        metrics = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "training_time": time.time() - self.start_time,
            "world_size": self.world_size,
            "rank": self.rank
        }
        
        if self.train_losses:
            final_train_loss = self.train_losses[-1][1]
            metrics["final_train_loss"] = final_train_loss
        
        if self.val_losses:
            final_val_loss = self.val_losses[-1][1]
            metrics["final_val_loss"] = final_val_loss
        
        return metrics


class SimpleMetricsCallback(TrainingCallback):
    """Simple callback for tracking metrics without distributed features."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.start_time = time.time()
    
    def on_train_loss_report(self, info: Dict[str, Any]):
        """Record training loss."""
        iteration = info.get("iteration")
        train_loss = info.get("train_loss")
        if iteration is not None and train_loss is not None:
            self.train_losses.append((iteration, train_loss))
            elapsed = time.time() - self.start_time
            print(f"[Train] Iteration {iteration}: Loss = {train_loss:.4f}, "
                  f"Time = {elapsed:.1f}s")
    
    def on_val_loss_report(self, info: Dict[str, Any]):
        """Record validation loss."""
        iteration = info.get("iteration")
        val_loss = info.get("val_loss")
        if iteration is not None and val_loss is not None:
            self.val_losses.append((iteration, val_loss))
            print(f"[Valid] Iteration {iteration}: Loss = {val_loss:.4f}")
