#!/usr/bin/env python3
"""
Distributed fine-tuning script for MLX based on the DaveAldon example.
This implements gradient averaging across multiple nodes using MPI.
"""

import argparse
import time
import types
import datetime
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.lora import run

# Initialize MPI world
world = mx.distributed.init()
size = world.size()
rank = world.rank()

def all_reduce_grads(grads):
    """Average gradients across all MPI ranks."""
    if size == 1:
        return grads
    # Sum across all ranks, then divide
    return tree_map(lambda g: mx.distributed.all_sum(g) / size, grads)

class DistributedCallback(TrainingCallback):
    """Training callback for distributed gradient averaging."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        
    def on_after_backward(self, model, grads, step):
        """Average gradients after backward pass."""
        new_grads = all_reduce_grads(grads)
        return new_grads
    
    def on_train_loss_report(self, info):
        """Log training loss."""
        iteration = info.get("iteration")
        train_loss = info.get("train_loss")
        if iteration is not None and train_loss is not None:
            self.train_losses.append((iteration, train_loss))
            if rank == 0:  # Only print from rank 0
                print(f"[Train] Iteration {iteration}: Loss = {train_loss:.4f}")
    
    def on_val_loss_report(self, info):
        """Log validation loss."""
        iteration = info.get("iteration")
        val_loss = info.get("val_loss")
        if iteration is not None and val_loss is not None:
            self.val_losses.append((iteration, val_loss))
            if rank == 0:  # Only print from rank 0
                print(f"[Valid] Iteration {iteration}: Loss = {val_loss:.4f}")

def main():
    # Print distributed setup info
    if size == 1:
        print("Single process mode: no gradient averaging needed.")
    else:
        print(f"Distributed mode: Rank {rank}/{size} - averaging gradients across {size} ranks.")
    
    parser = argparse.ArgumentParser(description="Distributed fine-tuning with MLX")
    parser.add_argument("--model", type=str, default="mlx-community/gemma-2-2b-it-4bit")
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--fine-tune-type", type=str, default="lora")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=3)  # Divisible by 3 nodes
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps-per-report", type=int, default=5)
    parser.add_argument("--steps-per-eval", type=int, default=10)
    parser.add_argument("--resume-adapter-file", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default="adapters/distributed")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--test-batches", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-parameters", type=dict,
                        default={"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0})
    parser.add_argument("--lr-schedule", type=str, default=None)
    parser.add_argument("--wandb", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--optimizer-config", type=dict,
                        default={"adam": {}, "adamw": {"weight_decay": 0.01}})
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Create callback for distributed training
    callback = DistributedCallback()
    
    # Run LoRA fine-tuning with gradient averaging
    try:
        run(types.SimpleNamespace(**vars(args)), training_callback=callback)
        
        if rank == 0:
            end_time = time.time()
            print(f"\nDistributed training completed in {end_time - start_time:.2f} seconds")
            print(f"Final training loss: {callback.train_losses[-1][1]:.4f}" if callback.train_losses else "No training loss recorded")
            print(f"Final validation loss: {callback.val_losses[-1][1]:.4f}" if callback.val_losses else "No validation loss recorded")
    
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
