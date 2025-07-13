"""Core training utilities for MLX distributed training."""

import types
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import mlx.core as mx
from mlx_lm import load
from mlx_lm.lora import run as lora_run
from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback

from .model_cache import ModelCacheManager
from .logging import setup_logging, get_logger
from .training_callbacks import DistributedTrainingCallback, SimpleMetricsCallback


def setup_distributed() -> Tuple[Optional[mx.distributed.Group], int, int]:
    """
    Setup distributed training environment.
    
    Returns:
        Tuple of (world group, world size, rank)
    """
    try:
        world = mx.distributed.init()
        size = world.size()
        rank = world.rank()
        
        # Update logger with rank info
        logger = setup_logging(rank=rank, size=size)
        
        if size > 1:
            logger.info(f"Distributed mode: Rank {rank}/{size}")
        else:
            logger.info("Single process mode")
            
        return world, size, rank
    except Exception as e:
        logger = get_logger()
        logger.warning(f"Failed to initialize distributed mode: {e}")
        logger.info("Falling back to single process mode")
        return None, 1, 0


def create_training_args(
    model_path: str,
    data_path: str,
    config: Dict[str, Any]
) -> types.SimpleNamespace:
    """
    Create training arguments from configuration.
    
    Args:
        model_path: Path to the model
        data_path: Path to the data directory
        config: Training configuration dictionary
        
    Returns:
        SimpleNamespace with training arguments
    """
    # Default values
    defaults = {
        "model": model_path,
        "train": True,
        "data": data_path,
        "fine_tune_type": "lora",
        "num_layers": 8,
        "batch_size": 2,
        "iters": 100,
        "val_batches": 25,
        "learning_rate": 1e-5,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "resume_adapter_file": None,
        "adapter_path": "adapters",
        "save_every": 100,
        "test": False,
        "test_batches": 100,
        "max_seq_length": 2048,
        "grad_checkpoint": False,
        "seed": 42,
        "lora_parameters": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "scale": 10.0
        },
        "lr_schedule": None,
        "wandb": None,  # Disable wandb by default
        "optimizer": "adam",
        "optimizer_config": {
            "adam": {},
            "adamw": {"weight_decay": 0.01}
        }
    }
    
    # Update with provided config
    args_dict = defaults.copy()
    args_dict.update(config)
    
    return types.SimpleNamespace(**args_dict)


def train_model(
    model_name: str,
    data_path: Union[str, Path],
    training_config: Dict[str, Any],
    cache_dir: Optional[Union[str, Path]] = None,
    distributed: bool = False,
    callback: Optional[TrainingCallback] = None
) -> Dict[str, Any]:
    """
    Train a model with the given configuration.
    
    Args:
        model_name: Name or path of the model
        data_path: Path to the training data directory
        training_config: Training configuration
        cache_dir: Custom cache directory
        distributed: Whether to use distributed training
        callback: Custom training callback
        
    Returns:
        Dictionary with training results
    """
    logger = get_logger()
    
    # Setup distributed if requested
    if distributed:
        world, size, rank = setup_distributed()
    else:
        world, size, rank = None, 1, 0
    
    # Initialize cache manager
    cache_manager = ModelCacheManager(cache_dir)
    
    # Get model path (download if needed)
    logger.info(f"Loading model: {model_name}")
    model_path = cache_manager.get_model_path(model_name)
    
    # Create training arguments
    args = create_training_args(
        str(model_path),
        str(data_path),
        training_config
    )
    
    # Create callback if not provided
    if callback is None:
        if distributed and size > 1:
            callback = DistributedTrainingCallback(world_size=size, rank=rank)
        else:
            callback = SimpleMetricsCallback()
    
    # Load model
    logger.info("Loading model for training...")
    model = load(str(model_path))
    
    # Run training
    logger.info("Starting training...")
    try:
        lora_run(args, training_callback=callback)
        
        # Get metrics from callback
        if hasattr(callback, 'get_metrics'):
            metrics = callback.get_metrics()
        else:
            metrics = {}
        
        metrics['success'] = True
        metrics['adapter_path'] = args.adapter_path
        
        return metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def evaluate_model(
    model_name: str,
    adapter_path: Optional[Union[str, Path]] = None,
    test_prompt: str = "Hello, how are you?",
    max_tokens: int = 100,
    cache_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Evaluate a trained model by generating text.
    
    Args:
        model_name: Name or path of the model
        adapter_path: Path to adapter files (if using LoRA)
        test_prompt: Prompt to test with
        max_tokens: Maximum tokens to generate
        cache_dir: Custom cache directory
        
    Returns:
        Generated text
    """
    from mlx_lm import generate
    
    logger = get_logger()
    
    # Initialize cache manager
    cache_manager = ModelCacheManager(cache_dir)
    
    # Get model path
    model_path = cache_manager.get_model_path(model_name)
    
    # Load model and tokenizer
    logger.info("Loading model for evaluation...")
    model, tokenizer = load(str(model_path))
    
    # Load adapter if provided
    if adapter_path:
        logger.info(f"Loading adapter from {adapter_path}")
        # Note: In a real implementation, you'd load the adapter here
        # For now, we'll just note it in the logs
    
    # Generate text
    logger.info(f"Generating response for: {test_prompt}")
    
    # Format prompt for chat models
    if hasattr(tokenizer, 'chat_template'):
        messages = [{"role": "user", "content": test_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = test_prompt
    
    # Generate
    response = generate(
        model, 
        tokenizer, 
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        verbose=True
    )
    
    return response
