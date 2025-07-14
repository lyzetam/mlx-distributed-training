# MLX Distributed Training Notebook

## Overview

The `mlx_distributed_training.ipynb` notebook provides an interactive interface for running MLX distributed training using all the existing modules and utilities in this repository.

## Features

- **Single Interface**: Run both local and distributed training from one notebook
- **Module Integration**: Uses all existing utility modules (data_preparation, logging, model_cache, training_callbacks, training_core)
- **Configuration Management**: Easy-to-modify training parameters in a single config dictionary
- **Data Validation**: Automatic data format validation and preparation
- **Model Testing**: Built-in model testing after training completion
- **Progress Monitoring**: Real-time training metrics and logging

## Usage

1. Open the notebook in Jupyter or VSCode
2. Run cells sequentially from top to bottom
3. Modify the configuration in Section 2 as needed:
   - Set `distributed: True` for multi-node training
   - Adjust training parameters (batch_size, learning_rate, etc.)
   - Change model or data paths

## Notebook Sections

1. **Setup and Imports**: Load all necessary modules
2. **Configuration**: Define training parameters
3. **Data Preparation**: Validate and prepare training data
4. **Model Setup**: Initialize directories and logging
5. **Training Functions**: Core training logic
6. **Run Training**: Execute local or distributed training
7. **Test Model**: Verify the fine-tuned model works
8. **Model Information**: Display training results and statistics
9. **Cleanup**: Optional cleanup of temporary files

## Requirements

- Python environment with MLX installed
- Training data in JSONL format
- For distributed training: properly configured hostfile and SSH access to nodes

## Output

The notebook creates:
- `models/fine-tuned/`: Fine-tuned model adapters
- `models/checkpoints/`: Training checkpoints
- `logs/`: Training logs
