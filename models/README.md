# MLX Model Directory Structure

This directory contains the trained and fine-tuned models from MLX distributed training.

## Directory Structure

```
models/
├── fine-tuned/          # Fine-tuned model adapters
│   ├── adapters.safetensors      # Final adapter weights
│   ├── adapter_config.json       # Adapter configuration
│   └── 000000X_adapters.safetensors  # Checkpoint files
├── checkpoints/         # Training checkpoints (if enabled)
└── README.md           # This file
```

## Fine-tuned Models

The `fine-tuned/` directory contains LoRA (Low-Rank Adaptation) adapters that can be applied to the base model. These files include:

- **adapters.safetensors**: The final trained adapter weights
- **adapter_config.json**: Configuration file specifying the adapter parameters
- **Checkpoint files**: Intermediate checkpoints saved during training (e.g., 0000002_adapters.safetensors)

## Usage

To use the fine-tuned model:

```python
from mlx_lm import load

# Load base model with adapters
model, tokenizer = load(
    "mlx-community/gemma-2-2b-it-4bit",
    adapter_path="models/fine-tuned"
)
```

## Training Details

- Base Model: mlx-community/gemma-2-2b-it-4bit
- Training Type: LoRA fine-tuning
- Training Data: Custom JSONL format data
- Distributed Training: Gradient averaging across multiple nodes (when using distributed script)

## Notes

- The adapter files are much smaller than full model weights, making them efficient for storage and deployment
- Checkpoints are saved periodically during training based on the `--save-every` parameter
- For distributed training, ensure all nodes have access to the same model directory
