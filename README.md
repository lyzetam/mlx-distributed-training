# MLX Distributed Training

A working implementation of distributed training for Apple MLX across multiple Mac nodes using MPI and gradient averaging.

## Overview

This repository provides a complete solution for distributed training of language models using Apple's MLX framework. It implements the gradient averaging approach demonstrated in the MLX documentation, allowing you to train models across multiple Mac machines efficiently.

## Features

- **Distributed Training**: Train models across multiple Mac nodes using MPI
- **Gradient Averaging**: Implements the proven gradient averaging approach from MLX examples
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **Home Assistant Integration**: Example dataset for training voice command models
- **Memory Optimization**: Configurable batch sizes and sequence lengths for different hardware

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- MPI installed on all nodes (`brew install open-mpi`)
- SSH access between nodes without password prompts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lyzetam/mlx-distributed-training.git
cd mlx-distributed-training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MPI hosts:
Edit the `hostfile` to include your node IPs:
```
10.85.100.220 slots=1
10.85.100.221 slots=1
10.85.100.222 slots=1
```

## Quick Start

Run distributed training across your configured nodes:

```bash
./run_distributed_fine_tune.sh
```

## Repository Structure

```
mlx-distributed-training/
├── scripts/                     # Training scripts
│   ├── fine_tune_distributed.py # Main distributed training script
│   └── train_homeassistant.py   # Single-node training option
├── config/                      # Configuration files
│   └── homeassistant_training_config.yaml
├── utils/                       # Training utilities
│   ├── training_core.py
│   ├── training_callbacks.py
│   └── data_preparation.py
├── data/                        # Example datasets
│   ├── example_train.jsonl
│   └── example_valid.jsonl
├── hostfile                     # MPI host configuration
├── requirements.txt             # Python dependencies
└── run_distributed_fine_tune.sh # Main execution script
```

## Configuration

### Training Parameters

Edit `config/homeassistant_training_config.yaml` to adjust:
- Model selection
- Batch size (must be divisible by number of nodes)
- Learning rate
- Number of iterations
- LoRA parameters

### Network Configuration

The training uses MPI for inter-node communication. Ensure:
1. All nodes can reach each other via the IPs in `hostfile`
2. SSH is configured for passwordless access
3. The same code and data exist on all nodes

## Usage Examples

### Basic Distributed Training
```bash
./run_distributed_fine_tune.sh
```

### Custom Configuration
```bash
./run_distributed_training.sh config/custom_config.yaml
```

### Single Node Training
```bash
python scripts/train_homeassistant.py --config config/homeassistant_training_config.yaml
```

## Results

After successful training:
- Adapter weights are saved to `adapters/distributed/`
- Training logs show loss progression
- Final model can be loaded with the base model + adapters

### Using the Trained Model

```python
from mlx_lm import load, generate

# Load base model with trained adapters
model, tokenizer = load(
    "mlx-community/gemma-2-2b-it-4bit",
    adapter_path="adapters/distributed/"
)

# Generate text
prompt = "turn on the living room lights"
response = generate(model, tokenizer, prompt=prompt, max_tokens=50)
print(response)
```

## Performance

- Training time: ~13 seconds for 10 iterations across 3 nodes
- Memory usage: ~2GB per node with current configuration
- Scalability: Tested with up to 3 nodes

## Troubleshooting

### Common Issues

1. **Network connectivity errors**
   - Verify all nodes can ping each other
   - Check firewall settings
   - Ensure MPI can communicate on required ports

2. **GPU timeout errors**
   - Reduce batch size or sequence length
   - Enable gradient checkpointing
   - Use fewer layers for fine-tuning

3. **Memory issues**
   - Use 4-bit quantized models
   - Reduce batch size
   - Enable memory offloading

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Apple MLX team for the framework and examples
- DaveAldon for the gradient averaging implementation reference
- The open-source community for MPI and related tools
