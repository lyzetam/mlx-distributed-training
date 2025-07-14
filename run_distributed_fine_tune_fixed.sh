#!/bin/bash

# Distributed fine-tuning script based on DaveAldon's approach

echo "=================================================="
echo "MLX Distributed Fine-Tuning (Gradient Averaging)"
echo "=================================================="

# Configuration
HOSTFILE="hostfile"
NUM_NODES=$(grep -v '^#' $HOSTFILE | grep -v '^$' | wc -l | tr -d ' ')

echo "Using MPI hostfile: $HOSTFILE"
echo "Configured hosts:"
grep -v '^#' $HOSTFILE | grep -v '^$' | awk '{print "  - " $1}'

# Make script executable
chmod +x scripts/fine_tune_distributed.py

# Get all unique nodes from hostfile
NODES=$(grep -v '^#' $HOSTFILE | grep -v '^$' | awk '{print $1}' | sort -u)

# Copy data files with expected names
echo "Preparing data files..."
cp data/example_train.jsonl data/train.jsonl
cp data/example_valid.jsonl data/valid.jsonl

echo ""
echo "Creating directories and syncing to all nodes..."
for node in $NODES; do
    # Skip local node
    if [[ "$node" == "10.85.100.220" ]] || [[ "$node" == "localhost" ]] || [[ "$node" == "$(hostname -I | awk '{print $1}')" ]]; then
        echo "Skipping local node $node"
        continue
    fi
    
    echo "Setting up $node..."
    # Create directory structure on remote node
    ssh $node "mkdir -p /Users/mm/mlx-distributed-training/scripts /Users/mm/mlx-distributed-training/data /Users/mm/mlx-distributed-training/adapters/distributed /Users/mm/mlx-distributed-training/models/fine-tuned" 2>/dev/null || {
        echo "Warning: Could not create directories on $node"
    }
    
    # Copy script to remote node
    scp scripts/fine_tune_distributed.py $node:/Users/mm/mlx-distributed-training/scripts/ 2>/dev/null || {
        echo "Warning: Could not copy script to $node"
    }
    
    # Copy data files
    scp data/train.jsonl data/valid.jsonl $node:/Users/mm/mlx-distributed-training/data/ 2>/dev/null || {
        echo "Warning: Could not copy data to $node"
    }
done

echo ""
echo "Starting distributed fine-tuning with gradient averaging..."
echo "=================================================="

# Check for mlx-env
MLX_ENV_PATH="/Users/mm/mlx-project/mlx-env"
if [ -d "$MLX_ENV_PATH" ]; then
    PYTHON_EXEC="$MLX_ENV_PATH/bin/python"
else
    PYTHON_EXEC="python3"
fi

# Create local output directory
mkdir -p adapters/distributed
mkdir -p models/fine-tuned

# Run distributed training
# Using the same approach as the PDF example
mpirun --hostfile $HOSTFILE \
    -np $NUM_NODES \
    --map-by node \
    --bind-to none \
    --mca btl self,tcp \
    --mca btl_tcp_if_include 10.85.100.0/24 \
    -x PYTHONPATH="$PYTHONPATH:$(pwd)" \
    -x PATH="$PATH" \
    -wdir $(pwd) \
    $PYTHON_EXEC scripts/fine_tune_distributed.py \
    --model mlx-community/gemma-2-2b-it-4bit \
    --data data \
    --batch-size 3 \
    --iters 10 \
    --num-layers 4 \
    --max-seq-length 128 \
    --learning-rate 1e-4 \
    --steps-per-report 2 \
    --steps-per-eval 5 \
    --save-every 5 \
    --adapter-path models/fine-tuned

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Distributed fine-tuning completed successfully!"
    echo "=================================================="
    echo ""
    echo "Results:"
    echo "- Check 'models/fine-tuned/' for output files"
    echo "- Training used gradient averaging across $NUM_NODES nodes"
    echo ""
    echo "Directory structure:"
    echo "- models/fine-tuned/    : Final fine-tuned model"
    echo "- models/checkpoints/   : Training checkpoints"
    echo "- adapters/distributed/ : Distributed training adapters"
else
    echo ""
    echo "=================================================="
    echo "❌ Distributed fine-tuning failed!"
    echo "=================================================="
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Ensure all nodes in hostfile are accessible via SSH"
    echo "2. Check that MPI is installed on all nodes"
    echo "3. Verify that Python and MLX are installed on all nodes"
    echo "4. Check network connectivity between nodes"
    echo ""
    echo "For testing, you can run the local version: ./run_local_fine_tune.sh"
fi
