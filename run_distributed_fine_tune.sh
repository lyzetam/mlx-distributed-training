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

# Make script executable and copy to all nodes
chmod +x scripts/fine_tune_distributed.py

echo ""
echo "Syncing script to all nodes..."
for node in 10.85.100.221 10.85.100.222; do
    echo "Copying to $node..."
    scp scripts/fine_tune_distributed.py $node:/Users/mm/mlx-distributed-training/scripts/
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
    --save-every 5

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Distributed fine-tuning completed successfully!"
    echo "=================================================="
    echo ""
    echo "Results:"
    echo "- Check 'adapters/distributed/' for output files"
    echo "- Training used gradient averaging across $NUM_NODES nodes"
else
    echo ""
    echo "=================================================="
    echo "❌ Distributed fine-tuning failed!"
    echo "=================================================="
fi
