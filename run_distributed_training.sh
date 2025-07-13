#!/bin/bash

# MLX Distributed Training Script
# This script runs training across multiple nodes using MPI

echo "=================================================="
echo "MLX Distributed Training"
echo "=================================================="

# Check if config file is provided
CONFIG_FILE=${1:-"config/training_config.yaml"}
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Usage: ./run_distributed_training.sh [config_file]"
    echo "Example: ./run_distributed_training.sh config/homeassistant_training_config.yaml"
    exit 1
fi

echo "Using configuration: $CONFIG_FILE"

# Check if hostfile exists (try both hostfile and hosts.json)
if [ -f "hostfile" ]; then
    HOSTFILE="hostfile"
    # Count number of nodes from hostfile
    NUM_NODES=$(grep -v '^#' hostfile | grep -v '^$' | wc -l | tr -d ' ')
    echo "Using MPI hostfile: $HOSTFILE"
    echo "Configured hosts:"
    grep -v '^#' hostfile | grep -v '^$' | awk '{print "  - " $1}'
elif [ -f "hosts.json" ]; then
    # Convert hosts.json to MPI hostfile format
    echo "Converting hosts.json to MPI hostfile format..."
    python3 -c "
import json
with open('hosts.json') as f:
    hosts = json.load(f)
with open('hostfile', 'w') as f:
    for host in hosts:
        ip = host.get('ssh', host.get('ips', [''])[0])
        f.write(f'{ip} slots=1\\n')
"
    HOSTFILE="hostfile"
    NUM_NODES=$(python3 -c "import json; print(len(json.load(open('hosts.json'))))")
    echo "Created MPI hostfile from hosts.json"
    echo "Configured hosts:"
    python3 -c "import json; hosts=json.load(open('hosts.json')); [print(f'  - {h.get(\"ssh\", h.get(\"ips\", [\"\"])[0])}') for h in hosts]"
else
    echo "Error: No hostfile or hosts.json found!"
    echo "Please create either:"
    echo "  - hostfile (MPI format): IP_ADDRESS slots=1"
    echo "  - hosts.json (JSON format)"
    exit 1
fi

# Check MPI installation
if ! command -v mpirun &> /dev/null; then
    echo "Error: MPI not found! Please install MPI first."
    echo "On macOS: brew install open-mpi"
    exit 1
fi

echo ""
echo "Starting distributed training with $NUM_NODES nodes..."
echo "=================================================="

# Determine which training script to use based on config
if [[ "$CONFIG_FILE" == *"homeassistant"* ]]; then
    TRAINING_SCRIPT="train_homeassistant.py"
    TRAINING_ARGS="--config $CONFIG_FILE --distributed"
else
    # For general training, we'll use the training module directly
    TRAINING_SCRIPT="-m utils.training_core"
    TRAINING_ARGS="--config $CONFIG_FILE"
fi

# Get the current directory for proper path resolution
CURRENT_DIR=$(pwd)

# Detect the Python environment
# Check if we're in a conda/virtual environment
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
    echo "Using conda environment: $CONDA_PREFIX"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
    echo "Using virtual environment: $VIRTUAL_ENV"
else
    # Try to find mlx-env specifically
    if [ -d "$HOME/mlx-env" ]; then
        PYTHON_EXEC="$HOME/mlx-env/bin/python"
        echo "Using mlx-env at: $HOME/mlx-env"
    elif [ -d "/Users/mm/mlx-env" ]; then
        PYTHON_EXEC="/Users/mm/mlx-env/bin/python"
        echo "Using mlx-env at: /Users/mm/mlx-env"
    elif [ -d "/Users/mm/mlx-project/mlx-env" ]; then
        PYTHON_EXEC="/Users/mm/mlx-project/mlx-env/bin/python"
        echo "Using mlx-env at: /Users/mm/mlx-project/mlx-env"
    else
        PYTHON_EXEC="python3"
        echo "Warning: No mlx-env found, using system python3"
    fi
fi

echo "Python executable: $PYTHON_EXEC"

# Run distributed training
mpirun --hostfile $HOSTFILE \
    -np $NUM_NODES \
    --map-by node \
    --bind-to none \
    -x PYTHONPATH=$PYTHONPATH:$CURRENT_DIR \
    -x PATH=$PATH \
    -wdir $CURRENT_DIR \
    $PYTHON_EXEC $TRAINING_SCRIPT $TRAINING_ARGS

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Distributed training completed successfully!"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "❌ Distributed training failed!"
    echo "=================================================="
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Ensure all nodes are accessible via SSH"
    echo "2. Check that Python and MLX are installed on all nodes"
    echo "3. Verify the same code is available on all nodes"
    echo "4. Check MPI configuration with: mpirun --hostfile $HOSTFILE -np $NUM_NODES hostname"
    exit 1
fi
