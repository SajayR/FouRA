#!/bin/bash

# Script to run CIFAR-100 training with FouRA
# This is a basic example. You might want to add more configurability (e.g., using command-line arguments for hyperparameters).

# Navigate to the root directory of the project if this script is elsewhere
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# cd "$SCRIPT_DIR/.." # Assuming this script is in examples/ and src/ is in ../

# Ensure the src directory is in PYTHONPATH if you are not running from the root or haven't installed the package
# export PYTHONPATH=$(pwd):$PYTHONPATH # If running from project root

# --- Configuration ---
MODEL_NAME="google/vit-base-patch16-224-in21k"
NUM_LABELS=100
BATCH_SIZE=32 # Adjusted for potentially limited resources, default was 64
NUM_EPOCHS=5  # Number of epochs to train for
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4

# FouRA Specific Config
RANK=16
FOURA_ALPHA=32
TRANSFORM_TYPE="dct" # Options: "none", "fft", "dct"
# TARGET_MODULES should be a Python list-like string if passed directly, e.g., '["query","value"]'
# The train.py script uses a Python list directly, so we will call it from python.
USE_GATE="False" # "True" or "False"

RUN_NAME="FouRA-CIFAR100-Rank${RANK}-${TRANSFORM_TYPE}-Gate${USE_GATE}-Epochs${NUM_EPOCHS}"
USE_WANDB="False" # Set to "True" to enable Weights & Biases logging

# --- Execute Training ---
echo "Starting CIFAR-100 training with FouRA..."
echo "Run Name: ${RUN_NAME}"

# It's generally better to call a Python script that handles argument parsing.
# For now, we directly call the train_model function from within a Python execution context.

python -c "
from src.foura.train import train_model

train_model(
    model_name='$MODEL_NAME',
    num_labels=$NUM_LABELS,
    batch_size=$BATCH_SIZE,
    num_epochs=$NUM_EPOCHS,
    learning_rate=$LEARNING_RATE,
    weight_decay=$WEIGHT_DECAY,
    rank=$RANK,
    foura_alpha=$FOURA_ALPHA,
    transform_type='$TRANSFORM_TYPE',
    target_modules=['query', 'value'],  # Default, or parse from script args
    use_gate=$USE_GATE,
    run_name='$RUN_NAME',
    use_wandb=$USE_WANDB
)
"

echo "Training finished." 