#!/bin/bash

# Direct approach using the full path to the Python interpreter
# Corrected path based on conda env list
PYTHON_PATH="/home/ai/miniconda3/envs/qwen_env/bin/python"

# Check if the Python interpreter exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Python interpreter not found at $PYTHON_PATH"
    echo "Please edit this script to specify the correct path"
    exit 1
fi

# Run the training script with improved parameters
$PYTHON_PATH Stonk_Trainer.py \
    --train \
    --quantize \
    --epochs 5 \
    --batch_size 8 \
    --lr 1e-5 \
    --kl_coef 0.15 \
    --save_steps 100 \
    --diverse_predictions \
    --max_train_samples 2000 \
    > training_large_dataset.log 2>&1

echo "Training completed! Check training_large_dataset.log for details." 