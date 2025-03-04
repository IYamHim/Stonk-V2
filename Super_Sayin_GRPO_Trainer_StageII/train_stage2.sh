#!/bin/bash

# Super Saiyan GRPO Stage II Training Script
# This script runs the second stage of training with natural market distribution

# Check if a Stage I model path is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_stage1_model> [options]"
    echo "Example: $0 ../checkpoints/best_model --epochs 3 --lr 3e-6"
    exit 1
fi

STAGE1_MODEL=$1
shift  # Remove the first argument (model path)

# Default paths and parameters
OUTPUT_DIR="./super_saiyan_grpo_stage2"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Print startup information
echo "=== Super Saiyan GRPO Trainer Stage II ===" | tee -a $LOG_FILE
echo "Starting training at $(date)" | tee -a $LOG_FILE
echo "Using Stage I model from: $STAGE1_MODEL" | tee -a $LOG_FILE
echo "Output directory: $OUTPUT_DIR" | tee -a $LOG_FILE

# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" | tee -a $LOG_FILE
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')" | tee -a $LOG_FILE
python -c "import torch; print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" | tee -a $LOG_FILE
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB' if torch.cuda.is_available() else 'No GPU')" | tee -a $LOG_FILE

# Run the training with natural distribution
echo "Starting Stage II training with natural market distribution..." | tee -a $LOG_FILE
python grpo_stage2.py \
    --stage1_model $STAGE1_MODEL \
    --quantize \
    --natural_distribution \
    --max_train_samples 5000 \
    "$@" | tee -a $LOG_FILE

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a $LOG_FILE
else
    echo "Training failed at $(date)" | tee -a $LOG_FILE
    exit 1
fi

# Set file as executable
chmod +x train_stage2.sh

echo "Training log saved to $LOG_FILE"
echo "=== Super Saiyan GRPO Trainer Stage II Complete ===" 