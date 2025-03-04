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

# Run the test script on the best model
$PYTHON_PATH Stonk_Trainer.py \
    --test \
    --quantize \
    --model_path stonk_trainer_grpo/best_model/ \
    > test_results.log 2>&1

echo "Testing completed! Check test_results.log for details." 