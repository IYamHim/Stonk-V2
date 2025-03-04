#!/bin/bash

# This script renames files from qwen_stock_test and similar naming to Stonk_Trainer

echo "Starting file renaming process..."

# Check if training is still running
TRAINING_RUNNING=$(ps -ef | grep "qwen_stock_test.py" | grep -v grep | wc -l)

if [ $TRAINING_RUNNING -gt 0 ]; then
    echo "Warning: Training seems to be still running."
    echo "Please wait until training completes before renaming files."
    echo "Current running process:"
    ps -ef | grep "qwen_stock_test.py" | grep -v grep
    exit 1
fi

# Main script file
if [ -f "qwen_stock_test.py" ]; then
    echo "Renaming qwen_stock_test.py to Stonk_Trainer.py"
    cp qwen_stock_test.py Stonk_Trainer.py
    
    # Update any internal references in the file
    sed -i 's/qwen_stock_test/Stonk_Trainer/g' Stonk_Trainer.py
    
    echo "Original file preserved as qwen_stock_test.py"
else
    echo "Warning: qwen_stock_test.py not found"
fi

# Update training scripts
for script in train_direct.sh train_large_dataset.sh; do
    if [ -f "$script" ]; then
        echo "Updating references in $script"
        sed -i 's/qwen_stock_test.py/Stonk_Trainer.py/g' "$script"
    fi
done

# Update testing scripts
for script in test_direct.sh test_model.sh; do
    if [ -f "$script" ]; then
        echo "Updating references in $script"
        sed -i 's/qwen_stock_test.py/Stonk_Trainer.py/g' "$script"
    fi
done

# Rename output directory if it exists
if [ -d "stonk_trainer_grpo" ]; then
    echo "Directory stonk_trainer_grpo already has the correct name"
else
    # Check for alternative naming pattern
    if [ -d "qwen_trainer_grpo" ]; then
        echo "Renaming qwen_trainer_grpo to stonk_trainer_grpo"
        mv qwen_trainer_grpo stonk_trainer_grpo
    fi
fi

# Update any potential references in other Python files
for pyfile in *.py; do
    if [ "$pyfile" != "Stonk_Trainer.py" ] && [ -f "$pyfile" ]; then
        echo "Updating references in $pyfile"
        sed -i 's/qwen_stock_test/Stonk_Trainer/g' "$pyfile"
        sed -i 's/qwen_trainer_grpo/stonk_trainer_grpo/g' "$pyfile"
    fi
done

# Update references in config files and READMEs
for txtfile in *.txt *.md; do
    if [ -f "$txtfile" ]; then
        echo "Updating references in $txtfile"
        sed -i 's/qwen_stock_test/Stonk_Trainer/g' "$txtfile"
        sed -i 's/qwen_trainer_grpo/stonk_trainer_grpo/g' "$txtfile"
    fi
done

echo "File renaming complete!"
echo "Please review the changes and test the renamed files." 