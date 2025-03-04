#!/bin/bash

# Two approaches to handle conda activation in script:
# OPTION 1: Source conda initialization first
source ~/miniconda3/etc/profile.d/conda.sh || source ~/.conda/etc/profile.d/conda.sh
conda activate qwen_env

# OPTION 2: If the above doesn't work, use this approach instead
# (Remove the # from these lines and comment out the above two lines)
#export PATH="/home/ai/miniconda3/envs/qwen_env/bin:$PATH" 
#export CONDA_PREFIX="/home/ai/miniconda3/envs/qwen_env"

# Run the training script with improved parameters
python Stonk_Trainer.py \
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