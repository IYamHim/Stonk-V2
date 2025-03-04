#!/bin/bash

# Two approaches to handle conda activation in script:
# OPTION 1: Source conda initialization first
source ~/miniconda3/etc/profile.d/conda.sh || source ~/.conda/etc/profile.d/conda.sh
conda activate qwen_env

# OPTION 2: If the above doesn't work, use this approach instead
# (Remove the # from these lines and comment out the above two lines)
#export PATH="/home/ai/miniconda3/envs/qwen_env/bin:$PATH" 
#export CONDA_PREFIX="/home/ai/miniconda3/envs/qwen_env"

# Run the test script on the best model
python Stonk_Trainer.py \
    --test \
    --quantize \
    --model_path stonk_trainer_grpo/best_model/ \
    > test_results.log 2>&1

echo "Testing completed! Check test_results.log for details." 