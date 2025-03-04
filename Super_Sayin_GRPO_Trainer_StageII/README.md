# Super Saiyan GRPO Trainer - Stage II

This repository contains the code and documentation for implementing Stage II of the Generative-Reinforcement-Policy-Optimization (GRPO) training for the stonk prediction model. Stage II focuses on fine-tuning a pre-trained Stage I model using a natural market distribution dataset, as opposed to the artificially balanced dataset used in Stage I.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training Process](#training-process)
- [Usage](#usage)
- [Monitoring and Evaluation](#monitoring-and-evaluation)
- [Learning Optimization Tips](#learning-optimization-tips)
- [Troubleshooting](#troubleshooting)
- [Further Research](#further-research)
- [Acknowledgements](#acknowledgements)

## Overview

Stage II training represents a critical advancement in the stonk prediction model's learning process. After Stage I where the model learned fundamental patterns from a balanced dataset, Stage II exposes the model to the natural distribution of market data, allowing it to adapt to real-world conditions while maintaining its foundational knowledge.

**Key improvements in Stage II:**

1. **Natural data distribution**: Trains on the actual market distribution of up/down movements
2. **Enhanced reward function**: Prioritizes data utilization and precision over binary correctness
3. **Adaptive learning**: Tracks and adapts to market bias during training
4. **Low-rank fine-tuning**: Uses smaller LoRA parameters for targeted refinement without catastrophic forgetting
5. **Reduced KL divergence coefficient**: Allows more adaptation while preventing overfitting

## Architecture

The Stage II training architecture is built on top of the Stage I model:

- **Base model**: Qwen2.5-1.5B-Instruct
- **Quantization**: 4-bit quantization for memory efficiency
- **Training technique**: GRPO (Generative Reinforcement Policy Optimization)
- **Adaptation method**: LoRA with reduced rank (r=8) for fine-tuning
- **KL divergence regularization**: Prevents deviation from Stage I model
- **Enhanced prompting**: More comprehensive prompts with additional historical data and metrics

## Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM
  - **Recommended**: GTX 1080 Ti (11GB) or better
  - The code has been optimized to run at 95-100% utilization on a GTX 1080 Ti without OOM errors
- **RAM**: 32GB recommended (16GB minimum)
  - Training processes can use 8-12GB RAM
  - Additional RAM needed for data processing
  - Less RAM may result in slower performance
- **Storage**: At least 50GB free space
  - Python environment: ~6-8GB for Conda environment
  - HuggingFace cache: ~15-20GB for models and datasets
  - Training datasets: ~12GB
  - Model checkpoints: ~5-10GB depending on saved epochs
  - Logs and evaluation results: ~1GB
- **CPU**: 4+ cores recommended for data preprocessing

### Software Requirements

- **CUDA**: 11.7+ (12.0+ recommended)
- **Python**: 3.10+
- **Key libraries**:
  - torch >= 2.0.0
  - transformers >= 4.38.0
  - datasets >= 2.14.0
  - peft >= 0.6.0
  - bitsandbytes >= 0.41.0
  - tqdm
  - numpy
  - re
  - json

## Setup

1. **Clone this repository**:
   ```bash
   git clone <repository-url>
   cd Super_Sayin_GRPO_Trainer_StageII
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Stage I model**:
   Ensure you have a trained Stage I model available. This model should be trained using `Stonk_Trainer.py` from the parent directory.

## Training Process

The training process follows these steps:

1. **Load Stage I model**: The previously trained balanced model is loaded and prepared
2. **Apply LoRA adapters**: Low-rank adaptation is applied for efficient fine-tuning
3. **Load unfiltered dataset**: The dataset is loaded without filtering for balance
4. **Training loop**:
   - Format comprehensive prompts with more historical data
   - Generate predictions with current model
   - Compute reward based on accuracy, confidence, and data utilization
   - Calculate policy gradient loss and KL divergence from reference model
   - Update model parameters with tracked market statistics
5. **Checkpointing**: Models are saved at regular intervals and at the end of each epoch
6. **Evaluation**: Performance is evaluated based on reward metrics and market adaptation

## Usage

### Basic Training

To train a Stage II model with default parameters:

```bash
./train_stage2.sh /path/to/stonk_trainer_grpo/best_model
```

### Advanced Options

For customized training:

```bash
./train_stage2.sh /path/to/stonk_trainer_grpo/best_model --epochs 5 --batch_size 2 --lr 3e-6 --kl_coef 0.03 --save_steps 100
```

### Parameters

- `--stage1_model`: Path to the Stage I model (required)
- `--quantize`: Enable 4-bit quantization (default: enabled)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 4)
- `--lr`: Learning rate (default: 5e-6)
- `--kl_coef`: KL divergence coefficient (default: 0.05)
- `--save_steps`: Steps between saving checkpoints (default: 50)
- `--natural_distribution`: Use natural market distribution (default: enabled)
- `--max_train_samples`: Maximum number of training samples (default: 5000)

## Monitoring and Evaluation

### Training Logs

Training progress is logged in detail:
- `training.log`: Overall training progress and system information
- `stage2_training_log.jsonl`: Detailed per-example results with predictions and rewards
- `market_stats.json`: Statistics on market distribution and model adaptation

### Key Metrics

The following metrics are tracked during training:
- **Average reward**: Overall performance metric
- **Market adaptation**: How well the model adapts to natural distributions
- **Up/down accuracy**: Separate tracking of accuracy for rising vs falling stonks
- **Data utilization**: How well the model leverages specific data points in reasoning

## Learning Optimization Tips

1. **Monitor reward components**: Pay attention to which aspects of the reward function are improving/declining
2. **Balance learning rate**: Too high may cause instability, too low may prevent adaptation
3. **Adjust KL coefficient**: Lower values allow more adaptation, higher values preserve more Stage I knowledge
4. **Track market bias**: Ensure the model isn't overfitting to majority class
5. **Review reasoning quality**: The content of justifications often reveals understanding gaps
6. **Consider gradient accumulation**: If batch sizes are limited by GPU memory
7. **Staged learning rate decay**: Consider reducing learning rate further in later epochs
8. **Use historical backtest**: Test models on realistic market sequences, not just random samples

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size
   - Enable quantization
   - Reduce prompt/response length

2. **NaN losses**:
   - Check for extreme reward values
   - Reduce learning rate
   - Increase epsilon in optimizer

3. **Model divergence**:
   - Increase KL coefficient
   - Clip gradients more aggressively
   - Check for data outliers

4. **Poor performance**:
   - Ensure Stage I model was well-trained
   - Verify dataset quality
   - Increase training samples
   - Consider more frequent checkpointing

## Further Research

Potential areas for improvement:

1. **Sequential training**: Exploring time-based sequences for better temporal understanding
2. **Enhanced metrics**: Incorporating additional financial metrics beyond direction and magnitude
3. **Multi-timeframe analysis**: Training on different prediction horizons simultaneously
4. **Ensemble approaches**: Combining Stage I and Stage II models with different specializations
5. **Uncertainty quantification**: Improving confidence calibration for better risk management
6. **Interpretability**: Enhancing reasoning capabilities and explanations
7. **Market regime detection**: Specialized sub-models for different market conditions

## Acknowledgements

- The Qwen team for the base model
- Special thanks to Lukas Nel the creator of the 2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2 dataset
- The PyTorch and Hugging Face communities
- All contributors to the open-source libraries used in this project

---

## Citation

If you use this code, please cite:

```
@misc{supersaiyangrpo,
  author = {Your Name},
  title = {Super Saiyan GRPO Trainer for Stonk Prediction},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{repository-url}}
}
``` 