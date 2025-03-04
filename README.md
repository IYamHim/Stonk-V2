The information provided here is for testing and educational purposes only and should not be construed as financial advice. Please consult with a licensed financial advisor before making any financial decisions. This is all theoretical and not proven to work! This is a work in progress and nothing is guaranteed and things may/will break. 

# Stonk-Trainer v2

An advanced stonk market prediction model using Generative Reinforcement Policy Optimization (GRPO) with the Qwen2.5-1.5B-Instruct model, featuring enhanced reward functions and training methodology.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Training Process](#training-process)
- [Training Architecture Diagram](#training-architecture-diagram)
- [Model Architecture](#model-architecture)
- [Reward Function](#reward-function)
- [Directory Structure](#directory-structure)
- [Results & Evaluation](#results--evaluation)
- [Super Saiyan Mode (Stage II)](#super-saiyan-mode-stage-ii)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

Stonk-Trainer v2 is a specialized framework for training language models to predict stonk market movements. It leverages GRPO (Generative Reinforcement Policy Optimization) to train LLMs specifically on stonk prediction tasks, incorporating concepts from reinforcement learning and behavioral economics.

The model is trained to analyze a company's stonk information (including historical data, price movements, and news) and predict whether the stonk will go up or down, along with percentage change estimates and confidence levels.

## Key Features

### 1. Enhanced Reward Function
The reward function has been significantly improved to encourage data-driven predictions and appropriate confidence calibration:

- **Confidence Penalty for Wrong Predictions**: Applies a sliding scale penalty (from -0.2 to -1.5) based on confidence level when the prediction direction is incorrect.
- **Data Utilization Reward (15%)**: Rewards the model for referencing specific data points in its reasoning.
- **Rebalanced Components**: Direction (35%), Magnitude (20%), Format (20%), Confidence (10%), Data Utilization (15%).

### 2. Diverse Training Examples
- Implements filtering to create a balanced dataset with equal representation of upward and downward price movements.
- Applies diversity penalties during training if the model shows bias toward consistently predicting one direction.

### 3. Improved Training Stability
- Enhanced KL divergence calculation for more stable GRPO training.
- Better gradient handling and loss computation.
- 4-bit quantization support for efficient training.

### 4. Two-Stage Training Pipeline
- **Stage I**: Balanced dataset training using modified GRPO.
- **Stage II**: "Super Saiyan" mode with natural distribution training (see [Super Saiyan Mode](#super-saiyan-mode-stage-ii)).

## Installation
Linux is the only OS that has been tested to work. It is possible that this will work on Windows but it has not been confirmed. The current implementation requires CUDA and therefore Mac is not supported at this time.

### Requirements
- Python 3.10+
- CUDA-capable GPU with 11GB+ VRAM (24GB recommended for larger batch sizes)
- 32GB+ System RAM (recommended)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Stonk-Trainer.git
cd Stonk-Trainer/Stonk-Trainer_v2
```

2. Create a conda environment (recommended):
```bash
conda create -n stonk_env python=3.10
conda activate stonk_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Make scripts executable:
```bash
chmod +x *.sh
```

## Usage

### Training

Two options are provided for running the training:

#### Option 1: Using the direct script (recommended)
```bash
./train_direct.sh
```

#### Option 2: Using the conda-based script
```bash
./train_large_dataset.sh
```

You may need to edit the Python path in the scripts to match your environment.

#### Training Parameters

Key parameters (customizable in the scripts):
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size for training (default: 8)
- `--lr`: Learning rate (default: 1e-5) # Experiment with other values ex.: 5e-5
- `--kl_coef`: KL divergence coefficient (default: 0.15) # Experiment with other values ex.: 0.05-0.2
- `--save_steps`: Steps between saving checkpoints (default: 100)
- `--diverse_predictions`: Enable diversity penalties
- `--max_train_samples`: Maximum number of training samples (default: 2000)

### Testing

After training completes, test the model with:

#### Option 1: Using the direct script (recommended)
```bash
./test_direct.sh
```

#### Option 2: Using the conda-based script
```bash
./test_model.sh
```

Test results will be saved to `test_results.log`.

## Training Process

1. **Data Loading**: The model loads the 2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2 dataset.
2. **Data Filtering**: Creates a balanced dataset with equal up/down examples (~505 up, ~504 down).
3. **Model Preparation**: Loads the Qwen2.5-1.5B-Instruct model with 4-bit quantization and applies LoRA adapters.
4. **GRPO Training Loop**:
   - Generates predictions for stonk data
   - Computes rewards based on prediction quality
   - Calculates policy gradient loss and KL divergence
   - Updates model parameters
5. **Checkpointing**: Saves the model regularly and keeps the best-performing version.

## Training Architecture Diagram

The Stonk-Trainer v2 employs a two-stage GRPO (Generative Reinforcement Policy Optimization) training architecture:

```
┌──────────────────────────── Stage I ────────────────────────────┐  ┌──────────────────────── Stage II ───────────────────────────┐
│                                                                 │  │                                                              │
│  ┌─────────┐     ┌──────────────┐     ┌────────────────┐        │  │  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ Qwen2.5 │────>│ 4-bit Quant  │────>│  Balanced      │        │  │  │ Stage I     │───>│ 4-bit Quant  │───>│ Natural        │  │
│  │ Model   │     │ LoRA (r=16)  │     │  Dataset       │        │  │  │ Model       │    │ LoRA (r=8)   │    │ Distribution   │  │
│  └─────────┘     └──────────────┘     └────────────────┘        │  │  └─────────────┘    └──────────────┘    └────────────────┘  │
│       │                │                     │                  │  │        │                 │                     │            │
│       │                │                     │                  │  │        │                 │                     │            │
│       v                v                     v                  │  │        v                 v                     v            │
│  ┌─────────┐     ┌──────────────┐     ┌────────────────┐        │  │  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │Reference│     │Reward Function│    │Policy Gradient │        │  │  │ Reference   │    │Enhanced      │    │Market Stats    │  │
│  │Model    │────>│Format:   20% │────>│KL coef: 0.15   │        │  │  │ Model       │───>│Reward        │───>│Tracking        │  │
│  │(Frozen) │     │Direction: 35%│     │                │        │  │  │ (Stage I)   │    │Direction: 25%│    │Up/Down Ratio   │  │
│  └─────────┘     │Magnitude: 20%│     └────────────────┘        │  │  └─────────────┘    │Magnitude: 20%│    └────────────────┘  │
│                  │Confidence:10%│              │                │  │                     │Confidence:25%│                │       │
│                  │Data Use: 15% │              │                │  │                     │Data Use: 30% │                │       │
│                  └──────────────┘              v                │  │                     └──────────────┘                v       │
│                                        ┌────────────────┐       │  │                                         ┌────────────────┐  │
│                                        │Best Model      │       │  │                                         │KL-Divergence   │  │
│                                        │Checkpointing   │───────┼──┼────────────────────────────────────────>│KL coef: 0.05   │  │
│                                        └────────────────┘       │  │                                         └────────────────┘  │
│                                                                 │  │                                                │            │
└─────────────────────────────────────────────────────────────────┘  └────────────────────────────────────────────────┼────────────┘
                                                                                                                      │
                                                                                                                      v
                                                                                                          ┌────────────────────┐
                                                                                                          │   Final Adapted    │
                                                                                                          │   Stonk Prediction │
                                                                                                          │   Model            │
                                                                                                          └────────────────────┘
```

### Key GRPO Elements:

1. **Two-Stage Process**:
   - Stage I: Training with balanced dataset (equal up/down examples)
   - Stage II: Fine-tuning with natural market distribution

2. **Reference Model**: Frozen copy of base model used to calculate KL divergence

3. **Reward Function Components**:
   - Direction reward (35%): Correctly predicting up/down movement
   - Magnitude reward (20%): Accuracy of percentage change prediction
   - Format reward (20%): Following correct output format
   - Confidence reward (10%): Appropriate confidence calibration
   - Data utilization reward (15%): Referencing specific data points

4. **Policy Gradient with KL Divergence**:
   - Stage I: Higher KL coefficient (0.15) for stability
   - Stage II: Lower KL coefficient (0.05) for adaptation

5. **Implementation Details**:
   - Stage I: 4-bit quantization with LoRA (r=16)
   - Stage II: 4-bit quantization with LoRA (r=8)
   - Best model checkpointing between stages

This unified GRPO approach ensures the model learns fundamental prediction patterns in a balanced setting before adapting to real-world market distributions while maintaining prediction quality.

## Model Architecture

- **Base Model**: Qwen2.5-1.5B-Instruct
- **Adaptation**: LoRA (Low-Rank Adaptation) with r=16
- **Quantization**: 4-bit precision for memory efficiency
- **Optimization**: AdamW with gradient clipping
- **Regularization**: KL divergence from reference model

## Reward Function

The reward function is a critical component that evaluates prediction quality across multiple dimensions:

### Format Reward (20%)
- Checks if model follows the correct format with thinking sections.

### Direction Reward (35%)
- Rewards correct prediction of price movement direction (up/down).

### Magnitude Reward (20%)
- Evaluates the accuracy of the percentage change prediction.

### Confidence Reward (10%)
- Rewards appropriate confidence levels and penalizes overconfidence on wrong predictions.

### Data Utilization Reward (15%)
- Rewards referencing specific data points (ticker, price, news, etc.).

## Directory Structure
- `Stonk_Trainer.py` - Main training and testing script with improved reward function
- `requirements.txt` - Dependencies required for running the model
- `train_direct.sh` / `train_large_dataset.sh` - Scripts to run training with optimized parameters
- `test_direct.sh` / `test_model.sh` - Scripts to test the trained model
- `Super_Sayin_GRPO_Trainer_StageII/` - Advanced Stage II training framework
- `stonk_trainer_grpo/` - Output directory for trained models and checkpoints

## Results & Evaluation

Training produces several outputs:
- `training_large_dataset.log`: Training progress and statistics
- `test_results.log`: Evaluation results after testing
- `stonk_trainer_grpo/`: Directory containing:
  - `checkpoints/`: Periodic model snapshots
  - `best_model/`: Best performing model based on avg reward
  - `evaluation_results/`: Visualizations and metrics (when using test scripts)

## Super Saiyan Mode (Stage II)

After training a model with the balanced dataset (Stage I), you can proceed to "Super Saiyan Mode" (Stage II) which:

- Trains on a natural market distribution (not artificially balanced)
- Uses enhanced reward functions that prioritize data utilization
- Adapts to market bias through advanced tracking
- Implements low-rank fine-tuning to prevent catastrophic forgetting

To use Super Saiyan Mode:
```bash
cd Super_Sayin_GRPO_Trainer_StageII
./setup.sh
./train_stage2.sh ../stonk_trainer_grpo/best_model
```

See the `Super_Sayin_GRPO_Trainer_StageII/README.md` for detailed instructions.

## Troubleshooting

### Common Issues

1. **CUDA out of memory errors**:
   - Reduce batch size in the training script
   - Ensure no other processes are using GPU memory
   - Enable gradient accumulation (modify Stonk_Trainer.py)

2. **NaN losses during training**:
   - Reduce learning rate
   - Check for extreme reward values
   - Increase KL coefficient for more stable training

3. **Model produces low-quality predictions**:
   - Ensure dataset is properly loaded and filtered
   - Check reward function components and weights
   - Try increasing the training samples

4. **Script path errors**:
   - Modify the PYTHON_PATH in the .sh files to match your environment

## Contributing

Contributions to improve Stonk-Trainer are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The Qwen team for the base model
- Special thanks to Lukas Nel the creator of the 2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2 dataset for the Stonk-Trainer
- The PyTorch and Hugging Face communities 
