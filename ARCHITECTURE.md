# Stonk-Trainer v2 Model Architecture

This document explains the architecture and design of the Stonk-Trainer v2 model, including both Stage I and Stage II components.

## Overall System Architecture

The Stonk-Trainer v2 project consists of a two-stage training approach using Generative Reinforcement Policy Optimization (GRPO):

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                               Stonk-Trainer v2                                 │
├───────────────────────────────────────────┬───────────────────────────────────┤
│                                           │                                   │
│            Stage I Training               │           Stage II Training       │
│         (Balanced Distribution)           │       (Natural Distribution)      │
│                                           │                                   │
├───────────────────────────────────────────┼───────────────────────────────────┤
│                                           │                                   │
│  • Base Model: Qwen2.5-1.5B-Instruct      │  • Base Model: Stage I Model      │
│  • Dataset: 50/50 Up/Down Balance         │  • Dataset: Natural Market Skew   │
│  • Learning Rate: Higher (1e-5)           │  • Learning Rate: Lower (5e-6)    │
│  • Reward Function: Basic GRPO            │  • Reward Function: Enhanced GRPO │
│                                           │                                   │
└───────────────────────────────────────────┴───────────────────────────────────┘
```

## Model Components

### Base Model

The foundation of the Stonk-Trainer is the Qwen2.5-1.5B-Instruct language model, which provides powerful language understanding and generation capabilities while being efficient enough to train on consumer-grade hardware.

```
┌─────────────────────────────────────────────────────────────┐
│                 Qwen2.5-1.5B-Instruct Model                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  • Parameters: 1.5 Billion                                  │
│  • Context Window: 2048 tokens                              │
│  • Architecture: Transformer-based language model           │
│  • Quantization: 4-bit precision (for memory efficiency)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### LoRA Adapter Configuration

Low-Rank Adaptation (LoRA) is used for efficient fine-tuning:

```
┌─────────────────────────────────────────────────────────────┐
│                     LoRA Configuration                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  • R: 8 (rank of low-rank matrices)                         │
│  • Alpha: 16 (scaling factor)                               │
│  • Dropout: 0.05                                            │
│  • Target Modules: Query, Key, Value, Output projections    │
│  • Bias: "none"                                             │
│  • Task_type: "CAUSAL_LM"                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Dataset Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Dataset Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐      ┌───────────────────────┐        │
│  │   Raw Market    │      │                       │        │
│  │   Data Source   │─────►│   Filtering Logic     │        │
│  │                 │      │                       │        │
│  └─────────────────┘      └───────────┬───────────┘        │
│                                       │                    │
│                                       ▼                    │
│  ┌─────────────────┐      ┌───────────────────────┐        │
│  │  Stage I:       │      │                       │        │
│  │  Balanced Set   │◄─────│   Data Processor      │        │
│  │  (50/50)        │      │                       │        │
│  └─────────────────┘      └───────────┬───────────┘        │
│                                       │                    │
│                                       ▼                    │
│  ┌─────────────────┐      ┌───────────────────────┐        │
│  │  Stage II:      │      │                       │        │
│  │  Natural Distr. │◄─────│   Formatting          │        │
│  │                 │      │                       │        │
│  └─────────────────┘      └───────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## GRPO Training Loop

The Generative Reinforcement Policy Optimization (GRPO) training loop is a specialized reinforcement learning approach for language models:

```
┌─────────────────────────────────────────────────────────────┐
│                     GRPO Training Loop                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────┐                              │
│  │                           │                              │
│  │    Forward Pass           │                              │
│  │    Generate Response      │───┐                          │
│  │                           │   │                          │
│  └───────────────────────────┘   │                          │
│                                  │                          │
│                                  ▼                          │
│  ┌───────────────────────────┐   │   ┌───────────────────┐  │
│  │                           │   │   │                   │  │
│  │    Compute Reward         │◄──┴───│  Extract Prediction│  │
│  │                           │       │  and Reasoning     │  │
│  └───────────────┬───────────┘       │                   │  │
│                  │                   └───────────────────┘  │
│                  │                                          │
│                  ▼                                          │
│  ┌───────────────────────────┐                              │
│  │                           │                              │
│  │    Compute Policy Loss    │                              │
│  │    with KL Penalty        │                              │
│  │                           │                              │
│  └───────────────┬───────────┘                              │
│                  │                                          │
│                  ▼                                          │
│  ┌───────────────────────────┐                              │
│  │                           │                              │
│  │    Backward Pass          │                              │
│  │    Update Parameters      │                              │
│  │                           │                              │
│  └───────────────────────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Reward Function Architecture

The reward function is composed of multiple components that evaluate different aspects of the model's performance:

```
┌─────────────────────────────────────────────────────────────┐
│                       Reward Function                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────┐  ┌────────────────────────┐  │
│  │                           │  │                        │  │
│  │    Prediction Accuracy    │  │   Reasoning Quality    │  │
│  │    Component (0.0-0.6)    │  │   Component (0.0-0.4)  │  │
│  │                           │  │                        │  │
│  └───────────────┬───────────┘  └────────────┬───────────┘  │
│                  │                           │              │
│                  └───────────┬───────────────┘              │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │                                                 │        │
│  │              Total Reward Score                 │        │
│  │                  (0.0-1.0)                      │        │
│  │                                                 │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Reasoning Quality Component Breakdown

The reasoning quality component evaluates the explanations provided by the model:

```
┌─────────────────────────────────────────────────────────────┐
│                  Reasoning Quality Evaluation               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────┐  ┌────────────────────────┐  │
│  │                           │  │                        │  │
│  │    Data Usage Score       │  │   Logical Structure    │  │
│  │    (0.0-0.2)              │  │   Score (0.0-0.2)      │  │
│  │                           │  │                        │  │
│  └───────────────┬───────────┘  └────────────┬───────────┘  │
│                  │                           │              │
│                  └───────────┬───────────────┘              │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │                                                 │        │
│  │         Total Reasoning Quality Score           │        │
│  │                  (0.0-0.4)                      │        │
│  │                                                 │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Stage I to Stage II Transition

```
┌─────────────────────────────────────────────────────────────┐
│                Stage I to Stage II Transition               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────┐  ┌────────────────────────┐  │
│  │                           │  │                        │  │
│  │    Save Best Stage I      │  │   Load Stage I Model   │  │
│  │    Model Checkpoint       │  │   for Stage II         │  │
│  │                           │  │                        │  │
│  └───────────────┬───────────┘  └────────────┬───────────┘  │
│                  │                           │              │
│                  └───────────┬───────────────┘              │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │                                                 │        │
│  │         Adapt Learning Rate and Dataset         │        │
│  │         for Natural Distribution Training       │        │
│  │                                                 │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Performance Metrics and Evaluation

```
┌─────────────────────────────────────────────────────────────┐
│                  Performance Evaluation                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────┐  ┌───────────────────┐               │
│  │                   │  │                   │               │
│  │   Accuracy        │  │   Average Reward  │               │
│  │   Metrics         │  │   Metrics         │               │
│  │                   │  │                   │               │
│  └─────────┬─────────┘  └────────┬──────────┘               │
│            │                     │                          │
│            │  ┌──────────────────┴──────────┐               │
│            │  │                             │               │
│            │  │  Reasoning Quality Analysis │               │
│            │  │                             │               │
│            │  └──────────────┬──────────────┘               │
│            │                 │                              │
│            └─────────┬───────┘                              │
│                      │                                      │
│                      ▼                                      │
│  ┌─────────────────────────────────────────────────┐        │
│  │                                                 │        │
│  │         Comprehensive Evaluation Report         │        │
│  │                                                 │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Hardware Requirements 

One advantage of using the Qwen2.5-1.5B-Instruct model is that it requires less hardware resources than larger models, making it accessible for more users:

```
┌─────────────────────────────────────────────────────────────┐
│                   Hardware Requirements                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  • GPU: NVIDIA GPU with at least 8GB VRAM                   │
│    - Recommended: GTX 1080 Ti (11GB) or better              │
│    - The code has been optimized to run at 95-100%          │
│      utilization on a GTX 1080 Ti without OOM errors        │
│                                                             │
│  • RAM: 32GB recommended (16GB minimum)                     │
│    - Training processes can use 8-12GB RAM                  │
│    - Additional RAM needed for data processing              │
│    - Less RAM may result in slower performance              │
│                                                             │
│  • Storage: At least 50GB free space                        │
│    - Python environment: ~6-8GB for Conda environment       │
│    - HuggingFace cache: ~15-20GB for models and datasets    │
│    - Training datasets: ~12GB                               │
│    - Model checkpoints: ~5-10GB depending on saved epochs   │
│    - Logs and evaluation results: ~1GB                      │
│                                                             │
│  • CPU: 4+ cores recommended for data preprocessing         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Software Environment

```
┌─────────────────────────────────────────────────────────────┐
│                   Software Environment                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  • Python: 3.10+                                            │
│                                                             │
│  • PyTorch: 2.0.0+                                          │
│                                                             │
│  • CUDA: 11.7+ (12.0+ recommended)                          │
│                                                             │
│  • Transformers: 4.38.0+                                    │
│                                                             │
│  • PEFT: 0.6.0+ (for LoRA adapter implementation)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```