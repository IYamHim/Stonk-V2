# Stonk-Trainer v2 Training Flow

This document provides a visual overview of the training process flow in the Stonk-Trainer v2 project, including both Stage I and Super Saiyan Stage II training.

## Training Process Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │
│  Data Preparation   │────►│  Stage I Training   │────►│   Model Evaluation  │
│                     │     │  (Balanced Dataset) │     │                     │
└─────────────────────┘     └─────────────────────┘     └──────────┬──────────┘
                                                                    │
                                                                    │
                                                                    ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │
│  Final Deployment   │◄────│   Stage II Training │◄────│  "Super Saiyan"     │
│                     │     │ (Natural Distr.)    │     │   Moment Achieved   │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## Detailed Stage I Training Flow

```
┌──────────────────────────────────────┐
│                                      │
│          Load Base Model             │
│     (Qwen with Quantization)         │
│                                      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│                                      │
│    Prepare Model for Training        │
│    (Configure LoRA Adapters)         │
│                                      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│                                      │
│       Process Training Data          │
│    (Create Balanced Dataset)         │
│                                      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│                                      │
│        Formatting Prompts            │
│                                      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Main Training Loop            │
├──────────────────────────────────────┤
│                                      │
│  ┌───────────────────────────────┐   │
│  │     Generate Responses        │   │
│  └───────────────┬───────────────┘   │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐   │
│  │  Extract Prediction & Reasoning│   │
│  └───────────────┬───────────────┘   │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐   │
│  │   Compute Rewards (GRPO)      │   │
│  └───────────────┬───────────────┘   │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐   │
│  │   Compute Policy Gradient     │   │
│  └───────────────┬───────────────┘   │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐   │
│  │     Update Model Weights      │   │
│  └───────────────┬───────────────┘   │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐   │
│  │    Save Model Checkpoint      │   │
│  └───────────────────────────────┘   │
│                                      │
└──────────────────────────────────────┘
```

## Stage I to Stage II Transition

The transition from Stage I to Stage II occurs when the model reaches what we call the "Super Saiyan aha moment" - a significant performance milestone where the model has learned the fundamental patterns in the balanced dataset.

### Indicators of "Super Saiyan" Moment:

1. **Consistent Reward Values**: Average rewards per batch consistently exceed 0.7
2. **High Accuracy**: Evaluation accuracy reaches above 80% on validation set
3. **Reasoning Quality**: The model provides coherent explanations for its predictions
4. **Low Loss**: Training loss stabilizes at a low value across multiple epochs

## Stage II Training Flow

```
┌──────────────────────────────────────┐
│                                      │
│      Load Stage I Trained Model      │
│                                      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│                                      │
│    Process Natural Distribution      │
│             Dataset                  │
│                                      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Stage II Training Loop        │
├──────────────────────────────────────┤
│                                      │
│  ┌───────────────────────────────┐   │
│  │ Generate Responses (Real Distr)│   │
│  └───────────────┬───────────────┘   │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐   │
│  │  Extract with Market-Aware     │   │
│  │       Confidence               │   │
│  └───────────────┬───────────────┘   │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐   │
│  │ Enhanced Reward Computation    │   │
│  │  (Includes Market Skew Aware)  │   │
│  └───────────────┬───────────────┘   │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐   │
│  │   Fine-Tune with Lower LR     │   │
│  └───────────────┬───────────────┘   │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐   │
│  │   Advanced Evaluation on      │   │
│  │    Real-World Distribution    │   │
│  └───────────────────────────────┘   │
│                                      │
└──────────────────────────────────────┘
```

## Reward Function Breakdown

The reward function is a critical component of the GRPO training process:

```
┌─────────────────────────────────────────────────────┐
│                 Total Reward                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────┐   ┌──────────────────────┐    │
│  │ Prediction      │   │ Reasoning Quality    │    │
│  │ Accuracy        │ + │ Component            │    │
│  │ (0.0 - 0.6)     │   │ (0.0 - 0.4)          │    │
│  └─────────────────┘   └──────────────────────┘    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Reasoning Quality Sub-Components

```
┌─────────────────────────────────────────────────────┐
│            Reasoning Quality (0.0 - 0.4)            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────┐   ┌──────────────────────┐    │
│  │ Data Usage      │   │ Logical Coherence    │    │
│  │ (0.0 - 0.2)     │ + │ (0.0 - 0.2)          │    │
│  └─────────────────┘   └──────────────────────┘    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Output Examples

### Example of Good Model Output:

```
Company: AAPL (Apple Inc.)
Day 1: $150.00
Day 2: $153.50
Day 3: $156.20
Day 4: ?

Model Prediction: UP
Confidence: 87%

Reasoning:
Apple shows a consistent upward trend over the 3-day period with increasing momentum:
1. Day 1 to Day 2: +2.33% increase
2. Day 2 to Day 3: +1.76% increase
While the growth rate is slightly decreasing, the stock maintains strong upward movement 
and has shown no negative days. Apple's recent product announcements and strong market
position further support this continued upward trajectory.
```

### Example of Bad Model Output:

```
Company: TSLA (Tesla Inc.)
Day 1: $220.00
Day 2: $210.50
Day 3: $204.75
Day 4: ?

Model Prediction: UP
Confidence: 65%

Reasoning:
It will go up.
```

## Common Errors and Solutions

| Error | Possible Cause | Solution |
|-------|----------------|----------|
| CUDA out of memory | Batch size too large | Reduce batch size in train_direct.sh |
| Dataset access failure | HuggingFace login required | Run `huggingface-cli login` |
| Poor initial performance | Model not properly initialized | Ensure quantization is enabled |
| Training too slow | GPU not being fully utilized | Check nvidia-smi during training |

## Evaluation Metrics

The model's performance is evaluated using these key metrics:

1. **Accuracy**: Percentage of correct UP/DOWN predictions
2. **Reward Score**: Average total reward across test samples
3. **Reasoning Quality**: Subjective assessment of prediction explanations
4. **ROC-AUC**: Area under the ROC curve (for probability calibration)

These metrics are tracked during training and compiled in the evaluation report after testing. 