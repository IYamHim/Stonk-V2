import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import os
from datetime import datetime
from datasets import load_dataset
import sys

# Add parent directory to path for imports
sys.path.append('../')
sys.path.append('.')

# Import functions from grpo_stage2.py
from grpo_stage2 import format_prompt, extract_prediction, find_next_trading_day, compute_reward

def load_model(model_path, device="auto", quantize=True):
    """Load the model and tokenizer from path"""
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with appropriate settings
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    return model, tokenizer

def evaluate_model(model, tokenizer, dataset, num_samples=100, output_dir="./evaluation_results"):
    """Evaluate the model on a dataset"""
    print(f"Evaluating model on {num_samples} samples...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    results_file = os.path.join(output_dir, "evaluation_results.jsonl")
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    
    # Determine samples to evaluate
    if num_samples >= len(dataset):
        num_samples = len(dataset)
        indices = list(range(len(dataset)))
    else:
        # Random sampling
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Stats tracking
    all_results = []
    stats = {
        "up_predictions": 0,
        "down_predictions": 0,
        "up_actual": 0,
        "down_actual": 0,
        "correct_up": 0,
        "correct_down": 0,
        "total_rewards": 0,
        "count": 0,
        "by_confidence": {
            "high_conf_correct": 0,
            "high_conf_incorrect": 0,
            "low_conf_correct": 0,
            "low_conf_incorrect": 0
        },
        "reward_components": {
            "direction": 0,
            "magnitude": 0,
            "confidence": 0,
            "data_utilization": 0,
            "format": 0
        }
    }
    
    # Process each sample
    for idx in tqdm(indices):
        try:
            # Get sample
            sample = dataset[idx]
            
            # Format prompt
            prompt = format_prompt(sample)
            
            # Generate prediction
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode generated text
            response_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract prediction
            prediction = extract_prediction(response_text, 1)  # Day 1 prediction
            
            # Get actual outcome
            next_date, next_price = find_next_trading_day(dataset, idx)
            
            if next_date and next_price:
                current_price = sample['company_info']['price']['close']
                actual_change_pct = ((next_price - current_price) / current_price) * 100
                actual_direction = "up" if actual_change_pct > 0 else "down"
                
                # Create actual outcome object
                actual_outcome = {
                    "direction": actual_direction,
                    "percentage": abs(actual_change_pct),
                    "formatted": f"{actual_direction} {abs(actual_change_pct):.1f}%"
                }
                
                # Compute reward
                reward, explanation, reward_breakdown = compute_reward(prediction, actual_outcome, response_text, sample)
                
                # Update stats
                if prediction["direction"] == "up":
                    stats["up_predictions"] += 1
                elif prediction["direction"] == "down":
                    stats["down_predictions"] += 1
                
                if actual_direction == "up":
                    stats["up_actual"] += 1
                    if prediction["direction"] == "up":
                        stats["correct_up"] += 1
                else:
                    stats["down_actual"] += 1
                    if prediction["direction"] == "down":
                        stats["correct_down"] += 1
                
                # Track confidence
                if prediction["confidence"] >= 70:
                    if prediction["direction"] == actual_direction:
                        stats["by_confidence"]["high_conf_correct"] += 1
                    else:
                        stats["by_confidence"]["high_conf_incorrect"] += 1
                else:
                    if prediction["direction"] == actual_direction:
                        stats["by_confidence"]["low_conf_correct"] += 1
                    else:
                        stats["by_confidence"]["low_conf_incorrect"] += 1
                
                # Track reward components
                for component, value in reward_breakdown.items():
                    stats["reward_components"][component] += value
                
                stats["total_rewards"] += reward
                stats["count"] += 1
                
                # Store full result
                result = {
                    "index": idx,
                    "ticker": sample['ticker'],
                    "date": str(sample['company_info'].get('current_date', '')),
                    "prediction": prediction,
                    "actual": actual_outcome,
                    "reward": reward,
                    "reward_breakdown": reward_breakdown,
                    "explanation": explanation,
                    "response": response_text
                }
                
                all_results.append(result)
                
                # Write to file
                with open(results_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Calculate aggregate statistics
    if stats["count"] > 0:
        # Overall accuracy
        total_correct = stats["correct_up"] + stats["correct_down"]
        accuracy = total_correct / stats["count"]
        
        # Direction-specific accuracy
        up_accuracy = stats["correct_up"] / max(1, stats["up_actual"])
        down_accuracy = stats["correct_down"] / max(1, stats["down_actual"])
        
        # Confidence calibration
        high_conf_accuracy = stats["by_confidence"]["high_conf_correct"] / max(1, stats["by_confidence"]["high_conf_correct"] + stats["by_confidence"]["high_conf_incorrect"])
        low_conf_accuracy = stats["by_confidence"]["low_conf_correct"] / max(1, stats["by_confidence"]["low_conf_correct"] + stats["by_confidence"]["low_conf_incorrect"])
        
        # Average reward
        avg_reward = stats["total_rewards"] / stats["count"]
        
        # Average reward components
        avg_components = {}
        for component, value in stats["reward_components"].items():
            avg_components[component] = value / stats["count"]
        
        # Summary statistics
        summary = {
            "datetime": datetime.now().isoformat(),
            "num_samples": stats["count"],
            "accuracy": accuracy,
            "up_accuracy": up_accuracy,
            "down_accuracy": down_accuracy,
            "up_predictions": stats["up_predictions"],
            "down_predictions": stats["down_predictions"],
            "up_actual": stats["up_actual"],
            "down_actual": stats["down_actual"],
            "avg_reward": avg_reward,
            "avg_reward_components": avg_components,
            "confidence_calibration": {
                "high_conf_accuracy": high_conf_accuracy,
                "low_conf_accuracy": low_conf_accuracy,
                "high_conf_total": stats["by_confidence"]["high_conf_correct"] + stats["by_confidence"]["high_conf_incorrect"],
                "low_conf_total": stats["by_confidence"]["low_conf_correct"] + stats["by_confidence"]["low_conf_incorrect"]
            }
        }
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create visualizations
        create_visualizations(summary, all_results, output_dir)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Accuracy: {accuracy:.2f} ({total_correct}/{stats['count']})")
        print(f"Up Accuracy: {up_accuracy:.2f} ({stats['correct_up']}/{stats['up_actual']})")
        print(f"Down Accuracy: {down_accuracy:.2f} ({stats['correct_down']}/{stats['down_actual']})")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Market Bias: {stats['up_actual']/(stats['up_actual'] + stats['down_actual']):.2f} up")
        print(f"Model Bias: {stats['up_predictions']/(stats['up_predictions'] + stats['down_predictions']):.2f} up")
        
        return summary, all_results
    else:
        print("No valid samples processed")
        return None, None

def create_visualizations(summary, results, output_dir):
    """Create visualizations of the evaluation results"""
    print("Creating visualizations...")
    
    # Plot accuracy by direction
    plt.figure(figsize=(10, 6))
    labels = ['Overall', 'Up', 'Down']
    accuracies = [summary['accuracy'], summary['up_accuracy'], summary['down_accuracy']]
    plt.bar(labels, accuracies, color=['blue', 'green', 'red'])
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy by Direction')
    plt.ylim(0, 1.0)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_direction.png'))
    
    # Plot market distribution vs model predictions
    plt.figure(figsize=(10, 6))
    labels = ['Actual Market', 'Model Predictions']
    up_pcts = [summary['up_actual']/(summary['up_actual'] + summary['down_actual']), 
               summary['up_predictions']/(summary['up_predictions'] + summary['down_predictions'])]
    
    up_bars = plt.bar(labels, up_pcts, color='green', label='Up')
    down_bars = plt.bar(labels, [1-up_pct for up_pct in up_pcts], bottom=up_pcts, color='red', label='Down')
    
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.ylabel('Proportion')
    plt.title('Market Distribution vs Model Predictions')
    plt.ylim(0, 1.0)
    plt.legend()
    
    for i, v in enumerate(up_pcts):
        plt.text(i, v/2, f'{v:.2f}', ha='center', color='white')
        plt.text(i, v + (1-v)/2, f'{1-v:.2f}', ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'))
    
    # Plot reward components
    plt.figure(figsize=(12, 6))
    components = list(summary['avg_reward_components'].keys())
    values = list(summary['avg_reward_components'].values())
    
    plt.barh(components, values, color='blue')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('Average Reward')
    plt.title('Reward Components')
    
    for i, v in enumerate(values):
        plt.text(max(v + 0.05, 0.05) if v >= 0 else v - 0.4, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_components.png'))
    
    # Plot confidence calibration
    plt.figure(figsize=(10, 6))
    conf_levels = ['High Confidence', 'Low Confidence']
    accuracies = [summary['confidence_calibration']['high_conf_accuracy'], 
                 summary['confidence_calibration']['low_conf_accuracy']]
    counts = [summary['confidence_calibration']['high_conf_total'],
              summary['confidence_calibration']['low_conf_total']]
    
    bars = plt.bar(conf_levels, accuracies, color=['purple', 'orange'])
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Confidence Level')
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.2f}\n(n={counts[i]})', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_calibration.png'))
    
    # Create rewards histogram
    if results:
        rewards = [r['reward'] for r in results]
        plt.figure(figsize=(10, 6))
        plt.hist(rewards, bins=20, color='blue', alpha=0.7)
        plt.axvline(x=summary['avg_reward'], color='red', linestyle='--')
        plt.text(summary['avg_reward'], plt.ylim()[1]*0.9, f'Mean: {summary["avg_reward"]:.2f}', color='red')
        plt.xlabel('Reward')
        plt.ylabel('Count')
        plt.title('Distribution of Rewards')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_distribution.png'))
    
    print(f"Visualizations saved to {output_dir}")

def compare_models(stage1_path, stage2_path, dataset, num_samples=100, output_dir="./comparison_results"):
    """Compare Stage I and Stage II models on the same dataset samples"""
    print(f"Comparing Stage I and Stage II models on {num_samples} samples...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Stage I model
    stage1_model, stage1_tokenizer = load_model(stage1_path)
    
    # Evaluate Stage I model
    stage1_summary, stage1_results = evaluate_model(
        stage1_model, 
        stage1_tokenizer, 
        dataset, 
        num_samples=num_samples, 
        output_dir=os.path.join(output_dir, "stage1")
    )
    
    # Load Stage II model
    stage2_model, stage2_tokenizer = load_model(stage2_path)
    
    # Evaluate Stage II model
    stage2_summary, stage2_results = evaluate_model(
        stage2_model,
        stage2_tokenizer,
        dataset,
        num_samples=num_samples,
        output_dir=os.path.join(output_dir, "stage2")
    )
    
    # Create comparison visualizations
    if stage1_summary and stage2_summary:
        create_comparison_visualizations(stage1_summary, stage2_summary, output_dir)
    
    return stage1_summary, stage2_summary

def create_comparison_visualizations(stage1_summary, stage2_summary, output_dir):
    """Create comparative visualizations between Stage I and Stage II models"""
    print("Creating comparison visualizations...")
    
    # Compare overall performance
    plt.figure(figsize=(12, 6))
    metrics = ['accuracy', 'avg_reward', 'up_accuracy', 'down_accuracy']
    labels = ['Overall Accuracy', 'Average Reward', 'Up Accuracy', 'Down Accuracy']
    
    stage1_values = [stage1_summary[m] for m in metrics]
    stage2_values = [stage2_summary[m] for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, stage1_values, width, label='Stage I (Balanced)')
    plt.bar(x + width/2, stage2_values, width, label='Stage II (Natural)')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel('Value')
    plt.title('Stage I vs Stage II Performance')
    plt.xticks(x, labels)
    plt.legend()
    
    for i, (v1, v2) in enumerate(zip(stage1_values, stage2_values)):
        plt.text(i - width/2, v1 + 0.02, f'{v1:.2f}', ha='center')
        plt.text(i + width/2, v2 + 0.02, f'{v2:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    
    # Compare market adaptation
    plt.figure(figsize=(12, 6))
    
    # Create data
    stage1_up_pct = stage1_summary['up_predictions'] / (stage1_summary['up_predictions'] + stage1_summary['down_predictions'])
    stage2_up_pct = stage2_summary['up_predictions'] / (stage2_summary['up_predictions'] + stage2_summary['down_predictions'])
    actual_up_pct = stage1_summary['up_actual'] / (stage1_summary['up_actual'] + stage1_summary['down_actual'])
    
    models = ['Market Reality', 'Stage I (Balanced)', 'Stage II (Natural)']
    up_pcts = [actual_up_pct, stage1_up_pct, stage2_up_pct]
    
    # Plot
    up_bars = plt.bar(models, up_pcts, color='green', label='Up')
    down_bars = plt.bar(models, [1-up_pct for up_pct in up_pcts], bottom=up_pcts, color='red', label='Down')
    
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.ylabel('Proportion')
    plt.title('Market Distribution Adaptation')
    plt.ylim(0, 1.0)
    plt.legend()
    
    for i, v in enumerate(up_pcts):
        plt.text(i, v/2, f'{v:.2f}', ha='center', color='white')
        plt.text(i, v + (1-v)/2, f'{1-v:.2f}', ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'market_adaptation.png'))
    
    # Compare reward components
    plt.figure(figsize=(12, 8))
    
    # Data
    components = list(stage1_summary['avg_reward_components'].keys())
    stage1_values = [stage1_summary['avg_reward_components'][c] for c in components]
    stage2_values = [stage2_summary['avg_reward_components'][c] for c in components]
    
    x = np.arange(len(components))
    width = 0.35
    
    plt.barh(x - width/2, stage1_values, width, label='Stage I (Balanced)', color='blue')
    plt.barh(x + width/2, stage2_values, width, label='Stage II (Natural)', color='orange')
    
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('Average Reward')
    plt.title('Reward Components Comparison')
    plt.yticks(x, components)
    plt.legend()
    
    for i, (v1, v2) in enumerate(zip(stage1_values, stage2_values)):
        plt.text(max(v1 + 0.05, 0.05) if v1 >= 0 else v1 - 0.4, i - width/2, f'{v1:.2f}', va='center')
        plt.text(max(v2 + 0.05, 0.05) if v2 >= 0 else v2 - 0.4, i + width/2, f'{v2:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_components_comparison.png'))
    
    print(f"Comparison visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage II GRPO model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model for evaluation")
    parser.add_argument("--stage1_model", type=str, help="Optional: Path to Stage I model for comparison")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory for results")
    parser.add_argument("--test_set", action="store_true", help="Use test set instead of training set")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load dataset
    print("Loading dataset...")
    if args.test_set:
        dataset = load_dataset("2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2", split="test")
    else:
        dataset = load_dataset("2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2", split="train")
    print(f"Loaded {len(dataset)} samples")
    
    # If comparing models
    if args.stage1_model:
        compare_models(
            args.stage1_model,
            args.model_path,
            dataset,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
    else:
        # Load model
        model, tokenizer = load_model(args.model_path, quantize=args.quantize)
        
        # Evaluate model
        evaluate_model(
            model,
            tokenizer,
            dataset,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main() 