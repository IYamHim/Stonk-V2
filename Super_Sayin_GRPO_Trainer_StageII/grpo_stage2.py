import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from datetime import datetime, timedelta
import random
import math
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from copy import deepcopy
import traceback
from datasets import load_dataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def load_stage1_model(model_path, quantize=True):
    """Load the trained Stage I model"""
    print("Loading Stage I trained model and tokenizer...")
    
    # Configure quantization
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quantization_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        torch_dtype=torch.float16
    )
    
    # Enable gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def prepare_model_for_training(model):
    """Prepare model for Stage II training with LoRA"""
    # Configure LoRA with lower rank for fine-tuning
    lora_config = LoraConfig(
        r=8,  # Smaller attention heads for fine-tuning
        lora_alpha=16,  # Lower alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adaptor
    model = get_peft_model(model, lora_config)
    
    # Ensure proper setup for gradient computation
    model.config.use_cache = False
    
    return model

def format_prompt(example):
    """Format a prompt for stonk prediction using dataset example"""
    
    # Get company info
    company_info = example.get('company_info', {})
    ticker = example.get('ticker', '')
    description = company_info.get('company_info', {}).get('description', '')
    
    # Get price info
    price_info = company_info.get('price', {})
    current_price = price_info.get('close', 0)
    previous_close = price_info.get('close_previous', 0)
    
    # Calculate day change percentage
    if previous_close and current_price:
        day_change_pct = ((current_price - previous_close) / previous_close) * 100
    else:
        day_change_pct = 0
    
    # Extract historical data - include more historical data for Stage II
    historical_data = []
    if 'historical_data' in company_info and company_info['historical_data']:
        historical_records = company_info['historical_data']
        if isinstance(historical_records, list) and len(historical_records) > 0:
            # Sort by date if available
            if 'date' in historical_records[0]:
                historical_records = sorted(historical_records, key=lambda x: x.get('date', ''), reverse=True)
            
            # Take up to 15 most recent records (more context for Stage II)
            for record in historical_records[:15]:
                date = record.get('date', '')
                close = record.get('close', 0)
                volume = record.get('volume', 0)
                if date and close:
                    historical_data.append(f"{date}: Close ${close:.2f}, Volume: {volume}")
    
    # Build prompt with more comprehensive context
    prompt = f"""You are a stonk market analyst tasked with predicting the price movement for {ticker}.

COMPANY INFORMATION:
Ticker: {ticker}
Description: {description}
Current Price: ${current_price:.2f}
Previous Close: ${previous_close:.2f}
Day Change: {day_change_pct:.2f}%

RECENT HISTORICAL DATA:
"""

    if historical_data:
        for data_point in historical_data:
            prompt += f"- {data_point}\n"
    else:
        prompt += "No historical data available\n"
    
    # Add news and sentiment if available - include more news for Stage II
    if 'news' in company_info and company_info['news']:
        prompt += "\nRECENT NEWS:\n"
        news_items = company_info['news']
        if isinstance(news_items, list):
            for i, news in enumerate(news_items[:8]):  # Include more news items
                title = news.get('title', '')
                date = news.get('date', '')
                if title:
                    prompt += f"- {date}: {title}\n"
    
    # Add specific financial metrics if available
    if 'financial_metrics' in company_info:
        metrics = company_info['financial_metrics']
        if isinstance(metrics, dict):
            prompt += "\nFINANCIAL METRICS:\n"
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    prompt += f"- {key}: {value}\n"
    
    # Task prompt with more detailed instructions
    prompt += """
TASK:
Based on the information provided, predict the stonk price movement for the next 3 trading days.
For each day, provide:
1. Direction (up or down)
2. Percentage change estimate (e.g., up 1.2%)
3. Your confidence level (0-100%)
4. Brief justification referencing specific data points

Use this format for each day:
Day X: [UP/DOWN] [PERCENTAGE]% with [CONFIDENCE]% confidence
Justification: [Your reasoning using specific data points]

Respond with your analysis and predictions below:
"""
    
    return prompt

def extract_prediction(response_text, day_number):
    """Extract the prediction from the response text for a given day"""
    day_pattern = re.compile(rf"Day {day_number}:?\s*(?:Prediction:?\s*)?([Uu][Pp]|[Dd][Oo][Ww][Nn])\s+([\d\.]+)%", re.IGNORECASE)
    
    # Extract direction and percentage
    day_match = day_pattern.search(response_text)
    
    # Default values
    direction = "unknown"
    percentage = 0.0
    confidence = 50.0  # Default confidence
    
    if day_match:
        direction = day_match.group(1).lower()
        percentage = float(day_match.group(2))
        
        # Try to extract confidence
        confidence_pattern = re.compile(rf"Day {day_number}:.*?(\d+)%\s+confidence", re.IGNORECASE | re.DOTALL)
        conf_match = confidence_pattern.search(response_text)
        if conf_match:
            confidence = float(conf_match.group(1))
    
    # Create formatted text
    formatted = f"{direction} {percentage:.1f}%"
    
    return {
        "direction": direction,
        "percentage": percentage,
        "confidence": confidence,
        "formatted": formatted
    }

def extract_thinking(response_text):
    """Extract any thinking or reasoning section from the response"""
    # Look for thinking pattern
    thinking_pattern = re.compile(r"(?:Thinking|Reasoning|Analysis):(.*?)(?=Day \d:|$)", re.IGNORECASE | re.DOTALL)
    thinking_match = thinking_pattern.search(response_text)
    
    if thinking_match:
        return thinking_match.group(1).strip()
    
    # Try justification pattern as alternative
    justification_pattern = re.compile(r"Justification:(.*?)(?=Day \d:|$)", re.IGNORECASE | re.DOTALL)
    justification_match = justification_pattern.search(response_text)
    
    if justification_match:
        return justification_match.group(1).strip()
    
    return "No thinking section found"

def compute_reward(prediction, actual_outcome, response_text, example):
    """Compute reward for a prediction compared to actual outcome - enhanced for Stage II"""
    total_reward = 0.0
    reward_explanation = []
    reward_breakdown = {}
    
    # 1. Direction reward (25% of total) - reduced weight for Stage II
    if prediction["direction"] == actual_outcome["direction"]:
        direction_reward = 2.5
        reward_explanation.append(f"Direction: {direction_reward} - Correct direction prediction")
    else:
        direction_reward = 0.0
        reward_explanation.append(f"Direction: {direction_reward} - Incorrect direction prediction")
    
    reward_breakdown["direction"] = direction_reward
    total_reward += direction_reward
    
    # 2. Magnitude reward (20% of total) - reduced weight for Stage II
    if prediction["direction"] == actual_outcome["direction"]:
        # Calculate magnitude error as percentage difference
        pred_pct = prediction["percentage"]
        actual_pct = actual_outcome["percentage"]
        
        # Error as percentage of the actual change
        error_ratio = abs(pred_pct - actual_pct) / max(0.5, actual_pct)
        
        # Reward inversely proportional to error
        if error_ratio <= 0.2:  # Within 20% of actual
            magnitude_reward = 2.0
            explanation = f"Magnitude: {magnitude_reward} - Excellent estimate (within 20% of actual)"
        elif error_ratio <= 0.5:  # Within 50% of actual
            magnitude_reward = 1.5
            explanation = f"Magnitude: {magnitude_reward} - Good estimate (within 50% of actual)"
        elif error_ratio <= 1.0:  # Within 100% of actual
            magnitude_reward = 1.0
            explanation = f"Magnitude: {magnitude_reward} - Fair estimate (within 100% of actual)"
        else:
            magnitude_reward = 0.5
            explanation = f"Magnitude: {magnitude_reward} - Poor estimate (more than 100% off)"
        
        reward_explanation.append(explanation)
    else:
        magnitude_reward = 0.0
        reward_explanation.append(f"Magnitude: {magnitude_reward} - No reward (wrong direction)")
    
    reward_breakdown["magnitude"] = magnitude_reward
    total_reward += magnitude_reward
    
    # 3. Confidence reward/penalty (25% of total) - increased weight for Stage II
    # - Reward high confidence on correct predictions
    # - Penalize high confidence on incorrect predictions
    confidence = prediction["confidence"]
    
    if prediction["direction"] == actual_outcome["direction"]:
        # Correct prediction - reward confidence
        if confidence >= 80:
            confidence_reward = 2.5
            explanation = f"Confidence: {confidence_reward} - High confidence ({confidence}%) with correct direction"
        elif confidence >= 60:
            confidence_reward = 2.0
            explanation = f"Confidence: {confidence_reward} - Good confidence ({confidence}%) with correct direction"
        elif confidence >= 40:
            confidence_reward = 1.5
            explanation = f"Confidence: {confidence_reward} - Moderate confidence ({confidence}%) with correct direction"
        else:
            confidence_reward = 1.0
            explanation = f"Confidence: {confidence_reward} - Low confidence ({confidence}%) with correct direction"
    else:
        # Incorrect prediction - penalize based on confidence
        if confidence >= 80:
            confidence_reward = -1.5
            explanation = f"Confidence: {confidence_reward} - Severe penalty for high confidence ({confidence}%) with incorrect direction"
        elif confidence >= 60:
            confidence_reward = -0.7
            explanation = f"Confidence: {confidence_reward} - Moderate penalty for medium confidence ({confidence}%) with incorrect direction"
        elif confidence >= 40:
            confidence_reward = -0.4
            explanation = f"Confidence: {confidence_reward} - Small penalty for medium-low confidence ({confidence}%) with incorrect direction"
        else:
            confidence_reward = -0.2
            explanation = f"Confidence: {confidence_reward} - Minimal penalty for low confidence ({confidence}%) with incorrect direction"
    
    reward_explanation.append(explanation)
    reward_breakdown["confidence"] = confidence_reward
    total_reward += confidence_reward
    
    # 4. Data utilization reward (30% of total) - significantly increased for Stage II
    # - Reward models for referencing specific data points
    reasoning = extract_thinking(response_text)
    
    # Check for references to historical data
    data_references = 0
    
    # Check for date mentions
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    date_matches = date_pattern.findall(reasoning)
    unique_dates = set(date_matches)
    data_references += len(unique_dates)
    
    # Check for price references
    price_pattern = re.compile(r"\$\d+\.\d+|\$\d+|\d+\.\d+\s+dollars|\d+\s+dollars")
    price_matches = price_pattern.findall(reasoning)
    data_references += len(price_matches)
    
    # Check for percentage references
    pct_pattern = re.compile(r"\d+\.\d+%|\d+%")
    pct_matches = pct_pattern.findall(reasoning)
    data_references += len(pct_matches)
    
    # Check for volume references
    volume_pattern = re.compile(r"volume of \d+|volume: \d+|\d+ shares|\d+ volume")
    volume_matches = volume_pattern.findall(reasoning.lower())
    data_references += len(volume_matches)
    
    # Calculate data utilization reward
    if data_references >= 5:
        data_reward = 3.0
        explanation = f"Data Utilization: {data_reward} - Excellent use of specific data points"
    elif data_references >= 3:
        data_reward = 2.0
        explanation = f"Data Utilization: {data_reward} - Good use of specific data points"
    elif data_references >= 1:
        data_reward = 1.0
        explanation = f"Data Utilization: {data_reward} - Referenced some specific data points"
    else:
        data_reward = 0.0
        explanation = f"Data Utilization: {data_reward} - Did not reference any specific data points"
    
    reward_explanation.append(explanation)
    reward_breakdown["data_utilization"] = data_reward
    total_reward += data_reward
    
    # Format bonus: clean, readable output (format is less important for Stage II)
    format_reward = 0.0
    day_pattern = re.compile(r"Day \d+: (Up|Down) \d+\.\d+% with \d+% confidence", re.IGNORECASE)
    if day_pattern.search(response_text):
        format_reward = 1.0
        reward_explanation.append(f"Format: {format_reward} - Followed the requested format")
    else:
        reward_explanation.append(f"Format: {format_reward} - Did not follow the requested format")
    
    reward_breakdown["format"] = format_reward
    total_reward += format_reward
    
    # Normalize total reward to be out of 10
    # total_reward = min(10.0, total_reward)
    
    return total_reward, reward_explanation, reward_breakdown

def find_next_trading_day(dataset, current_idx):
    """Find the next trading day's data for a given example"""
    try:
        current_example = dataset[current_idx]
        ticker = current_example.get('ticker', '')
        current_date = current_example.get('company_info', {}).get('current_date', '')
        
        if not ticker or not current_date:
            return None, None
        
        # Convert date format if needed
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, "%Y-%m-%d").date()
        
        # Find examples for the same ticker
        next_day = None
        next_price = None
        
        # Build a dictionary of dates for this ticker
        dates_by_ticker = {}
        for i, example in enumerate(dataset):
            ex_ticker = example.get('ticker', '')
            ex_date = example.get('company_info', {}).get('current_date', '')
            ex_price = example.get('company_info', {}).get('price', {}).get('close', 0)
            
            if ex_ticker == ticker and ex_date and ex_price:
                if isinstance(ex_date, str):
                    ex_date = datetime.strptime(ex_date, "%Y-%m-%d").date()
                
                if ex_ticker not in dates_by_ticker:
                    dates_by_ticker[ex_ticker] = []
                
                dates_by_ticker[ex_ticker].append((ex_date, ex_price, i))
        
        # Sort dates for this ticker
        if ticker in dates_by_ticker:
            sorted_dates = sorted(dates_by_ticker[ticker], key=lambda x: x[0])
            
            # Find the next trading day
            for date, price, idx in sorted_dates:
                if date > current_date:
                    next_day = date
                    next_price = price
                    break
        
        return next_day, next_price
        
    except Exception as e:
        print(f"Error finding next trading day: {str(e)}")
        return None, None

def stage2_training(model, tokenizer, dataset, epochs=3, batch_size=4, learning_rate=5e-6, kl_coef=0.05, save_steps=50, natural_distribution=True, output_dir = "./super_saiyan_grpo_stage2"):
    """
    Stage II GRPO Training with natural market distribution
    """
    print("Starting Stage II GRPO Training with Natural Market Distribution...")
    
    # Create output directory
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file
    log_file = os.path.join(output_dir, "stage2_training_log.jsonl")
    
    # Create reference model (frozen copy)
    reference_model = deepcopy(model)
    reference_model.eval()
    
    # Freeze reference model
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # Create optimizer with gradient clipping and stability improvements
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        eps=1e-8,  # Increase epsilon for stability
        weight_decay=0.01  # Add weight decay for regularization
    )
    
    # Training statistics
    total_samples = 0
    total_reward = 0
    best_avg_reward = 0
    step = 0
    
    # For logging market bias adaptation
    market_stats = {
        "up_examples": 0,
        "down_examples": 0,
        "up_predictions": 0,
        "down_predictions": 0,
        "correct_up": 0,
        "correct_down": 0
    }
    
    # GRPO training loop
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")
        model.train()
        
        # Sample examples for this epoch - no filtering for balanced dataset in Stage II
        num_samples = min(1000, len(dataset))
        indices = [int(idx) for idx in np.random.choice(len(dataset), num_samples, replace=False)]
        
        # Process in batches
        for batch_idx in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[batch_idx:batch_idx + batch_size]
            batch_rewards = []
            batch_losses = []
            batch_logs = []
            
            # Process each example in the batch
            for idx in batch_indices:
                try:
                    # Get example and format prompt
                    sample = dataset[idx]
                    prompt = format_prompt(sample)
                    
                    # Tokenize prompt
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    # Generate response from current model
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=400,  # Increased tokens for more detailed responses
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    
                    # Get generated text
                    response_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    # Extract prediction from response
                    prediction = extract_prediction(response_text, 1)  # Get Day 1 prediction
                    
                    # Track prediction distribution
                    if prediction['direction'] == 'up':
                        market_stats["up_predictions"] += 1
                    elif prediction['direction'] == 'down':
                        market_stats["down_predictions"] += 1
                    
                    # Get actual outcome
                    next_date, next_price = find_next_trading_day(dataset, idx)
                    
                    if next_date and next_price:
                        current_price = sample['company_info']['price']['close']
                        actual_change_pct = ((next_price - current_price) / current_price) * 100
                        actual_direction = "up" if actual_change_pct > 0 else "down"
                        
                        # Track actual distribution
                        if actual_direction == "up":
                            market_stats["up_examples"] += 1
                            if prediction['direction'] == 'up':
                                market_stats["correct_up"] += 1
                        else:
                            market_stats["down_examples"] += 1
                            if prediction['direction'] == 'down':
                                market_stats["correct_down"] += 1
                        
                        actual_outcome = {
                            "direction": actual_direction,
                            "percentage": abs(actual_change_pct),
                            "formatted": f"{actual_direction} {abs(actual_change_pct):.1f}%"
                        }
                        
                        # Compute reward
                        reward, explanation, _ = compute_reward(prediction, actual_outcome, response_text, sample)
                        reward = torch.tensor(reward, dtype=torch.float32, device=model.device)
                        batch_rewards.append(reward)
                        
                        # Create combined sequence for training (input followed by response)
                        # First, tokenize the response text
                        response_tokens = tokenizer(response_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                        
                        # Combine input and response IDs for full sequence
                        full_input_ids = torch.cat([inputs.input_ids, response_tokens], dim=1)
                        
                        # Create attention mask for full sequence
                        full_attention_mask = torch.ones_like(full_input_ids)
                        
                        # Create labels with -100 for input tokens (we don't want to compute loss on them)
                        labels = torch.full_like(full_input_ids, -100)
                        # Set labels for response tokens to their token ids
                        labels[:, inputs.input_ids.shape[1]:] = response_tokens
                        
                        # Forward pass for both policy and reference model
                        outputs = model(
                            input_ids=full_input_ids,
                            attention_mask=full_attention_mask,
                            labels=labels,
                        )
                        
                        # Get loss directly from outputs
                        loss = outputs.loss
                        
                        # Skip if NaN loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: NaN or Inf in model loss for sample {idx}")
                            continue
                            
                        # Get reference model outputs
                        with torch.no_grad():
                            ref_outputs = reference_model(
                                input_ids=full_input_ids,
                                attention_mask=full_attention_mask,
                            )
                        
                        # Get response start position
                        response_start = inputs.input_ids.shape[1]
                        
                        # Only consider response tokens for KL divergence
                        # Improved KL divergence calculation with better numerical stability
                        try:
                            # Get logits for response tokens only
                            policy_logits = outputs.logits[:, (response_start-1):-1, :]  # shape [batch, seq_len, vocab]
                            ref_logits = ref_outputs.logits[:, (response_start-1):-1, :]
                            
                            # Get target token indices for response tokens
                            target_ids = response_tokens
                            
                            # Initialize KL divergence
                            kl_div_loss = 0.0
                            
                            # Compute token-by-token KL divergence only on actual tokens 
                            # (this is more stable than computing over the entire vocab)
                            for pos in range(target_ids.shape[1]):
                                # Get token ID at this position
                                token_id = target_ids[0, pos].item()
                                
                                # Get policy and reference logits for this position
                                p_logits = policy_logits[0, pos, :]  # [vocab_size]
                                r_logits = ref_logits[0, pos, :]  # [vocab_size]
                                
                                # Apply temperature to sharpen/soften distributions
                                temperature = 1.0
                                p_logits = p_logits / temperature
                                r_logits = r_logits / temperature
                                
                                # Convert to probabilities
                                p_probs = torch.nn.functional.softmax(p_logits, dim=-1)
                                r_probs = torch.nn.functional.softmax(r_logits, dim=-1)
                                
                                # Add small epsilon for numerical stability
                                epsilon = 1e-6
                                p_probs = p_probs + epsilon
                                r_probs = r_probs + epsilon
                                
                                # Normalize after adding epsilon
                                p_probs = p_probs / p_probs.sum()
                                r_probs = r_probs / r_probs.sum()
                                
                                # Compute KL for this token position
                                token_kl = torch.sum(p_probs * (torch.log(p_probs) - torch.log(r_probs)))
                                
                                # Accumulate
                                kl_div_loss += token_kl
                            
                            # Average over sequence length
                            kl_div_loss = kl_div_loss / target_ids.shape[1]
                            
                            # Clip to avoid extreme values
                            kl_div_loss = torch.clamp(kl_div_loss, min=0.0, max=10.0)
                            
                            # Compute policy gradient loss with reward scaling
                            # Use a constant factor to scale reward for better gradient flow
                            reward_scaling = 0.1
                            pg_loss = -reward_scaling * reward * loss
                            
                            # Add KL penalty - lower KL coefficient for Stage II
                            total_loss = pg_loss + kl_coef * kl_div_loss
                            
                            # Check for invalid loss values
                            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                                batch_losses.append(total_loss)
                                # Log KL and total loss
                                print(f"Sample {idx} - KL: {kl_div_loss.item():.4f}, Loss: {total_loss.item():.4f}")
                            else:
                                print(f"Warning: Invalid total loss: {total_loss}")
                            
                        except Exception as kl_error:
                            print(f"Error computing KL divergence: {kl_error}")
                            # Fall back to just using policy gradient loss without KL
                            if not torch.isnan(loss) and not torch.isinf(loss):
                                total_loss = -reward * loss
                                batch_losses.append(total_loss)
                                print(f"Using fallback loss for sample {idx}: {total_loss.item():.4f}")
                        
                        # Log example details
                        batch_logs.append({
                            "step": step,
                            "ticker": sample['ticker'],
                            "actual_change": float(actual_change_pct),
                            "predicted_direction": prediction["direction"],
                            "predicted_pct": float(prediction["percentage"]),
                            "predicted_confidence": float(prediction["confidence"]),
                            "reward": float(reward.item()),
                            "loss": float(total_loss.item()) if not torch.isnan(total_loss) else "NaN",
                            "thinking": extract_thinking(response_text),
                            "response": response_text
                        })
                        
                        total_samples += 1
                        total_reward += reward.item()
                        
                except Exception as e:
                    print(f"Error processing sample {idx}: {str(e)}")
                    traceback.print_exc()
                    continue
            
            # Update model if we have valid losses
            if batch_losses:
                optimizer.zero_grad()
                batch_loss = torch.stack(batch_losses).mean()
                
                # Skip update if loss is invalid
                if not torch.isnan(batch_loss) and not torch.isinf(batch_loss):
                    batch_loss.backward()
                    
                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Calculate batch statistics
                    avg_reward = torch.stack(batch_rewards).mean().item() if batch_rewards else 0
                    
                    # Log batch results
                    print(f"Batch {batch_idx//batch_size + 1} - Avg Reward: {avg_reward:.2f}, Loss: {batch_loss.item():.4f}")
                else:
                    print(f"Skipping update for batch {batch_idx//batch_size + 1} due to invalid loss")
                
                # Save logs
                with open(log_file, 'a') as f:
                    for log in batch_logs:
                        f.write(json.dumps(log) + '\n')
                
                # Save checkpoint periodically
                if (step + 1) % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step+1}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print(f"Saved checkpoint to {checkpoint_dir}")
                    
                    # Log market adaptation statistics
                    with open(os.path.join(checkpoint_dir, "market_stats.json"), 'w') as f:
                        json.dump(market_stats, f, indent=2)
                
                step += 1
        
        # Calculate epoch statistics
        epoch_avg_reward = total_reward / max(1, total_samples)
        print(f"Epoch {epoch+1} - Avg Reward: {epoch_avg_reward:.2f}")
        
        # Save epoch checkpoint
        epoch_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"Saved epoch checkpoint to {epoch_dir}")
        
        # Save as best model if best so far
        if epoch_avg_reward > best_avg_reward:
            best_avg_reward = epoch_avg_reward
            best_dir = os.path.join(output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"New best model with avg reward {best_avg_reward:.2f}")
    
    # Save final model
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final model to {final_dir}")
    
    # Log final market adaptation statistics
    print("\nMarket Adaptation Statistics:")
    print(f"Up examples: {market_stats['up_examples']}, Down examples: {market_stats['down_examples']}")
    print(f"Up predictions: {market_stats['up_predictions']}, Down predictions: {market_stats['down_predictions']}")
    
    up_accuracy = market_stats['correct_up'] / max(1, market_stats['up_examples'])
    down_accuracy = market_stats['correct_down'] / max(1, market_stats['down_examples'])
    print(f"Up accuracy: {up_accuracy:.2f}, Down accuracy: {down_accuracy:.2f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Super Saiyan GRPO Stage II Training")
    parser.add_argument("--stage1_model", type=str, required=True, help="Path to Stage I trained model")
    parser.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--kl_coef", type=float, default=0.05, help="KL divergence coefficient")
    parser.add_argument("--save_steps", type=int, default=50, help="Steps between saving checkpoints")
    parser.add_argument("--natural_distribution", action="store_true", help="Use natural market distribution")
    parser.add_argument("--max_train_samples", type=int, default=5000, help="Maximum number of training samples")
    parser.add_argument("--output_path", type=str, default="./super_saiyan_grpo_stage2", help="Output path for saving model")
    
    args = parser.parse_args()
    
    print("Starting Super Saiyan GRPO Stage II training...")
    
    # Load the Stage I trained model
    model, tokenizer = load_stage1_model(args.stage1_model, quantize=args.quantize)
    
    # Prepare model for Stage II training (fine-tuning)
    model = prepare_model_for_training(model)
    
    # Load dataset - using the same dataset source but without filtering
    print("Loading 2084Collective dataset...")
    dataset = load_dataset("2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2", split="train")
    print(f"Loaded {len(dataset)} samples")
    
    # Limit dataset size if specified
    if args.max_train_samples:
        max_samples = min(args.max_train_samples, len(dataset))
        dataset = dataset.select(range(max_samples))
        print(f"Using {len(dataset)} samples for training")
    
    # Run Stage II training
    model = stage2_training(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        save_steps=args.save_steps,
        natural_distribution=args.natural_distribution,
        output_dir=args.output_path
    )

if __name__ == "__main__":
    main() 