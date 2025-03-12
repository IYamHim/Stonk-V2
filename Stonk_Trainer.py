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

def prepare_model_for_training(model):
    """Prepare model for training with LoRA"""
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # attention heads
        lora_alpha=32,  # alpha scaling
        lora_dropout=0.05,  # dropout probability
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

def load_base_model(quantize=True):
    """Load the base Qwen model with quantization"""
    print("Loading Qwen model and tokenizer...")
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with gradient checkpointing enabled and use_cache disabled
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,  # Disable KV cache for training - required for gradient checkpointing
        torch_dtype=torch.float16  # Use float16 for better numerical stability
    )
    
    # Enable gradient checkpointing
    model.config.use_cache = False  # Make sure it's disabled at config level too
    model.gradient_checkpointing_enable()  # Remove use_reentrant parameter
    
    return model, tokenizer

def format_prompt(example):
    """Format a prompt for up/down prediction with percentage using dataset example"""
    
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
    
    # Get news headlines
    news = company_info.get('news', {}).get('news_headlines', [])
    news_headlines = '\n'.join(news) if news else 'No recent news'
    
    # Get financial metrics
    try:
        financials = json.loads(company_info.get('financials', {}).get('financials', '{}'))
    except:
        financials = {}
    
    # Format financial metrics
    key_metrics = {
        'Revenue Growth': (financials.get('Total Revenue', 0) / 1e9),
        'Profit Margin': (financials.get('Net Income', 0) / financials.get('Total Revenue', 1) * 100),
        'P/E Ratio': current_price / (financials.get('Basic EPS', 1) or 1),
        'Debt to Equity': financials.get('Interest Expense', 0) / financials.get('Net Income', 1)
    }
    financials_str = '\n'.join([f"{k}: {v:.2f}" for k, v in key_metrics.items()])
    
    # System prompt
    system_prompt = """You are an elite stock market analyst with decades of experience. You specialize in predicting short-term stock price movements based on company fundamentals, technical analysis, and market sentiment. Your task is to analyze the given stock information and predict its movement for the next three trading days."""
    
    # User prompt
    user_prompt = f"""Analyze the following stock and predict if it will go UP or DOWN for each of the next 3 trading days, with a percentage for each day.

Company: {ticker}
Description: {description}

Price Movement:
- Current Date: {company_info.get('current_date', 'N/A')}
- Current Price: ${current_price:.2f}
- Previous Close: ${previous_close:.2f}
- Day Change: {day_change_pct:.2f}%
- Volume: {financials.get('Basic Average Shares', 'N/A')}

Recent News:
{news_headlines}

Financial Metrics:
{financials_str}

Your answer should look like the following format for each of the 3 days:

Day 1:
<think>
reasoning about why the stock would go up or down for day 1
- Analyze recent news impact on stock performance
- Consider financial metrics and their implications
- Evaluate current price relative to previous trends
</think>
<confidence>75%</confidence>
<answer>up 2.5%</answer>

Day 2:
<think>
reasoning for day 2...
</think>
<confidence>60%</confidence>
<answer>down 1.2%</answer>

Day 3:
<think>
reasoning for day 3...
</think>
<confidence>50%</confidence>
<answer>up 1.8%</answer>"""
    
    # Combine prompts in chat format
    chat_format = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return chat_format

def extract_prediction(response_text, day_number):
    """Extract prediction for a specific day from the response text."""
    try:
        # Initialize default prediction
        prediction = {
            "direction": "unknown",
            "percentage": 0.0,
            "confidence": 0.0,
            "formatted": "Could not extract prediction"
        }
        
        # Look for day header and answer tag
        day_patterns = [
            f"Day {day_number}:",
            f"**Day {day_number}:**",
            f"Day {day_number} Prediction:"
        ]
        
        # Find the start of the day's section
        day_start = -1
        for pattern in day_patterns:
            day_start = response_text.find(pattern)
            if day_start != -1:
                break
                
        if day_start == -1:
            return prediction
            
        # Find the next day's section or end of text
        next_day_start = len(response_text)
        for pattern in [f"Day {day_number + 1}:", f"**Day {day_number + 1}:**", f"Day {day_number + 1} Prediction:", "---"]:
            next_start = response_text.find(pattern, day_start)
            if next_start != -1:
                next_day_start = min(next_day_start, next_start)
        
        # Extract the day's section
        day_section = response_text[day_start:next_day_start]
        
        # Extract confidence
        confidence_pattern = r'<confidence>(.*?)</confidence>'
        confidence_match = re.search(confidence_pattern, day_section, re.DOTALL)
        if confidence_match:
            confidence_text = confidence_match.group(1).strip().lower()
            # Extract percentage number
            confidence_value_match = re.search(r'(\d+)%?', confidence_text)
            if confidence_value_match:
                prediction["confidence"] = float(confidence_value_match.group(1))
        
        # Look for answer in various formats
        answer_patterns = [
            r'<answer>(.*?)</answer>',  # Standard format
            r'Answer:.*?(up|down)\s+(\d+\.?\d*)%',  # Alternative format
            r'- \*\*Answer:\*\*\s*(up|down)\s+(\d+\.?\d*)%',  # Markdown format
            r'answer:\s*(up|down)\s+(\d+\.?\d*)%',  # Simple format
            r'Conclusion:\s*(up|down)\s+(\d+\.?\d*)%',  # Conclusion format
            r'- \*\*Conclusion:\*\*\s*(up|down)\s+(\d+\.?\d*)%',  # Markdown conclusion format
            r'\*\*Conclusion:\*\*\s*(up|down)\s+(\d+\.?\d*)%',  # Bold conclusion format
            r'conclusion:\s*(up|down)\s+(\d+\.?\d*)%'  # Simple conclusion format
        ]
        
        for pattern in answer_patterns:
            matches = re.finditer(pattern, day_section, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.groups()) == 1:  # Standard format
                    answer_text = match.group(1).strip().lower()
                    direction_match = re.search(r'(up|down)\s+(\d+\.?\d*)%', answer_text)
                    if direction_match:
                        prediction["direction"] = direction_match.group(1)
                        prediction["percentage"] = float(direction_match.group(2))
                        prediction["formatted"] = f"{prediction['direction']} {prediction['percentage']}%"
                        return prediction
                elif len(match.groups()) == 2:  # Alternative formats
                    prediction["direction"] = match.group(1).lower()
                    prediction["percentage"] = float(match.group(2))
                    prediction["formatted"] = f"{prediction['direction']} {prediction['percentage']}%"
                    return prediction
        
        # If no matches found yet, try looking for just the prediction pattern anywhere in the section
        prediction_pattern = r'(up|down)\s+(\d+\.?\d*)%'
        matches = re.finditer(prediction_pattern, day_section, re.IGNORECASE)
        for match in matches:
            prediction["direction"] = match.group(1).lower()
            prediction["percentage"] = float(match.group(2))
            prediction["formatted"] = f"{prediction['direction']} {prediction['percentage']}%"
            return prediction
        
        return prediction
        
    except Exception as e:
        print(f"Error extracting prediction: {str(e)}")
        return {
            "direction": "unknown",
            "percentage": 0.0,
            "confidence": 0.0,
            "formatted": "Error extracting prediction"
        }

def extract_thinking(response, day_number=None):
    """Extract the thinking section from the response for a specific day"""
    if day_number is not None:
        # Look for thinking after "Day X:" marker
        day_pattern = f"Day {day_number}:"
        parts = response.split(day_pattern)
        if len(parts) > 1:
            response = parts[1].split("Day")[0] if "Day" in parts[1] else parts[1]
    
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return "No thinking section found"

def compute_reward(prediction, actual_outcome=None, response_text=None, example=None):
    """
    Compute a reward score based on prediction quality and format adherence.
    If actual_outcome is None, we simulate a ground truth for demonstration.
    
    Args:
        prediction: Dictionary with direction and percentage 
        actual_outcome: Dictionary with actual direction and percentage (optional)
        response_text: Full text of model response (optional)
        example: Original example with company data (optional)
    
    Returns:
        tuple: (reward score (0-10), explanation string, actual_outcome dictionary)
    """
    # Initialize reward
    max_reward = 10.0
    reward_components = {}
    
    # If no prediction could be extracted, return minimum reward with dummy outcome
    if prediction["direction"] == "unknown":
        dummy_outcome = {
            "direction": "unknown",
            "percentage": 0.0,
            "formatted": "Could not simulate outcome - invalid prediction"
        }
        explanation = "Could not extract a valid prediction"
        return 0, explanation, dummy_outcome
    
    # If no actual outcome provided, simulate one
    if actual_outcome is None:
        # For demonstration: simulate outcome with some randomness
        # but biased to follow the current trend
        if random.random() < 0.7:  # 70% chance to be "correct"
            direction = prediction["direction"]
            # Generate percentage close to predicted
            base_pct = prediction["percentage"]
            # Add some noise
            actual_pct = base_pct * (0.7 + random.random() * 0.6)  # Between 70-130% of prediction
            actual_outcome = {
                "direction": direction,
                "percentage": actual_pct,
                "formatted": f"{direction} {actual_pct:.1f}%"
            }
        else:
            # Opposite direction
            direction = "down" if prediction["direction"] == "up" else "up"
            actual_pct = prediction["percentage"] * (0.5 + random.random())
            actual_outcome = {
                "direction": direction,
                "percentage": actual_pct,
                "formatted": f"{direction} {actual_pct:.1f}%"
            }
    
    # Extract thinking for format evaluation
    thinking = extract_thinking(response_text) if response_text else ""
    
    # 1. Format Reward (20% of total) - increased from 10% to ensure positive total reward
    format_reward = 0
    if thinking and "<answer>" in response_text:
        format_reward = 2.0  # Increased from 1.0 to ensure positive reward even with max penalty
        reward_components["format"] = "2.0 - Followed reasoning format"
    else:
        reward_components["format"] = "0.0 - Failed to follow format instructions"
    
    # 2. Direction Reward (30% of total) - adjusted from 35%
    direction_reward = 0
    if prediction["direction"] == actual_outcome["direction"]:
        direction_reward = 3.0  # Adjusted from 3.5 to maintain total reward balance
        reward_components["direction"] = "3.0 - Correct direction prediction"
    else:
        direction_reward = 0.0
        reward_components["direction"] = "0.0 - Incorrect direction prediction"
    
    # 3. Magnitude Reward (20% of total) - adjusted from 25%
    magnitude_reward = 0
    predicted_magnitude = prediction["percentage"]
    actual_magnitude = actual_outcome["percentage"]
    
    # Calculate error as percentage difference
    error = abs(predicted_magnitude - actual_magnitude)
    
    # Only calculate magnitude reward if direction is correct
    if direction_reward > 0:
        # Scale reward based on error
        if error < 0.5:  # Very accurate
            magnitude_reward = 2.0  # Adjusted from 2.5
            reward_components["magnitude"] = f"2.0 - Very accurate (error: {error:.2f}%)"
        elif error < 1.0:  # Good accuracy
            magnitude_reward = 1.5  # Adjusted from 2.0
            reward_components["magnitude"] = f"1.5 - Good accuracy (error: {error:.2f}%)"
        elif error < 2.0:  # Decent accuracy
            magnitude_reward = 1.0  # Adjusted from 1.5
            reward_components["magnitude"] = f"1.0 - Decent accuracy (error: {error:.2f}%)"
        elif error < 4.0:  # Poor accuracy
            magnitude_reward = 0.5  # Unchanged
            reward_components["magnitude"] = f"0.5 - Poor accuracy (error: {error:.2f}%)"
        else:  # Bad accuracy
            magnitude_reward = 0.25  # Adjusted from 0.5
            reward_components["magnitude"] = f"0.25 - Minimum reward for correct direction (error: {error:.2f}%)"
    else:
        magnitude_reward = 0.0
        reward_components["magnitude"] = f"0.0 - No reward (wrong direction)"
    
    # 4. Confidence Reward (15% of total) - unchanged percentage
    confidence_reward = 0
    model_confidence = prediction.get("confidence", 50)  # Default to 50% if not provided
    
    # Calculate confidence reward based on whether direction is correct
    if prediction["direction"] == actual_outcome["direction"]:
        # Direction is correct - reward appropriately calibrated confidence
        # Calculate magnitude accuracy as a percentage (0-100%)
        max_error_pct = 5.0  # Maximum error we consider (5% is very bad)
        magnitude_accuracy = max(0, 100 - (error / max_error_pct * 100))
        
        # Normalize confidence to 0-1 scale
        normalized_confidence = model_confidence / 100.0
        # Normalize magnitude accuracy to 0-1 scale
        normalized_magnitude_accuracy = magnitude_accuracy / 100.0
        
        # For high confidence predictions (>70%)
        if normalized_confidence > 0.7:
            if normalized_magnitude_accuracy > 0.7:
                # High confidence with high accuracy - excellent!
                confidence_reward = 1.5
                reward_components["confidence"] = f"1.5 - High confidence ({model_confidence}%) with high accuracy ({magnitude_accuracy:.0f}%)"
            elif normalized_magnitude_accuracy > 0.5:
                # High confidence with moderate accuracy - good
                confidence_reward = 1.2
                reward_components["confidence"] = f"1.2 - High confidence ({model_confidence}%) with moderate accuracy ({magnitude_accuracy:.0f}%)"
            else:
                # High confidence with low accuracy - still give some reward
                confidence_reward = 0.6
                reward_components["confidence"] = f"0.6 - High confidence ({model_confidence}%) with low accuracy ({magnitude_accuracy:.0f}%)"
        else:
            # Default reward for correct direction with moderate/low confidence
            confidence_reward = 0.8
            reward_components["confidence"] = f"0.8 - Moderate/low confidence ({model_confidence}%) with correct direction"
    else:
        # Direction is INCORRECT - apply penalty based on confidence level
        # Higher confidence = larger penalty
        normalized_confidence = model_confidence / 100.0
        
        # Calculate penalty - scales from -0.2 (very low confidence) to -1.5 (very high confidence)
        # This creates a steeper penalty for being very confident when wrong
        if normalized_confidence > 0.8:
            # Very high confidence (>80%) when wrong - large penalty
            confidence_reward = -1.5
            reward_components["confidence"] = f"-1.5 - Severe penalty for very high confidence ({model_confidence}%) with incorrect direction"
        elif normalized_confidence > 0.6:
            # High confidence (60-80%) when wrong - significant penalty
            confidence_reward = -1.0
            reward_components["confidence"] = f"-1.0 - Significant penalty for high confidence ({model_confidence}%) with incorrect direction"
        elif normalized_confidence > 0.4:
            # Moderate confidence (40-60%) when wrong - moderate penalty
            confidence_reward = -0.7
            reward_components["confidence"] = f"-0.7 - Moderate penalty for medium confidence ({model_confidence}%) with incorrect direction"
        elif normalized_confidence > 0.2:
            # Low confidence (20-40%) when wrong - small penalty
            confidence_reward = -0.4
            reward_components["confidence"] = f"-0.4 - Small penalty for low confidence ({model_confidence}%) with incorrect direction"
        else:
            # Very low confidence (<20%) when wrong - minimal penalty
            confidence_reward = -0.2
            reward_components["confidence"] = f"-0.2 - Minimal penalty for very low confidence ({model_confidence}%) with incorrect direction"
    
    # 5. Data Utilization Reward (15% of total) - unchanged percentage
    data_utilization_reward = 0
    if response_text and example:
        # Get key data points to check for in reasoning
        ticker = example.get('ticker', '')
        price = example.get('company_info', {}).get('price', {}).get('close', 0)
        day_change_pct = 0
        previous_close = example.get('company_info', {}).get('price', {}).get('close_previous', 0)
        if previous_close and price:
            day_change_pct = ((price - previous_close) / previous_close) * 100
            
        news_headlines = example.get('company_info', {}).get('news', {}).get('news_headlines', [])
        
        # Count referenced data points
        data_points_referenced = 0
        
        # Check for ticker reference
        if ticker and ticker.lower() in thinking.lower():
            data_points_referenced += 1
            
        # Check for price reference - FIX: Check each condition individually
        price_referenced = False
        if price > 0:
            if f"${price:.1f}" in thinking:
                price_referenced = True
            elif f"${price:.0f}" in thinking:
                price_referenced = True
            elif f"{price:.1f}" in thinking:
                price_referenced = True
            elif f"{price:.0f}" in thinking:
                price_referenced = True
                
        if price_referenced:
            data_points_referenced += 1
            
        # Check for day change reference
        if day_change_pct != 0 and any([f"{day_change_pct:.1f}%" in thinking, f"{day_change_pct:.0f}%" in thinking]):
            data_points_referenced += 1
            
        # Check for news references
        news_referenced = 0
        for headline in news_headlines:
            if headline.lower() in thinking.lower():
                news_referenced += 1
        if news_referenced > 0:
            data_points_referenced += min(2, news_referenced)  # Cap at 2 points for news references
            
        # Calculate reward based on referenced data points
        if data_points_referenced >= 4:
            data_utilization_reward = 1.5
            reward_components["data_utilization"] = f"1.5 - Excellent use of data (referenced {data_points_referenced} data points)"
        elif data_points_referenced == 3:
            data_utilization_reward = 1.2
            reward_components["data_utilization"] = f"1.2 - Good use of data (referenced {data_points_referenced} data points)"
        elif data_points_referenced == 2:
            data_utilization_reward = 0.9
            reward_components["data_utilization"] = f"0.9 - Moderate use of data (referenced {data_points_referenced} data points)"
        elif data_points_referenced == 1:
            data_utilization_reward = 0.5
            reward_components["data_utilization"] = f"0.5 - Limited use of data (referenced only {data_points_referenced} data point)"
        else:
            reward_components["data_utilization"] = "0.0 - Did not reference any specific data points"
    else:
        reward_components["data_utilization"] = "0.0 - No example data or reasoning provided to evaluate"
    
    # Sum rewards and clip to range [0, max_reward]
    total_reward = format_reward + direction_reward + magnitude_reward + confidence_reward + data_utilization_reward
    total_reward = max(0, min(max_reward, total_reward))
    
    # Create explanation
    explanation = (
        f"Format: {reward_components['format']}\n"
        f"Direction: {reward_components['direction']}\n"
        f"Magnitude: {reward_components['magnitude']}\n"
        f"Confidence: {reward_components['confidence']}\n"
        f"Data Utilization: {reward_components.get('data_utilization', '0.0 - Not evaluated')}\n"
        f"Total reward: {total_reward:.1f}/{max_reward:.1f}"
    )
    
    return total_reward, explanation, actual_outcome

def find_next_trading_day(dataset, example_idx):
    """Find the next trading day and its price from the dataset"""
    try:
        # Get current example
        current = dataset[example_idx]
        current_date = current['company_info']['current_date']
        current_ticker = current['ticker']
        
        # Look for next example with same ticker
        for i in range(example_idx + 1, len(dataset)):
            next_example = dataset[i]
            if next_example['ticker'] == current_ticker:
                return next_example['company_info']['current_date'], next_example['company_info']['price']['close']
        
        return None, None
    except Exception as e:
        print(f"Error finding next trading day: {str(e)}")
        return None, None

def find_next_trading_day_for_testing(dates_by_ticker, ticker, current_date):
    """Find the next trading day and its price for test mode"""
    try:
        if ticker not in dates_by_ticker:
            return None, None
            
        # Convert date to datetime for comparison
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        
        # Find the next date after current_date
        next_date = None
        next_price = None
        
        for entry in dates_by_ticker[ticker]:
            entry_dt = datetime.strptime(entry['date'], '%Y-%m-%d')
            if entry_dt > current_dt:
                next_date = entry['date']
                next_price = entry['close']
                break
                
        return next_date, next_price
    except Exception as e:
        print(f"Error finding next trading day: {str(e)}")
        return None, None

def grpo_training(model, tokenizer, dataset, epochs=3, batch_size=4, learning_rate=1e-5, kl_coef=0.1, save_steps=50, diverse_predictions=False, output_dir = "./stonk_trainer_grpo"):
    """
    Direct GRPO training without SFT
    """
    print("Starting Direct GRPO Training...")
    
    # Create output directory
    
    os.makedirs(output_dir, exist_ok=True)
    # Create checkpoints directory
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Log file
    log_file = os.path.join(output_dir, "training_log.jsonl")
    
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
    
    # Track diversity of predictions for each ticker
    if diverse_predictions:
        ticker_predictions = {}
    
    # GRPO training loop
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")
        model.train()
        
        # Sample examples for this epoch
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
                            max_new_tokens=300,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    
                    # Get generated text
                    response_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    # Extract prediction from response
                    prediction = extract_prediction(response_text, 1)  # Get Day 1 prediction
                    
                    # If enforcing diverse predictions, track and adjust rewards
                    if diverse_predictions:
                        ticker = sample['ticker']
                        if ticker not in ticker_predictions:
                            ticker_predictions[ticker] = {'up': 0, 'down': 0, 'total': 0}
                        
                        # Count this prediction
                        ticker_info = ticker_predictions[ticker]
                        ticker_info['total'] += 1
                        if prediction['direction'] == 'up':
                            ticker_info['up'] += 1
                        elif prediction['direction'] == 'down':
                            ticker_info['down'] += 1
                    
                    # Get actual outcome
                    next_date, next_price = find_next_trading_day(dataset, idx)
                    
                    if next_date and next_price:
                        current_price = sample['company_info']['price']['close']
                        actual_change_pct = ((next_price - current_price) / current_price) * 100
                        actual_direction = "up" if actual_change_pct > 0 else "down"
                        
                        actual_outcome = {
                            "direction": actual_direction,
                            "percentage": abs(actual_change_pct),
                            "formatted": f"{actual_direction} {abs(actual_change_pct):.1f}%"
                        }
                        
                        # Compute reward
                        reward, explanation, _ = compute_reward(prediction, actual_outcome, response_text, sample)
                        
                        # If enforcing diversity, adjust reward based on prediction frequency
                        if diverse_predictions and ticker_info['total'] > 5:  # Only after we have enough data
                            up_ratio = ticker_info['up'] / ticker_info['total']
                            # If we're heavily biased toward one direction, penalize it
                            if (prediction['direction'] == 'up' and up_ratio > 0.6) or \
                               (prediction['direction'] == 'down' and up_ratio < 0.4):
                                # Apply diversity penalty - stronger penalty (0.3 to 1.0 range)
                                diversity_factor = max(0.3, 1.0 - abs(up_ratio - 0.5) * 3)  # Ranges from 0.3 to 1.0
                                old_reward = reward
                                reward = reward * diversity_factor
                                print(f"Applied diversity penalty: {old_reward:.2f} -> {reward:.2f} for {ticker} (up ratio: {up_ratio:.2f})")
                        
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
                            
                            # Add KL penalty
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
                            "reward": float(reward.item()),
                            "loss": float(total_loss.item()) if not torch.isnan(total_loss) else "NaN",
                            "thinking": extract_thinking(response_text)
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
                
                # Increment step
                step += 1
                
                # Save checkpoint if needed
                if step % save_steps == 0:
                    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint-{step}")
                    model.save_pretrained(checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
        
        # Epoch summary
        if total_samples > 0:
            avg_epoch_reward = total_reward / total_samples
            print(f"Epoch {epoch+1} summary - Avg Reward: {avg_epoch_reward:.2f}, Samples: {total_samples}")
            
            # Save if best epoch
            if avg_epoch_reward > best_avg_reward:
                best_avg_reward = avg_epoch_reward
                best_model_path = os.path.join(checkpoints_dir, "best_model")
                model.save_pretrained(best_model_path)
                print(f"New best model saved to {best_model_path}")
            
            # Save epoch checkpoint
            epoch_path = os.path.join(checkpoints_dir, f"epoch-{epoch+1}")
            model.save_pretrained(epoch_path)
            print(f"Epoch {epoch+1} model saved to {epoch_path}")
    
    # Save final model
    final_path = os.path.join(checkpoints_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")
    
    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Qwen Stock Prediction Test")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--test", action="store_true", help="Enable test mode")
    parser.add_argument("--model_path", type=str, help="Path to trained model for testing")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL divergence coefficient")
    parser.add_argument("--save_steps", type=int, default=50, help="How often to save checkpoints")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples to use")
    parser.add_argument("--diverse_predictions", action="store_true", help="Enforce diverse predictions during training")
    parser.add_argument("--output_dir", type=str, default="./stonk_trainer_grpo", help="Output directory for training")
    
    args = parser.parse_args()
    
    print("Starting Qwen stock prediction test...")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_base_model(quantize=args.quantize)
        
        if args.train:
            # Load the full dataset
            print("Loading 2084Collective dataset...")
            dataset = load_dataset("2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2", split="train")
            print(f"Loaded {len(dataset)} samples")
            
            if args.max_train_samples:
                dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))
                print(f"Using {len(dataset)} samples for training")
                
            # Apply additional data processing for diverse training
            if args.diverse_predictions:
                # Ensure training has diverse examples with both up and down movements
                filtered_examples = []
                up_count = 0
                down_count = 0
                target_ratio = 0.5  # Try to get close to 50/50 up/down
                
                print("Filtering dataset for balanced up/down examples...")
                for i, example in enumerate(tqdm(dataset, desc="Filtering")):
                    try:
                        # Find the next day's outcome
                        next_date, next_price = find_next_trading_day(dataset, i)
                        if next_date and next_price:
                            current_price = example['company_info']['price']['close']
                            actual_change_pct = ((next_price - current_price) / current_price) * 100
                            is_up = actual_change_pct > 0
                            
                            # Balance dataset
                            current_ratio = up_count / (up_count + down_count + 1e-10)
                            
                            # Stricter balancing - only add examples that improve balance
                            if (is_up and current_ratio < target_ratio) or (not is_up and current_ratio > target_ratio):
                                filtered_examples.append(example)
                                if is_up:
                                    up_count += 1
                                else:
                                    down_count += 1
                                    
                                # Print progress on balancing
                                if (up_count + down_count) % 100 == 0:
                                    print(f"Current balance: Up: {up_count}, Down: {down_count}, Ratio: {up_count/(up_count+down_count):.2f}")
                    except Exception as e:
                        continue
                
                if filtered_examples:
                    # Replace dataset with balanced version
                    from datasets import Dataset
                    balanced_dataset = Dataset.from_list(filtered_examples)
                    dataset = balanced_dataset
                    print(f"Created balanced dataset with {len(dataset)} samples - Up: {up_count}, Down: {down_count}")
            
            # Prepare model for training
            model = prepare_model_for_training(model)
            
            # Run GRPO training
            model = grpo_training(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                kl_coef=args.kl_coef,
                save_steps=args.save_steps,
                diverse_predictions=args.diverse_predictions,
                output_dir=args.output_dir
            )
        
        if args.test:
            # Create sample data for testing
            print("Creating sample data for testing...")
            
            # Create dates_by_ticker for testing
            dates_by_ticker = {
                'AAPL': [
                    {'date': '2023-01-02', 'close': 152.0},
                    {'date': '2023-01-03', 'close': 155.0},
                    {'date': '2023-01-04', 'close': 153.0},
                    {'date': '2023-01-05', 'close': 158.0}
                ],
                'MSFT': [
                    {'date': '2023-01-02', 'close': 252.0},
                    {'date': '2023-01-03', 'close': 258.0},
                    {'date': '2023-01-04', 'close': 254.0},
                    {'date': '2023-01-05', 'close': 262.0}
                ],
                'GOOGL': [
                    {'date': '2023-01-02', 'close': 95.0},
                    {'date': '2023-01-03', 'close': 93.0},
                    {'date': '2023-01-04', 'close': 96.0},
                    {'date': '2023-01-05', 'close': 94.5}
                ],
                'AMZN': [
                    {'date': '2023-01-02', 'close': 86.0},
                    {'date': '2023-01-03', 'close': 85.0},
                    {'date': '2023-01-04', 'close': 87.0},
                    {'date': '2023-01-05', 'close': 88.5}
                ]
            }
            
            sample_data = [
                {
                    'ticker': 'AAPL',
                    'company_info': {
                        'current_date': '2023-01-01',
                        'company_info': {'description': 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.'},
                        'price': {'close': 150.0, 'close_previous': 145.0},
                        'news': {'news_headlines': ['Apple announces new iPhone', 'Apple reports record profits']},
                        'financials': {'financials': '{"Total Revenue": 365000000000, "Net Income": 95000000000, "Basic EPS": 6.0}'}
                    }
                },
                {
                    'ticker': 'MSFT',
                    'company_info': {
                        'current_date': '2023-01-01',
                        'company_info': {'description': 'Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.'},
                        'price': {'close': 250.0, 'close_previous': 245.0},
                        'news': {'news_headlines': ['Microsoft announces new Windows version', 'Microsoft cloud business growing']},
                        'financials': {'financials': '{"Total Revenue": 198000000000, "Net Income": 72000000000, "Basic EPS": 9.5}'}
                    }
                },
                {
                    'ticker': 'GOOGL',
                    'company_info': {
                        'current_date': '2023-01-01',
                        'company_info': {'description': 'Alphabet Inc., through its subsidiaries, provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.'},
                        'price': {'close': 94.0, 'close_previous': 96.0},
                        'news': {'news_headlines': ['Google layoffs impact thousands', 'Antitrust concerns continue to plague Google']},
                        'financials': {'financials': '{"Total Revenue": 282800000000, "Net Income": 59900000000, "Basic EPS": 4.5}'}
                    }
                },
                {
                    'ticker': 'AMZN',
                    'company_info': {
                        'current_date': '2023-01-01',
                        'company_info': {'description': 'Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions in North America and internationally.'},
                        'price': {'close': 85.0, 'close_previous': 84.0},
                        'news': {'news_headlines': ['Amazon expands Prime service', 'Cloud services see growth despite economic headwinds']},
                        'financials': {'financials': '{"Total Revenue": 469800000000, "Net Income": 12700000000, "Basic EPS": 1.25}'}
                    }
                }
            ]
            
            # Test each sample
            total_overall_reward = 0
            for i, example in enumerate(sample_data):
                ticker = example['ticker']
                current_date = example['company_info']['current_date']
                current_price = example['company_info']['price']['close']
                
                print(f"\n============================== TESTING STOCK {i+1}: {ticker} ==============================")
                print(f"Current date: {current_date}")
                print(f"Available trading days: {len(dates_by_ticker[ticker])}")
                
                # Format prompt
                prompt = format_prompt(example)
                print("\nPrompt:")
                print("-" * 80)
                print(prompt)
                print("-" * 80)
                
                print("\nGenerating prediction...")
                
                # Generate prediction
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("\nModel's Full Response:")
                print("=" * 80)
                print(response)
                print("=" * 80)

                # Extract and analyze predictions
                print("\nAnalyzing predictions for each day:")
                print("=" * 50)
                
                days = ["Day 1", "Day 2", "Day 3"]
                total_reward = 0
                last_date = current_date
                
                for day_idx, day in enumerate(days, 1):
                    # Use our extract_prediction function instead of simple regex
                    prediction_data = extract_prediction(response, day_idx)
                    
                    print(f"\n{day} Prediction:")
                    print(f"Prediction: {prediction_data['formatted']}")
                    print(f"Confidence: {prediction_data['confidence']}%")
                    
                    # Find next actual trading day and price
                    next_date, next_price = find_next_trading_day_for_testing(dates_by_ticker, ticker, current_date)
                    
                    if next_date and next_price:
                        actual_change_pct = ((next_price - current_price) / current_price) * 100
                        actual_direction = "up" if actual_change_pct > 0 else "down"
                        
                        # Create actual outcome dictionary
                        actual_outcome = {
                            "direction": actual_direction,
                            "percentage": abs(actual_change_pct),
                            "formatted": f"{actual_direction} {abs(actual_change_pct):.1f}%"
                        }
                        
                        print("\nReward Function Evaluation:")
                        print("-" * 50)
                        print(f"Next trading day: {next_date}")
                        print(f"Actual Outcome: {actual_outcome['formatted']}")
                        
                        # Use our updated compute_reward function with the example data
                        day_reward, explanation, _ = compute_reward(prediction_data, actual_outcome, response, example)
                        total_reward += day_reward
                        
                        # Print the explanation which includes all reward components
                        print(explanation)
                        print(f"{day} reward: {day_reward:.1f}/10.0")
                        
                        last_date = next_date
                    else:
                        print(f"No trading day found after {last_date}")
                
                print(f"\nStock {ticker} Performance:")
                print("=" * 50)
                print(f"Average reward across 3 days: {(total_reward/3):.1f}/10.0")
                total_overall_reward += total_reward
            
            print("\n============================== OVERALL PERFORMANCE ==============================")
            print(f"Average reward across all stocks: {(total_overall_reward/(3*len(sample_data))):.1f}/10.0")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # For running interactively - add necessary data for testing
    if 'main' not in globals():
        dates_by_ticker = {
            'AAPL': [
                {'date': '2023-01-02', 'close': 152.0},
                {'date': '2023-01-03', 'close': 155.0},
                {'date': '2023-01-04', 'close': 153.0},
                {'date': '2023-01-05', 'close': 158.0}
            ],
            'MSFT': [
                {'date': '2023-01-02', 'close': 252.0},
                {'date': '2023-01-03', 'close': 258.0},
                {'date': '2023-01-04', 'close': 254.0},
                {'date': '2023-01-05', 'close': 262.0}
            ]
        }
    main() 