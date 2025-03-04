import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from datetime import datetime, timedelta
import random
import math
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import json

def format_prompt(example):
    """Format a prompt for up/down prediction with percentage using dataset example"""
    
    # Get company info
    company_info = example['company_info']
    price_info = company_info['price']
    financials = json.loads(company_info['financials']['financials'])
    
    # Calculate day change percentage
    day_change_pct = ((price_info['close'] - price_info['close_previous']) / price_info['close_previous']) * 100
    
    # Format news headlines
    news_headlines = '\n'.join(company_info['news']['news_headlines'])
    
    # Format financial metrics
    key_metrics = {
        'Revenue Growth': (financials.get('Total Revenue', 0) / 1e9),
        'Profit Margin': (financials.get('Net Income', 0) / financials.get('Total Revenue', 1) * 100),
        'P/E Ratio': price_info['close'] / (financials.get('Basic EPS', 1) or 1),
        'Debt to Equity': financials.get('Interest Expense', 0) / financials.get('Net Income', 1)
    }
    financials_str = '\n'.join([f"{k}: {v:.2f}" for k, v in key_metrics.items()])
    
    # System prompt
    system_prompt = """You are an elite stock market analyst and trader with decades of experience in financial markets. You have:
1. A proven track record of accurate stock predictions
2. Deep expertise in technical and fundamental analysis
3. Advanced understanding of market psychology and sentiment analysis"""
    
    # User prompt
    user_prompt = f"""Analyze the following stock and predict if it will go UP or DOWN for each of the next 3 trading days, with a percentage for each day.

Company: {example['ticker']}
Description: {company_info['company_info']['description']}

Price Movement:
- Current Date: {company_info['current_date']}
- Current Price: ${price_info['close']:.2f}
- Previous Close: ${price_info['close_previous']:.2f}
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
<answer>up 1.8%</answer>

Please provide detailed reasoning for each day in the <think></think> tags, considering how previous days' predictions might affect subsequent days. Then provide your confidence level (0-100%) in the <confidence></confidence> tags. Finally, provide your answer as a direction (up/down) and percentage in the <answer></answer> tags."""
    
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

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

def estimate_confidence(prediction, thinking):
    """
    Estimate the model's confidence in its prediction based on thinking and prediction
    Returns a confidence percentage from 0-100%
    """
    # Start with base confidence
    confidence = 50  # More neutral starting point
    
    # If no valid prediction or thinking, low confidence
    if prediction["direction"] == "unknown" or not thinking or thinking == "No thinking section found":
        return 20  # Lower base confidence for invalid predictions
    
    # Check length and detail of reasoning (more detailed reasoning often indicates more confidence)
    words = thinking.split()
    if len(words) > 150:  # Very detailed analysis
        confidence += 15
    elif len(words) > 100:  # Detailed analysis
        confidence += 10
    elif len(words) < 30:  # Very brief analysis
        confidence -= 15
    
    # Look for hedging language in thinking
    hedging_terms = ["may", "might", "could", "possibly", "perhaps", "uncertain", 
                     "unclear", "potential", "not sure", "difficult to predict",
                     "speculative", "risky", "volatile", "unpredictable"]
    hedging_count = sum(1 for term in hedging_terms if term in thinking.lower())
    confidence -= hedging_count * 4  # Increased penalty for hedging terms
    
    # Look for confident language
    confident_terms = ["certainly", "definitely", "clearly", "strong", "confident", 
                       "evident", "obvious", "will", "expect", "likely", "bullish",
                       "bearish", "trend", "pattern", "consistent", "reliable"]
    confident_count = sum(1 for term in confident_terms if term in thinking.lower())
    confidence += confident_count * 3  # Increased reward for confident terms
    
    # Check for data-driven analysis
    data_terms = ["data", "metrics", "analysis", "statistics", "numbers", 
                 "percentage", "growth", "decline", "earnings", "revenue", "profit"]
    data_count = sum(1 for term in data_terms if term in thinking.lower())
    confidence += data_count * 2  # Reward for data-driven analysis
    
    # Check for mixed signals in the analysis
    if "however" in thinking.lower() or "but" in thinking.lower() or "although" in thinking.lower():
        confidence -= 8  # Increased penalty for mixed signals
    
    # Check for extreme predictions (very high or low percentages may indicate less confidence)
    if prediction["percentage"] is not None:
        if prediction["percentage"] > 5:
            confidence -= 10  # Increased penalty for extreme predictions
        elif prediction["percentage"] > 3:
            confidence -= 5  # Moderate penalty for high predictions
        if prediction["percentage"] < 0.5:
            confidence -= 8  # Increased penalty for very small predictions
    
    # Check for market condition awareness
    market_terms = ["market conditions", "sector performance", "industry trends", 
                   "economic indicators", "market sentiment", "broader market"]
    market_awareness = sum(1 for term in market_terms if term in thinking.lower())
    confidence += market_awareness * 3  # Reward for market awareness
    
    # Ensure confidence stays within bounds
    confidence = max(min(confidence, 95), 5)  # Cap between 5% and 95%
    
    return confidence

def compute_reward(prediction, actual_outcome=None, response_text=None):
    """
    Compute a reward score based on prediction quality and format adherence.
    If actual_outcome is None, we simulate a ground truth for demonstration.
    
    Args:
        prediction: Dictionary with direction and percentage 
        actual_outcome: Dictionary with actual direction and percentage (optional)
        response_text: Full text of model response (optional)
    
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
    
    # 1. Format Reward (10% of total) - binary reward for following format
    format_reward = 0
    if thinking and "<answer>" in response_text:
        format_reward = 1.0
        reward_components["format"] = "1.0 - Followed reasoning format"
    else:
        reward_components["format"] = "0.0 - Failed to follow format instructions"
    
    # 2. Direction Reward (40% of total)
    direction_reward = 0
    if prediction["direction"] == actual_outcome["direction"]:
        direction_reward = 4.0
        reward_components["direction"] = "4.0 - Correct direction prediction"
    else:
        direction_reward = 0.0
        reward_components["direction"] = "0.0 - Incorrect direction prediction"
    
    # 3. Magnitude Reward (30% of total)
    magnitude_reward = 0
    predicted_magnitude = prediction["percentage"]
    actual_magnitude = actual_outcome["percentage"]
    
    # Calculate error as percentage difference
    error = abs(predicted_magnitude - actual_magnitude)
    
    # Only calculate magnitude reward if direction is correct
    if direction_reward > 0:
        # Scale reward based on error
        if error < 0.5:  # Very accurate
            magnitude_reward = 3.0
            reward_components["magnitude"] = f"3.0 - Very accurate (error: {error:.2f}%)"
        elif error < 1.0:  # Good accuracy
            magnitude_reward = 2.25
            reward_components["magnitude"] = f"2.25 - Good accuracy (error: {error:.2f}%)"
        elif error < 2.0:  # Decent accuracy
            magnitude_reward = 1.5
            reward_components["magnitude"] = f"1.5 - Decent accuracy (error: {error:.2f}%)"
        elif error < 4.0:  # Poor accuracy
            magnitude_reward = 1.0  # Minimum 1 point for correct direction
            reward_components["magnitude"] = f"1.0 - Poor accuracy (error: {error:.2f}%)"
        else:  # Bad accuracy
            magnitude_reward = 1.0  # Minimum 1 point for correct direction
            reward_components["magnitude"] = f"1.0 - Minimum reward for correct direction (error: {error:.2f}%)"
    else:
        magnitude_reward = 0.0
        reward_components["magnitude"] = f"0.0 - No reward (wrong direction)"
    
    # 4. Confidence Reward (20% of total) - MORE LENIENT VERSION
    confidence_reward = 0
    
    # Get model's predicted confidence (if available) or estimate it
    model_confidence = prediction.get("confidence", 0)
    if model_confidence == 0:
        # Fallback to estimated confidence if not provided
        model_confidence = estimate_confidence(prediction, thinking)
    
    # Only calculate confidence reward if direction is correct
    if direction_reward > 0:
        # Calculate magnitude accuracy as a percentage (0-100%)
        max_error_pct = 5.0  # Maximum error we consider (5% is very bad)
        magnitude_accuracy = max(0, 100 - (error / max_error_pct * 100))
        
        # Normalize confidence to 0-1 scale
        normalized_confidence = model_confidence / 100.0
        # Normalize magnitude accuracy to 0-1 scale
        normalized_magnitude_accuracy = magnitude_accuracy / 100.0
        
        # MORE LENIENT CONFIDENCE REWARD CALCULATION
        
        # For high confidence predictions (>70%)
        if normalized_confidence > 0.7:
            if normalized_magnitude_accuracy > 0.7:
                # High confidence with high accuracy - excellent!
                confidence_reward = 2.0
                reward_components["confidence"] = f"2.0 - High confidence ({model_confidence}%) with high accuracy ({magnitude_accuracy:.0f}%)"
            elif normalized_magnitude_accuracy > 0.5:
                # High confidence with moderate accuracy - good
                confidence_reward = 1.5
                reward_components["confidence"] = f"1.5 - High confidence ({model_confidence}%) with moderate accuracy ({magnitude_accuracy:.0f}%)"
            else:
                # High confidence with low accuracy - still give some reward
                confidence_reward = 0.75
                reward_components["confidence"] = f"0.75 - High confidence ({model_confidence}%) with low accuracy ({magnitude_accuracy:.0f}%)"
        
        # For moderate confidence predictions (50-70%)
        elif normalized_confidence > 0.5:
            if normalized_magnitude_accuracy > 0.7:
                # Moderate confidence with high accuracy - should be more confident
                confidence_reward = 1.75
                reward_components["confidence"] = f"1.75 - Moderate confidence ({model_confidence}%) with high accuracy ({magnitude_accuracy:.0f}%)"
            elif normalized_magnitude_accuracy > 0.5:
                # Moderate confidence with moderate accuracy - appropriate
                confidence_reward = 1.5
                reward_components["confidence"] = f"1.5 - Appropriate moderate confidence ({model_confidence}%) with moderate accuracy ({magnitude_accuracy:.0f}%)"
            else:
                # Moderate confidence with low accuracy - slightly overconfident
                confidence_reward = 1.0
                reward_components["confidence"] = f"1.0 - Moderate confidence ({model_confidence}%) with low accuracy ({magnitude_accuracy:.0f}%)"
        
        # For low confidence predictions (<50%)
        else:
            if normalized_magnitude_accuracy > 0.7:
                # Low confidence with high accuracy - too cautious, but still reward
                confidence_reward = 1.2
                reward_components["confidence"] = f"1.2 - Low confidence ({model_confidence}%) despite high accuracy ({magnitude_accuracy:.0f}%)"
            elif normalized_magnitude_accuracy > 0.5:
                # Low confidence with moderate accuracy - too cautious
                confidence_reward = 1.2
                reward_components["confidence"] = f"1.2 - Low confidence ({model_confidence}%) with moderate accuracy ({magnitude_accuracy:.0f}%)"
            else:
                # Low confidence with low accuracy - appropriate caution
                confidence_reward = 1.5
                reward_components["confidence"] = f"1.5 - Appropriately low confidence ({model_confidence}%) with low accuracy ({magnitude_accuracy:.0f}%)"
        
        # Direction correctness bonus
        if prediction["direction"] == actual_outcome["direction"]:
            # If direction is correct, add bonus
            confidence_reward = min(2.0, confidence_reward + 0.25)
            reward_components["confidence"] = f"{confidence_reward} - Bonus for correct direction"
    else:
        confidence_reward = 0.0
        reward_components["confidence"] = f"0.0 - No reward (wrong direction)"
    
    # Sum rewards and clip to range [0, max_reward]
    total_reward = format_reward + direction_reward + magnitude_reward + confidence_reward
    total_reward = max(0, min(max_reward, total_reward))
    
    # Create explanation
    explanation = (
        f"Format: {reward_components['format']}\n"
        f"Direction: {reward_components['direction']}\n"
        f"Magnitude: {reward_components['magnitude']}\n"
        f"Confidence: {reward_components['confidence']}\n"
        f"Total reward: {total_reward:.1f}/{max_reward:.1f}"
    )
    
    return total_reward, explanation, actual_outcome

def find_next_trading_day(dates_by_ticker, ticker, current_date):
    """Find the next trading day and its price for a given ticker and date"""
    if ticker not in dates_by_ticker:
        return None, None
        
    dates = sorted(dates_by_ticker[ticker].keys())
    for date in dates:
        if date > current_date:
            return date, dates_by_ticker[ticker][date]
    return None, None

def main():
    print("Starting quick stock prediction test...")
    
    try:
        print("Loading model (this may take a moment)...")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", 
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Model loaded successfully!")
        
        # Load dataset with all parts
        dataset_path = "/home/ai/.cache/huggingface/datasets/2084Collective___deepstock-sp500-companies-with-info-and-user-prompt"
        print(f"Loading dataset from {dataset_path}...")
        
        # Create sample data for testing since we can't access the actual dataset
        print("Creating sample data for testing...")
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
                    'company_info': {'description': 'Alphabet Inc. provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.'},
                    'price': {'close': 120.0, 'close_previous': 118.0},
                    'news': {'news_headlines': ['Google announces new AI features', 'Google ad revenue increases']},
                    'financials': {'financials': '{"Total Revenue": 282000000000, "Net Income": 76000000000, "Basic EPS": 5.8}'}
                }
            }
        ]
        
        # Create a dataset from the sample data
        from datasets import Dataset
        dataset = Dataset.from_list(sample_data)
        print(f"Created sample dataset with {len(dataset)} examples")
        
        # Build price lookup table for all tickers
        print("Building price lookup table...")
        dates_by_ticker = {}
        
        # Add some simulated future dates and prices for our sample data
        dates_by_ticker['AAPL'] = {
            '2023-01-01': 150.0,
            '2023-01-02': 153.0,  # up 2%
            '2023-01-03': 149.0,  # down 2.6%
            '2023-01-04': 151.0,  # up 1.3%
        }
        
        dates_by_ticker['MSFT'] = {
            '2023-01-01': 250.0,
            '2023-01-02': 255.0,  # up 2%
            '2023-01-03': 260.0,  # up 2%
            '2023-01-04': 258.0,  # down 0.8%
        }
        
        dates_by_ticker['GOOGL'] = {
            '2023-01-01': 120.0,
            '2023-01-02': 118.0,  # down 1.7%
            '2023-01-03': 121.0,  # up 2.5%
            '2023-01-04': 123.0,  # up 1.7%
        }
        
        # Select all samples for testing
        num_samples = 3
        test_samples = dataset
        
        # Test each sample
        total_overall_reward = 0
        for i, example in enumerate(test_samples):
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
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
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
                next_date, next_price = find_next_trading_day(dates_by_ticker, ticker, last_date)
                
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
                    
                    # Use our compute_reward function
                    day_reward, explanation, _ = compute_reward(prediction_data, actual_outcome, response)
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
        print(f"Average reward across all stocks: {(total_overall_reward/(3*num_samples)):.1f}/10.0")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 