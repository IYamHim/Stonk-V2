#!/bin/bash

# Super Saiyan GRPO Trainer - Stage II Setup Script
# This script sets up the environment for Stage II training

echo "=== Setting up Super Saiyan GRPO Trainer Stage II ==="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found, setting up environment..."
    
    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "stonk_super_saiyan"; then
        echo "Creating new conda environment: stonk_super_saiyan"
        conda create -y -n stonk_super_saiyan python=3.10
    else
        echo "Environment stonk_super_saiyan already exists"
    fi
    
    # Activate environment and install dependencies
    echo "Installing dependencies in stonk_super_saiyan environment"
    conda run -n stonk_super_saiyan pip install -r requirements.txt
    
    # Check for CUDA
    if conda run -n stonk_super_saiyan python -c "import torch; print(torch.cuda.is_available())"; then
        echo "CUDA is available in the environment"
    else
        echo "WARNING: CUDA not detected, training will be slow without GPU acceleration"
    fi
    
    # Instructions for activation
    echo -e "\nSetup complete! To use this environment, run:"
    echo "conda activate stonk_super_saiyan"
    
else
    echo "Conda not found, using system Python"
    
    # Check if in virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Creating and activating virtual environment..."
        python -m venv venv
        source venv/bin/activate
    else
        echo "Using existing virtual environment: $VIRTUAL_ENV"
    fi
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Check for CUDA
    if python -c "import torch; print(torch.cuda.is_available())"; then
        echo "CUDA is available in the environment"
    else
        echo "WARNING: CUDA not detected, training will be slow without GPU acceleration"
    fi
    
    # Instructions for next time
    echo -e "\nSetup complete! In the future, activate the environment with:"
    echo "source venv/bin/activate"
fi

# Make scripts executable
chmod +x train_stage2.sh

# Set up logging directory
mkdir -p super_saiyan_grpo_stage2/logs

# Check for Stage I models
echo -e "\nChecking for Stage I models..."
if [ -d "../checkpoints/best_model" ]; then
    echo "Found Stage I model at ../checkpoints/best_model"
    echo "You can use this model for Stage II training"
elif [ -d "../best_model" ]; then
    echo "Found Stage I model at ../best_model"
    echo "You can use this model for Stage II training"
else
    echo "No Stage I model found in expected locations."
    echo "Please provide a path to your Stage I model when running the training script."
fi

echo -e "\n=== Setup Complete ==="
echo "Run the training with: ./train_stage2.sh /path/to/stage1/model" 