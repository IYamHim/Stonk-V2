#!/bin/bash

# Stonk-Trainer v2 Setup Script
# This script helps set up your environment and configure paths for training

echo "=== Stonk-Trainer v2 Setup ==="

# Function to detect conda and choose environment
setup_conda_env() {
    echo "Detecting conda installation..."
    
    # Check if conda is available
    if command -v conda &> /dev/null; then
        echo "Conda found!"
        
        # List existing environments
        echo "Existing conda environments:"
        conda env list
        
        # Prompt for environment name
        read -p "Enter conda environment name to use/create (default: stonk_env): " env_name
        env_name=${env_name:-stonk_env}
        
        # Check if the environment exists
        if conda env list | grep -q "$env_name"; then
            echo "Environment $env_name exists. Using it for setup."
        else
            echo "Creating new conda environment: $env_name"
            conda create -y -n $env_name python=3.10
            if [ $? -ne 0 ]; then
                echo "Failed to create conda environment."
                exit 1
            fi
        fi
        
        # Install dependencies
        echo "Installing required packages in $env_name..."
        conda run -n $env_name pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Warning: Some packages may not have installed correctly."
        fi
        
        # Get Python path from conda
        CONDA_PYTHON_PATH=$(conda run -n $env_name which python)
        echo "Python path: $CONDA_PYTHON_PATH"
        
        # Update scripts with the correct Python path
        update_python_path "$CONDA_PYTHON_PATH"
        
        # Print activation instruction
        echo -e "\nTo activate this environment, use:"
        echo "conda activate $env_name"
    else
        echo "Conda not found. Using system Python instead."
        setup_venv
    fi
}

# Function to set up virtual environment if conda is not available
setup_venv() {
    echo "Setting up a virtual environment..."
    
    # Check if Python 3 is available
    if command -v python3 &> /dev/null; then
        # Create virtual environment if it doesn't exist
        if [ ! -d "venv" ]; then
            python3 -m venv venv
            if [ $? -ne 0 ]; then
                echo "Failed to create virtual environment. Make sure python3-venv is installed."
                exit 1
            fi
        else
            echo "Using existing virtual environment."
        fi
        
        # Activate the environment
        source venv/bin/activate
        
        # Install dependencies
        echo "Installing required packages..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Warning: Some packages may not have installed correctly."
        fi
        
        # Get Python path
        VENV_PYTHON_PATH=$(which python)
        echo "Python path: $VENV_PYTHON_PATH"
        
        # Update scripts with the correct Python path
        update_python_path "$VENV_PYTHON_PATH"
        
        # Print activation instruction
        echo -e "\nTo activate this environment in the future, use:"
        echo "source venv/bin/activate"
    else
        echo "Python 3 not found. Please install Python 3 before proceeding."
        exit 1
    fi
}

# Function to update Python path in scripts
update_python_path() {
    PYTHON_PATH=$1
    
    # Update Python path in all shell scripts
    echo "Updating Python path in shell scripts..."
    
    # For each shell script
    for script in train_direct.sh test_direct.sh train_large_dataset.sh test_model.sh; do
        if [ -f "$script" ]; then
            # Replace the Python path line
            sed -i "s|^PYTHON_PATH=.*|PYTHON_PATH=\"$PYTHON_PATH\"|" $script
            echo "Updated $script"
        fi
    done
    
    # Make scripts executable
    chmod +x *.sh
}

# Check CUDA
check_cuda() {
    echo -e "\nChecking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA devices found:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    else
        echo "WARNING: nvidia-smi not found. CUDA may not be available."
        echo "Training will be extremely slow without GPU acceleration."
    fi
}

# Check for HuggingFace datasets
check_datasets() {
    echo -e "\nChecking for dataset accessibility..."
    
    # Try to access the dataset info
    echo "Testing access to the training dataset..."
    python -c "from datasets import load_dataset; print(load_dataset('2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2', split='train').info.description[:100] + '...')" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "Dataset access successful!"
    else
        echo "WARNING: Could not access the dataset. Make sure you have an internet connection and HuggingFace access."
        echo "You might need to log in with 'huggingface-cli login'."
    fi
}

# Create data directories
create_directories() {
    echo -e "\nCreating necessary directories..."
    
    # Create output directories if they don't exist
    mkdir -p stonk_trainer_grpo/checkpoints
    mkdir -p stonk_trainer_grpo/best_model
    mkdir -p stonk_trainer_grpo/evaluation_results
    
    echo "Directory structure created."
}

# Main setup process
echo "This script will set up your environment for Stonk-Trainer v2."
echo "It will install required packages and configure training scripts."
echo -e "\nChoose your setup method:"
echo "1. Use conda (recommended)"
echo "2. Use system Python with venv"
read -p "Enter your choice (1 or 2): " setup_choice

case $setup_choice in
    1)
        setup_conda_env
        ;;
    2)
        setup_venv
        ;;
    *)
        echo "Invalid choice. Using conda setup as default."
        setup_conda_env
        ;;
esac

# Check CUDA availability
check_cuda

# Check dataset access
check_datasets

# Create directories
create_directories

# Final instructions
echo -e "\n=== Setup Complete ==="
echo "You can now run the training with:"
echo "./train_direct.sh"
echo -e "\nTo monitor training progress:"
echo "tail -f training_large_dataset.log"
echo -e "\nAfter training, test the model with:"
echo "./test_direct.sh"
echo -e "\nFor advanced (Stage II) training, see:"
echo "cd Super_Sayin_GRPO_Trainer_StageII"
echo "./setup.sh"
echo -e "\nHappy training!" 