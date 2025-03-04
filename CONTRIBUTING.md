# Contributing to Stonk-Trainer v2

Thank you for your interest in contributing to the Stonk-Trainer v2 project! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Workflow](#development-workflow)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please:

- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community and project
- Show empathy towards other community members

## How Can I Contribute?

There are many ways you can contribute to the Stonk-Trainer v2 project:

### Reporting Bugs

If you find a bug, please create an issue with the following information:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- System information (OS, GPU, CUDA version, etc.)
- Any relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- A clear description of the enhancement
- The motivation behind it
- Any implementation ideas you have

### Code Contributions

Code contributions can be made through pull requests. Some areas where contributions are particularly welcome:

1. **Reward Function Improvements**: Enhancing the reward function to better evaluate predictions and reasoning.
2. **Performance Optimizations**: Improving training efficiency and memory usage.
3. **Data Processing**: Creating better data filtering mechanisms or preprocessing techniques.
4. **Model Architecture Improvements**: Suggestions for better adapters or training approaches.
5. **Evaluation Metrics**: Adding new evaluation metrics or improving existing ones.
6. **Stage II Training Enhancements**: Improving the transition between stages or the natural distribution training.

## Development Workflow

1. **Fork the repository**: Create your own fork of the project.
2. **Create a branch**: Make a branch for your feature or bugfix.
3. **Develop your changes**: Implement your feature or fix.
4. **Test your changes**: Ensure your changes work as expected.
5. **Create a pull request**: Submit a PR with a clear description of your changes.

## Style Guidelines

### Python Code

- Follow PEP 8 style guidelines.
- Use meaningful variable and function names.
- Include docstrings for functions and classes.
- Keep functions focused on a single responsibility.
- Add comments for complex code sections.

### Commit Messages

- Use the imperative mood ("Add feature" not "Added feature").
- Keep the first line under 50 characters.
- Reference issues and pull requests where appropriate.

## Testing

Before submitting a pull request, please:

1. Run the model on a small test dataset to ensure it trains without errors.
2. Verify that your changes don't break existing functionality.
3. For significant changes, include test results showing the impact of your changes.

## Documentation

Improvements to documentation are always welcome! This includes:

- Updating the README with clearer instructions or new information.
- Adding comments to explain complex code sections.
- Creating or improving diagrams that explain the system architecture.
- Writing guides for specific aspects of the project.

## Pull Request Process

1. Ensure your code follows the style guidelines.
2. Update the documentation to reflect any changes.
3. Include a clear description of what your PR accomplishes.
4. Link any relevant issues that your PR addresses.
5. Be responsive to feedback and be willing to make requested changes.

## Model Training and Contribution Guidelines

When contributing changes that affect model training:

1. **Hyperparameters**: If changing hyperparameters, provide evidence of improved performance.
2. **Memory Usage**: Consider the impact on memory usage, especially for users with limited GPU VRAM.
3. **Training Speed**: Document any significant impacts on training speed.
4. **Compatibility**: Ensure compatibility with both Stage I and Stage II training approaches.

## Dataset and Financial Considerations

When contributing code related to financial predictions:

1. **No Financial Advice**: Make it clear that the model outputs should not be considered financial advice.
2. **Data Ethics**: Be mindful of data sources and usage rights.
3. **Evaluation**: Be transparent about evaluation metrics and performance.

Thank you for contributing to the Stonk-Trainer v2 project! 