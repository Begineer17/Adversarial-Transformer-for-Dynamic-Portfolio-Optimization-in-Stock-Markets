# Adversarial-Transformer-for-Dynamic-Portfolio-Optimization-in-Stock-Markets

# Stock Investment Transformer

This project implements a stock investment strategy using transformer models. The goal is to maximize the Sharpe ratio while minimizing market volatility. The project uses PyTorch for model implementation and Sacred for experiment management.

## Table of Contents

- Installation
- Usage
- [Project Structure](#project-structure)
- Models
- [Loss Functions](#loss-functions)
- [Training and Testing](#training-and-testing)
- Configuration
- License

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/stock-investment-transformer.git
    cd stock-investment-transformer
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the `stocks.txt`: file with the list of stock symbols you want to analyze.

2. Run the experiment:
    ```sh
    python portfolio_model.py
    ```

## Project Structure

- `portfolio_model.py`: Main script containing model definitions, loss functions, and training/testing logic.
- `experiments/`: Directory where experiment logs are saved.
- `stocks.txt`: File containing the list of stock symbols.

## Models

### CrossStockTransformer

A transformer model that processes stock data across multiple stocks and time steps.

### InvestorTransformer

A model that uses CrossStockTransformer to predict stock allocations.

### MarketTransformer

A model that uses CrossStockTransformer to predict market impact.

## Loss Functions

- sharpe_ratio_loss: Maximizes the Sharpe ratio by minimizing its negative value.
- market_stability_loss: Minimizes volatility by penalizing return standard deviation.

## Training and Testing

### Training

The training process involves optimizing both the investor and market models to maximize the Sharpe ratio and minimize market volatility.

### Testing

The testing process evaluates the models on future data to ensure they generalize well.

## Configuration

The configuration for the experiment is managed using Sacred. Key configurations include:

- seed: Random seed for reproducibility.
- stocks_file: Path to the file containing stock symbols.
- start_date, end_date: Date range for training data.
- future_start_date, future_end_date: Date range for testing data.
- input_dim, embed_dim, num_heads, num_layers, dropout, window_size, batch_size, learning_rate, epochs: Model and training hyperparameters.

---

Feel free to customize this README file further to suit your project's specific needs.
