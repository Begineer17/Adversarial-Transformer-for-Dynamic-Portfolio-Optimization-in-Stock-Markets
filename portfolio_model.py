import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import vnquant.data as dt
import os
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Initialize a sacred experiment
ex = Experiment("Stock_Investment_Transformer")
ex.observers.append(FileStorageObserver("experiments"))  # Save experiment logs to 'experiments/' folder

# Set random seeds
@ex.config
def config():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

# Data loading configuration
@ex.config
def data_config():
    stocks_file = 'stocks.txt'
    start_date = '2021-12-10'
    end_date = '2023-01-01'
    future_start_date = '2022-01-05'
    future_end_date = '2023-01-25'

    with open(stocks_file, 'r') as f:
        stocks = f.read().splitlines()
    num_stocks = len(stocks)

    stock_data = dt.DataLoader(symbols=stocks, start=start_date, end=end_date, data_source='CAFE', minimal=True, table_style='stack').download()
    stock_data = stock_data.sort_index()

    future_data = dt.DataLoader(symbols=stocks, start=future_start_date, end=future_end_date, data_source='CAFE', minimal=True, table_style='stack').download()
    future_data = future_data.sort_index()


# Model and training configuration
@ex.config
def model_config():
    input_dim = 4  # Number of features: open, high, low, close
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    window_size = 10
    batch_size = 4
    learning_rate = 0.001
    epochs = 10



# Define Models
class CrossStockTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_stocks, num_heads, num_layers, dropout):
        super(CrossStockTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_stocks, embed_dim))
        self.time_step_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, x_real):
        x = x_real.clone()
        x = x.float()

        batch_size, T, num_stocks, num_features = x.shape
        x = x.view(batch_size * T, num_stocks, num_features)
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :num_stocks, :]
        x = x.permute(1, 0, 2)
        x, _ = self.time_step_attention(x, x, x)
        x = x.permute(1, 0, 2)
        x = x.view(batch_size, T, num_stocks, -1)
        x = x.permute(0, 2, 1, 3).flatten(0, 1)
        x = self.transformer_encoder(x).mean(dim=1)
        x = self.fc_out(x).view(batch_size, num_stocks)
        return x

class InvestorTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_stocks, num_heads, num_layers, dropout):
        super(InvestorTransformer, self).__init__()
        self.cross_stock_transformer = CrossStockTransformer(input_dim, embed_dim, num_stocks, num_heads, num_layers, dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.cross_stock_transformer(x))

class MarketTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_stocks, num_heads, num_layers, dropout):
        super(MarketTransformer, self).__init__()
        self.cross_stock_transformer = CrossStockTransformer(input_dim, embed_dim, num_stocks, num_heads, num_layers, dropout)

    def forward(self, x):
        return self.cross_stock_transformer(x)

# Loss Functions
def sharpe_ratio_loss(returns):
    """
    Maximizes the Sharpe ratio by minimizing its negative value.
    :param returns: Tensor of returns (batch_size x num_stocks).
    :return: Scalar loss value (negative Sharpe ratio).
    """
    mean_return = torch.mean(returns)
    std_return = torch.std(returns) + 1e-6  # Add a small epsilon to avoid division by zero
    sharpe_ratio = mean_return / std_return
    return -sharpe_ratio  # Negative Sharpe ratio to maximize it

# Volatility minimization loss for Market
def market_stability_loss(returns):
    """
    Minimizes volatility by penalizing return standard deviation.
    :param returns: Tensor of returns (batch_size x num_stocks).
    :return: Scalar loss value (variance of returns).
    """
    # variance = torch.var(returns, dim=1).mean()  # Variance across stocks, averaged over the batch
    variance = torch.std(returns, dim=1).mean()  # Standard deviation across stocks, averaged over the batch
    return variance

# Function to initialize weights using He initialization
def initialize_weights_he(model):
    import torch.nn.init as init
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

def reshape_data(data):
    # Prepare data
    unique_dates = data.index.unique()
    reshaped_data = []
    for date in unique_dates:
        daily_data = data.loc[date]
        daily_features = daily_data[['open', 'high', 'low', 'close']].values
        reshaped_data.append(daily_features)
    x = np.array(reshaped_data) # Shape: [num_days, num_stocks, num_features]
    return x

def create_rolling_window_data(x, window_size):
    x_windows = []
    y_targets = []
    for i in range(x.shape[0] - window_size):
        x_windows.append(torch.tensor(x[i:i + window_size]))
        y_targets.append(torch.tensor((x[i + window_size, :, 3] - x[i, :, 3]) / x[i, :, 3]))
    return torch.stack(x_windows), torch.stack(y_targets)

@ex.capture
def train(
    x_data,
    y_data,
    investor,
    market,
    optimizer_investor,
    optimizer_market,
    batch_size,
    epochs
):  
    print("Starting training...")
    # Training Loop
    for epoch in range(epochs):
        for i in range(0, x_data.size(0), batch_size):
            x_batch = x_data[i:i + batch_size] # Shape: [batch_size, window_size, num_stocks, num_features]
            y_batch = y_data[i:i + batch_size] # Shape: [batch_size, num_stocks]

            market_impact = market(x_batch)
            allocations = investor(x_batch)
            # allocations shape: [batch_size, num_stocks]
            adjusted_returns = (y_batch + market_impact) * allocations.detach()

            # Investor training
            optimizer_investor.zero_grad()
            investor_loss = sharpe_ratio_loss(adjusted_returns)
            # Retain graph to prevent memory leak
            # Backpropagate investor loss to update investor weights
            investor_loss.backward(retain_graph=True)
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(investor.parameters(), max_norm=1.0)
            optimizer_investor.step()

            # Market training
            optimizer_market.zero_grad()
            market_loss = market_stability_loss(adjusted_returns)
            market_loss.backward()
            torch.nn.utils.clip_grad_norm_(market.parameters(), max_norm=1.0)
            optimizer_market.step()

        # Log metrics
        ex.log_scalar("investor_loss", investor_loss.item(), epoch)
        ex.log_scalar("market_loss", market_loss.item(), epoch)
        print(f"Epoch {epoch}: Investor Loss = {investor_loss.item()}, Market Loss = {market_loss.item()}")
    
    print("Training complete!")
    # For validation
    return investor, market

@ex.capture
def test(
    x_data,
    y_data,
    investor,
    market
):
    print("Starting testing...")
    investor.eval()
    market.eval()
    with torch.no_grad():
        market_impact = market(x_data)
        allocations = investor(x_data)
        adjusted_returns = (y_data + market_impact) * allocations.detach()
        investor_loss = sharpe_ratio_loss(adjusted_returns)
        market_loss = market_stability_loss(adjusted_returns)
    print(f"Test results: Investor Loss = {investor_loss.item()}, Market Loss = {market_loss.item()}")
    print("Testing complete!")

# Experiment main logic
@ex.main
def main(stock_data, future_data, input_dim, embed_dim, num_stocks, num_heads, num_layers, dropout, window_size, learning_rate):
    # Output user's hardware configuration
    print("--Device configuration--")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        # Activate GPU
        torch.cuda.set_device(0)
    print("------------------------")

    # Reshape data
    train_data = reshape_data(stock_data) # Shape: [num_days, num_stocks, num_features]

    # Create rolling windows
    # x_data: [num_samples, window_size, num_stocks, num_features]
    # y_data: [num_samples, num_stocks]
    x_data, y_data = create_rolling_window_data(train_data, window_size=window_size)

    investor = InvestorTransformer(input_dim, embed_dim, num_stocks, num_heads, num_layers, dropout)
    market = MarketTransformer(input_dim, embed_dim, num_stocks, num_heads, num_layers, dropout)
    initialize_weights_he(investor)
    initialize_weights_he(market)

    optimizer_investor = optim.Adam(investor.parameters(), lr=learning_rate)
    optimizer_market = optim.Adam(market.parameters(), lr=learning_rate)

    # Train the model
    investor, market = train(x_data, y_data, investor, market, optimizer_investor, optimizer_market)

    # Test the model
    future_data = reshape_data(future_data)
    x_data_future, y_data_future = create_rolling_window_data(future_data, window_size=window_size)
    test(x_data_future, y_data_future, investor, market)

    # Probability allocations of what stocks to invest in
    investor.eval()
    market.eval()
    with torch.no_grad():
        x_batch = x_data_future[-1].clone().unsqueeze(0) # Shape: [1, window_size, num_stocks, num_features]
        allocations = investor(x_batch)
        print(f"Allocations: {allocations}")
        ex.log_scalar("allocations", allocations.numpy().tolist())

# Run the experiment
ex.run()
