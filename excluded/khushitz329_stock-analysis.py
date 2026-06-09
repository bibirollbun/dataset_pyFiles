# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install yfinance PyPortfolioOpt openpyxl



# =============================
# Step 0: Install libraries (only run once)
# =============================
!pip install yfinance PyPortfolioOpt openpyxl --quiet

# =============================
# Step 1: Import libraries
# =============================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns

# =============================
# Step 2: Download stock data
# =============================
# Beginner-friendly: 3 Indian stocks from NSE
stocks = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS']

# Download last 5 years daily prices, auto-adjusted
data = yf.download(stocks, start='2018-01-01', end='2023-12-31', auto_adjust=True)['Close']

# Drop rows with missing values
data = data.dropna()

# Show stock prices
print("Stock Prices:")
display(data.head())

# Save to Excel (optional)
data.to_excel('stock_prices.xlsx')

# =============================
# Step 3: Calculate daily returns
# =============================
returns = data.pct_change().dropna()
print("\nDaily Returns:")
display(returns.head())
returns.to_excel('stock_returns.xlsx')

# =============================
# Step 4: Risk Metrics
# =============================
mean_returns = returns.mean()
volatility = returns.std()

risk_metrics = pd.DataFrame({
    'Mean Return (daily)': mean_returns,
    'Volatility (daily)': volatility
})
print("\nRisk Metrics:")
display(risk_metrics)
risk_metrics.to_excel('risk_metrics.xlsx')

# =============================
# Step 5: Portfolio Optimization
# =============================
# Calculate expected annual returns and covariance
mu = expected_returns.mean_historical_return(data)  # annual expected returns
S = risk_models.sample_cov(data)  # covariance matrix

# Maximize Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print("\nOptimal Portfolio Weights:")
display(cleaned_weights)

# Portfolio performance
print("\nPortfolio Performance:")
ef.portfolio_performance(verbose=True)

# =============================
# Step 6: Visualizations
# =============================

# --- 6a: Line chart of stock prices ---
plt.figure(figsize=(10,6))
for stock in stocks:
    plt.plot(data.index, data[stock], label=stock)
plt.title("Stock Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.show()

# --- 6b: Bar chart of daily volatility ---
plt.figure(figsize=(8,5))
plt.bar(risk_metrics.index, risk_metrics['Volatility (daily)'])
plt.title("Daily Volatility of Stocks")
plt.ylabel("Volatility")
plt.show()

# --- 6c: Pie chart of portfolio allocation ---
plt.figure(figsize=(6,6))
plt.pie(list(cleaned_weights.values()), labels=list(cleaned_weights.keys()), autopct='%1.1f%%')
plt.title("Optimal Portfolio Allocation")
plt.show()

# =============================
# ✅ Notes:
# 1. All NaNs are removed for accurate calculations.
# 2. Adjusted Close prices are used for realistic returns.
# 3. Portfolio is optimized to maximize Sharpe Ratio (risk-adjusted returns).
# =============================











