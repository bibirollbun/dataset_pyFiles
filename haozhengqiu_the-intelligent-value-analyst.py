import sys
import os
import time
import json
import math
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown

print("âœ“ Libraries Loaded")

# ==========================================
# 1. Configuration & API Setup
# ==========================================
CONFIG = {
    "team": "TheIntelligentValueAnalyst",
    "model": "models/gemini-2.5-flash", 
    "max_tokens": 2000,
    "temperature": 0.3,
    "version": "2.0.0"
}

try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ“ API Key Configured")
except Exception as e:
    print(f"âš  API Key Error: {str(e)}")
    print("ğŸ“Œ To fix: Go to Add-ons â†’ Secrets â†’ Add 'GOOGLE_API_KEY'")
    GOOGLE_API_KEY = None


def get_stock_price(ticker: str):
    """
    Fetches the current stock price for a given ticker symbol.
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'NVDA', 'INTC').
    """
    print(f"\n[Tool Call] ğŸ”� Fetching Price for {ticker}...") # Observability Log
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if not history.empty:
            price = round(history['Close'].iloc[0], 2)
            print(f"[Tool Result] Price: ${price}")
            return price
        return "Error: Could not fetch price data."
    except Exception as e:
        return f"Error: {str(e)}"

def get_financial_metrics(ticker: str):
    """
    Retrieves key financial metrics: P/E, P/B, EPS, ROE, and Debt-to-Equity.
    Args:
        ticker: The stock ticker symbol.
    """
    print(f"\n[Tool Call] ğŸ“Š Fetching Financials for {ticker}...") # Observability Log
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        metrics = {
            "ticker": ticker,
            "forwardPE": info.get("forwardPE", "N/A"),
            "priceToBook": info.get("priceToBook", "N/A"),
            "trailingEps": info.get("trailingEps", 0),
            "bookValue": info.get("bookValue", 0),
            "debtToEquity": info.get("debtToEquity", "N/A"),
            "returnOnEquity": info.get("returnOnEquity", "N/A")
        }
        print(f"[Tool Result] Metrics retrieved successfully.")
        return metrics
    except Exception as e:
        return f"Error fetching metrics: {str(e)}"

def calculate_graham_number(eps: float, book_value_per_share: float):
    """
    Calculates the Benjamin Graham Number. Formula: Sqrt(22.5 * EPS * Book Value).
    Args:
        eps: Earnings Per Share.
        book_value_per_share: Book Value Per Share.
    """
    print(f"\n[Tool Call] ğŸ§® Calculating Graham Number...") # Observability Log
    
    if eps is None or book_value_per_share is None:
        return "Error: Missing data."
    if eps < 0:
        return "Error: Negative EPS. Company is unprofitable."
        
    try:
        graham_number = math.sqrt(22.5 * eps * book_value_per_share)
        res = round(graham_number, 2)
        print(f"[Tool Result] Graham Number: ${res}")
        return res
    except ValueError:
        return "Calculation Error."

# Create the Tool Dictionary for Gemini
tools_list = [get_stock_price, get_financial_metrics, calculate_graham_number]
print("âœ“ Tools Registered")


system_instruction = """
You are a strict **Value Investing Analyst** following the principles of Benjamin Graham.
Your goal is to analyze a stock ticker provided by the user.

**Workflow:**
1. Call `get_stock_price` to get the current price.
2. Call `get_financial_metrics` to get fundamentals.
3. Call `calculate_graham_number` using the EPS and Book Value from step 2.
4. **Reasoning:**
   - Compare Current Price vs. Graham Number.
   - Check Financial Health (Is ROE > 15%? Is Debt/Equity > 100%?).
5. **Output:** A professional Investment Memo in Markdown. 
   - Start with a clear "Recommendation: BUY/HOLD/SELL".
   - Use tables for data.
   - Be concise and data-driven.
"""

# Initialize the Gemini Model
model = genai.GenerativeModel(
    model_name=CONFIG["model"],
    tools=tools_list,
    system_instruction=system_instruction,
    generation_config=genai.GenerationConfig(
        temperature=CONFIG["temperature"],
        max_output_tokens=CONFIG["max_tokens"]
    )
)

print(f"âœ“ Agent {CONFIG['model']} Online")


def analyze_stock(ticker_symbol):
    print(f"{'='*40}")
    print(f"ğŸ¤– AGENT STARTING ANALYSIS: {ticker_symbol}")
    print(f"{'='*40}")
    
    # Start a chat session with automatic tool use enabled
    chat = model.start_chat(enable_automatic_function_calling=True)
    
    try:
        # Send the user prompt
        response = chat.send_message(f"Analyze {ticker_symbol} for a potential long-term value investment.")
        
        # Display the final result nicely
        print(f"\n{'='*40}")
        print(f"ğŸ“„ FINAL INVESTMENT MEMO")
        print(f"{'='*40}\n")
        display(Markdown(response.text))
        
    except Exception as e:
        print(f"â�Œ Analysis Failed: {str(e)}")


# ==========================================
# Run the Demo
# ==========================================

# Test Case 1: Intel (A common Value Trap example)
analyze_stock("INTC")

# Test Case 2: Optional (e.g., Ford)
# analyze_stock("F")

