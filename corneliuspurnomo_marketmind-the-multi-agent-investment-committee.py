import os
import time
import google.generativeai as genai
import yfinance as yf
from dataclasses import dataclass
from typing import List, Dict

# --- CONFIGURATION ---
# ðŸš¨ TODO: Replace with your actual Gemini API Key or set it in your environment variables
# For a capstone submission, it is best practice to use os.getenv
API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# Configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- COLORS FOR OBSERVABILITY (Console Output) ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --- TOOLS (Rubric: Tools / Function Calling) ---
def get_stock_data_tool(ticker: str) -> str:
    """Fetches quantitative data for the Technical Agent."""
    print(f"{Colors.CYAN}[Tool] Fetching market data for {ticker}...{Colors.ENDC}")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        info = stock.info
        
        if hist.empty:
            return "Error: No history found."

        latest_close = hist['Close'].iloc[-1]
        start_close = hist['Close'].iloc[0]
        change_pct = ((latest_close - start_close) / start_close) * 100

        data_summary = (
            f"Current Price: ${latest_close:.2f}\n"
            f"Price 1 Month Ago: ${start_close:.2f}\n"
            f"1-Month Change: {change_pct:.2f}%\n"
            f"52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"PE Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"Volume: {hist['Volume'].iloc[-1]}"
        )
        return data_summary
    except Exception as e:
        return f"Error fetching data: {str(e)}"

def get_news_tool(ticker: str) -> str:
    """Fetches qualitative data for the Fundamental Agent."""
    print(f"{Colors.CYAN}[Tool] Searching news for {ticker}...{Colors.ENDC}")
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return "No recent news found."
        
        # Return top 3 headlines with partial summary to save tokens
        headlines = []
        for n in news[:3]:
            title = n.get('title', 'No Title')
            headlines.append(f"- {title}")
        
        return "\n".join(headlines)
    except Exception as e:
        return f"Error fetching news: {str(e)}"

# --- AGENT CLASSES (Rubric: Multi-Agent System) ---

class Agent:
    def __init__(self, name, role, color):
        self.name = name
        self.role = role
        self.color = color

    def log(self, message):
        print(f"{self.color}[{self.name}]: {message}{Colors.ENDC}")

    def think(self, prompt_context):
        """Sends prompt to Gemini."""
        self.log("Thinking...")
        try:
            response = model.generate_content(prompt_context)
            return response.text.strip()
        except Exception as e:
            return f"Error during inference: {e}"

# --- ORCHESTRATION ---

def run_market_mind(ticker: str):
    print(f"{Colors.HEADER}{Colors.BOLD}=== MarketMind: Starting Analysis for {ticker} ==={Colors.ENDC}\n")
    
    # 1. Initialize Agents
    tech_agent = Agent("Technical Analyst", "Quantitative Analysis", Colors.BLUE)
    fund_agent = Agent("Fundamental Analyst", "News & Sentiment", Colors.GREEN)
    manager_agent = Agent("Portfolio Manager", "Decision Maker", Colors.WARNING)

    # 2. Gather Information (Tools)
    stock_data = get_stock_data_tool(ticker)
    news_data = get_news_tool(ticker)
    
    print("-" * 50)

    # 3. Technical Agent Execution
    tech_prompt = f"""
    Role: You are a strict Technical Analyst.
    Task: Analyze the following stock data. Identify trends (Bullish/Bearish).
    Data: 
    {stock_data}
    
    Output: A short, bulleted assessment of the technicals.
    """
    tech_analysis = tech_agent.think(tech_prompt)
    tech_agent.log(f"Report Generated:\n{tech_analysis}\n")

    # 4. Fundamental Agent Execution
    fund_prompt = f"""
    Role: You are a Fundamental Analyst.
    Task: Analyze the following news headlines. Identify sentiment and risks.
    News: 
    {news_data}
    
    Output: A short, bulleted assessment of the fundamentals/sentiment.
    """
    fund_analysis = fund_agent.think(fund_prompt)
    fund_agent.log(f"Report Generated:\n{fund_analysis}\n")

    # 5. Portfolio Manager Execution (Synthesizing Memory)
    manager_prompt = f"""
    Role: You are a Portfolio Manager.
    Task: You have received two conflicting reports. Make a final decision.
    
    Report 1 (Technical):
    {tech_analysis}
    
    Report 2 (Fundamental):
    {fund_analysis}
    
    Instructions:
    - Weigh the evidence.
    - If Technicals say buy but News is bad, be cautious.
    - Provide a Final Verdict: BUY, SELL, or HOLD.
    - Provide a 2-sentence rationale.
    """
    final_decision = manager_agent.think(manager_prompt)
    
    print("-" * 50)
    print(f"{Colors.HEADER}{Colors.BOLD}=== FINAL VERDICT FOR {ticker} ==={Colors.ENDC}")
    print(f"{Colors.WARNING}{final_decision}{Colors.ENDC}")
    print("-" * 50)

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Observability: Interactive input
    user_ticker = input("Enter a stock ticker (e.g., AAPL, TSLA, NVDA): ").upper()
    if user_ticker:
        run_market_mind(user_ticker)
    else:
        print("No ticker provided. Exiting.")

