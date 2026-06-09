# Cell 1: Setup & API Configuration
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
import os

# Get Google API Key from Kaggle Secrets
os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")

# Install Google Generative AI library
!pip install -q google-generativeai

# Import and configure Gemini API
import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("âœ… Setup complete! API configured successfully.")



# Cell 3: Data Fetcher Agent
import pandas as pd

def data_fetch_agent(file_path):
    """
    Loads CSV business data and returns preview.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"âœ… Data loaded successfully! Total rows: {len(df)}, Columns: {len(df.columns)}")
        return df.head().to_string()
    except Exception as e:
        return f"â�Œ Error loading data: {str(e)}"

# Updated path for your dataset
file_path = "/kaggle/input/just-test/Suppier_Disruption_LogTable_Coded.csv"
data_preview = data_fetch_agent(file_path)
print("\nğŸ“Š Data Preview:\n")
print(data_preview)




# Cell 4: Data Analyzer Agent
import google.generativeai as genai

def data_analyze_agent(data_preview):
    """
    Analyzes business data and extracts key insights using Gemini API.
    """
    try:
        # Use the correct model name from available models
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
        You are a business data analyst. Analyze the following supplier disruption data and provide:
        - Key insights about supplier risks
        - Business trends and patterns
        - Actionable recommendations for risk management
        
        Data Preview:
        {data_preview}
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"â�Œ Error in analysis: {str(e)}"

# Test the Data Analyzer Agent
print("ğŸ”� Analyzing data...\n")
analysis = data_analyze_agent(data_preview)
print("ğŸ“ˆ Analysis Results:\n")
print(analysis)




# Cell 5: Summary Agent
import google.generativeai as genai

def summary_agent(analysis_text):
    """
    Summarizes the analysis into concise, actionable bullet points.
    """
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
        You are a business executive assistant. Summarize the following analysis into:
        - 5-7 concise bullet points
        - Focus on actionable insights
        - Use clear, business-friendly language
        
        Analysis:
        {analysis_text}
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"â�Œ Error in summary: {str(e)}"

# Generate Summary
print("ğŸ“‹ Generating Executive Summary...\n")
summary = summary_agent(analysis)
print("âœ… Executive Summary:\n")
print(summary)


