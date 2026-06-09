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



import os
import google.generativeai as genai

# Function to initialize Gemini API
def initialize_gemini():
    %env GEMINI_API_KEY=AIzaSyCvGu9KD4AkwVYtkl0KG3FLGusb-Mr5d1M
    api_key = os.getenv('GEMINI_API_KEY')  # Get API key from environment variable
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable with your Google AI Studio API key.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')  # Use the latest Gemini model
    return model

# Function to analyze business data and generate insights
def analyze_business_data(model, user_input):
    # Craft a prompt to guide Gemini towards business analysis
    prompt = f"""
    You are a business improvement AI agent. Analyze the following user input related to a business scenario.
    Provide insights on data trends, potential issues, and actionable recommendations for improvement.
    Focus on areas like revenue growth, cost reduction, customer satisfaction, and operational efficiency.
    User Input: {user_input}
    Response Format: Start with a summary, then list key insights and recommendations in bullet points.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}. Please check your API key or input."

# Main function to run the AI agent
def main():
    print("Welcome to BizInsight AI Agent!")
    print("This tool analyzes business data and suggests improvements using Google's Gemini AI.")
    print("Enter your business-related query or data (e.g., 'Sales: $100K, Expenses: $80K. How to improve profit?').")
    print("Type 'exit' to quit.\n")
    
    model = initialize_gemini()
    
    while True:
        user_input = input("Your Input: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if not user_input.strip():
            print("Please provide some input.\n")
            continue
        
        print("Analyzing... (This may take a few seconds)")
        result = analyze_business_data(model, user_input)
        print("AI Insights:\n" + result + "\n")

if __name__ == "__main__":
    main()


