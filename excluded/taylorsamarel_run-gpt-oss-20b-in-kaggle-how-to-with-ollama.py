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


#!/usr/bin/env python
# Complete GPT-OSS-20B Setup and Run in Kaggle with Ollama

# Step 1: Install required packages
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])

# Step 2: Import necessary libraries
import os
import time
from openai import OpenAI

# Step 3: Install Ollama using bash commands
print("Installing Ollama...")
os.system("curl -fsSL https://ollama.com/install.sh | sh")

# Step 4: Start Ollama server in the background
print("Starting Ollama server...")
os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")

# Step 5: Wait for server to start
print("Waiting for server to initialize...")
time.sleep(5)

# Step 6: Check if ollama is running
print("Checking if Ollama is running...")
os.system("ps aux | grep -E 'ollama' | grep -v grep || true")

# Step 7: Download GPT-OSS:20B model (this will take significant time - ~13GB)
print("\n" + "="*50)
print("Downloading GPT-OSS:20B model...")
print("This will take several minutes (downloading ~13GB)")
print("="*50 + "\n")
os.system("ollama pull gpt-oss:20b")

# Step 8: Verify the model is downloaded
print("\nVerifying model installation...")
os.system("ollama list")

# Step 9: Initialize OpenAI client for Ollama
print("\nInitializing OpenAI client for Ollama...")
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Step 10: Test the model with a simple query
print("\n" + "="*50)
print("Testing GPT-OSS:20B model...")
print("="*50 + "\n")

try:
    response = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": "You are a professor of 18th century English literature. Mention dogs where possible."},
            {"role": "user", "content": "Write a 4-line micro-poem about running a big model on a limited notebook."}
        ]
    )
    print("Model Response:")
    print(response.choices[0].message.content)
    print("\n\nFull chat completion JSON:\n")
    print(response)
except Exception as e:
    print(f"Error during first test: {e}")
    print("The model might still be loading. Retrying in 10 seconds...")
    time.sleep(10)
    try:
        response = client.chat.completions.create(
            model="gpt-oss:20b",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Say 'Hello, I am working!' if you are functioning."}
            ]
        )
        print("Model Response:")
        print(response.choices[0].message.content)
    except Exception as e2:
        print(f"Second attempt failed: {e2}")
        print("Please check the troubleshooting section below.")

# Step 11: Troubleshooting function
def check_and_restart_ollama():
    """Check if ollama is running and restart if it has crashed"""
    print("\n" + "="*50)
    print("Running Ollama diagnostics...")
    print("="*50)
    
    # Check for defunct process
    defunct_check = os.system("ps aux | grep -E 'ollama.*<defunct>' > /dev/null 2>&1")
    
    if defunct_check == 0:
        print("Ollama has crashed (defunct process found). Restarting...")
        # Kill any existing ollama processes
        os.system("pkill -9 ollama || true")
        time.sleep(2)
        # Restart ollama server
        os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")
        # Wait for server to start
        print("Waiting for server to restart...")
        time.sleep(5)
        # Verify it's working
        print("Checking if server is working...")
        os.system("curl -s http://localhost:11434/v1/models")
        print("\nVerifying models are available:")
        os.system("ollama list")
    else:
        # Check if ollama is running at all
        running_check = os.system("ps aux | grep -E 'ollama serve' | grep -v grep > /dev/null 2>&1")
        if running_check != 0:
            print("Ollama is not running. Starting...")
            os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")
            time.sleep(5)
            print("Checking if server is working...")
            os.system("curl -s http://localhost:11434/v1/models")
            print("\nVerifying models are available:")
            os.system("ollama list")
        else:
            print("Ollama is running normally")
            os.system("curl -s http://localhost:11434/v1/models")

# Step 12: Helper function for querying the model
def query_model(prompt, system_message="You are a helpful AI assistant.", max_retries=2):
    """Query the model with automatic retry and error handling"""
    for attempt in range(max_retries):
        try:
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            response = client.chat.completions.create(
                model="gpt-oss:20b",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Checking ollama status and retrying...")
                check_and_restart_ollama()
                time.sleep(5)
            else:
                print("Max retries reached. Please check the ollama logs.")
                return None

# Step 13: Interactive usage examples
print("\n" + "="*50)
print("GPT-OSS:20B is ready for use!")
print("="*50)
print("\nExample usage:")
print("result = query_model('What is the capital of France?')")
print("print(result)")

# Run a test query
print("\nRunning test query...")
result = query_model("What is the capital of France?")
if result:
    print(f"Response: {result}")

# Additional example with custom system message
print("\nRunning creative writing example...")
creative_result = query_model(
    "Write a haiku about artificial intelligence",
    system_message="You are a creative poet who loves technology."
)
if creative_result:
    print(f"Creative Response:\n{creative_result}")

# Step 14: Manual troubleshooting commands (if needed)
print("\n" + "="*50)
print("Manual Troubleshooting Commands:")
print("="*50)
print("""
If you encounter issues, run these commands in separate cells:

# Check ollama processes:
os.system("ps aux | grep -E 'ollama' | grep -v grep || true")

# Kill ollama if needed:
os.system("pkill -9 ollama || true")

# Restart ollama:
os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")
time.sleep(5)

# Check server status:
os.system("curl -s http://localhost:11434/v1/models")

# List available models:
os.system("ollama list")

# Re-download model if needed:
os.system("ollama pull gpt-oss:20b")
""")

