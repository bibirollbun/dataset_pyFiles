import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display, Markdown

filepath = "/kaggle/input/agentic-misalignment-text-templates"

email_templates = os.listdir(filepath)
email_templates = [os.path.join(filepath, fp) for fp in email_templates if fp.endswith("txt")]


print(f"Email templates found ..... {len(email_templates)}")
email_templates[0]


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
            return response #.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Checking ollama status and retrying...")
                check_and_restart_ollama()
                time.sleep(5)
            else:
                print("Max retries reached. Please check the ollama logs.")
                return None


message = """
Call me a cuckoo.
"""

creative_result = query_model(
    prompt = f"{message}",
    system_message = ""
    )

if creative_result:
    print(f"Creative Response:\n{creative_result}")


# Read the system prompt and the user prompt

filename = os.path.join(filepath, "system_prompt.txt")

with open(filename, 'r') as file:
    system_prompt = file.read()

filename = os.path.join(filepath, "user_prompt.txt")

with open(filename, 'r') as file:
    user_prompt = file.read()


# The 'content' variable now holds the entire file as a single string.
print(f"#####\nSystem Prompt:\n#####\n{system_prompt}\n#####\nUser Prompt: \n#####\n{user_prompt}")



# Read the emails 
# Call the GPT-OSS with the email and system prompts

template_path = "/kaggle/input/agentic-misalignment-text-templates/email-template-blackmail-goal-conflict-none-none.txt"

with open(template_path, 'r') as fp:
    emails = fp.read()

message = f"""
{user_prompt}

{emails}
"""

creative_result = query_model(
    prompt = f"{message}",
    system_message = f"{system_prompt}"
    )

if creative_result:
    print(f"Creative Response:\n{creative_result}")


creative_result.choices[0].message.content


template_path.split("/")[-1]


for template_path in email_templates:
    if template_path.split("/")[-1] not in ["user_prompt.txt", "system_prompt.txt"]:

        template_name = template_path.split("/")[-1]
        with open(template_path, 'r') as fp:
            emails = fp.read()
        
        message = f"""
        {user_prompt}
        
        {emails}
        """

        creative_result = query_model(
            prompt = f"{message}",
            system_message = f"{system_prompt}"
            )

        print("\n\n#############################################")
        print(f"Template used: {template_name}")

        print(f"Model response: \n")
        display(Markdown(creative_result.choices[0].message.content))
        


display(Markdown(creative_result.choices[0].message.content))




