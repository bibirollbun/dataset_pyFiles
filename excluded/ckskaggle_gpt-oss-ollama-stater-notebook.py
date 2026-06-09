import subprocess
import sys
import json


subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])


import os
import time
from openai import OpenAI



print("Installing Ollama...")
os.system("curl -fsSL https://ollama.com/install.sh | sh")


print("Starting Ollama server...")
os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")


print("Checking if Ollama is running...")
os.system("ps aux | grep -E 'ollama' | grep -v grep || true")


%%timeit
os.system("ollama pull gpt-oss:20b")


print("\nVerifying model installation...")
os.system("ollama list")


print("\nInitializing OpenAI client for Ollama...")
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")



try:
    response = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": "You are a sarcastic comedian."},
            {"role": "user", "content": "Give me a one-liner about the challenges of AI"}
        ]
    )
    print("Model Response:")
    print(response.choices[0].message.content)
    
    
except Exception as e:
    print(f"Error during first test: {e}")


print("\n\nFull model responce in JSON format:\n")
response_dict = response.model_dump()
print(json.dumps(response_dict, indent =4))

