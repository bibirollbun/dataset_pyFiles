!git clone https://github.com/lmassaron/Agentic-AI-Kaggle-Competition-Assistant.git


!pip install -r Agentic-AI-Kaggle-Competition-Assistant/requirements.txt > /dev/null


import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

try:    
    my_username = user_secrets.get_secret("kaggle_username")
    
    my_key = user_secrets.get_secret("kaggle_key") 

    os.environ["KAGGLE_USERNAME"] = my_username
    os.environ["KAGGLE_KEY"] = my_key
    
    print("Kaggle Credentials set successfully!")

except Exception as e:
    print(f"Error retrieving secrets: {e}")
    print("Please check the 'Add-ons -> Secrets' menu to verify your secret labels.")


import sys
import os
import glob
import importlib

project_root = os.path.abspath('Agentic-AI-Kaggle-Competition-Assistant')

if project_root not in sys.path:
    sys.path.append(project_root)

src_folder = os.path.join(project_root, 'src')
py_files = glob.glob(os.path.join(src_folder, '*.py'))

for file_path in py_files:
    filename = os.path.basename(file_path)[:-3]
    
    if filename == "__init__":
        continue
        
    module_name = f"src.{filename}"
    
    print(f"Importing {module_name}...")
    try:
        module = importlib.import_module(module_name)
        # Optional: Add to global namespace if you really need to use functions directly
        globals().update(vars(module)) 
    except Exception as e:
        print(f"Failed to import {module_name}: {e}")


import os
import sys
import json
from dotenv import load_dotenv

# Ensure the 'src' module is in the python path
sys.path.append(os.getcwd())

from src.agent import KaggleAgent

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    # Try to load from Kaggle secrets if running in a Kaggle notebook
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        google_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
        print("API Key loaded from Kaggle Secrets.")
    except ImportError:
        print("Error: GOOGLE_API_KEY not found. Please set it in .env or environment variables.")

# Initialize the Agent
if google_api_key:
    agent = KaggleAgent(api_key=google_api_key)
    print("Agent initialized successfully.")


response = agent.run("Find competitions similar to 'titanic' and tell me about them.")
print(response)


response = agent.run("What are the winning solutions for the 'titanic' competition?")
print(response)


response = agent.run("Show me the top scoring Python kernels for 'titanic'.")
print(response)


response = agent.run("What libraries are most commonly used in the 'titanic' competition?")
print(response)


response = agent.run("Search for code snippets using 'RandomForestClassifier' in the 'titanic' competition.")
print(response)


response = agent.run("What is the competition slug for 'https://www.kaggle.com/c/titanic'?")
print(response)


response = agent.run("Summarize this competition: https://www.kaggle.com/c/titanic")
print(response)


if 'agent' in locals():
    print(json.dumps(agent.get_session_stats(), indent=2))

