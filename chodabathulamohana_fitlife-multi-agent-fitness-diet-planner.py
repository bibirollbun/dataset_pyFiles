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


from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")



from google import genai



from kaggle_secrets import UserSecretsClient
from google import genai

# Load key
secrets = UserSecretsClient()
key = secrets.get_secret("GOOGLE_API_KEY")

client = genai.Client(api_key=key)

# Working model call
response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents="Give one simple fitness tip."
)

print(response.text)



from kaggle_secrets import UserSecretsClient
from google import genai

secrets = UserSecretsClient()
key = secrets.get_secret("GOOGLE_API_KEY")

client = genai.Client(api_key=key)

models = client.models.list()
for m in models:
    print(m.name)



while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=user_input
    )
    print("AI:", response.text)



from kaggle_secrets import UserSecretsClient
from google import genai

# Load key
secrets = UserSecretsClient()
key = secrets.get_secret("GOOGLE_API_KEY")

client = genai.Client(api_key=key)

# Predefined queries (no input needed)
queries = [
    "Give me a fitness tip for today.",
    "Suggest a 10-minute home workout.",
    "Give a healthy breakfast idea."
]

for q in queries:
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=q
    )
    print("You:", q)
    print("AI:", response.text)
    print("-"*50)



from kaggle_secrets import UserSecretsClient
from google import genai

# Load key
secrets = UserSecretsClient()
key = secrets.get_secret("GOOGLE_API_KEY")

client = genai.Client(api_key=key)

# List of queries
queries = [
    "Give me a fitness tip for today.",
    "Suggest a 10-minute home workout.",
    "Give a healthy breakfast idea.",
    "Give a healthy lunch idea.",
    "Give a quick evening snack idea.",
]

# Loop through queries
for q in queries:
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=q
    )
    print("You:", q)
    print("AI:", response.text)
    print("-"*50)



conversation_history = []

for q in queries:
    # Combine previous history
    full_prompt = "\n".join(conversation_history + [q])
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=full_prompt
    )
    print("You:", q)
    print("AI:", response.text)
    print("-"*50)
    
    # Save to history
    conversation_history.append(f"You: {q}")
    conversation_history.append(f"AI: {response.text}")



with open("daily_fitness_plan.txt", "w") as f:
    for entry in conversation_history:
        f.write(entry + "\n\n")



queries.extend([
    "Give a healthy dinner idea.",
    "Suggest a bedtime relaxation tip.",
    "Suggest a hydration schedule for the day."
])



user_input = input("Ask for a fitness/diet tip: ")
response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents="\n".join(conversation_history + [user_input])
)
print(response.text)
conversation_history.append(f"You: {user_input}")
conversation_history.append(f"AI: {response.text}")



from IPython.display import Markdown, display

for entry in conversation_history:
    display(Markdown(entry))


