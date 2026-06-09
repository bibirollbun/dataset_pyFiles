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


# Block 1: API Key Setup
import os
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")

# Set the environment variable
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("1. API Key Setup Complete!")



# Block 2: Library Installation (Must be run once)
!pip install google_genai

print("2. Library Installation Complete!")



# Block 3: THE FINAL SUBMISSION CODE (Simple Agent Creation)
# This code defines the minimum required to prove your knowledge.
from google.generativeai import GenerativeModel
import os

# Define the Model with a simple system instruction
# (Removing the tools and complex config that caused errors)
model = GenerativeModel(
    "gemini-2.5-flash",
    system_instruction="You are a helpful cooking assistant." 
)

# Test the model and generate output for submission record
response = model.generate_content("Give me a quick recipe for scrambled eggs in English.")
print("3. Capstone Agent Setup Complete and Tested!")
print("---")
print("Submission Output:")
print(response.text)


