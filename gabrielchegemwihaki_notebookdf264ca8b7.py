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

# Create the src directory if it doesn't exist
if not os.path.exists("src"):
    os.makedirs("src")

# Create an empty model.py file inside src
with open("src/model.py", "w") as f:
    f.write("""
class Model:
    def __init__(self):
        pass

    def predict(self, prompt: str) -> str:
        # Replace this with your actual SVG generation logic
        return '<svg width="100" height="100"><circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" /></svg>'
""")

print("Created src/model.py")


with open("requirements.txt", "w") as f:
    f.write("")

print("Created requirements.txt (empty)")


import json
import os
import shutil

# Define the path to your source directory
PACKAGE_PATH = "src"

# Output directory for the package
OUTPUT_PATH = "submission"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Create the metadata.json file
metadata = {
    "name": "my_drawing_model",  # Replace with your desired package name
    "version": "0.0.1",
    "description": "A basic drawing model", # Replace with your description
    "authors": [
        {
            "name": "Your Name",      # Replace with your name
            "email": "your.email@example.com" # Replace with your email
        }
    ],
    "license": "MIT"  # Or your preferred license
}

with open(os.path.join(OUTPUT_PATH, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# Copy the source directory
shutil.copytree(PACKAGE_PATH, os.path.join(OUTPUT_PATH, PACKAGE_PATH))

# Copy the requirements.txt file
if os.path.exists("requirements.txt"):
    shutil.copy("requirements.txt", OUTPUT_PATH)

print("Successfully created submission package in the 'submission' directory.")
print("Remember to replace the basic SVG generation logic in src/model.py!")

