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


# ðŸ§  Smart Task Prioritizer Agent
# Author: Haikuxpp

# -----------------------------
# 1. Import required libraries
# -----------------------------
import pandas as pd

# -----------------------------
# 2. Define the prioritization logic
# -----------------------------
def classify_priority(task):
    """Classify a task into High, Medium, or Low priority based on keywords."""
    task = task.lower()
    high_keywords = ["urgent", "today", "now", "immediately", "asap", "deadline", "submit", "exam"]
    medium_keywords = ["soon", "important", "this week", "project", "meeting", "prepare"]
    low_keywords = ["later", "someday", "optional", "maybe", "whenever"]

    if any(word in task for word in high_keywords):
        return "High"
    elif any(word in task for word in medium_keywords):
        return "Medium"
    elif any(word in task for word in low_keywords):
        return "Low"
    else:
        return "Medium"  # Default to Medium if unclear

# -----------------------------
# 3. Create a small sample dataset
# -----------------------------
tasks = [
    "Submit science project today",
    "Prepare for math test next week",
    "Buy new notebook later",
    "Complete coding assignment asap",
    "Plan birthday party",
    "Clean my room whenever I can"
]

# -----------------------------
# 4. Apply the agent logic
# -----------------------------
df = pd.DataFrame(tasks, columns=["Task"])
df["Priority"] = df["Task"].apply(classify_priority)

# -----------------------------
# 5. Display results
# -----------------------------
print("ðŸ§¾ Task Prioritization Results:\n")
print(df)

# -----------------------------
# 6. Optional: Give suggestions
# -----------------------------
def suggest_action(priority):
    if priority == "High":
        return

