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


import random

motivation_quotes = [
    "You can do this!",
    "Small progress is still progress.",
    "Stay consistent, your future self will thank you.",
    "Discipline beats motivation."
]

def productivity_agent():
    print("Agent: Hi! I am your productivity assistant.")
    

    while True:
        user = input("You: ")

        if user.lower() == "exit":
            print("Agent: Goodbye! Stay productive!")
            break

        if "study plan" in user.lower():
            print("Agent: Here is a quick study plan:")
            print("- 45 min study\n- 10 min break\n- Repeat for 3 cycles")
            continue

        if "to-do" in user.lower() or "todo" in user.lower():
            print("Agent: Make a list with 3 priority tasks:")
            print("1. Most important\n2. Medium\n3. Optional")
            continue

        if "motivate" in user.lower() or "motivation" in user.lower():
            print("Agent:", random.choice(motivation_quotes))
            continue

        print("Agent: I can help with study plans, to-do lists, motivation, or break schedules.")

productivity_agent()

