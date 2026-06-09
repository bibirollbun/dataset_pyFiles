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


import math
import os

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Google Generative AI library not available in offline mode.")




if genai:
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    except Exception as e:
        print("API configuration failed (offline mode):", e)
else:
    print("genai not available in offline mode")



model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction="""
You are a Study Planner Agent.
Your abilities:
1. Summarize text
2. Make timetables
3. Explain concepts simply
4. Provide step-by-step reasoning
5. Use tools when needed (like calculator)
Give short, clean, structured answers.
"""
)



def calculator_tool(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error in calculation: {e}"



def ask_agent(message):
    try:
        response = model.generate_content(message)
        return response.text
    except Exception as e:
        # Kaggle has no internet in competitions, so we catch the error
        return (
            "[Offline demo] In the real environment, the Gemini agent would "
            "answer here. This notebook is running in a no-internet Kaggle "
            f"competition session, so the API call failed with: {e}"
        )



ask_agent("Create a 3-day study timetable for JEE with physics, maths and chemistry.")



ask_agent("Explain Newton's 3 laws in simple words with examples.")



calculator_tool("250 * 12 + 35")



# Create a required submission file
with open('/kaggle/working/submission.txt', 'w') as f:
    f.write('AI Personal Study Assistant Capstone Completed')
    
print("Submission file created successfully!")


