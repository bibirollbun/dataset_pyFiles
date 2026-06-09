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


studyalpha/



import time
from IPython.display import Markdown

Markdown("""
# ğŸ�“ StudyAlpha â€” Personal AI Study Coach  
### Shubham Mahajan Â· Freestyle Track Â· Kaggle x Google Agents Intensive  
---
""")



!pip install scikit-learn joblib matplotlib pandas --quiet



from studyalpha.agents import StudyOrchestrator
from studyalpha.predictor import train_and_save_model



print("Initializing StudyAlpha agent...")
alpha = StudyOrchestrator()
train_and_save_model()
print("Ready âœ”")



topics = [
    {"topic": "Arrays", "priority": 2},
    {"topic": "Dynamic Programming", "priority": 3},
    {"topic": "Graphs", "priority": 1},
]
topics



flow = alpha.full_plan_flow(topics, hours_per_day=2.0, days=7)
plan = flow["plan"]

print("ğŸ“˜ Study Plan (7 Days):")
plan



quiz = flow["sample_quiz"]

print("ğŸ“� Sample Quiz for:", quiz["topic"])
quiz



user_answers = [
    "It is a linear data structure", 
    "We use indexing to access elements", 
    "They store homogeneous data"
]
user_answers



result = alpha.tracker.record_quiz(quiz, user_answers)
result



print("ğŸ§  MemoryBank Contents:")
alpha.memory.long_term



alpha.memory.query("Arrays", top_k=3)



revision = alpha.revision.generate("Arrays")
revision



Markdown("""
# âœ… Demo Complete  

### What the StudyAlpha Agent just did:
- Generated a personalized 7-day study plan  
- Created quiz questions dynamically  
- Evaluated your answers  
- Stored results in long-term memory (RAG)  
- Predicted weak topics using ML  
- Generated revision sessions  

This is the full multi-agent pipeline working end-to-end.
""")


