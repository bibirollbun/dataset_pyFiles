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


# TutorAgent â€” AI-Powered Personalized Learning Assistant  
### Agents for Good â€” Kaggle AI Agents Intensive Capstone

Welcome! This notebook demonstrates *TutorAgent*, an AI-powered agent that:

Creates personalized study plans  
Retrieves relevant notes via vector search  
Explains concepts in simple language  
Quizzes the student and scores answers  
Adapts difficulty based on performance  

This project showcases:
- *Planning & task decomposition*
- *Retrieval-Augmented Generation (RAG)*
- *Evaluation & adaptive feedback loops*
- *Memory-based personalization*

Scroll down and run the demo cell to see TutorAgent in action.

!pip install chromadb faiss-cpu tiktoken

import chromadb
from chromadb.utils import embedding_functions
import json
importÂ random

# Simple Chroma in-memory database
client = chromadb.Client()
collection = client.create_collection(
    name="study_notes",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# Sample content (you can replace with real study notes later)
documents = [
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "Chlorophyll absorbs sunlight to initiate photosynthesis.",
    "Mitochondria are the powerhouse of the cell.",
    "A noun is a word that represents a person, place, or thing."
]

collection.add(
    documents=documents,
    ids=[str(i) for i in range(len(documents))]
)

def mock_llm(prompt):
    # Replace with your ADK model call in real version
    return "AI Response: "Â +Â prompt[:150]

    def generate_study_plan(topic):
    plan = [
        {"step": 1, "task": f"Explain core idea of {topic}"},
        {"step": 2, "task": f"Retrieve important notes on {topic}"},
        {"step": 3, "task": f"Create a short quiz on {topic}"}
    ]
Â Â Â Â returnÂ plan

    def retrieve_notes(query):
    results = collection.query(query_texts=[query], n_results=2)
    return results["documents"][0]

    def make_quiz(topic):
    return [
        {"question": f"What is the main idea of {topic}?", "answer": "photosynthesis" if "photo" in topic.lower() else topic}
    ]

def evaluate_quiz(user_answer, correct_answer):
    return "Correct!" if user_answer.lower().strip() == correct_answer.lower().strip() else "Almost there! ReviewÂ onceÂ more."

    def run_tutoragent(topic):
    output = {}

    # Step 1 â€” Study plan
    plan = generate_study_plan(topic)
    output["plan"] = plan

    # Step 2 â€” Retrieval
    notes = retrieve_notes(topic)
    output["notes"] = notes

    # Step 3 â€” Explanation
    explanation = mock_llm(f"Explain: {notes}")
    output["explanation"] = explanation

    # Step 4 â€” Quiz
    quiz = make_quiz(topic)
    output["quiz"] = quiz

returnÂ output

# ðŸŽ¬ Demo: Run TutorAgent

Enter a topic and let the agent generate a plan, retrieve notes, explain the concept, andÂ createÂ aÂ quiz.

topic = "Photosynthesis"
response = run_tutoragent(topic)

response

user_answer = input("Your answer: ")
correct = response["quiz"][0]["answer"]
evaluate_quiz(user_answer,Â correct)

## ðŸš€ Future Enhancements
- Add adaptive difficulty based on student history  
- Multi-chapter learning paths  
- Voice-based tutoring  
- Parent/teacher report generation  
- Full deploymentÂ asÂ aÂ webÂ app

## ðŸ“š References
- Kaggle AI Agents Intensive Course  
- ChromaDB Documentation  
- FAISSÂ VectorÂ Search

