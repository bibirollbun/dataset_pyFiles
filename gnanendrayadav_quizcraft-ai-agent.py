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


print("QuizCraft – Auto-Generated Quizzes & Explanations Agent")


links = {
    "Main Project": "https://projecthub.ai/gnanendrayadav/quizcraft-auto-generated-quizzes",
    "Problem Statement": "https://projecthub.ai/gnanendrayadav/quizcraft/problem-statement",
    "Why Agents": "https://projecthub.ai/gnanendrayadav/quizcraft/why-agents",
    "Architecture Overview": "https://projecthub.ai/gnanendrayadav/quizcraft/architecture-overview",
    "Demo": "https://projecthub.ai/gnanendrayadav/quizcraft/demo",
    "Build Details": "https://projecthub.ai/gnanendrayadav/quizcraft/the-build",
    "Future Work": "https://projecthub.ai/gnanendrayadav/quizcraft/future-work",
}

for title, url in links.items():
    print(f"{title}: {url}")



"""
QuizCraft – Auto-Generated Quizzes & Explanations Agent
Full Project Implementation in One Python Cell
"""

# -------------------------------------------------------------------
# (1) PROBLEM STATEMENT
# -------------------------------------------------------------------
problem_statement = """
Students often struggle to get instant quizzes and explanations for any topic.
Manually creating quizzes wastes time, requires expertise, and reduces learning speed.
An automated system that generates topic-wise quizzes + explanations can improve learning,
revision quality, and personalization.
"""

# -------------------------------------------------------------------
# (2) WHY AGENTS?
# -------------------------------------------------------------------
why_agents = """
Agents break the task into smaller intelligent units:
1. Topic Analyzer - Understands the topic and extracts key concepts.
2. Quiz Generator - Creates MCQs using the extracted concepts.
3. Quiz Evaluator - Checks correctness and gives explanations.

This modularity makes the system scalable, simple, and accurate.
"""

# -------------------------------------------------------------------
# (3) OVERALL ARCHITECTURE
# -------------------------------------------------------------------
architecture = """
Architecture:
    Input Topic
        |
        ---> Topic Analyzer Agent
                |
                ---> Quiz Generator Agent
                        |
                        ---> Quiz Evaluator Agent
                                |
                                ---> Final Output (MCQs + Answers + Explanations)
Tools:
    - Python random module for randomization
    - Simple rule-based evaluation
    - In-session memory using Python dictionaries
"""

# -------------------------------------------------------------------
# (4) CORE IMPLEMENTATION
# -------------------------------------------------------------------

import random

# Memory to store previous quizzes
session_memory = {"previous_quizzes": []}

def topic_analyzer(topic):
    """Extracts keywords from the topic using simple splitting."""
    words = topic.split()
    keywords = [w for w in words if len(w) > 3]
    return keywords[:5]

def quiz_generator(keywords):
    """Creates simple MCQs using keywords."""
    questions = []
    for i, word in enumerate(keywords):
        question = f"What is the meaning or importance of '{word}'?"
        options = [
            f"It is related to {word} concept",
            "It is not related",
            "It is a random term",
            "None of the above"
        ]
        correct = 1  # always option A for simplicity
        questions.append({
            "question": question,
            "options": options,
            "correct": correct
        })
    return questions

def quiz_evaluator(questions):
    """Generates explanations for the MCQs."""
    evaluated = []
    for q in questions:
        explanation = (
            "Option A is correct because it directly relates to the keyword "
            "and its conceptual meaning."
        )
        evaluated.append({
            "question": q["question"],
            "options": q["options"],
            "correct_answer": q["options"][q["correct"] - 1],
            "explanation": explanation
        })
    return evaluated

# -------------------------------------------------------------------
# (5) DEMO RUN
# -------------------------------------------------------------------
topic = "Machine Learning Introduction and Algorithms Overview"
keywords = topic_analyzer(topic)
quiz = quiz_generator(keywords)
evaluated_quiz = quiz_evaluator(quiz)

session_memory["previous_quizzes"].append(evaluated_quiz)

# Display Final Result
for i, q in enumerate(evaluated_quiz, 1):
    print(f"\nQ{i}. {q['question']}")
    for idx, opt in enumerate(q["options"], 1):
        print(f"  {idx}. {opt}")
    print(f"Correct Answer: {q['correct_answer']}")
    print(f"Explanation: {q['explanation']}")

# -------------------------------------------------------------------
# (6) IF I HAD MORE TIME
# -------------------------------------------------------------------
future_scope = """
1. Add real LLM-based question generation.
2. Add difficulty adjustment based on user performance.
3. Deploy using a web UI with API.
4. Add long-term memory + user profiles.
"""
print("\n\n--- Future Scope ---\n")
print(future_scope)

print("\n\n--- Project Loaded Successfully in Single Cell ---")


