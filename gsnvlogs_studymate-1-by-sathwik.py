# ------------------------------------------------------------
#            STUDYMATE â€” MULTI AGENT STUDY PROJECT
# ------------------------------------------------------------

import requests
import json

# ------------------------------------------------------------
# 1. Load Gemini API Key from Kaggle Secrets
# ------------------------------------------------------------
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# ------------------------------------------------------------
# 2. LLM Function (Stable â€“ Works with Gemini 1.5 Flash)
# ------------------------------------------------------------
def llm(prompt):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent?key=" + GOOGLE_API_KEY
    )

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    # Return API error message if any
    if "error" in result:
        return "API Error: " + result["error"]["message"]

    # Try to extract text from multiple valid response formats
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except:
        pass

    try:
        return result["contents"][0]["parts"][0]["text"]
    except:
        pass

    try:
        return result["candidates"][0]["output"][0]["text"]
    except:
        pass

    # If nothing matches, return raw result
    return "Unexpected response:\n" + json.dumps(result, indent=2)


# ------------------------------------------------------------
# 3. Agent 1 â€” Smart Study Plan Agent
# ------------------------------------------------------------
def study_plan_agent(topic):
    prompt = f"""
    You are Study Plan Agent.

    Create a 3-day structured medium-level study plan for: {topic}

    Include:
    - Day-wise focused tasks
    - Practice problems count
    - Study duration
    - Mini revision work
    - Motivation line for each day
    """
    return llm(prompt)


# ------------------------------------------------------------
# 4. Agent 2 â€” Concept Explanation Agent
# ------------------------------------------------------------
def explain_agent(concept):
    prompt = f"""
    Explain the concept: {concept}

    Provide:
    - Definition
    - Easy explanation
    - Real-life analogy
    - Why it is important in learning
    - A simple example
    """
    return llm(prompt)


# ------------------------------------------------------------
# 5. Agent 3 â€” Question Answer Agent
# ------------------------------------------------------------
def question_agent(question):
    prompt = f"""
    Answer the following question clearly and accurately:

    {question}

    Provide:
    - Straightforward explanation
    - If useful, add tips or steps
    """
    return llm(prompt)


# ------------------------------------------------------------
# 6. Main StudyMate Controller
# ------------------------------------------------------------
def StudyMate():
    print("ðŸ“˜ Welcome to StudyMate â€” Your Multi-Agent Study Partner\n")

    # ---- Agent 1: Study Plan ----
    topic = input("Enter a study topic: ")
    print("\nðŸ“… Your 3-Day Study Plan:\n")
    print(study_plan_agent(topic))
    print("\n" + "-"*70)

    # ---- Agent 2: Concept Explanation ----
    concept = input("\nEnter a concept to explain: ")
    print("\nðŸ“– Concept Explanation:\n")
    print(explain_agent(concept))
    print("\n" + "-"*70)

    # ---- Agent 3: Ask any question ----
    question = input("\nAsk any study question: ")
    print("\nðŸ’¬ Answer:\n")
    print(question_agent(question))


# ------------------------------------------------------------
# Run the Full Program
# ------------------------------------------------------------
StudyMate()

