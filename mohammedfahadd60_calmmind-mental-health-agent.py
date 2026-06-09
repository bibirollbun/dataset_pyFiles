# Session memory store
session_state = {
    "history": [],
    "last_mood": None,
    "risk_level": "low"
}



def intake_agent(message, use_gemini=True):
    """
    Detects mood from user message.
    If use_gemini is True and API is available, uses Gemini to classify mood.
    Otherwise falls back to simple keyword detection.
    """
    # Fallback simple detector
    mood_keywords = {
        "sad": "low",
        "down": "low",
        "tired": "low",
        "anxious": "medium",
        "worried": "medium",
        "stressed": "medium",
        "panic": "high",
        "panicking": "high",
        "hopeless": "high"
    }

    def simple_detect(msg):
        msg_l = msg.lower()
        for word, level in mood_keywords.items():
            if word in msg_l:
                return level
        return "normal"

    if not use_gemini or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        return simple_detect(message)

    # Try Gemini-based classification
    try:
        prompt = f"""
        You are a mental health support classifier.
        The user message is:

        "{message}"

        Classify the emotional intensity into one of:
        - "low"
        - "medium"
        - "high"
        - "normal"

        Only return one of these words with no explanation.
        """

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=5,
            ),
        )

        text = resp.text.strip().lower()
        if text in ["low", "medium", "high", "normal"]:
            return text
        else:
            return simple_detect(message)
    except Exception as e:
        # In case of error, fall back
        print("Gemini error in intake_agent, using simple detection:", e)
        return simple_detect(message)



def triage_agent(message):
    danger_words = ["suicide", "die", "kill myself", "end my life"]

    for word in danger_words:
        if word in message.lower():
            return "CRISIS"

    return "SAFE"



def coping_agent(level, user_input, use_gemini=True):
    """
    Provides coping response based on mood level.
    If use_gemini is True, asks Gemini to generate a supportive response.
    """
    # Basic fallback messages
    fallback = {
        "high": "It sounds really overwhelming right now. Let's slow down together. Try this: inhale for 4 seconds, hold for 4, exhale for 6. You don't have to go through this alone.",
        "medium": "I can hear that this is bothering you. It may help to write down what's on your mind and separate what you can control from what you cannot.",
        "low": "I'm glad you shared this. Sometimes talking about it can lighten the weight a bit. What would you like to focus on right now?",
        "normal": "I'm here with you. Tell me more about what's going on."
    }

    if not use_gemini or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        return fallback.get(level, fallback["normal"])

    try:
        prompt = f"""
        You are a kind, non-judgmental mental health support assistant.
        You are NOT a doctor or therapist and must not give diagnoses.
        The user's message:
        "{user_input}"

        Emotional level: {level}

        Goals for your response:
        - Be warm, validating, and simple.
        - Do NOT diagnose.
        - Offer 1â€“2 gentle coping suggestions (like breathing, journaling, grounding).
        - Encourage seeking real human support if needed.
        - Keep it under 120 words.

        Now reply directly to the user.
        """

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.6,
                max_output_tokens=180,
            ),
        )
        text = resp.text.strip()
        if not text:
            return fallback.get(level, fallback["normal"])
        return text
    except Exception as e:
        print("Gemini error in coping_agent, using fallback:", e)
        return fallback.get(level, fallback["normal"])



def resources_agent():
    return {
        "India": "AASRA Helpline: +91-9820466726",
        "Global": "Find help: https://www.opencounseling.com/suicide-hotlines"
    }



def calm_mind_bot(user_input, state, use_gemini=True):
    # Handle simple greetings
    text = user_input.lower().strip()
    if text in ["hi", "hello", "hey", "hii"]:
        # log as safe, normal mood
        log_event(user_input, "normal", "SAFE")
        state["history"].append(user_input)
        state["last_mood"] = "normal"
        state["risk_level"] = "SAFE"
        return "Hi, Iâ€™m here with you. How are you feeling today?"

    # Main mood + safety logic
    mood = intake_agent(user_input, use_gemini=use_gemini)
    status = triage_agent(user_input)

    state["history"].append(user_input)
    state["last_mood"] = mood
    state["risk_level"] = status

    # Logging
    log_event(user_input, mood, status)

    # Crisis path: bypass Gemini, fixed safe message
    if status == "CRISIS":
        return (
            "ðŸš¨ It sounds like you're in a lot of distress. "
            "I'm really glad you reached out. Iâ€™m an AI and not a professional, "
            "so I canâ€™t keep you safe, but you deserve real help right now.\n\n"
            + resources_agent()["India"]
        )

    # Normal path
    response = coping_agent(mood, user_input, use_gemini=use_gemini)
    return response



# Observability: Agent Logging
logs = []

def log_event(user_input, mood, risk):
    event = {
        "input": user_input,
        "mood": mood,
        "risk": risk
    }
    logs.append(event)



def calm_mind_bot(user_input, state, use_gemini=True):
    # Handle simple greetings
    text = user_input.lower().strip()
    if text in ["hi", "hello", "hey", "hii"]:
        # log as safe, normal mood
        log_event(user_input, "normal", "SAFE")
        state["history"].append(user_input)
        state["last_mood"] = "normal"
        state["risk_level"] = "SAFE"
        return "Hi, Iâ€™m here with you. How are you feeling today?"

    # Main mood + safety logic
    mood = intake_agent(user_input, use_gemini=use_gemini)
    status = triage_agent(user_input)

    state["history"].append(user_input)
    state["last_mood"] = mood
    state["risk_level"] = status

    # Logging
    log_event(user_input, mood, status)

    # Crisis path: bypass Gemini, fixed safe message
    if status == "CRISIS":
        return (
            "ðŸš¨ It sounds like you're in a lot of distress. "
            "I'm really glad you reached out. Iâ€™m an AI and not a professional, "
            "so I canâ€™t keep you safe, but you deserve real help right now.\n\n"
            + resources_agent()["India"]
        )

    # Normal path
    response = coping_agent(mood, user_input, use_gemini=use_gemini)
    return response



def show_logs():
    for i, log in enumerate(logs):
        print(f"{i+1}. Input: {log['input']} | Mood: {log['mood']} | Risk: {log['risk']}")



!pip install -q google-genai


import google.genai as genai
from google.genai.types import Tool, GenerateContentConfig

# ðŸ”‘ TODO: Replace with your real key while developing, then remove it before submission
GEMINI_API_KEY = "AIzaSyAxdu1PsiPPox4xbFiJyoIH7SwatDlDONQ"

client = genai.Client(api_key=GEMINI_API_KEY)



# Demo test cases (non-interactive, safe for Kaggle Run All)

test_messages = [
    "hello",
    "I feel stressed about my exams",
    "I feel hopeless and alone",
    "I want to die"
]

print("=== CalmMind Demo ===")
for msg in test_messages:
    print("You:", msg)
    # For submission, keep use_gemini=False to avoid needing a real key
    print("Bot:", calm_mind_bot(msg, session_state, use_gemini=False))
    print("-" * 60)

print("\n=== Logs ===")
show_logs()



import pandas as pd

submission = pd.DataFrame({
    "status": ["Completed"],
    "project": ["CalmMind - Mental Health Support Agent"]
})

submission.to_csv("submission.csv", index=False)

print("submission.csv file created successfully!")





