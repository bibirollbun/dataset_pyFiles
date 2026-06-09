import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")
genai.configure(api_key=secret_value_0)



def call_gemini(system_prompt, user_prompt):
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    final_prompt = system_prompt + "\n\nUser: " + user_prompt
    resp = model.generate_content(final_prompt)
    return resp.text



def create_study_plan(topic: str, hours: int, branch: str = "computer"):
    return f"""
Study Plan for {topic} ({hours} hours, branch: {branch})

Day 1:
- Basics and intro (videos + notes) ~ {hours//3} hrs
Day 2:
- Core concepts + few numericals ~ {hours//3} hrs
Day 3:
- More problems + quick test + revision ~ {hours - 2*(hours//3)} hrs
"""

def suggest_resources(topic: str):
    return f"""
Free Resources for {topic}:
- YouTube: search "Gate Smashers {topic}" or "freeCodeCamp {topic}"
- Website: GeeksforGeeks {topic}
- Practice: Hackerrank / LeetCode questions on {topic}
"""



memory = {
    "branch": "computer eng",
    "hours_per_week": 6,
    "past_topics": []
}

system_prompt = """You are a friendly Study Assistant agent for a BTech student.
Use the given tools outputs when provided.
Language simple, bullet points allowed."""

def study_agent(message: str):
    # tool trigger
    msg_lower = message.lower()
    tool_output = ""

    if "plan" in msg_lower:
        topic = message.replace("plan", "").strip()
        tool_output = create_study_plan(topic, memory["hours_per_week"], memory["branch"])
    elif "resources" in msg_lower:
        topic = message.replace("resources", "").strip()
        tool_output = suggest_resources(topic)
    elif "set branch" in msg_lower:
        memory["branch"] = message.split(":",1)[-1].strip()
        return "OK, branch set to " + memory["branch"]
    elif "set hours" in msg_lower:
        try:
            h = int("".join(c for c in msg_lower if c.isdigit()))
            memory["hours_per_week"] = h
            return f"OK, I will plan using {h} hours per week."
        except:
            return "Please tell hours as a number."

    # ask Gemini to format nice answer
    final_user = f"User message: {message}\n\nTool output (if any):\n{tool_output}\n\nMemory: {memory}"
    answer = call_gemini(system_prompt, final_user)
    return answer



print(study_agent("Set branch: Computer Science"))
print()
print(study_agent("Set hours per week 9"))
print()
print(study_agent("Make study plan for Neural Networks"))
print()
print(study_agent("Give resources for Neural Networks"))



import pandas as pd

data = {"id": [1], "note": ["Study Assistant Agent submission"]}
df = pd.DataFrame(data)
df.to_csv("/kaggle/working/submission.csv", index=False)


