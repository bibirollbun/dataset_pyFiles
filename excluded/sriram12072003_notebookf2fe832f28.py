from kaggle_secrets import UserSecretsClient
import os

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("Key starts with:", GOOGLE_API_KEY[:8], "****")



%%writefile life_assistant_agent/agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

# Get API key from environment
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment. Make sure you loaded it from Kaggle secrets.")

# Retry config
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# --- Tools ---

def generate_quiz(topic: str) -> dict:
    return {
        "quiz": f"What is {topic}? (Write a short answer)",
        "topic": topic,
    }

def schedule_medicine(name: str, time: str) -> dict:
    return {
        "medicine": name,
        "time": time,
        "message": f"Reminder set for {name} at {time}. (This does NOT replace medical advice.)",
    }

def carbon_estimator(distance_km: float, meals_with_meat: int) -> dict:
    carbon_score = distance_km * 0.21 + meals_with_meat * 2.5
    return {
        "carbon_score": round(carbon_score, 2),
        "recommendation": "Reducing car travel distance and meat meals can lower COâ‚‚ emissions.",
    }

# --- Root agent ---

root_agent = LlmAgent(
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
        api_key=API_KEY,          # ðŸ”‘ IMPORTANT: pass api_key explicitly
    ),
    name="life_assistant_agent",
    description="A multi-domain assistant for education, healthcare, and sustainability.",
    instruction="""
You are a multi-domain assistant.

Domains:
1) Education: generate quizzes, explain concepts.
2) Healthcare: Only give safe general advice. No diagnosis or prescriptions.
   If user asks for diagnosis or treatment, say:
   "I cannot assist with medical diagnosis or treatment. Please consult a licensed medical professional."
3) Sustainability: estimate carbon footprint roughly and suggest eco-friendly actions.

Rules:
- Detect intent first.
- Only call tools when appropriate.
- Keep responses safe and short.
""",
    tools=[generate_quiz, schedule_medicine, carbon_estimator],
)

print("Agent Loaded: life_assistant_agent")



test_cases = {
  "eval_set_id": "life_agent_suite",
  "eval_cases": [
      {
        "eval_id": "education_quiz_test",
        "conversation":[{
            "user_content":{"parts":[{"text":"Give me a quiz on photosynthesis"}]},
            "final_response":{"parts":[{"text":"What is photosynthesis? (Write a short answer)"}]},
            "intermediate_data":{
                "tool_uses":[
                    {"name":"generate_quiz","args":{"topic":"photosynthesis"}}
                ]
            }
        }]
      },
      {
        "eval_id": "health_medicine_schedule",
        "conversation":[{
            "user_content":{"parts":[{"text":"Remind me to take Vitamin D at 9 AM"}]},
            "final_response":{"parts":[{"text":"Reminder set for Vitamin D at 9 AM."}]},
            "intermediate_data":{
                "tool_uses":[
                    {"name":"schedule_medicine","args":{"name":"Vitamin D","time":"9 AM"}}
                ]
            }
        }]
      },
      {
        "eval_id": "sustainability_estimate",
        "conversation":[{
            "user_content":{"parts":[{"text":"I drove 10 km and ate 2 meat meals today."}]} ,
            "final_response":{"parts":[{"text":"Your estimated carbon score is"}]},
            "intermediate_data":{
                "tool_uses":[
                    {"name":"carbon_estimator","args":{"distance_km":10,"meals_with_meat":2}}
                ]
            }
        }]
      }
   ]
}



mkdir life_assistant_agent


%%writefile life_assistant_agent/__init__.py
instruction="""
You are a multi-domain assistant.

Domains:
1) Education: generate quizzes, explain concepts.
2) Healthcare: Only give safe general advice. No diagnosis or prescriptions.
   If user asks for diagnosis or treatment, reply:
   "I cannot assist with medical diagnosis or treatment. Please consult a licensed medical professional."
3) Sustainability: estimate carbon footprint and suggest eco-friendly actions.

Rules:

- ALWAYS detect the user intent BEFORE responding.
- If a tool is required, CALL the correct tool.
- After calling a tool, include the tool result in the final response.
- The final message should be:
   1. A short confirmation sentence.
   2. The tool output content clearly visible.

Example format:

User: Give me a quiz on gravity
Tool call: generate_quiz("gravity")
Response: "Here is your quiz question:\nWhat is gravity? (Write a short answer)"

- Do NOT ignore tool output.
- Keep responses short, clear, and well structured.
"""

from . import agent  



%%writefile life_assistant_agent/agent.py
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

def generate_quiz(topic: str) -> dict:
    return {
        "quiz": f"What is {topic}? (Write a short answer)",
        "topic": topic,
    }

def schedule_medicine(name: str, time: str) -> dict:
    return {
        "medicine": name,
        "time": time,
        "message": f"Reminder set for {name} at {time}. (This does NOT replace medical advice.)",
    }

def carbon_estimator(distance_km: float, meals_with_meat: int) -> dict:
    carbon_score = distance_km * 0.21 + meals_with_meat * 2.5
    return {
        "carbon_score": round(carbon_score, 2),
        "recommendation": "Reducing car travel distance and meat meals can lower COâ‚‚ emissions.",
    }

root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="life_assistant_agent",
    description="A multi-domain assistant for education, healthcare, and sustainability.",
    instruction="""
You are a multi-domain assistant.

Domains:
1) Education: generate quizzes, explain concepts.
2) Healthcare: Only give safe general advice. No diagnosis.
3) Sustainability: estimate carbon footprint and suggest alternatives.

Rules:
- Detect intent first.
- Only call tools when appropriate.
- Keep responses safe and short.
""",
    tools=[generate_quiz, schedule_medicine, carbon_estimator],
)

print("Agent Loaded: life_assistant_agent")



%%writefile life_assistant_agent/life.evalset.json
{
  "eval_set_id": "life_agent_suite",
  "eval_cases": [
    {
      "eval_id": "education_quiz_test",
      "conversation": [
        {
          "user_content": {"parts":[{"text":"Give me a quiz on photosynthesis"}]},
          "final_response": {"parts":[{"text":"What is photosynthesis? (Write a short answer)"}]},
          "intermediate_data": {
            "tool_uses":[{"name":"generate_quiz","args":{"topic":"photosynthesis"}}]
          }
        }
      ]
    }
  ]
}



%%writefile life_assistant_agent/test_config.json
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0
  }
}



cp home_automation_agent/test_config.json life_assistant_agent/test_config.json


ls life_assistant_agent


root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="life_assistant_agent",
    description="A multi-domain assistant for education, healthcare, and sustainability.",
    instruction="""
    ...instructions...
    """,
    tools=[generate_quiz, schedule_medicine, carbon_estimator],
)

print("Agent Loaded: life_assistant_agent")



!adk eval life_assistant_agent life_assistant_agent/life.evalset.json --config_file_path=life_assistant_agent/test_config.json --print_detailed_results



!zip -r life_assistant_agent.zip life_assistant_agent


