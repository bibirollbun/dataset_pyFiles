import os
import random
import time
import vertexai
from kaggle_secrets import UserSecretsClient
from vertexai import agent_engines

print("âœ… Imports completed successfully")


# Set up Cloud Credentials in Kaggle
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

print("âœ… Cloud credentials configured")


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


## Create simple agent - all code for the agent will live in this directory
!mkdir -p edu_agent

print(f"âœ… Education Agent directory created")


%%writefile edu_agent/requirements.txt

google-adk
opentelemetry-instrumentation-google-genai


%%writefile edu_agent/.env

# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#global-endpoint
GOOGLE_CLOUD_LOCATION="global"

# Set to 1 to use Vertex AI, or 0 to use Google AI Studio
GOOGLE_GENAI_USE_VERTEXAI=1


%%writefile edu_agent/agent.py

import random
import os
from dotenv import load_dotenv

import datetime

class MCPContext:
    def __init__(self, user_name, goals, constraints=None, memory=None, session_id=None):
        self.user = {"name": user_name, "goals": goals}
        self.session_history = []
        self.current_task = None
        self.constraints = constraints or {}
        self.agent_outputs = {}
        # Persistent learner data (memory)
        self.memory = memory or {}
        # Session metadata
        self.session = {
            "id": session_id or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": None,
            "end_time": None,
            "events": []
        }

    def update_history(self, event):
        self.session_history.append(event)
        self.session["events"].append(event)

    def set_task(self, task):
        self.current_task = task

    def record_output(self, agent_name, output):
        self.agent_outputs[agent_name] = output

    def start_session(self):
        self.session["start_time"] = datetime.datetime.now().isoformat()
        self.update_history(f"Session started at {self.session['start_time']}")

    def end_session(self):
        self.session["end_time"] = datetime.datetime.now().isoformat()
        self.update_history(f"Session ended at {self.session['end_time']}")

    def update_memory(self, key, value):
        self.memory[key] = value

class LearningPathDesignerAgent:
    def run(self, context):
        context.set_task("Designing learning path")
        output = "Learning path created for: " + ", ".join(context.user["goals"])
        context.update_history(output)
        context.record_output("LearningPathDesignerAgent", output)
        # Optionally update memory
        context.update_memory("last_learning_path", context.user["goals"])

class AssessmentAgent:
    def run(self, context):
        context.set_task("Assessing student knowledge")
        # Simulate assessment scores
        scores = {goal: 70 + hash(goal) % 30 for goal in context.user["goals"]}
        output = f"Assessment completed: {scores}"
        context.update_history(output)
        context.record_output("AssessmentAgent", scores)
        context.agent_outputs["scores"] = scores
        # Save scores to memory
        context.update_memory("last_scores", scores)

class ContentAdaptationAgent:
    def run(self, context):
        context.set_task("Adapting content based on assessment")
        scores = context.agent_outputs.get("scores", {})
        adaptation = {topic: "review" if score < 80 else "continue" for topic, score in scores.items()}
        output = f"Content adaptation: {adaptation}"
        context.update_history(output)
        context.record_output("ContentAdaptationAgent", adaptation)
        context.agent_outputs["adaptation"] = adaptation
        # Save adaptation to memory
        context.update_memory("last_adaptation", adaptation)

class FeedbackAgent:
    def run(self, context):
        context.set_task("Providing feedback to student")
        adaptation = context.agent_outputs.get("adaptation", {})
        feedback = []
        for topic, action in adaptation.items():
            if action == "review":
                feedback.append(f"Let's review {topic} together.")
            else:
                feedback.append(f"Great job on {topic}!")
        output = " ".join(feedback)
        context.update_history(output)
        context.record_output("FeedbackAgent", feedback)
        # Save feedback to memory
        context.update_memory("last_feedback", feedback)

class RootAgent:
    def __init__(self, agents):
        self.agents = agents

    def run(self, context):
        context.set_task("Starting root agent orchestration")
        context.update_history("RootAgent: Orchestration started.")
        context.start_session()
        for agent in self.agents:
            agent_name = agent.__class__.__name__
            context.update_history(f"RootAgent: Running {agent_name}.")
            agent.run(context)
            context.update_history(f"RootAgent: Completed {agent_name}.")
        context.set_task("Root agent orchestration completed")
        context.update_history("RootAgent: Orchestration finished.")
        context.record_output("RootAgent", "Workflow complete")
        context.end_session()

context = MCPContext(
    user_name="Alex",
    goals=["learn fractions", "improve geometry"],
    memory={"preferred_language": "English"}
)

agents = [
    LearningPathDesignerAgent(),
    AssessmentAgent(),
    ContentAdaptationAgent(),
    FeedbackAgent()
]

root_agent = RootAgent(agents)
root_agent.run(context)

print("Agent Outputs:")
for agent, output in context.agent_outputs.items():
    print(f"{agent}: {output}")

print("\nMemory (Persistent Learner Data):")
for key, value in context.memory.items():
    print(f"{key}: {value}")

print("\nSession History:")
for event in context.session_history:
    print("-", event)

print("\nSession Metadata:")
for key, value in context.session.items():
    print(f"{key}: {value}")


%run edu_agent/agent.py

