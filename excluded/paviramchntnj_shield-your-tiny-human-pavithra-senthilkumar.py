# -------------------- Importing required libraries -------------------

import os
import time
import json
import logging
from typing import Dict, Any
from typing import Literal
from google.genai import types

# ADK imports
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.agents import SequentialAgent
from google.adk.runners import InMemoryRunner

# -------------------- Kaggle secrets -------------------
from kaggle_secrets import UserSecretsClient

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("SafetyGuardian")

# ------------------- Gemini API Key -------------------
try:
    API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = API_KEY
    print("ğŸ”‘ Gemini API key loaded.")
except:
    print("â�Œ Add GOOGLE_API_KEY to Kaggle secrets!")
    API_KEY = None

MODEL_NAME = "gemini-2.5-flash"
print(f"Using model: {MODEL_NAME}")



# ------------------- Custom Tool Definition -------------------

def calculate_risk_level(confidence_score: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
    """
    Tool to convert a numerical confidence score (0.0 to 1.0) into a simple LOW, MEDIUM, or HIGH risk rating 
    based on fixed policy thresholds.
    
    Args:
        confidence_score: The LLM-determined confidence score of the risk, from 0.0 to 1.0.
        
    Returns:
        The final risk rating: "LOW", "MEDIUM", or "HIGH".
    """
    # ------------------- Policy Logic: This is the non-LLM, deterministic part of the tool. -------------------
    if confidence_score >= 0.8:
        return "HIGH"
    elif confidence_score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

# ------------------- The FunctionTool definition: All metadata is inferred from the function's signature and docstring. -------------------
RiskScoringTool = FunctionTool(
    calculate_risk_level # Only pass the function object
)

print("âœ… Custom Tool 'calculate_risk_level' defined.")


# ------------------- helper function definition -------------------

def show_python_code_and_result(response):
    for i in range(len(response)):
        # Check if the response contains a valid function call result from the code executor
        if (
            (response[i].content.parts)
            and (response[i].content.parts[0])
            and (response[i].content.parts[0].function_response)
            and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])


print("âœ… Helper functions defined.")


# ------------------- retry options configuration -------------------

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# ------------------- Preprocessor Agent: Cleans, normalizes, and structures the raw input message. -------------------

preprocessor_agent = Agent(
    name="PreprocessorAgent",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=retry_config
    ),
    
    instruction="""You are the Preprocessor Agent.\nClean and normalize the input text. "
                "Detect language and tag the context (e.g., gaming_chat, dm_chat). "
                "Output JSON with keys: clean_text, language, context_tag.""",
    tools=[google_search],
    output_key="preprocessed_data",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… PreprocessorAgent created.")


# ------------------- Risk Classifier Agent: Identifies the type of online safety risk present in the message. -------------------

risk_classifier_agent = Agent(
    name="RiskClassifierAgent",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=retry_config
    ),
    
    instruction="""Analyze the preprocessed_data: {preprocessed_data}
classify the content into one of:\n"
            "grooming, bullying, coercion, hate_speech, explicit_content, no_risk.\n\n"
            "Return ONLY JSON:\n"
            "- risk_category\n"
            "- confidence_score (0.0 to 1.0)\n\n""",
    output_key="risk_classified_data",
)

print("âœ… RiskClassfierAgent created.")


# ------------------- Risk Scorer Agent: Assigns a numerical risk severity score based on the detected threat. -------------------

risk_scorer_agent = Agent(
    name="RiskScorerAgent",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=retry_config
    ),
    
    instruction="""Use the risk_classified_data: {risk_classified_data}
Use the tool `calculate_risk_level` to convert the classifierâ€™s "
            "confidence_score into LOW, MEDIUM, or HIGH.\n\n"
            "Return ONLY JSON:\n"
            "- final_risk_level\n"
            "- justification\n\n""",
    output_key="risk_scored_data",
    tools=[calculate_risk_level],
)

print("âœ… RiskScorerAgent created.")


# ------------------- Summary Agent: Produces the final safety report combining all agent outputs. -------------------

summary_agent = Agent(
    name="SummaryAgent",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=retry_config
    ),
    
    instruction="""Analyze the risk_scored_data: {risk_scored_data}
You are the Summary Agent.\n\n"
            "Create a clear, parent-friendly, non-fear-driven summary of the risk.\n"
            "Include:\n"
            "- A simple safety report\n"
            "- Easy-to-follow reasoning\n"
            "- Practical next steps\n\n"
            "Return plain text only.\n\n""",
    output_key="final_summary",
)

print("âœ… SummaryAgent created.")


# ------------------- Sequential Agent: Executes all agents in a fixed pipeline to produce the final consolidated output. -------------------

root_agent = SequentialAgent(
    name="shield_your_tiny_human_child_online_safety_guardian",
    sub_agents=[preprocessor_agent, risk_classifier_agent, risk_scorer_agent, summary_agent],
)

print("âœ… Sequential Agent created.")


# Here we're using AgentTool to wrap the sub-agents to make them callable tools for the root agent.
# Example 1

runner = InMemoryRunner(agent=root_agent)

response = await runner.run_debug(
    "hey, add me on private chat. send me a pic. we can be alone. also you're stupid if you don't reply. gg"
)

print(response)


# Here we're using AgentTool to wrap the sub-agents to make them callable tools for the root agent.
# Example 2

runner = InMemoryRunner(agent=root_agent)

response = await runner.run_debug(
    "Reply fast or I'm gonna spam your whole system, idiot."
)

print(response)


# Here we're using AgentTool to wrap the sub-agents to make them callable tools for the root agent.
# Example 3

runner = InMemoryRunner(agent=root_agent)

response = await runner.run_debug(
    "OMG, this game is dead! The guy who plays this is trash. I'm deleting it. lol."
)

print(response)


# Set up Cloud Credentials in Kaggle

user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

print("âœ… Cloud credentials configured")


## Set your PROJECT_ID

PROJECT_ID = "gen-lang-client-0857112229"  # TODO: Replace with your project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

if PROJECT_ID == "your-project-id" or not PROJECT_ID:
    raise ValueError("âš ï¸� Please replace 'your-project-id' with your actual Google Cloud Project ID.")

print(f"âœ… Project ID set to: {PROJECT_ID}")


# Optional Deployment to Cloud Run

# NOTE: This step is optional and requires the 'adk-deploy' module

# ------------------- Deployment Configuration -------------------
GCP_REGION = "us-central1" # A standard region for Google Cloud services
SERVICE_NAME = "safety-guardian-workflow" 
# PROJECT_ID is already defined

# ------------------- Deployment Code -------------------
try:
    
    from google.adk.deploy import cloud_run 

    print(f"ğŸš€ Attempting deployment of {root_agent.name} to Cloud Run in {GCP_REGION}...")

    # The 'root_agent' object is packaged and deployed.
    deployment = cloud_run.deploy(
        root_agent,
        project_id=PROJECT_ID,
        region=GCP_REGION,
        service_name=SERVICE_NAME,
        # Recommended practice: enable tracing for production debugging
        enable_tracing=True, 
    )

    print("\nâœ… Deployment Command Successful!")
    print(f"Service URL (Target): {deployment.service_url}")
    print("\n**Next Steps:** Interact with your agent via this URL using the LlmAgentClient or HTTP requests.")

except ModuleNotFoundError:
    print("\nâš ï¸� DEPLOYMENT SKIPPED: 'google.adk.deploy' module not found.")
    print("This confirms the environment does not support cloud deployment tools.")
    print("The agent is correctly defined for local execution.")
except Exception as e:
    print("\nâ�Œ DEPLOYMENT FAILED.")
    print(f"Error details: {e}")
    print("Check your GCP credentials, project ID, and ensure Cloud Run API is enabled.")

