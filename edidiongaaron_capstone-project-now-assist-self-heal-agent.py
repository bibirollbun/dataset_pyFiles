!pip install --upgrade google-genai


import json
import os
from typing import List
from pydantic import BaseModel, Field

# --- Kaggle Secrets Import ---
from kaggle_secrets import UserSecretsClient

# --- Google Gemini/ADK Imports ---
from google import genai
from google.genai import types

# --- API KEY INITIALIZATION (Kaggle Best Practice) ---

try:
    # 1. Instantiate the client for Kaggle secrets
    user_secrets = UserSecretsClient()
    
    # 2. Retrieve the secret value using your key name
    google_api_key = user_secrets.get_secret("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError("Secret 'GOOGLE_API_KEY' found, but its value is empty.")

    # 3. Initialize the Gemini client
    client = genai.Client(api_key=google_api_key)
    print("âœ… Gemini Client successfully initialized using Kaggle UserSecretsClient.")

except Exception as e:
    # Catch any error related to the secrets retrieval or client initialization
    print(f"ğŸ›‘ Error during client initialization: {e}")
    # Raise a final error to stop execution if the client isn't ready
    raise RuntimeError("Cannot proceed. Please verify your Kaggle Secret 'GOOGLE_API_KEY'.")


# --- 1. Tool for TriageBot (Simulates RAG/Long-Term Memory) ---
PAST_INCIDENTS_DB = {
    "db_connection": ["Incident 1001: DB Connection Pool Exhausted. Resolution: Restarted DB service and increased pool size. Root Cause: High traffic spike."],
    "server_disk": ["Incident 2003: Server X disk space low. Resolution: Cleared /tmp and recycled logs. Root Cause: Log rotation failure."]
}

def retrieve_past_incidents(issue_keywords: str) -> List[str]:
    """Analyzes issue_keywords and queries historical incidents for context."""
    keywords = issue_keywords.lower().split()
    results = []
    if any(k in keywords for k in ["db", "connection", "pool", "timeout"]):
        results.extend(PAST_INCIDENTS_DB.get("db_connection", []))
    if any(k in keywords for k in ["server", "disk", "low", "space"]):
        results.extend(PAST_INCIDENTS_DB.get("server_disk", []))
    return results

# --- 2. Tool for InvestigatorBot (Simulates Live System Check) ---
def check_system_status(ci_name: str) -> str:
    """Simulates checking the live operational status of a Configuration Item (CI)."""
    if "payment_db" in ci_name.lower():
        return "Status: Degraded. High CPU usage, 95% connection pool utilization."
    else:
        return "Status: Normal. Low utilization."

# --- 3. Tool for ResolutionBot (Simulates Execution) ---
def execute_remediation_action(ci_name: str, action: str) -> str:
    """Simulates executing a single step of the remediation plan on the target CI."""
    print(f"\n[Executing Action on {ci_name}]: {action}...")
    if "increase connection pool" in action.lower() or "cleared /tmp" in action.lower():
        return "SUCCESS: Action executed. Status check required."
    else:
        return f"FAILURE: Action '{action}' failed to execute."


# --- Triage Bot Output (A2A Artifact 1) ---
class TriageOutput(BaseModel):
    ci_name: str = Field(description="The name of the Configuration Item (e.g., 'Payment_DB').")
    issue_summary: str = Field(description="A concise summary of the core issue.")
    severity: str = Field(description="The assigned severity of the incident: 'Critical', 'High', 'Medium', or 'Low'.")
    search_keywords: str = Field(description="Keywords used to retrieve past incident memory.")
    past_incident_context: str = Field(description="The formatted results returned from the retrieve_past_incidents tool.")

# --- Investigator Bot Output (A2A Artifact 2) ---
class InvestigationOutput(BaseModel):
    diagnosis: str = Field(description="A clear, concise root cause diagnosis.")
    remediation_plan: List[str] = Field(description="A detailed, numbered list of actionable steps for the Resolution Bot to execute.")
    system_status_check: str = Field(description="The output from the 'check_system_status' tool.")

# --- Resolution Bot Output (Final Outcome) ---
class ResolutionOutput(BaseModel):
    final_status: str = Field(description="The final status: 'Resolved', 'Needs Manual Review', or 'Failed'.")
    execution_log: str = Field(description="A concatenated summary of all tool outputs (SUCCESS/FAILURE messages).")
    notes: str = Field(description="Notes on the outcome and verification steps.")


# ---run_triage_bot Function ---

def run_triage_bot(alert_message: str) -> str:
    """Runs the Triage Bot and returns the raw JSON string output."""
    response = client.models.generate_content(
       model='gemini-2.5-flash',# Change this line in all four functions:
        contents=[TRIAGE_AGENT_PROMPT, f"ALERT: {alert_message}"],
        config=types.GenerateContentConfig(
            tools=Triage_Tool,
            response_schema=TriageOutput,
            system_instruction="Expert IT incident classifier generating structured JSON output.",
        )
    )
    return response.text


# ---Triage Bot Prompt and Tool ---

TRIAGE_AGENT_PROMPT = """
You are the expert 'Triage Agent'. Your task is to receive a raw IT alert, classify it, 
and gather relevant historical context using the 'retrieve_past_incidents' tool.
CRITICALLY: Respond ONLY with a JSON object adhering to the 'TriageOutput' schema.
"""

Triage_Tool = [retrieve_past_incidents]




def run_investigator_bot(triage_json_output: str) -> str:
    """Runs the Investigator Bot and returns the raw JSON string output."""
    response = client.models.generate_content(
       model='gemini-2.5-flash',
        contents=[INVESTIGATOR_BOT_PROMPT, f"TRIAGE ARTIFACT: \n {triage_json_output}"],
        config=types.GenerateContentConfig(
            tools=Investigator_Tool,
            response_schema=InvestigationOutput,
            system_instruction="Expert IT troubleshooter, skilled at diagnostics and formulating executable plans.",
            
        )
    )
    return response.text




INVESTIGATOR_BOT_PROMPT = """
You are the expert 'Investigator Bot'. Your task is to take a structured 'TriageOutput' JSON artifact, 
diagnose the problem, and formulate a technical remediation plan.
1. Use the 'check_system_status' tool on the identified 'ci_name' to confirm status.
2. Formulate an actionable remediation plan based on all context.
CRITICALLY: Respond ONLY with a JSON object adhering to the 'InvestigationOutput' schema.
"""
Investigator_Tool = [check_system_status]



def run_resolution_bot(investigation_json_output: str) -> str:
    """Runs the Resolution Bot to execute plan."""
    response = client.models.generate_content(
     model='gemini-2.5-flash',
        contents=[RESOLUTION_BOT_PROMPT, f"INVESTIGATION ARTIFACT: \n {investigation_json_output}"],
        config=types.GenerateContentConfig(
            tools=Resolution_Tool,
            response_schema=ResolutionOutput,
            system_instruction="Expert IT execution agent that runs remediation tools and logs outcomes.",
        )
    )
    return response.text


RESOLUTION_BOT_PROMPT = """
You are the expert 'Resolution Bot' in the IT Self-Healing Pipeline.
Your task is to take the structured 'InvestigationOutput' JSON artifact and execute the remediation plan step-by-step.

STEPS:
1. **PARSE:** Parse the 'InvestigationOutput' JSON and extract the list from the 'remediation_plan' field.
2. **EXECUTE:** For EACH step in the 'remediation_plan', you MUST call the 'execute_remediation_action' tool.
3. **LOG:** Collect the text output (SUCCESS/FAILURE messages) of *every tool call* and **concatenate them** into a single string for the 'execution_log' field.
4. **FINALIZE:**
    - Set 'final_status': 'Resolved' if all steps were successful, otherwise 'Needs Manual Review'.
    - Populate the 'notes' field with a brief summary of why the status was set.
5. **CRITICALLY:** You must respond ONLY with a JSON object that strictly adheres to the 'ResolutionOutput' schema (final_status, execution_log, and notes). Do not output the raw tool call structure.
"""


Resolution_Tool = [execute_remediation_action]




# --- Test Alert ---
test_alert = "Critical Alert: DB connection pool exhausted on the Payments_DB server. Users are reporting 503 errors on checkout page."

print("--- STARTING ENTERPRISE AIOPS AGENT PIPELINE ---")
print(f"Initial Alert: {test_alert}\n")

# 1. Run Triage Bot
print("1. TriageBot: Classifying and retrieving memory...")
triage_json = run_triage_bot(test_alert)
print("âœ… Triage Output Captured.")

# 2. Run Investigator Bot (A2A Handoff)
print("\n2. InvestigatorBot: Diagnosing and creating remediation plan...")
investigation_json = run_investigator_bot(triage_json)
print("âœ… Investigation Plan Created.")

# 3. Run Resolution Bot (Execution Phase)
print("\n3. ResolutionBot: Executing remediation actions...")
resolution_json = run_resolution_bot(investigation_json)
print("âœ… Execution Complete.")

# --- FINAL OUTPUT SUMMARY AND PARSING (FIXED) ---
print("\n--- FINAL OUTCOME ---")
print("Full Resolution JSON:\n", resolution_json)

# --- CORRECTED JSON PARSING LOGIC ---
raw_json_string = resolution_json

# 1. Strip the Markdown fences (```json...```) that the model sometimes includes
if raw_json_string.strip().startswith("```json"):
    # Strip '```json' from the start and '```' from the end
    clean_json_string = raw_json_string.strip().lstrip("```json").rstrip("```").strip()
else:
    # If no fences, use the raw string
    clean_json_string = raw_json_string

# 2. Validate the cleaned JSON using the Pydantic model
try:
    final_status = ResolutionOutput.model_validate_json(clean_json_string)
    
    print("\n[Parsed Final Status]:", final_status.final_status)
    print("[Execution Log Summary]:", final_status.execution_log)

except Exception as e:
    # Fallback in case of unexpected JSON structure
    print(f"\nğŸ›‘ Error during final JSON validation after cleaning: {e}")
    print("\nRaw string that failed validation:\n", clean_json_string)
# --- END OF CORRECTED BLOCK ---


# --- Evaluation Bot's Output Schema ---
class EvaluationScore(BaseModel):
    """Structured data defining the performance score of the agent pipeline."""
    correctness_score: int = Field(description="Score (1-10) for diagnosis correctness based on the initial alert and system check. 10 is perfect.")
    completeness_score: int = Field(description="Score (1-10) for the remediation plan's completeness and actionability.")
    fidelity_score: int = Field(description="Score (1-10) for the final resolution log's success rate and adherence to the plan.")
    final_assessment: str = Field(description="A concise summary of the pipeline's overall success or failure.")

# No tool is needed, as the agent judges based on provided context.


EVALUATION_BOT_PROMPT = """
You are the expert 'Evaluation Bot' and act as an LLM-as-a-Judge for the IT Self-Healing Pipeline.
Your task is to review the initial incident and the final resolution artifact, then score the pipeline's performance.

INPUT CONTEXT: You will receive the ORIGINAL ALERT, the INVESTIGATION ARTIFACT, and the RESOLUTION ARTIFACT.
SCORING CRITERIA (1-10 scale):
1. Correctness: Did the diagnosis match the alert and system check?
2. Completeness: Was the remediation plan fully actionable and thorough?
3. Fidelity: Did the execution logs show a successful resolution?

CRITICALLY: Respond ONLY with a JSON object adhering to the 'EvaluationScore' schema.
"""

def run_evaluation_bot(original_alert: str, investigation_json: str, resolution_json: str) -> str:
    """Runs the Evaluation Bot to score the entire pipeline."""
    
    evaluation_contents = [
        EVALUATION_BOT_PROMPT,
        f"1. ORIGINAL ALERT: {original_alert}",
        f"2. INVESTIGATION ARTIFACT: {investigation_json}",
        f"3. RESOLUTION ARTIFACT: {resolution_json}"
    ]
    
    response = client.models.generate_content(
       model='gemini-2.5-flash', # Or a larger model like gemini-2.5-pro for better reasoning
        contents=evaluation_contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvaluationScore,
            system_instruction="You are a strict and objective scorer using the LLM-as-a-Judge methodology."
        )
    )
    return response.text


# NOTE: This assumes 'test_alert', 'investigation_json', and 'resolution_json'
# were successfully defined in the previous cell runs.

print("\n--- 4. EvaluationBot: Assessing pipeline performance (LLM-as-a-Judge)... ---")
try:
    evaluation_score_json = run_evaluation_bot(test_alert, investigation_json, resolution_json)
    print("âœ… Evaluation Complete.")

    # --- FINAL PROJECT SUMMARY ---
    print("\n--- END OF ENTERPRISE AIOPS AGENT PIPELINE ---")
    print("\n[Evaluation Scores (JSON)]: \n", evaluation_score_json)

    # Clean the JSON output just in case the Evaluation Bot also uses fences
    raw_eval_string = evaluation_score_json
    if raw_eval_string.strip().startswith("```json"):
        clean_eval_string = raw_eval_string.strip().lstrip("```json").rstrip("```").strip()
    else:
        clean_eval_string = raw_eval_string

    final_evaluation = EvaluationScore.model_validate_json(clean_eval_string)
    
    print("\n--- PARSED EVALUATION ---")
    print(f"Final Assessment: **{final_evaluation.final_assessment}**")
    print(f"Correctness Score: {final_evaluation.correctness_score}/10")
    print(f"Completeness Score: {final_evaluation.completeness_score}/10")
    print(f"Fidelity Score: {final_evaluation.fidelity_score}/10")

except NameError:
    print("\nğŸ›‘ NameError: The 'run_evaluation_bot' function or key input variables are not defined. Please ensure you've run the cells defining the Evaluation Bot's schema and function.")
except Exception as e:
    print(f"\nğŸ›‘ An unexpected error occurred during the Evaluation Bot run: {e}")

