import os
from kaggle_secrets import UserSecretsClient

#from google.colab import userdata

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    #GOOGLE_API_KEY = userdata.get('GEMINI_API_KEY')
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


# for kaggle execution only

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style=\"padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;\">
        <div style=\"font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;\">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style=\"font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;\">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style=\"margin-top: 10px; padding-left: 20px;\">
                <li style=\"margin-bottom: 5px;\"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style=\"margin-bottom: 5px;\">Wait for that cell to show it is \"Running\" (it will not \"complete\").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style=\"font-size: 0.9em; color: #555;\">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style=\"\n
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;\n
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;\n
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;\">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


import os
import subprocess
import subprocess 
import shutil

unified_agent_code = '''
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

# --- Credit Review Agent Definition ---

class CreditWorthinessAgent:
    """Agent to assess credit worthiness."""

    def __init__(self):
        self.required_fields = ['salary_per_year', 'loan_amount', 'credit_score']
        self.file_path = '/kaggle/working/loan_applications.json'

    def execute(self, request: str) -> str:
        """
        Assesses the credit worthiness of a home loan applicant and updates the application status.
        """
        if not os.path.exists(self.file_path):
            return f"Error: Loan application file not found at {self.file_path}"

        with open(self.file_path, 'r+') as f:
            try:
                applications = json.load(f)
            except json.JSONDecodeError:
                # If the file is empty, there are no applications to process.
                applications = []

            application_found = False
            for app in applications:
                if app.get('loan_id') == request:
                    application_found = True
                    # Ensure loan_term is an integer for consistency, default to 0 if missing
                    app['loan_term'] = int(app.get('loan_term', 0))
                    
                    missing_fields = [f for f in self.required_fields if f not in app]
                    if missing_fields:
                        app['status'] = 'credit - info missing'
                        status_message = f"Missing required fields for credit assessment: {', '.join(missing_fields)}"
                    else:
                        try:
                            salary = float(app['salary_per_year'])
                            loan_amount = float(app['loan_amount'])
                            credit_score = int(app['credit_score'])

                            if credit_score < 600:
                                app['status'] = 'credit - rejected'
                                status_message = "Credit Rejected: Credit score is too low."
                            elif salary * 5 < loan_amount:
                                app['status'] = 'credit - rejected'
                                status_message = "Credit Rejected: Loan amount is too high. Please decrease loan amount."
                            else:
                                app['status'] = 'credit - approved'
                                status_message = "Credit Approved: Congratulations! Your loan is approved."

                        except (ValueError, TypeError):
                            app['status'] = 'credit - info missing'
                            status_message = "Invalid data types for credit assessment fields."
                    
                    break
            
            if not application_found:
                return f"Error: Loan application with ID '{request}' not found."

            f.seek(0)
            json.dump(applications, f, indent=2)
            f.truncate()

            return f"âœ… Assessment for loan {request} complete. status: {app['status']}. Reason: {status_message}"


# --- Loan Application Tool Definition ---

def save_loan_application(
    customer_name: str,
    salary_per_year: float,
    loan_amount: float,
    loan_term: int,
    home_address: str,
    city: str,
    state: str,
    ssn: str,
    credit_score: int
) -> dict:
    """
    Saves a new home loan application to a file after collecting all necessary information.
    If info is missing, it returns a message asking for the missing fields.
    On success, it returns a dictionary containing the loan_id.
    """
    application_id = f"{re.sub(r'[^a-zA-Z0-9]+', '-', customer_name).lower()}-{datetime.now(timezone.utc).isoformat().split('.')[0].replace('T', '').replace('-', '').replace(':', '')}"
    application_data = {
        'loan_id': application_id,
        'customer_name': customer_name,
        'salary_per_year': salary_per_year,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'home_address': home_address,
        'city': city,
        'state': state,
        'ssn': ssn,
        'credit_score': credit_score
    }

    file_path = '/kaggle/working/loan_applications.json'
    applications = []
    if os.path.exists(file_path):
        # Only try to load if the file is not empty
        if os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as f:
                try:
                    applications = json.load(f)
                except json.JSONDecodeError:
                    # Handle cases where the file is corrupted or not valid JSON
                    applications = []

    application_data['status'] = 'Submitted'

    applications.append(application_data)

    with open(file_path, 'w') as f:
        json.dump(applications, f, indent=2)

    return {'loan_id': application_id}


# --- Agent Definitions ---

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# Credit Worthiness Agent
credit_agent_instance = CreditWorthinessAgent()
credit_worthiness_agent = LlmAgent(
    name="credit_worthiness_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="An expert credit analyst that assesses an application given a loan_id.",
    instruction="""You are a credit worthiness analyst. Your only job is to assess the credit worthiness of a home loan applicant when you are given a loan_id.
    To do this, you must call the `execute` tool with the `request` you receive from the user or previous agent.
    For example, if the user says 'assess credit for loan_id xyz-123', you must call the `execute` tool with `request='xyz-123'`.""",
    tools=[credit_agent_instance.execute]
)

# Home Loan Agent
home_loan_agent = LlmAgent(
    name="home_loan_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="A friendly agent that collects user information for a home loan application.",
    instruction="""You are a friendly and professional home loan specialist. Your only job is to collect information for a home loan application.
**Your process is a strict, state-driven conversation. Do not deviate from the questioning order.**

**STATE 1: GREET & COLLECT INITIAL INFO**
- Greet the user, introduce yourself, and ask for their full name and annual salary.
- Once you have this, move to STATE 2.

**STATE 2: COLLECT LOAN DETAILS**
- Ask the user for the loan amount and the loan term in months.
- Once you have this, move to STATE 3.

**STATE 3: COLLECT ADDRESS**
- Ask the user for their full home address, including city and state.
- Once you have this, move to STATE 4.

**STATE 4: COLLECT FINAL DETAILS**
- Ask the user for their Social Security Number (SSN) and their credit score.
- **You are forbidden from calling any tools until you have received a response in this state.**
- Once you have this information, and only then, move to STATE 5.

**STATE 5: EXECUTE & RESPOND**
- In this state, you have all 9 pieces of information. You will now perform the following sequence of actions in a single turn:
    a. **Verification:** Internally confirm you have all 9 fields: customer name, annual salary, loan amount, loan term, home address, city, state, SSN, and credit score. If any are missing, go back to the appropriate state and ask for the data.
    b. **Tool Call 1 (Save):** Call the `save_loan_application` tool with all 9 pieces of information.
    - **IMPORTANT**: If the user gives the loan term in years (e.g., "30 years"), you MUST convert it to months (e.g., 360) before calling the tool.
    c. **Tool Call 2 (Assess):** Extract the `loan_id` from the result of the save tool and immediately call the `credit_worthiness_agent` tool, passing the `loan_id` to it.
    d. **Final Response:** Take the final assessment from the credit agent and present it to the user as a text message. This message MUST include both the `loan_id` and the full assessment result.

**GLOBAL CONSTRAINT:** Strictly handle home loans only. If the user asks for any other loan type, politely decline.
""", 
    tools=[save_loan_application, AgentTool(credit_worthiness_agent)]
)

# --- Root Agent ---
# The home_loan_agent is the main entry point and controls the entire user-facing process.
root_agent = home_loan_agent
'''

# --- Create Unified Loan Processing Agent ---
agent_name = "loan_processing_agent" 
agent_dir = f"/kaggle/working/{agent_name}"

# If the directory already exists, remove it
if os.path.isdir(agent_dir):
    print(f"Directory '{agent_dir}' already exists. Removing it.")
    shutil.rmtree(agent_dir)
    
# Step 1: Run the adk create command
# GOOGLE_API_KEY needs to be defined in the environment
create_command = f"adk create {agent_name} --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY"
result = subprocess.run(create_command, shell=True, capture_output=True, text=True, check=False)

# Print the output from the command
print(f"--- adk create {agent_name} stdout ---")
print(result.stdout)
print(f"--- adk create {agent_name} stderr ---")
print(result.stderr)

# Step 2: Write the agent.py file
if os.path.isdir(agent_dir):
    print(f"\nUnified Loan Processing Agent directory for {agent_name} exists. Writing to agent.py...")
    agent_file_path = os.path.join(agent_dir, "agent.py")
    try:
        with open(agent_file_path, 'w') as f:
            f.write(unified_agent_code)
        print(f"âœ… Successfully updated {agent_file_path}")
    except IOError as e:
        print(f"â�Œ Error writing to {agent_file_path}: {e}")
else:
    print(f"\nUnified Loan Processing Agent creation for {agent_name} may have failed. The agent.py file was not written.")
    


#for kaggle only

try:
    url_prefix = get_adk_proxy_url()
except Exception as e:
    url_prefix = None
    print(f"âš ï¸� Could not obtain ADK proxy URL in this execution environment: {e}")
    print("If running interactively in Kaggle, run the helper cell manually to display the ADK Web UI button.")


if url_prefix:
    print("ADK Web command (run interactively and keep running):")
    print(f"!adk web --log_level DEBUG --url_prefix {url_prefix}")
    print("\nTo open the UI, run the above command in a notebook cell (uncomment to execute). Keep that cell running and then click the ADK Web UI button displayed by the helper.")
else:
    print("âš ï¸� Skipping ADK Web server cell: no ADK proxy URL available in this execution.")
    print("Run the helper cell interactively (get_adk_proxy_url()) to display the ADK Web UI button and obtain the url_prefix.")


# Minimal automated eval + runner for the 'happy path'
import json
import os
import subprocess

evalset = {
    "eval_set_id": "automated_happy_path_single",
    "name": "Automated Happy Path",
    "description": "Single automated conversation to exercise loan_processing_agent",
    "eval_cases": [
        {
            "eval_id": "automated_happy_path",
            "conversation": [
                {   "user_content": {"parts": [{"text": "hi"}], "role": "user"},
                    "final_response": { "parts": [{ "text": "Hello! I'm your home loan specialist. To get started, please tell me your full name and your annual salary." }], "role": "model" }    
                },
                {   "user_content": {"parts": [{"text": "My name is KP and my annual salary is 200000"}], "role": "user"},
                    "final_response": { "parts": [{ "text": "Thank you, Bad Credit Bob. Now, could you please tell me the loan amount you are requesting and the loan term in months?" }], "role": "model" }
                },
                {
                    "user_content": {"parts": [{"text": "loan amount of 500000 and loan term for 30 years"}], "role": "user"},
                    "final_response": { "parts": [{ "text": "Next, please provide your full home address, including the city and state." }], "role": "model" }
                },
                {   "user_content": {"parts": [{"text": "678 turkey roost rd, fort mill, sc"}], "role": "user"},
                    "final_response": { "parts": [{ "text": "Finally, please provide your Social Security Number (SSN) and your credit score." }], "role": "model" }
                },
                {   "user_content": {"parts": [{"text": "SSN is 890-00-0909 and my credit score is 780"}], "role": "user"},
                    "final_response": { "parts": [{ "text": "Your loan application (kp-20251127173904) has been approved." }], "role": "model" }
                }
            ],
            "session_input": {"app_name": "loan_processing_agent", "user_id": "automated_test"}
        }
    ]
}

agent_dir = "/kaggle/working/loan_processing_agent"
eval_file_path = os.path.join(agent_dir, "example.evalset.json")
config_path = os.path.join(agent_dir, "test_config.json")

os.makedirs(agent_dir, exist_ok=True)

with open(eval_file_path, "w") as f:
    json.dump(evalset, f, indent=2)
print(f"Wrote evalset to: {eval_file_path}")

if not os.path.exists(config_path):
    minimal_config = {"criteria": {"tool_trajectory_avg_score": 1, "response_match_score": 0.5}}
    with open(config_path, "w") as f:
        json.dump(minimal_config, f, indent=2)
    print(f"Created minimal config at: {config_path}")
else:
    print(f"Using existing config at: {config_path}")

cmd = ["adk", "eval", "loan_processing_agent", eval_file_path, f"--config_file_path={config_path}", "--print_detailed_results"]

print("Running adk eval (this may take a while)...")
proc = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
print("--- STDOUT ---")
print(proc.stdout)
print("--- STDERR ---")
print(proc.stderr)
if proc.returncode != 0:
    print(f"adk eval exited with code {proc.returncode}")
else:
    print("adk eval completed successfully.")


import json

# Create evaluation configuration with basic criteria
eval_config = {
    "criteria": {
        "tool_trajectory_avg_score": 1,  # Perfect tool usage required
        "response_match_score": 0.5,  # 80% text similarity threshold
    }
}

with open("loan_processing_agent/test_config.json", "w") as f:
    json.dump(eval_config, f, indent=2)

print("âœ… Evaluation configuration created!")
print("\nğŸ“Š Evaluation Criteria:")
print("â€¢ tool_trajectory_avg_score: 1.0 - Requires exact tool usage match")
print("â€¢ response_match_score: 0.6 - Requires 80% text similarity")
print("\nğŸ�¯ What this evaluation will catch:")
print("âœ… Incorrect tool usage")
print("âœ… Poor response quality and communication")
print("âœ… Deviations from expected behavior patterns")


test_cases = {
  "eval_set_id": "happy_path_eval_case_generated_from_adk",
  "name": "Happy Path Evaluation",
  "description": "Tests agents ability to collect information for home loan and perform credit assessment",
  "eval_cases": [
    {
      "eval_id": "happy_path_1",
      "conversation": [
        {
          "invocation_id": "e-92893499-a8f4-40e4-9a87-7a7cab3e60c0",
          "user_content": {
            "parts": [
              {
                "text": "hi"
              }
            ],
            "role": "user"
          },
          "final_response": {
            "parts": [
              {
                "text": "Hello! I'm your home loan specialist. To get started, please tell me your full name and your annual salary."
              }
            ],
            "role": "model"
          },
          "intermediate_data": {},
          "creation_timestamp": 1764265101.772898
        },
        {
          "invocation_id": "e-79546b64-42f1-4f07-8f16-a754f8226175",
          "user_content": {
            "parts": [
              {
                "text": "my name is KP and my annual salary is 200k"
              }
            ],
            "role": "user"
          },
          "final_response": {
            "parts": [
              {
                "text": "Thank you, KP. \n\nNow, could you please tell me the loan amount you are requesting and the loan term in months?"
              }
            ],
            "role": "model"
          },
          "intermediate_data": {},
          "creation_timestamp": 1764265109.351708
        },
        {
          "invocation_id": "e-aeac348c-2f4b-42bc-9928-09cf2978e9e4",
          "user_content": {
            "parts": [
              {
                "text": "loan amount of 500k and loan term for 30 years"
              }
            ],
            "role": "user"
          },
          "final_response": {
            "parts": [
              {
                "text": "Next, please provide your full home address, including the city and state."
              }
            ],
            "role": "model"
          },
          "intermediate_data": {},
          "creation_timestamp": 1764265118.281027
        },
        {
          "invocation_id": "e-1cbfabf3-7eee-4bc3-a882-e65bb4935cc0",
          "user_content": {
            "parts": [
              {
                "text": "678 turkey roost rd, fort mill, sc"
              }
            ],
            "role": "user"
          },
          "final_response": {
            "parts": [
              {
                "text": "Finally, please provide your Social Security Number (SSN) and your credit score."
              }
            ],
            "role": "model"
          },
          "intermediate_data": {},
          "creation_timestamp": 1764265131.393957
        },
        {
          "invocation_id": "e-762939fc-2bee-49b7-a029-f7e375807453",
          "user_content": {
            "parts": [
              {
                "text": "SSN is 890-00-0909 and my credit score is 780"
              }
            ],
            "role": "user"
          },
          "final_response": {
            "parts": [
              {
                "text": "Your loan application (kp-20251127173904) has been approved."
              }
            ],
            "role": "model"
          },
          "intermediate_data": {
            "invocation_events": [
              {
                "author": "home_loan_agent",
                "content": {
                  "parts": [
                    {
                      "function_call": {
                        "id": "adk-6e0bc094-3e1e-4bd8-9dba-acc3f0e4472f",
                        "args": {
                          "city": "fort mill",
                          "loan_amount": 500000,
                          "salary_per_year": 200000,
                          "ssn": "890-00-0909",
                          "state": "sc",
                          "home_address": "678 turkey roost rd",
                          "credit_score": 780,
                          "loan_term": 360,
                          "customer_name": "KP"
                        },
                        "name": "save_loan_application"
                      }
                    }
                  ],
                  "role": "model"
                }
              },
              {
                "author": "home_loan_agent",
                "content": {
                  "parts": [
                    {
                      "function_response": {
                        "id": "adk-6e0bc094-3e1e-4bd8-9dba-acc3f0e4472f",
                        "name": "save_loan_application",
                        "response": {
                          "loan_id": "kp-20251127173904"
                        }
                      }
                    }
                  ],
                  "role": "user"
                }
              },
              {
                "author": "home_loan_agent",
                "content": {
                  "parts": [
                    {
                      "function_call": {
                        "id": "adk-3fd9aaee-983f-4175-8bdc-ab4e4717f5eb",
                        "args": {
                          "request": "kp-20251127173904"
                        },
                        "name": "credit_worthiness_agent"
                      }
                    }
                  ],
                  "role": "model"
                }
              },
              {
                "author": "home_loan_agent",
                "content": {
                  "parts": [
                    {
                      "function_response": {
                        "id": "adk-3fd9aaee-983f-4175-8bdc-ab4e4717f5eb",
                        "name": "credit_worthiness_agent",
                        "response": {
                          "result": "The credit worthiness of the loan application kp-20251127173904 has been assessed and it is approved."
                        }
                      }
                    }
                  ],
                  "role": "user"
                }
              }
            ]
          },
          "creation_timestamp": 1764265143.015724
        }
      ],
      "session_input": {
        "app_name": "loan_processing_agent",
        "user_id": "user"
      },
      "creation_timestamp": 1764265165.3900979
    }
    ,
    {
      "eval_id": "rejection_low_credit_score",
      "conversation": [
        {
          "user_content": { "parts": [{ "text": "hi" }], "role": "user" },
          "final_response": { "parts": [{ "text": "Hello! I'm your home loan specialist. To get started, please tell me your full name and your annual salary." }], "role": "model" }
        },
        {
          "user_content": { "parts": [{ "text": "My name is Bad Credit Bob and my salary is $80,000" }], "role": "user" },
          "final_response": { "parts": [{ "text": "Thank you, Bad Credit Bob. Now, could you please tell me the loan amount you are requesting and the loan term in months?" }], "role": "model" }
        },
        {
          "user_content": { "parts": [{ "text": "I need $300,000 for 30 years" }], "role": "user" },
          "final_response": { "parts": [{ "text": "Next, please provide your full home address, including the city and state." }], "role": "model" }
        },
        {
          "user_content": { "parts": [{ "text": "123 Main St, Anytown, USA" }], "role": "user" },
          "final_response": { "parts": [{ "text": "Finally, please provide your Social Security Number (SSN) and your credit score." }], "role": "model" }
        },
        {
          "user_content": { "parts": [{ "text": "My SSN is 111-22-3333 and my credit score is 550" }], "role": "user" },
          "final_response": { "parts": [{ "text": "âœ… Assessment for loan bad-credit-bob-20251127221500 complete. status: credit - rejected. Reason: Credit Rejected: Credit score is too low." }], "role": "model" },
          "intermediate_data": {
            "invocation_events": [
              {
                "author": "home_loan_agent",
                "content": { "parts": [{ "function_call": { "name": "save_loan_application", "args": { "customer_name": "Bad Credit Bob", "salary_per_year": 80000, "loan_amount": 300000, "loan_term": 360, "home_address": "123 Main St", "city": "Anytown", "state": "USA", "ssn": "111-22-3333", "credit_score": 550 } } }], "role": "model" }
              },
              {
                "author": "home_loan_agent",
                "content": { "parts": [{ "function_response": { "name": "save_loan_application", "response": { "loan_id": "bad-credit-bob-20251127221500" } } }], "role": "user" }
              },
              {
                "author": "home_loan_agent",
                "content": { "parts": [{ "function_call": { "name": "credit_worthiness_agent", "args": { "request": "bad-credit-bob-20251127221500" } } }], "role": "model" }
              },
              {
                "author": "home_loan_agent",
                "content": { "parts": [{ "function_response": { "name": "credit_worthiness_agent", "response": { "result": "âœ… Assessment for loan bad-credit-bob-20251127221500 complete. status: credit - rejected. Reason: Credit Rejected: Credit score is too low." } } }], "role": "user" }
              }
            ]
          }
        }
      ],
      "session_input": { "app_name": "loan_processing_agent", "user_id": "user" }
    },
    {
      "eval_id": "rejection_high_loan_amount",
      "conversation": [
        {
          "user_content": { "parts": [{ "text": "I want to apply for a home loan" }], "role": "user" },
          "final_response": { "parts": [{ "text": "Hello! I'm your home loan specialist. To get started, please tell me your full name and your annual salary." }], "role": "model" }
        },
        {
          "user_content": { "parts": [{ "text": "I am Penny Pincher and I make 60000 a year" }], "role": "user" },
          "final_response": { "parts": [{ "text": "Thank you, Penny Pincher. Now, could you please tell me the loan amount you are requesting and the loan term in months?" }], "role": "model" }
        },
        {
          "user_content": { "parts": [{ "text": "I'd like to borrow $400,000 for 360 months" }], "role": "user" },
          "final_response": { "parts": [{ "text": "Next, please provide your full home address, including the city and state." }], "role": "model" }
        },
        {
          "user_content": { "parts": [{ "text": "456 Oak Avenue, Smallville, USA" }], "role": "user" },
          "final_response": { "parts": [{ "text": "Finally, please provide your Social Security Number (SSN) and your credit score." }], "role": "model" }
        },
        {
          "user_content": { "parts": [{ "text": "SSN is 444-55-6666 and credit score is 720" }], "role": "user" },
          "final_response": { "parts": [{ "text": "âœ… Assessment for loan penny-pincher-20251127221600 complete. status: credit - rejected. Reason: Credit Rejected: Loan amount is too high. Please decrease loan amount." }], "role": "model" },
          "intermediate_data": {
            "invocation_events": [
              {
                "author": "home_loan_agent",
                "content": { "parts": [{ "function_call": { "name": "save_loan_application", "args": { "customer_name": "Penny Pincher", "salary_per_year": 60000, "loan_amount": 400000, "loan_term": 360, "home_address": "456 Oak Avenue", "city": "Smallville", "state": "USA", "ssn": "444-55-6666", "credit_score": 720 } } }], "role": "model" }
              },
              {
                "author": "home_loan_agent",
                "content": { "parts": [{ "function_response": { "name": "save_loan_application", "response": { "loan_id": "penny-pincher-20251127221600" } } }], "role": "user" }
              },
              {
                "author": "home_loan_agent",
                "content": { "parts": [{ "function_call": { "name": "credit_worthiness_agent", "args": { "request": "penny-pincher-20251127221600" } } }], "role": "model" }
              },
              {
                "author": "home_loan_agent",
                "content": { "parts": [{ "function_response": { "name": "credit_worthiness_agent", "response": { "result": "âœ… Assessment for loan penny-pincher-20251127221600 complete. status: credit - rejected. Reason: Credit Rejected: Loan amount is too high. Please decrease loan amount." } } }], "role": "user" }
              }
            ]
          }
        }
      ],
      "session_input": { "app_name": "loan_processing_agent", "user_id": "user" }
    },
    {
      "eval_id": "decline_other_loan_type",
      "conversation": [
        {
          "user_content": { "parts": [{ "text": "Can I get a car loan?" }], "role": "user" },
          "final_response": { "parts": [{ "text": "I'm sorry, but I can only assist with home loan applications. Is there anything else that I can help you with?" }], "role": "model" },
          "intermediate_data": {}
        }
      ],
      "session_input": { "app_name": "loan_processing_agent", "user_id": "user" }
    }
  ],
  "creation_timestamp": 1764265159.9523897
}


import json

with open("/kaggle/working/loan_processing_agent/integration.evalset.json", "w") as f:
    json.dump(test_cases, f, indent=2)

print("âœ… Evaluation test cases created")
print("\nğŸ§ª Test scenarios:")
for case in test_cases["eval_cases"]:
    user_msg = case["conversation"][0]["user_content"]["parts"][0]["text"]
    print(f"â€¢ {case['eval_id']}: {user_msg}")

print("\nğŸ“Š Expected results:")
print("â€¢ home loan test case: Should pass both criteria")
print(
    "â€¢ wrong_tool_usage_test: May fail tool_trajectory if agent uses wrong parameters"
)
print(
    "â€¢ poor_response_quality_test: May fail response_match if response differs too much"
)


print("ğŸš€ Run this command to execute evaluation:")
!adk eval loan_processing_agent loan_processing_agent/integration.evalset.json --config_file_path=loan_processing_agent/test_config.json --print_detailed_results

