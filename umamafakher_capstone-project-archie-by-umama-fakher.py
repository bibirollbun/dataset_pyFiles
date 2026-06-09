import os
import base64
import random
import time
import vertexai
from kaggle_secrets import UserSecretsClient
from IPython.display import Image, display, Markdown
from vertexai import agent_engines


print("âœ… Imports completed successfully")

# Set up Cloud Credentials in Kaggle
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

print("âœ… Cloud credentials configured")

# 1. Authenticate (Make sure you added 'GOOGLE_API_KEY' in Add-ons > Secrets)
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"â�Œ Error: {e}. Please add your GOOGLE_API_KEY in the Secrets menu.")

# 2. Define a Helper to Render Mermaid Diagrams
# This allows us to see the Architecture visually!
print("âœ… Defined Mermaid Helper function to render generated diagrams.")
def render_mermaid(mermaid_code):
    """Renders a Mermaid.js diagram using the mermaid.ink service."""
    try:
        # Clean up code block markers if present
        clean_code = mermaid_code.replace("```mermaid", "").replace("```", "").strip()
        graphbytes = clean_code.encode("utf8")
        base64_bytes = base64.urlsafe_b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        url = "https://mermaid.ink/img/" + base64_string
        display(Image(url=url))
    except Exception as e:
        print(f"Could not render diagram: {e}")


import base64
from IPython.display import Image, display
from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

# --- Configuration ---
model_config = Gemini(model="gemini-2.5-flash-lite")

# --- Concept 1: Tools ---

# Tool A: The Loop Breaker
def exit_loop():
    """Call this function ONLY when the reviewer has explicitly APPROVED the design."""
    return {"status": "approved", "message": "Design approved. Exiting loop."}

# Tool B: The Visualization Renderer (The new tool)
def render_mermaid(mermaid_code: str):
    """Renders a Mermaid.js diagram code block into an image."""
    try:
        # Cleanup: Remove markdown code block symbols if the LLM left them in
        clean_code = mermaid_code.replace("```mermaid", "").replace("```", "").strip()
        
        # Encoding logic for mermaid.ink
        graphbytes = clean_code.encode("utf8")
        base64_bytes = base64.urlsafe_b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        url = "https://mermaid.ink/img/" + base64_string
        
        # Display directly in the notebook
        display(Image(url=url))
        return "Diagram successfully rendered and displayed to the user."
    except Exception as e:
        return f"Error rendering diagram: {e}"

# --- Concept 2: Multi-Agent System ---

# Agent A: The Architect (Drafts)
architect_agent = Agent(
    name="Architect",
    model=model_config,
    instruction="""You are a Senior Cloud Architect.
    Your goal is to generate Mermaid.js diagram code based on the user's request.
    
    GUIDELINES:
    1. Focus on Google Cloud Platform (GCP) services.
    2. Use standard Mermaid syntax (graph TD or flowhart LR).
    3. Output ONLY the valid Mermaid code block. Do not add conversational filler.
    """,
    output_key="current_design" # Stores the design in session state
)

# Agent B: The Reviewer (Critiques)
reviewer_agent = Agent(
    name="Reviewer",
    model=model_config,
    instruction="""You are a Security and Reliability Engineer. 
    Review the architecture provided in {current_design}.
    
    CHECKLIST:
    1. Are there Single Points of Failure (SPOF)?
    2. Is the database exposed to the public internet? (It shouldn't be).
    3. Are there Load Balancers where appropriate?
    4. Is the Mermaid syntax valid?

    OUTPUT FORMAT:
    - If the design is solid, respond EXACTLY with: "APPROVED".
    - Otherwise, provide a numbered list of specific fixes required.""",
    output_key="critique"
)

# Agent C: The Refiner (Fixes)
refiner_agent = Agent(
    name="Refiner",
    model=model_config,
    instruction="""You are the coordinator. Read the critique: {critique}.
    
    LOGIC:
    1. If the critique says "APPROVED", you MUST call the `exit_loop` tool.
    2. If there is feedback, act as the Architect. Rewrite the Mermaid.js code in {current_design} to address the feedback.
    """,
    output_key="current_design",
    tools=[FunctionTool(exit_loop)] 
)

# Agent D: The Visualizer (NEW! Renders the result)
visualizer_agent = Agent(
    name="Visualizer",
    model=model_config,
    instruction="""You are a Documentation Specialist.
    Your job is to take the final approved design code from {current_design} and render it.
    
    ACTION:
    1. Look at the variable {current_design}.
    2. Call the `render_mermaid` tool passing that code as the argument.
    3. Confirm to the user that the architecture is ready.
    """,
    tools=[FunctionTool(render_mermaid)]
)

# --- Concept 3: Architecture Patterns ---

# The Quality Assurance Loop (Review <-> Refine)
qa_loop = LoopAgent(
    name="QualityAssuranceLoop",
    sub_agents=[reviewer_agent, refiner_agent],
    max_iterations=4
)

# The Main Pipeline
# 1. Draft -> 2. Refine Loop -> 3. Render Final Image
root_agent = SequentialAgent(
    name="Archie_The_Architect",
    sub_agents=[architect_agent, qa_loop, visualizer_agent] 
)

print("âœ… Archie is online and ready to design & render!")


# --- Execution ---
# Concept 4: Sessions & Memory (Handled by InMemoryRunner)
runner = InMemoryRunner(agent=root_agent)

# Define the user prompt
prompt = "Design a highly available web application on Google Cloud. It needs a frontend, a backend API, and a database."

print(f"ğŸ�¨ User Request: {prompt}\n")
print("Thinking... (This may take 30-60 seconds as agents debate the design)...\n")

# Run the agent
response = await runner.run_debug(prompt)


# --- Agent Engine Configuration ---
import vertexai

# 1. Project Config
PROJECT_ID = "root-wharf-271420"
REGION = "us-central1"
# 2. Staging Bucket (Required for Agent Engine)
# Format: "gs://your-unique-bucket-name"
STAGING_BUCKET = "gs://archie-agent-staging-bucket" 

# 3. Initialize Vertex AI SDK
try:
    vertexai.init(
        project=PROJECT_ID, 
        location=REGION, 
        staging_bucket=STAGING_BUCKET
    )
    print(f"âœ… Vertex AI initialized for project: {PROJECT_ID}")
except Exception as e:
    print(f"âš ï¸� Initialization skipped (Run this in a cloud environment): {e}")


!mkdir -p "generate-my-archie"
print("âœ… Created directory: generate-my-archie")


%%writefile generate-my-archie/.env
# Use the global endpoint for best compatibility
GOOGLE_CLOUD_LOCATION="global"

# CRITICAL: This tells ADK to use Vertex AI permissions (Service Account)
# instead of an API Key.
GOOGLE_GENAI_USE_VERTEXAI=1


%%writefile generate-my-archie/.agent_engine_config.json
{
    "min_instances": 0,
    "max_instances": 1,
    "resource_limits": {
        "cpu": "1",
        "memory": "2Gi"
    }
}


%%writefile generate-my-archie/requirements.txt
google-adk>=0.1.4
google-cloud-aiplatform>=1.70.0
pydantic>=2.9.0
opentelemetry-instrumentation-google-genai


import inspect
from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.genai import types

# 1. Define the content of agent.py dynamically
# NOTE: We use double curly braces {{ }} for agent placeholders so the f-string ignores them.
agent_file_content = f"""
import os
import vertexai
import base64
from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.genai import types

# --- CLOUD SETUP ---
PROJECT_ID = "{PROJECT_ID}"
LOCATION = "{REGION}"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- CONFIGURATION ---
retry_config = types.HttpRetryOptions(attempts=3, initial_delay=1)
model_config = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# --- TOOLS ---

# Tool 1: Loop Breaker
{inspect.getsource(exit_loop)}

# Tool 2: Cloud-Safe Renderer (Re-written for the server)
# This version returns a URL string instead of crashing the server
def render_mermaid(mermaid_code: str):
    \"\"\"Generates a viewable link for the Mermaid diagram code.\"\"\"
    try:
        # Cleanup
        clean_code = mermaid_code.replace("```mermaid", "").replace("```", "").strip()
        
        # Encode
        graphbytes = clean_code.encode("utf8")
        base64_bytes = base64.urlsafe_b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        url = "https://mermaid.ink/img/" + base64_string
        
        # Return the link as Markdown so the UI can render it later
        return f"Diagram Link Generated: {{url}}"
    except Exception as e:
        return f"Error generating diagram link: {{e}}"

# --- AGENTS ---
architect_agent = Agent(
    name="Architect",
    model=model_config,
    instruction=\"\"\"You are a Senior Cloud Architect.
    Your goal is to generate Mermaid.js diagram code based on the user's request.
    
    GUIDELINES:
    1. Focus on Google Cloud Platform (GCP) services.
    2. Use standard Mermaid syntax (graph TD or flowhart LR).
    3. Output ONLY the valid Mermaid code block. Do not add conversational filler.
    \"\"\",
    output_key="current_design" # Stores the design in session state
)

reviewer_agent = Agent(
    name="Reviewer",
    model=model_config,
    instruction=\"\"\"You are a Security and Reliability Engineer. 
    Review the architecture provided in {{current_design}}.
    
    CHECKLIST:
    1. Are there Single Points of Failure (SPOF)?
    2. Is the database exposed to the public internet? (It shouldn't be).
    3. Are there Load Balancers where appropriate?
    4. Is the Mermaid syntax valid?

    OUTPUT FORMAT:
    - If the design is solid, respond EXACTLY with: "APPROVED".
    - Otherwise, provide a numbered list of specific fixes required.\"\"\",
    output_key="critique"
)

refiner_agent = Agent(
    name="Refiner",
    model=model_config,
    instruction=\"\"\"You are the coordinator. Read the critique: {{critique}}.
    
    LOGIC:
    1. If the critique says "APPROVED", you MUST call the `exit_loop` tool.
    2. If there is feedback, act as the Architect. Rewrite the Mermaid.js code in {{current_design}} to address the feedback.
    \"\"\",
    output_key="current_design",
    tools=[FunctionTool(exit_loop)] 
)

visualizer_agent = Agent(
    name="Visualizer",
    model=model_config,
    instruction=\"\"\"You are a Documentation Specialist.
    Your job is to take the final approved design code from {{current_design}} and generate a viewable image link for it.
    
    ACTION:
    1. Look at the variable {{current_design}}.
    2. Call the `render_mermaid` tool passing that code as the argument.
    3. Output the result (the URL) to the user.
    \"\"\",
    tools=[FunctionTool(render_mermaid)]
)

# --- WORKFLOW ---
qa_loop = LoopAgent(
    name="DesignLoop",
    sub_agents=[reviewer_agent, refiner_agent],
    max_iterations=4
)

# --- ROOT ENTRY POINT ---
# Pipeline: Architect -> QA Loop -> Visualizer
root_agent = SequentialAgent(
    name="Archie_The_Architect",
    sub_agents=[architect_agent, qa_loop, visualizer_agent]
)
"""

# 2. Write it to the file
with open("generate-my-archie/agent.py", "w") as f:
    f.write(agent_file_content)

print("âœ… Successfully exported Cloud-Safe logic to generate-my-archie/agent.py")


import os

# TODO: UPDATE THIS PATH to match your uploaded dataset file
KEY_PATH = "/kaggle/input/serviceaccountkey/root-wharf-271420-683efe1a6b90.json" 
PROJECT_ID = "root-wharf-271420" 
REGION = "us-central1"

# Set environment variable for Python SDKs
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

# Activate gcloud for the CLI command
# We add --quiet to prevent "Do you want to continue?" prompts
!gcloud auth activate-service-account --key-file=$KEY_PATH --quiet
!gcloud config set project $PROJECT_ID --quiet

print("âœ… Authentication successful.")


print("ğŸš€ Deploying Archie to Agent Engine...")

!adk deploy agent_engine \
  --project=$PROJECT_ID \
  --region=$REGION \
  generate-my-archie \
  --agent_engine_config_file=generate-my-archie/.agent_engine_config.json


import vertexai
from vertexai import agent_engines
from IPython.display import Image, display
import base64

# --- Configuration ---
PROJECT_ID = "root-wharf-271420"
REGION = "us-central1"
# YOUR DEPLOYED NEW RESOURCE ID (Paste the green ID from your last output here)  
NEW_RESOURCE_ID = "projects/399354965493/locations/us-central1/reasoningEngines/385739465349398528" 

# 1. Setup
print(f"ğŸ”„ Connecting to Agent Engine...")
vertexai.init(project=PROJECT_ID, location=REGION)

# 2. Connect to the deployed agent
agents = list(agent_engines.list())

if agents:
    remote_archie = agent_engines.get(NEW_RESOURCE_ID)
    print(f"âœ… Connected to: {remote_archie.resource_name}")
    
    print("\nğŸ¤– Archie is thinking ...")
    
    try:
        # 2. Prepare Payload
        payload = {
            "message": "Design a secure GCP storage system",
            "user_id": "test_user_001"
        }
        
        # Store full response text
        full_response_text = ""    
        
        # 3. Stream and Print Text
        async for event in remote_archie.async_stream_query(**payload):
            if 'content' in event and 'parts' in event['content']:
                for part in event['content']['parts']:
                    if 'text' in part:
                        text_chunk = part['text']
                        print(text_chunk, end="", flush=True)
                        full_response_text += text_chunk
        
        # 4. Post-Process: Robust Visualization
        print("\n\nâœ¨ --- VISUALIZING ARCHITECTURE --- âœ¨")
        
        # Split by markdown blocks
        code_blocks = full_response_text.split("```")
        mermaid_code = None
        
        for block in reversed(code_blocks):
            if block.strip().startswith("mermaid"):
                # Remove "mermaid" tag
                raw_code = block.replace("mermaid", "", 1).strip()
                
                # --- THE FIX: Sanitizing the Code ---
                # We filter out lines that cause syntax errors (like connecting subgraphs with spaces)
                safe_lines = []
                for line in raw_code.splitlines():
                    # This removes the specific line causing your error
                    if "GCP Services --" not in line and "Integrated with" not in line:
                        safe_lines.append(line)
                
                mermaid_code = "\n".join(safe_lines)
                break
                
        if mermaid_code:
            try:
              render_mermaid(mermaid_code)  
            except Exception as img_err:
                print(f"âš ï¸� Image rendering failed: {img_err}")
                print("ğŸ“� Raw Mermaid Code:\n", mermaid_code)
        else:
            print("âš ï¸� No valid Mermaid code block detected.")
    
    except Exception as e:
        print(f"\nâ�Œ Execution Error: {e}")
else:
    print("â�Œ No deployed agents found.")


url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

