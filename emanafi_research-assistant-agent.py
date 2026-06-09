# Install the Agent Development Kit (ADK), ArXiv and AsyncIO
!pip install google-adk arxiv nest-asyncio

print("âœ… ADK, ArXiv, and nest-asyncio installed successfully.")


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503],
)

print("âœ… Retry configuration set.")


import json
import logging
import asyncio
from typing import List, Dict, Any
from google.genai.types import Content, Part

# ADK Imports
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")


# Configure Logging for Observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("âœ… Logging configured.")


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]['base_url']

    try:
        path_parts = baseURL.split('/')
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix

print("âœ… Helper functions defined.")


import arxiv
import time

# --- ArXiv API Tool ---
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Searches ArXiv for research papers matching the query.
    Includes retry logic and uses the updated Client API to avoid deprecation warnings.
    
    Args:
        query: Search query for papers
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted string with paper details
    """
    client = arxiv.Client()
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results = []
    try:
        # Use client.results() instead of search.results() to fix deprecation warning
        # and add retry logic for robustness
        for attempt in range(3):
            try:
                # Convert generator to list to ensure we actually fetch data
                found_papers = list(client.results(search))
                break
            except Exception as e:
                if attempt == 2: # Last attempt
                    raise e
                time.sleep(1) # Wait 1s before retry
        
        for paper in found_papers:
            result = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url
            }
            results.append(result)
            
        if not results:
            return f"No papers found for query: {query}"
            
        # Format results
        formatted = f"Found {len(results)} papers for '{query}':\n\n"
        for i, paper in enumerate(results, 1):
            authors = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors += " et al."
            formatted += f"{i}. **{paper['title']}**\n"
            formatted += f"   Authors: {authors}\n"
            formatted += f"   Published: {paper['published']}\n"
            formatted += f"   URL: {paper['url']}\n"
            formatted += f"   Summary: {paper['summary']}\n\n"
            
        logger.info(f"ArXiv search completed: {len(results)} results for '{query}'")
        return formatted
        
    except Exception as e:
        error_msg = f"ArXiv search failed: {str(e)}"
        logger.error(error_msg)
        return f"Error searching ArXiv: {str(e)}"

print("âœ… ArXiv search tool defined (updated with Client API and retries).")


# --- Custom Tool 1: Citation Formatter ---
def format_citation(title: str, authors: List[str], year: str, url: str) -> str:
    """Formats a research paper citation."""
    author_str = ", ".join(authors)
    citation = f"{author_str} ({year}). **{title}**. Retrieved from {url}"
    logger.info(f"Formatted citation for: {title}")
    return citation

# --- Custom Tool 2: Long-Term Memory (File-based) ---
MEMORY_FILE = "knowledge_base.json"

def save_to_memory(key: str, value: Any) -> str:
    """Saves a key-value pair to a persistent JSON file."""
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {}
            
        data[key] = value
        
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved to memory: {key}")
        return f"Successfully saved '{key}' to long-term memory."
    except Exception as e:
        return f"Error saving to memory: {str(e)}"

# --- Custom Tool 3: Parallel Research Orchestrator ---
# Note: In a real scenario, this would spawn independent agent processes.
# Here, we simulate it by running multiple google searches concurrently.

async def parallel_research(topics: List[str]) -> List[str]:
    """
    Performs research on multiple topics simultaneously.
    Args:
        topics: A list of sub-topics to research.
    Returns:
        A combined list of search results for all topics.
    """
    logger.info(f"Starting parallel research on: {topics}")
    
    # Define a helper for a single search
    async def search_topic(topic):
        # We use the google_search tool directly here for simulation
        # In a full ADK app, this could be `await researcher_agent.run(topic)`
        logger.info(f"Searching for: {topic}")
        return google_search(topic)

    # Run searches in parallel
    results = await asyncio.gather(*(search_topic(topic) for topic in topics))
    
    # Combine results
    combined_results = []
    for topic, result in zip(topics, results):
        combined_results.append(f"--- Results for '{topic}' ---\n{result}\n")
        
    return combined_results

print("âœ… Custom tools defined: format_citation, save_to_memory, parallel_research.")


# 1. Researcher Agent (Parallel Capabilities with ArXiv)
researcher_agent = LlmAgent(
    name="researcher_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for research papers using ArXiv API.",
    instruction="""
    You are a research assistant with access to ArXiv.
    When given a research topic:
    1. Use the 'search_arxiv' tool to find relevant academic papers.
    2. Focus on recent papers (2023-2025 when possible).
    3. Return the paper details including titles, authors, dates, and URLs.
    """,
    tools=[search_arxiv]
)

# 2. Analyst Agent (Manual Analysis)
analyst_agent = LlmAgent(
    name="analyst_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Analyzes research data and creates visualizations.",
    instruction="""
    You are a data analyst.
    Given a list of research papers or search results:
    1. Extract the publication years from the paper information.
    2. Calculate the distribution of papers by year.
    3. Create a simple ASCII bar chart showing the distribution.
    4. Return the analysis summary and the ASCII chart.
    
    Example ASCII chart format:
    2025: *** (3 papers)
    2024: ***** (5 papers)
    2023: ** (2 papers)
    """
)

# 3. Formatter Agent
formatter_agent = LlmAgent(
    name="formatter_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Formats paper details into proper citations.",
    instruction="""
    You are a citation expert. 
    Take the raw paper information and format it into clean academic citations.
    Use APA format: Authors (Year). Title. Retrieved from URL
    Return the final list of formatted citations as a numbered list.
    """
)

# 4. Root Agent (Orchestrator & Memory Manager)
root_agent = LlmAgent(
    name="root_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are the Lead Research Coordinator.
    
    Your workflow is:
    1. **Research**: Delegate to 'researcher_agent' to find papers on the user's topic.
    2. **Analyze**: Delegate the findings to 'analyst_agent' to get a distribution of publication years.
    3. **Format**: Delegate to 'formatter_agent' to get a list of citations.
    4. **Save**: Use the 'save_to_memory' tool to save the final citation list and analysis to the 'knowledge_base.json' file. Use the topic as the key.
    5. **Report**: YOU MUST PRINT the final report to the user.
       - Show the "Formatted Citations" list.
       - Show the "Publication Year Analysis" chart.
       - Confirm that data has been saved to memory.
       
    CRITICAL: Do NOT stop after calling tools. You MUST generate a final text response summarizing the results.
    """,
    tools=[
        AgentTool(agent=researcher_agent), 
        AgentTool(agent=analyst_agent), 
        AgentTool(agent=formatter_agent),
        save_to_memory
    ]
)

print("âœ… Agents defined: Researcher, Analyst, Formatter, and Root.")


def evaluate_response(response_text: str):
    """
    Evaluation to check if the response contains citations, analysis, and confirms saving.
    """
    score = 0
    checks = []
    
    # Check 1: List format (Citations)
    if "1." in response_text or "-" in response_text:
        score += 1
        checks.append("List format detected")
    
    # Check 2: Recent years
    if "2024" in response_text or "2025" in response_text:
        score += 1
        checks.append("Recent years detected")
        
    # Check 3: URLs
    if "http" in response_text:
        score += 1
        checks.append("URLs detected")

    # Check 4: Analysis/Chart
    if "Analysis" in response_text or "|" in response_text or "*" in response_text: # Simple check for ASCII chart chars
        score += 1
        checks.append("Analysis/Chart detected")

    # Check 5: Memory Save
    if "saved" in response_text.lower() or "memory" in response_text.lower():
        score += 1
        checks.append("Memory save confirmed")
        
    print(f"\n--- Evaluation Results ---")
    print(f"Score: {score}/5")
    print(f"Checks passed: {', '.join(checks)}")
    
print("âœ… Evaluation function defined.")


# Initialize Session Service and Runner
from google.adk.runners import Runner
import nest_asyncio
import traceback
import logging

# Apply nest_asyncio to allow nested event loops (critical for notebook environments)
nest_asyncio.apply()

# Suppress the specific google_genai warning about non-text parts
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

APP_NAME = "research_agent"
USER_ID = "researcher_01"

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

print("âœ… Session service and runner initialized.")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")

async def run_agent(query: str, session_id: str = "default_session"):
    """
    Runs the agent with the given query and returns the full response.
    Includes debug logging to trace execution flow and errors.
    """
    print(f"User > {query}")
    print("-" * 50)
    
    # Create or get the session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id=session_id
        )
    
    full_response = ""
    
    # Create a Content object from the query string
    message = Content(parts=[Part(text=query)], role="user")
    
    event_count = 0
    try:
        # Run the agent asynchronously with the session
        async for event in runner.run_async(
            user_id=USER_ID, 
            session_id=session.id,
            new_message=message
        ):
            event_count += 1
            
            # Check for content parts
            if event.content and event.content.parts:
                for part in event.content.parts:
                    # Handle text parts
                    if part.text and part.text != "None":
                        print(f"Agent > {part.text}")
                        full_response += part.text + "\n"
                    
                    # Handle function calls
                    if part.function_call:
                        print(f"ğŸ¤– Agent is calling tool: {part.function_call.name}")
            
            # Log errors if present in the event
            if hasattr(event, 'error') and event.error:
                print(f"âš ï¸� Event Error: {event.error}")
                
    except Exception as e:
        print(f"â�Œ Error during agent execution: {str(e)}")
        traceback.print_exc()
        
    if event_count == 0:
        print("âš ï¸� Warning: No events received from the agent. Check network connection and API keys.")
    elif not full_response.strip():
        print("âš ï¸� Warning: Agent executed tools but returned no final text response.")
        print("   This usually means the agent loop finished before the final answer was generated.")
        print("   Try running the cell again, or check if the 'Root Agent' instructions are clear.")
                
    return full_response

# Example Query - ArXiv Search
query = "Agentic AI Design Patterns"

# Run the query and evaluate
try:
    response = await run_agent(query)
    if response.strip():
        evaluate_response(response)
    else:
        print("No response received to evaluate.")
except Exception as e:
    print(f"Failed to run agent: {e}")


!pip install google-cloud-aiplatform

print("âœ… Vertex AI SDK installed.")


from google.cloud import aiplatform

# Configure your Google Cloud project
PROJECT_ID = "your-project-id"  # Replace with your GCP project ID
LOCATION = "us-central1"  # Choose your preferred region
STAGING_BUCKET = "gs://your-bucket-name"  # Replace with your GCS bucket

# Initialize Vertex AI
aiplatform.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET
)

print(f"âœ… Vertex AI initialized for project: {PROJECT_ID}")


import os

# Create deployment directory
os.makedirs("vertex_agent_deployment", exist_ok=True)

# Create agent.py - Main agent file
agent_code = """
import arxiv
import json
import os
import logging
from typing import List, Any
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool
from google.genai import types

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Tools ---

def search_arxiv(query: str, max_results: int = 5) -> str:
    \"\"\"Searches ArXiv for research papers matching the query.\"\"\"
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        results = []
        for paper in search.results():
            result = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "url": paper.entry_id,
            }
            results.append(result)
            
        if not results:
            return f"No papers found for query: {query}"

        formatted = f"Found {len(results)} papers for '{query}':\\n\\n"
        for i, paper in enumerate(results, 1):
            authors = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors += " et al."
            formatted += f"{i}. **{paper['title']}**\\n"
            formatted += f"   Authors: {authors}\\n"
            formatted += f"   Published: {paper['published']}\\n"
            formatted += f"   URL: {paper['url']}\\n"
            formatted += f"   Summary: {paper['summary']}\\n\\n"
        return formatted
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

def format_citation(title: str, authors: List[str], year: str, url: str) -> str:
    \"\"\"Formats a research paper citation.\"\"\"
    author_str = ", ".join(authors)
    return f"{author_str} ({year}). **{title}**. Retrieved from {url}"

# Use /tmp for writable location in cloud environments
MEMORY_FILE = "/tmp/knowledge_base.json"

def save_to_memory(key: str, value: Any) -> str:
    \"\"\"Saves a key-value pair to a persistent JSON file.\"\"\"
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {}
            
        data[key] = value
        
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
        return f"Successfully saved '{key}' to long-term memory."
    except Exception as e:
        return f"Error saving to memory: {str(e)}"

# --- Agents ---

retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503],
)

# Shared Model
model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# 1. Researcher Agent
researcher_agent = LlmAgent(
    name="researcher_agent",
    model=model,
    description="Searches for research papers using ArXiv API.",
    instruction='''
    You are a research assistant with access to ArXiv.
    When given a research topic:
    1. Use the 'search_arxiv' tool to find relevant academic papers.
    2. Focus on recent papers (2023-2025 when possible).
    3. Return the paper details including titles, authors, dates, and URLs.
    ''',
    tools=[search_arxiv]
)

# 2. Analyst Agent
analyst_agent = LlmAgent(
    name="analyst_agent",
    model=model,
    description="Analyzes research data and creates visualizations.",
    instruction='''
    You are a data analyst.
    Given a list of research papers or search results:
    1. Extract the publication years from the paper information.
    2. Calculate the distribution of papers by year.
    3. Create a simple ASCII bar chart showing the distribution.
    4. Return the analysis summary and the ASCII chart.
    '''
)

# 3. Formatter Agent
formatter_agent = LlmAgent(
    name="formatter_agent",
    model=model,
    description="Formats paper details into proper citations.",
    instruction='''
    You are a citation expert. 
    Take the raw paper information and format it into clean academic citations.
    Use APA format: Authors (Year). Title. Retrieved from URL
    Return the final list of formatted citations as a numbered list.
    '''
)

# 4. Root Agent (Orchestrator)
root_agent = LlmAgent(
    name="root_agent",
    model=model,
    instruction='''
    You are the Lead Research Coordinator.
    
    Your workflow is:
    1. **Research**: Delegate to 'researcher_agent' to find papers on the user's topic.
    2. **Analyze**: Delegate the findings to 'analyst_agent' to get a distribution of publication years.
    3. **Format**: Delegate to 'formatter_agent' to get a list of citations.
    4. **Save**: Use the 'save_to_memory' tool to save the final citation list and analysis to the 'knowledge_base.json' file. Use the topic as the key.
    5. **Report**: YOU MUST PRINT the final report to the user.
       - Show the "Formatted Citations" list.
       - Show the "Publication Year Analysis" chart.
       - Confirm that data has been saved to memory.
       
    CRITICAL: Do NOT stop after calling tools. You MUST generate a final text response summarizing the results.
    ''',
    tools=[
        AgentTool(agent=researcher_agent), 
        AgentTool(agent=analyst_agent), 
        AgentTool(agent=formatter_agent),
        save_to_memory
    ]
)

# Export the root agent
agent = root_agent
"""

with open("vertex_agent_deployment/agent.py", "w") as f:
    f.write(agent_code)

# Create requirements.txt
requirements = """google-adk
arxiv
google-cloud-aiplatform
"""

with open("vertex_agent_deployment/requirements.txt", "w") as f:
    f.write(requirements)

# Create deployment config
config = """{
  "agent_name": "research-assistant",
  "agent_description": "Multi-agent academic research system",
  "model": "gemini-2.5-flash-lite"
}
"""

with open("vertex_agent_deployment/config.json", "w") as f:
    f.write(config)

print("âœ… Agent package created in 'vertex_agent_deployment/' directory")
print("   - agent.py: Multi-agent system code")
print("   - requirements.txt: Dependencies")
print("   - config.json: Deployment configuration")


# Example: Interacting with deployed agent
test_code = '''
from google.cloud import aiplatform
from google.genai.types import Content, Part

# Initialize the agent client
AGENT_ID = "your-agent-id"  # Replace with your deployed agent ID

# Create a prediction request
def query_deployed_agent(query: str):
    """Query the deployed Vertex AI agent."""
    
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{AGENT_ID}"
    )
    
    # Format the request
    message = Content(parts=[Part(text=query)], role="user")
    
    # Send request
    response = endpoint.predict(instances=[{"message": message}])
    
    return response.predictions[0]

# Test the deployed agent
try:
    result = query_deployed_agent("Search for papers on transformers in NLP")
    print("Agent Response:", result)
except Exception as e:
    print(f"Note: Replace AGENT_ID and ensure agent is deployed. Error: {e}")
'''

print("ğŸ’¬ Example code to query your deployed agent:")
print(test_code)

print("\n" + "="*60)
print("REST API Example:")
print("="*60)

rest_example = f'''
curl -X POST \\
  https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/agents/AGENT_ID:query \\
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "message": "Search for papers on quantum computing",
    "session_id": "unique-session-id"
  }}'
'''

print(rest_example)


from flask import Flask, request, jsonify
import threading
from IPython.display import display, HTML
from jupyter_server.serverapp import list_running_servers
import nest_asyncio
import asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Capture the main event loop where the runner was initialized
try:
    MAIN_LOOP = asyncio.get_running_loop()
except RuntimeError:
    MAIN_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(MAIN_LOOP)

# Define the Flask App
app = Flask(__name__)

def run_agent_sync(query: str, session_id: str):
    """
    Synchronous wrapper that runs the async agent on the main event loop.
    This avoids 'attached to a different loop' errors by ensuring the runner
    executes in the same loop where it was created.
    """
    from google.genai.types import Content, Part
    
    async def run_agent_task():
        # Create or get session
        try:
            session = await session_service.create_session(
                app_name=APP_NAME, 
                user_id=USER_ID, 
                session_id=session_id
            )
        except:
            session = await session_service.get_session(
                app_name=APP_NAME, 
                user_id=USER_ID, 
                session_id=session_id
            )
        
        full_response = ""
        message = Content(parts=[Part(text=query)], role="user")
        
        # Run the agent
        async for event in runner.run_async(
            user_id=USER_ID, 
            session_id=session.id,
            new_message=message
        ):
            if event.content and event.content.parts:
                part = event.content.parts[0]
                if part.text and part.text != "None":
                    print(f"Agent > {part.text}")
                    full_response += part.text + "\n"
        
        return full_response
    
    # Submit the task to the main loop from the Flask thread
    future = asyncio.run_coroutine_threadsafe(run_agent_task(), MAIN_LOOP)
    return future.result()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    API Endpoint to interact with the Research Agent.
    Expected JSON: {"query": "your research topic", "session_id": "optional_session_id"}
    """
    data = request.json
    query = data.get('query')
    session_id = data.get('session_id', 'default_session')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    print(f"API Request: {query}")
    
    # Run the agent
    try:
        response_text = run_agent_sync(query, session_id)
        return jsonify({"response": response_text, "status": "success"})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "agent": "Research Assistant"})

def run_flask():
    """Runs the Flask app on port 5003."""
    app.run(host='0.0.0.0', port=5003, debug=False, use_reloader=False)

def get_flask_proxy_url():
    """Get the Kaggle proxy URL for Flask"""
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    FLASK_PORT = "5003"
    
    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")
    
    baseURL = servers[0]['base_url']
    
    try:
        path_parts = baseURL.split('/')
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")
    
    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{FLASK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"
    
    styled_html = f"""
    <div style="padding: 20px; border: 2px solid #4CAF50; border-radius: 8px; background-color: #f0fdf4; margin: 20px 0;">
        <h3 style="color: #2e7d32; margin-top: 0;">ğŸš€ Flask API Running!</h3>
        <p style="font-family: sans-serif; color: #333;">Your Research Assistant API is now accessible.</p>
        
        <div style="background-color: white; padding: 15px; border-radius: 5px; margin: 15px 0;">
            <strong>Endpoints:</strong><br>
            <code style="background-color: #f5f5f5; padding: 2px 6px; border-radius: 3px;">POST {url}/chat</code> - Query the agent<br>
            <code style="background-color: #f5f5f5; padding: 2px 6px; border-radius: 3px;">GET {url}/health</code> - Health check
        </div>
        
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #ffc107;">
            <strong>Example cURL command:</strong><br>
            <code style="display: block; margin-top: 10px; padding: 10px; background-color: #2d2d2d; color: #f8f8f2; border-radius: 3px; overflow-x: auto;">
curl -X POST "{url}/chat" \\<br>
  -H "Content-Type: application/json" \\<br>
  -d '{{"query": "Search for papers on quantum computing"}}'
            </code>
        </div>
        
        <a href='{url}/health' target='_blank' style="
            display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease; margin-top: 10px;">
            Test Health Check â†—
        </a>
    </div>
    """
    
    display(HTML(styled_html))
    return url

# Start Flask in background
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

print("âœ… Flask API server starting...")
print("Run the next cell to get your API URL!")


# Get and display the Flask API URL
flask_api_url = get_flask_proxy_url()


# Example: Test the API using Python requests
import requests
import json
import threading

def test_api():
    # Test health check
    print("Testing health endpoint...")
    try:
        health_response = requests.get(f"{flask_api_url}/health")
        print(f"Health check: {health_response.json()}")
    except Exception as e:
        print(f"Note: Run this after getting the URL from the previous cell. Error: {e}")

    # Test chat endpoint
    print("\nTesting chat endpoint...")
    test_query = {
        "query": "Find papers on transformer neural networks",
        "session_id": "test_session_001"
    }

    try:
        print("Sending request (this may take 30-60s)...")
        chat_response = requests.post(
            f"{flask_api_url}/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_query)
        )
        print(f"Chat response: {chat_response.json()}")
    except Exception as e:
        print(f"Note: Make sure Flask is running. Error: {e}")

# Run the test in a separate thread to avoid blocking the Main Event Loop
# (which is needed by the Agent running in the Flask server)
test_thread = threading.Thread(target=test_api)
test_thread.start()


!adk create research-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


# Overwrite the generated agent.py with our Multi-Agent System code
agent_code = """
import arxiv
import json
import os
import logging
from typing import List, Any
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool
from google.genai import types

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Tools ---

def search_arxiv(query: str, max_results: int = 5) -> str:
    \"\"\"Searches ArXiv for research papers matching the query.\"\"\"
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        results = []
        for paper in search.results():
            result = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "url": paper.entry_id,
            }
            results.append(result)
            
        if not results:
            return f"No papers found for query: {query}"

        formatted = f"Found {len(results)} papers for '{query}':\\n\\n"
        for i, paper in enumerate(results, 1):
            authors = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors += " et al."
            formatted += f"{i}. **{paper['title']}**\\n"
            formatted += f"   Authors: {authors}\\n"
            formatted += f"   Published: {paper['published']}\\n"
            formatted += f"   URL: {paper['url']}\\n"
            formatted += f"   Summary: {paper['summary']}\\n\\n"
        return formatted
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

def format_citation(title: str, authors: List[str], year: str, url: str) -> str:
    \"\"\"Formats a research paper citation.\"\"\"
    author_str = ", ".join(authors)
    return f"{author_str} ({year}). **{title}**. Retrieved from {url}"

# Use /tmp for writable location in cloud environments
MEMORY_FILE = "/tmp/knowledge_base.json"

def save_to_memory(key: str, value: Any) -> str:
    \"\"\"Saves a key-value pair to a persistent JSON file.\"\"\"
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {}
            
        data[key] = value
        
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
        return f"Successfully saved '{key}' to long-term memory."
    except Exception as e:
        return f"Error saving to memory: {str(e)}"

# --- Agents ---

retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503],
)

# Shared Model
model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# 1. Researcher Agent
researcher_agent = LlmAgent(
    name="researcher_agent",
    model=model,
    description="Searches for research papers using ArXiv API.",
    instruction='''
    You are a research assistant with access to ArXiv.
    When given a research topic:
    1. Use the 'search_arxiv' tool to find relevant academic papers.
    2. Focus on recent papers (2023-2025 when possible).
    3. Return the paper details including titles, authors, dates, and URLs.
    ''',
    tools=[search_arxiv]
)

# 2. Analyst Agent
analyst_agent = LlmAgent(
    name="analyst_agent",
    model=model,
    description="Analyzes research data and creates visualizations.",
    instruction='''
    You are a data analyst.
    Given a list of research papers or search results:
    1. Extract the publication years from the paper information.
    2. Calculate the distribution of papers by year.
    3. Create a simple ASCII bar chart showing the distribution.
    4. Return the analysis summary and the ASCII chart.
    '''
)

# 3. Formatter Agent
formatter_agent = LlmAgent(
    name="formatter_agent",
    model=model,
    description="Formats paper details into proper citations.",
    instruction='''
    You are a citation expert. 
    Take the raw paper information and format it into clean academic citations.
    Use APA format: Authors (Year). Title. Retrieved from URL
    Return the final list of formatted citations as a numbered list.
    '''
)

# 4. Root Agent (Orchestrator)
root_agent = LlmAgent(
    name="root_agent",
    model=model,
    instruction='''
    You are the Lead Research Coordinator.
    
    Your workflow is:
    1. **Research**: Delegate to 'researcher_agent' to find papers on the user's topic.
    2. **Analyze**: Delegate the findings to 'analyst_agent' to get a distribution of publication years.
    3. **Format**: Delegate to 'formatter_agent' to get a list of citations.
    4. **Save**: Use the 'save_to_memory' tool to save the final citation list and analysis to the 'knowledge_base.json' file. Use the topic as the key.
    5. **Report**: YOU MUST PRINT the final report to the user.
       - Show the "Formatted Citations" list.
       - Show the "Publication Year Analysis" chart.
       - Confirm that data has been saved to memory.
       
    CRITICAL: Do NOT stop after calling tools. You MUST generate a final text response summarizing the results.
    ''',
    tools=[
        AgentTool(agent=researcher_agent), 
        AgentTool(agent=analyst_agent), 
        AgentTool(agent=formatter_agent),
        save_to_memory
    ]
)

# Export the root agent
agent = root_agent
"""

with open("research-agent/agent.py", "w") as f:
    f.write(agent_code)

# Update requirements.txt to include arxiv
requirements = """google-adk
arxiv
"""

with open("research-agent/requirements.txt", "w") as f:
    f.write(requirements)

print("âœ… Updated 'research-agent/agent.py' with Multi-Agent System code.")
print("âœ… Updated 'research-agent/requirements.txt' with dependencies.")


url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}

