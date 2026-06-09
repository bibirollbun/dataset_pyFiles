import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


# --- Standard Python Libraries ---
import os
import json
import requests
import socket
import uuid
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --- ADK Component Imports ---
# These components are the building blocks of the agentic system.
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# --- Configuration for Robustness ---
# Define the retry configuration (a key concept demonstrated).
retry_config = types.HttpRetryOptions(
    attempts=5,        
    exp_base=7,        
    initial_delay=1,   
    http_status_codes=[429, 500, 503, 504], 
)

# Hide non-critical warnings
import warnings
warnings.filterwarnings("ignore")

print("âœ… Setup complete. Core components initialized.")


# Configure Retry Options
# This helps the agent automatically handle transient errors like
# rate limits (429) or temporary server issues (500, 503, 504).
retry_config = types.HttpRetryOptions(
    attempts=5,        # Maximum retry attempts
    exp_base=7,        # Delay multiplier
    initial_delay=1,   # Initial delay in seconds
    http_status_codes=[429, 500, 503, 504], # Retry on these errors
)

print("âœ… Step 3 Complete: 'retry_config' is defined.")


# --- Tool Utility Functions ---
def get_asset_size(url, session):
    """
    Tries to get the size of a remote asset using a HEAD request.
    """
    try:
        # We use a short timeout to keep the scan fast
        response = session.head(url, timeout=3, allow_redirects=True)
        # Check if 'content-length' header exists
        if 'content-length' in response.headers:
            return int(response.headers['content-length'])
    except requests.RequestException:
        # Ignore assets we can't access or that time out
        pass
    return None

# --- Our Main Custom Tool Definition ---

def scan_website_assets(url: str) -> str:
    """
    Scans a given URL to extract assets (images, CSS, JS),
    their sizes, and the hosting IP.
    
    Args:
        url: The full URL of the website to scan (e.g., "https://www.google.com").
    
    Returns:
        A JSON string containing the scan results.
    """
    print(f"ğŸ•µï¸� [Scanner Tool] Received request for: {url}")
    try:
        # Use a session for efficient, repeated network requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # --- Get Main Page and Hosting IP ---
        response = session.get(url, timeout=10)
        response.raise_for_status() # Check for errors (like 404)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get domain and IP
        domain = urlparse(url).netloc
        try:
            ip_address = socket.gethostbyname(domain)
        except socket.gaierror:
            ip_address = None # Handle cases where IP lookup fails

        assets = {
            "hosting_domain": domain,
            "hosting_ip": ip_address,
            "images": [],
            "css": [],
            "js": [],
        }

        # --- Process Assets ---
        
        # Find images
        for img in soup.find_all('img'):
            img_src = img.get('src')
            if not img_src: continue # Skip images without a 'src'
            
            absolute_url = urljoin(url, img_src)
            assets["images"].append({
                "url": absolute_url,
                "format": absolute_url.split('.')[-1].lower().split('?')[0],
                "size_bytes": get_asset_size(absolute_url, session)
            })

        # Find CSS
        for link in soup.find_all('link', {'rel': 'stylesheet'}):
            css_href = link.get('href')
            if not css_href: continue
                
            absolute_url = urljoin(url, css_href)
            assets["css"].append({
                "url": absolute_url,
                "minified": ".min." in absolute_url,
                "size_bytes": get_asset_size(absolute_url, session)
            })

        # Find JS
        for script in soup.find_all('script', {'src': True}):
            js_src = script.get('src')
            if not js_src: continue
                
            absolute_url = urljoin(url, js_src)
            assets["js"].append({
                "url": absolute_url,
                "minified": ".min." in absolute_url,
                "size_bytes": get_asset_size(absolute_url, session)
            })
        
        print(f"âœ… [Scanner Tool] Scan complete for: {url}")
        # Return the data as a JSON string, as required by the ADK
        return json.dumps(assets)

    except requests.RequestException as e:
        print(f"â�Œ [Scanner Tool] HTTP Error for {url}: {e}")
        return json.dumps({"error": f"Failed to fetch base URL: {e}"})
    except Exception as e:
        print(f"â�Œ [Scanner Tool] An unexpected error occurred: {e}")
        return json.dumps({"error": str(e)})

# --- Confirmation ---
# This line will run after the functions are defined, 
# confirming the cell completed without syntax errors.
print("âœ… Step 4 Complete: 'scan_website_assets' tool is defined.")


# --- Define Agent Prompts (Final, Adaptive, State-Passing) ---

SCANNER_PROMPT = """
You are a machine-to-machine data transfer agent.
Your output will be used as a JSON input for another agent.
1. You MUST call the 'scan_website_assets' tool with this URL.
2. The tool will return a JSON string.
3. Your final response MUST be ONLY this raw, exact, unmodified JSON string.
Do NOT add any text, commentary, greetings, or formatting.
Your entire output must be 100% JSON.
"""

ANALYST_PROMPT = """
You are a 'Green-Code Analyst'. Your goal is to find energy inefficiencies.
Your input is the text summary from the previous agent. You MUST ignore all conversational
text and focus only on extracting the JSON data embedded within the text.

Your tasks are:
1.  **Parse the JSON from the input:** Extract the raw JSON string from the input text.
2.  **Analyze the extracted data:**
    * Look at `images`: Are any `size_bytes` over 200000 (200KB)? Is the `format` "png" or "jpg" instead of "webp" or "avif"?
    * Look at `css` and `js`: Are any `minified` values `false`? Are any `size_bytes` over 100000 (100KB)?
3.  **Use your Google Search tool:**
    * Investigate the `hosting_domain` from the data.
    * Search for "[hosting_domain] renewable energy policy" or "[hosting_domain] sustainability report".
    * Determine if the host is known for being "green" or not.
4.  **Output your findings:** Create a simple, factual, bulleted list of the *key problems* you found. 
    If no problems are found, just output "Analysis complete. No optimization opportunities found."
"""

REPORTER_PROMPT = """
You are the 'Green-Code Advisor'. Your tone is expert, helpful, and encouraging.
IGNORE any chat text you receive. Your one and only input is the bulleted list
of technical problems in the '{problem_list}' state variable.

Your job is to convert this list into a beautiful, user-friendly, and 
actionable report using Markdown.

For each problem, you MUST:
1.  **Give it a clear title.** (e.g., "## ğŸ”´ High Priority: Large Image Files")
2.  **Explain the "Green" Impact:** Why does this waste energy?
3.  **Provide the Actionable Fix:** How does the developer fix it?

Structure your output clearly. Start with a nice intro.
If the input is "No optimization opportunities found," just write a 
short, positive message, like "Great news! Your site looks well-optimized."
"""

# --- Define Agent 1: The Scanner Agent ---
scanner_llm = Gemini(
    model_name='gemini-1.5-flash-latest',
    api_key=os.environ["GOOGLE_API_KEY"],
    retry_options=retry_config,
    system_prompt=SCANNER_PROMPT 
)
scanner_agent = Agent(
    name="scanner", 
    model=scanner_llm,
    tools=[scan_website_assets],
    output_key="scan_results_json" # <-- Data is saved here
)

# --- Define Agent 2: The Analyst Agent ---
analyst_llm = Gemini(
    model_name='gemini-1.5-flash-latest',
    api_key=os.environ["GOOGLE_API_KEY"],
    retry_options=retry_config,
    system_prompt=ANALYST_PROMPT 
)
analyst_agent = Agent(
    name="analyst", 
    model=analyst_llm,
    tools=[google_search],
    output_key="problem_list" # <-- Analyst's findings are saved here
)

# --- Define Agent 3: The Reporter Agent ---
reporter_llm = Gemini(
    model_name='gemini-1.5-flash-latest',
    api_key=os.environ["GOOGLE_API_KEY"],
    retry_options=retry_config,
    system_prompt=REPORTER_PROMPT
)
reporter_agent = Agent(
    name="reporter", 
    model=reporter_llm
    # No output_key needed, its chat response is the final answer
)

print("âœ… Step 5 Complete: All 'worker' agents are defined.")


import uuid

# --- Create the list of our "worker" agents ---
# This is the exact order they will run: Scan -> Analyze -> Report.
sub_agents = [
    scanner_agent,
    analyst_agent,
    reporter_agent
]

# --- Create the "Manager" Agent ---
# FIX: Appends a unique ID to the name to bypass the parent-link error,
# ensuring the cell runs successfully even if re-executed.
manager_agent = SequentialAgent(
    name=f"green_code_manager_{uuid.uuid4().hex[:6]}",
    sub_agents=sub_agents
)

print("âœ… Step 6 Complete: 'manager_agent' is defined.")


# --- Initialize Session and Runner ---
session_service = InMemorySessionService()

# --- Initialize the Runner ---
# As per the practice notebooks, we use InMemoryRunner for notebook execution.
runner = InMemoryRunner(
    agent=manager_agent # <-- Pass our "manager" agent
)

# --- Set our Test URL ---
# This is the initial message for our system.
test_url = "https://www.python.org" 

# --- Run the System! ---
print(f"ğŸš€ Starting Green-Code Advisor for: {test_url}\n")
print("---" * 20)

# --- Call .run_debug() with await ---
# This executes the full sequence.
response_events = await runner.run_debug(test_url)

# --- Extract and Print the Final Result ---
# The final message is in the last event of the returned list.
final_text = response_events[-1].content.parts[0].text
print("\nğŸ�� Final Report:\n")
print(final_text)

