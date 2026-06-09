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


from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
print("âœ… ADK components imported successfully.")


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"


# 1. Code Reader
code_reader = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="CodeReader",
    instruction="You are given raw source code. Return ONLY valid JSON:\n"
                "{\n  \"code_lines\": [\"1: line content...\", \"2: next line...\", ...]\n}\n"
                "Preserve exact indentation and content. Number every line starting from 1.",
)

# 2. Three parallel analysis agents
vuln_scanner = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="VulnScanner",
    instruction="Find security vulnerabilities. Return ONLY valid JSON with key 'vulnerabilities' (list of dicts with line_number, issue, severity, suggestion).",
    tools=[google_search]
)

sanitation_checker = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="SanitationChecker",
    instruction="Check input validation/sanitization. Return ONLY valid JSON with key 'sanitation_issues' (list of dicts).",
    tools=[google_search]
)

hackable_checker = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="HackableChecker",
    instruction="Find hard-coded secrets, weak crypto, etc. Return ONLY valid JSON with key 'hackable_parts' (list of dicts).",
    tools=[google_search]
)

# Parallel analysis
parallel_analysis = ParallelAgent(
    name="ParallelAnalysis",
    sub_agents=[vuln_scanner, sanitation_checker, hackable_checker]
)

# Final report generator
report_generator = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="ReportGenerator",
    instruction="Compile all findings into a beautiful markdown security report with sections, line numbers, and clear fix suggestions.",
)

# Full sequential workflow
root_agent = SequentialAgent(
    name="CodeSecurityAnalyzer",
    sub_agents=[code_reader, parallel_analysis, report_generator]
)


# Step 2: Set up Session Management
# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()

# Step 3: Create the Runner
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

print("âœ… Stateful agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")


# Run a conversation with two queries in the same session
# Notice: Both queries are part of the SAME session, so context is maintained
await run_session(
    runner,
    ["<?php echo $abcd;"],
    "stateful-agentic-session",
)


# Run a conversation with two queries in the same session
# Notice: Both queries are part of the SAME session, so context is maintained
await run_session(
    runner,
    [
        """
<?php
// Super vulnerable PHP application - DO NOT USE IN PRODUCTION!!

$db_host = "localhost";
$db_user = "root";
$db_pass = "admin123";          // Hard-coded credentials
$db_name = "myapp";

$link = mysqli_connect($db_host, $db_user, $db_pass, $db_name);

// No input validation whatsoever
$id     = $_GET['id'];
$query  = "SELECT * FROM users WHERE id = $id";   // Direct SQL injection
$result = mysqli_query($link, $query);

// Direct command injection
$ip = $_GET['ip'];
system("ping -c 4 " . $ip);                       // RCE via system()

// Reflected XSS everywhere
$name = $_GET['name'];
echo "<h1>Welcome, $name!</h1>";                  // XSS

// File inclusion vulnerability
$page = $_GET['page'];
include($page . ".php");                          // LFI / RFI

// Path traversal + arbitrary file read
$file = $_GET['file'];
readfile("/var/www/uploads/" . $file);            // Directory traversal

// Insecure deserialization (simplified example)
$data = $_POST['data'];
unserialize($data);                               // Potential gadget chain

// Hard-coded API keys
define("STRIPE_SECRET", "sk_live_51J2k9...real_key_here");
define("AWS_SECRET", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");

// Weak cryptography
$password = md5($_POST['pass']);                  // MD5 is broken

// Debug mode left on
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Dangerous eval usage
$code = $_POST['code'];
eval($code);                                      // Full RCE

echo "Debug: Current user IP is " . $_SERVER['REMOTE_ADDR'];
?>
        """
    ],
    "stateful-agentic-session",
)

