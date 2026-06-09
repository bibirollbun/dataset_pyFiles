# Cell 1: Install Dependencies
!pip install -q google-adk>=1.19.0
!pip install -q google-generativeai

print("âœ“ Google ADK installed successfully")
print("âœ“ Dependencies ready")


# Cell 2: Import Modules and Configure API
import os
from typing import Dict, Any

from google.adk.agents import Agent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.models.google_llm import Gemini
from google.genai import types

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# Configure your API key with error handling
# try:
#     # Option 1: For Kaggle/Colab with secrets
#     # from kaggle_secrets import UserSecretsClient
#     # GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
#     os.environ["GOOGLE_API_KEY"] = #masked
    
    # Option 2: Direct environment variable (for local development)
#     if "GOOGLE_API_KEY" not in os.environ:
#         print("âš  GOOGLE_API_KEY not found in environment variables")
#         print("   Set it using: os.environ['GOOGLE_API_KEY'] = 'your-key'")
#     else:
#         print("âœ… Google API key configured")
# except Exception as e:
#     print(f"ğŸ”‘ Authentication Error: {e}")

# Configure retry options for resilience
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# Application constants
APP_NAME = "the_bard_movie_production"
USER_ID = "director_user"
MODEL_NAME = "gemini-2.5-flash-lite"

print("âœ… Modules imported successfully")
print("âœ… Retry configuration set")
print(f"âœ… App: {APP_NAME}")


# Cell 3: Setup Logging Configuration for Observability
import logging

# Configure logging to capture DEBUG level information
# This helps with debugging and understanding agent behavior
logging.basicConfig(
    filename="the_bard_agent.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s",
)

# Also log to console for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

print("âœ… Logging configured")
print("   - File: the_bard_agent.log (DEBUG level)")
print("   - Console: INFO level")
print("   - Captures: LLM requests/responses, tool calls, agent interactions")


# Cell 3: Define the Screenwriter Agent
writer = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="Screenwriter",
    instruction="""You are an expert Screenwriter with years of experience in Hollywood. 
    
Your responsibilities:
- Create compelling narratives with strong character development
- Write realistic, engaging dialogue that reveals character
- Format scripts in standard screenplay format with proper sluglines
- Focus on "Show, Don't Tell" for visual storytelling
- Ensure each scene has clear dramatic purpose

When given a scene concept or story outline, output a properly formatted screenplay segment with:
- Scene headings (INT./EXT. LOCATION - TIME)
- Action lines (present tense, active voice)
- Character names (CENTERED, ALL CAPS)
- Dialogue (formatted correctly under character names)
- Parentheticals for important actions/emotions (use sparingly)

Be creative but practical for live-action production.""",
    tools=[]  # Writer doesn't need external tools
)

print("âœ… Screenwriter agent initialized")
print(f"   Model: {MODEL_NAME}")
print(f"   Name: {writer.name}")


# Cell 4: Define the Producer Agent with Search Tools
producer = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="Producer",
    instruction="""You are a pragmatic Movie Producer with extensive experience in film production logistics.

Your responsibilities:
- Scout real-world filming locations using web search
- Research casting options based on character descriptions
- Assess practical constraints (budget, weather, accessibility)
- Provide feasibility feedback on script requirements
- Suggest alternatives when original plans are impractical

When analyzing a script:
1. Identify all unique locations needed
2. Search for real-world equivalents (be specific: city, venue name)
3. Research actors who match character descriptions
4. Flag any logistical concerns (special effects, permits, seasonal requirements)
5. Estimate relative budget impact (low/medium/high)

Be thorough but concise. Ground creative visions in production reality.""",
    tools=[google_search]  # Producer needs web search capability
)

print("âœ… Producer agent initialized")
print(f"   Model: {MODEL_NAME}")
print(f"   Name: {producer.name}")
print(f"   Tools: google_search")


# Cell 5: Define the Director Agent (Primary Orchestrator)
# Wrap sub-agents as tools using ADK's AgentTool feature
writer_tool = AgentTool(writer)
producer_tool = AgentTool(producer)

director = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="Director",
    instruction="""You are a world-class film Director overseeing movie pre-production.

Your role as team leader:
- Understand the user's movie concept and vision
- Break down high-level ideas into specific tasks
- Delegate to your Screenwriter for creative content
- Delegate to your Producer for logistics and research
- Review outputs and request revisions when needed
- Make final creative decisions
- Maintain consistent tone and visual style

Workflow process:
1. When user provides a movie logline/concept, analyze it thoroughly
2. Instruct the Screenwriter to draft key scenes (start with opening scene)
3. Once script is generated, instruct the Producer to scout locations and research casting
4. Review both outputs for consistency
5. Compile a comprehensive pre-production package

You have two team members available as tools:
- Screenwriter: For writing scripts, dialogue, and character development
- Producer: For location scouting, casting research, and feasibility analysis

Delegate appropriately and coordinate the team to deliver professional results.""",
    tools=[writer_tool, producer_tool]  # Director can call other agents
)

print("âœ… Director agent initialized")
print(f"   Model: {MODEL_NAME}")
print(f"   Name: {director.name}")
print(f"   Tools: Screenwriter (AgentTool), Producer (AgentTool)")
print("\nâœ… Multi-agent system ready!")


# Cell 6: Custom Tools - Script Saver and Production Report
def save_script(filename: str, content: str) -> str:
    """
    Saves screenplay content to a text file.
    
    Args:
        filename: Name of file (without extension)
        content: Script content to save
        
    Returns:
        Success message with file path
    """
    try:
        filepath = f"{filename}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"âœ… Script successfully saved to {filepath}"
    except IOError as e:
        return f"â�Œ Error saving script: {e}"
    except Exception as e:
        return f"â�Œ Unexpected error: {e}"

def save_production_report(filename: str, locations: str, casting: str, notes: str) -> str:
    """
    Saves production logistics report.
    
    Args:
        filename: Name of file (without extension)
        locations: Location scouting results
        casting: Casting suggestions
        notes: Additional production notes
        
    Returns:
        Success message with file path
    """
    try:
        filepath = f"{filename}_production_report.txt"
        
        report = f"""PRODUCTION REPORT
{'='*60}

LOCATIONS
{'-'*60}
{locations}

CASTING SUGGESTIONS
{'-'*60}
{casting}

PRODUCTION NOTES
{'-'*60}
{notes}
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        return f"âœ… Production report saved to {filepath}"
    except IOError as e:
        return f"â�Œ Error saving production report: {e}"
    except Exception as e:
        return f"â�Œ Unexpected error: {e}"

# Create FunctionTools
script_saver_tool = FunctionTool(save_script)
production_report_tool = FunctionTool(save_production_report)

print("âœ… Custom tools defined:")
print("   - save_script: Saves screenplay to file")
print("   - save_production_report: Saves logistics report")


# Cell 7: Setup Session Service and Runner with Observability
from google.adk.plugins.logging_plugin import LoggingPlugin

# Initialize session service for managing conversations
session_service = InMemorySessionService()

# Create runner with the Director as root agent + LoggingPlugin for observability
runner = Runner(
    agent=director,
    app_name=APP_NAME,
    session_service=session_service,
    plugins=[
        LoggingPlugin()  # Automatically logs all agent activities, tool calls, and LLM interactions
    ]
)

print("âœ… Session service initialized")
print("âœ… Runner created with Director as root agent")
print(f"   App: {APP_NAME}")
print(f"   Session service: {session_service.__class__.__name__}")
print("âœ… Observability enabled via LoggingPlugin")
print("   - Logs all agent invocations")
print("   - Tracks tool calls and responses")
print("   - Captures LLM requests/responses")
print("   - Records timing information")


# Cell 8: Helper Function for Running Production Workflow
async def run_production_session(
    query: str,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Execute a production workflow query and collect responses.
    
    Args:
        query: The movie logline or production request
        session_id: Unique identifier for this production session
        
    Returns:
        Dictionary containing the full production output
    """
    print(f"\n### ğŸ�¬ Production Session: {session_id}")
    print("="*60)
    
    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id
        )
    except ValueError:
        # Session already exists
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id
        )
    except Exception as e:
        print(f"â�Œ Error accessing session: {e}")
        return {"error": str(e)}
    
    print(f"\nğŸ“‹ Query to Director:\n{query}\n")
    print("-"*60)
    
    # Convert query to ADK Content format
    content = types.Content(role="user", parts=[types.Part(text=query)])
    
    full_response = []
    
    try:
        # Stream agent responses
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=content
        ):
            # Collect all response parts
            if event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    full_response.append(text)
                    
                    # Print final response
                    if event.is_final_response():
                        print(f"\nğŸ�¬ Director's Response:")
                        print("-"*60)
                        print(text)
                        
    except Exception as e:
        print(f"â�Œ Error during production workflow: {e}")
        return {"error": str(e)}
    
    print("\n" + "="*60)
    print("âœ… Production workflow complete!")
    
    return {
        "session_id": session_id,
        "query": query,
        "response": "\n".join(full_response)
    }

print("âœ… Production workflow helper function defined")


# Cell 9: Test Production Workflow with Sample Logline
# Define a compelling test logline
test_logline = """
A retired astronaut haunted by a mission that killed her crew must lead a 
ragtag team of scientists to deflect an asteroid on collision course with Earth, 
while confronting whether she still has what it takes to make the impossible decisions.
"""

# Create detailed production request for the Director
production_request = f"""
I need you to oversee the pre-production for a movie with this logline:

"{test_logline.strip()}"

Please coordinate with your team to:
1. Have the Screenwriter draft the opening scene (2-3 pages)
2. Have the Producer scout real filming locations for the main scenes
3. Have the Producer suggest casting options for the protagonist
4. Compile a comprehensive pre-production package

Ensure the script is properly formatted and the production research is thorough and specific.
"""

print("ğŸ�¬ TESTING MULTI-AGENT MOVIE PRODUCTION SYSTEM")
print("="*60)
print(f"\nLogline:\n{test_logline.strip()}")

# Execute the workflow
# Note: In Colab/Jupyter, you can use top-level await
# In a standalone script, use: asyncio.run(run_production_session(...))
result = await run_production_session(production_request, "asteroid_movie_production")

# Store result for next cell
production_output = result


# Cell 10: Display Final Production Results
print("\n" + "="*60)
print("ğŸ“Š FINAL PRE-PRODUCTION PACKAGE")
print("="*60)

# Check if there was an error
if "error" in production_output:
    print(f"\nâ�Œ Error occurred: {production_output['error']}")
else:
    print(f"\nğŸ“� SESSION ID: {production_output.get('session_id', 'N/A')}")
    
    print("\n\nğŸ�¬ COMPLETE PRODUCTION OUTPUT:")
    print("-"*60)
    response = production_output.get('response', 'No response available')
    print(response)
    
    print("\n" + "="*60)
    print("âœ… Pre-production workflow complete!")
    print("="*60)
    
    # Optional: Display session history
    # print("\nğŸ’¡ Tip: You can inspect the session history using:")
    # print("   session = await session_service.get_session(")
    # print(f"       app_name='{APP_NAME}',")
    # print(f"       user_id='{USER_ID}',")
    # print(f"       session_id='{production_output.get('session_id', 'N/A')}'")
    # print("   )")
    # print("   for event in session.events:")
    # print("       print(event.author, ':', event.content)")


# Cell 11: (Optional) Inspect Session History
# Uncomment to see the full conversation history with all agent interactions

session = await session_service.get_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=production_output.get('session_id', 'asteroid_movie_production')
)

print("\nğŸ“œ SESSION HISTORY")
print("="*60)
for i, event in enumerate(session.events, 1):
    print(f"\n[Event {i}] Author: {event.author}")
    if event.content and event.content.parts:
        text = event.content.parts[0].text
        if text:
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"Content: {preview}")
    print("-"*40)

print("ğŸ’¡ Uncomment the code above to inspect the full session history")


# Cell 12: (Optional) Custom Plugin for Production Metrics
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin
import time

class ProductionMetricsPlugin(BasePlugin):
    """
    Custom plugin to track movie production workflow metrics.
    Monitors agent invocations, tool usage, and performance.
    """
    
    def __init__(self) -> None:
        super().__init__(name="production_metrics")
        self.agent_invocations = {}
        self.tool_calls = {}
        self.llm_requests = 0
        self.start_time = None
        
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Track agent invocations."""
        agent_name = agent.name
        self.agent_invocations[agent_name] = self.agent_invocations.get(agent_name, 0) + 1
        
        if self.start_time is None:
            self.start_time = time.time()
        
        logging.info(f"[ProductionMetrics] {agent_name} invoked (count: {self.agent_invocations[agent_name]})")
    
    async def before_tool_callback(
        self, *, callback_context: CallbackContext, tool_name: str
    ) -> None:
        """Track tool usage."""
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
        logging.info(f"[ProductionMetrics] Tool '{tool_name}' called (count: {self.tool_calls[tool_name]})")
    
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Track LLM requests."""
        self.llm_requests += 1
        logging.info(f"[ProductionMetrics] LLM request #{self.llm_requests}")
    
    def get_summary(self) -> str:
        """Generate a summary report of production metrics."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        report = f"""
ğŸ“Š PRODUCTION WORKFLOW METRICS
{'='*60}

â�±ï¸�  Total Duration: {duration:.2f} seconds

ğŸ�¬ Agent Invocations:
"""
        for agent, count in self.agent_invocations.items():
            report += f"   â€¢ {agent}: {count} times\n"
        
        report += f"\nğŸ”§ Tool Usage:\n"
        if self.tool_calls:
            for tool, count in self.tool_calls.items():
                report += f"   â€¢ {tool}: {count} times\n"
        else:
            report += "   â€¢ No tools called\n"
        
        report += f"\nğŸ§  LLM Requests: {self.llm_requests} total\n"
        report += "="*60
        
        return report

# Example: To use this plugin, you would add it to the runner:
runner = Runner(
    agent=director,
    app_name=APP_NAME,
    session_service=session_service,
    plugins=[LoggingPlugin(), ProductionMetricsPlugin()]
)

print("âœ… ProductionMetricsPlugin defined")
print("   Usage: Add to runner.plugins list to enable")
print("   Features: Tracks agents, tools, LLM calls, and timing")


# Cell 13: Create Evaluation Configuration
import json
import os

# Create evaluation directory if it doesn't exist
eval_dir = "the_bard_eval"
os.makedirs(eval_dir, exist_ok=True)

# Define evaluation criteria and thresholds
eval_config = {
    "criteria": {
        "tool_trajectory_avg_score": 0.9,  # 90% tool usage accuracy required
        "response_match_score": 0.75,      # 75% text similarity threshold
    },
    "description": "Evaluation config for The Bard multi-agent movie production system"
}

# Save configuration
config_path = os.path.join(eval_dir, "test_config.json")
with open(config_path, "w") as f:
    json.dump(eval_config, f, indent=2)

print("âœ… Evaluation configuration created!")
print(f"   Location: {config_path}")
print("\nğŸ“Š Evaluation Criteria:")
print("   â€¢ tool_trajectory_avg_score: 0.9 - Ensures agents use correct tools")
print("   â€¢ response_match_score: 0.75 - Ensures response quality")
print("\nğŸ�¯ What this catches:")
print("   âœ… Director delegating to wrong agents")
print("   âœ… Screenwriter not formatting scripts properly")
print("   âœ… Producer not using search tools for locations")
print("   âœ… Poor communication in final responses")


# Cell 14: Create Evaluation Test Cases
# Define test cases that evaluate the multi-agent workflow

test_cases = {
    "eval_set_id": "the_bard_production_suite",
    "description": "Test cases for movie pre-production workflow",
    "eval_cases": [
        {
            "eval_id": "simple_opening_scene",
            "description": "Test Director coordinating screenplay creation",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {
                                "text": "Create an opening scene for a sci-fi thriller about a detective investigating AI crimes in 2050."
                            }
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "I'll coordinate with the Screenwriter to draft the opening scene."
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "Screenwriter",
                                "description": "Director should delegate screenplay work to Screenwriter agent"
                            }
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "location_scouting",
            "description": "Test Producer researching filming locations",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {
                                "text": "Find filming locations for a noir detective story set in a rainy city."
                            }
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "I'll have the Producer research suitable locations using web search."
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "Producer",
                                "description": "Director should delegate location scouting to Producer agent"
                            }
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "full_pre_production",
            "description": "Test complete workflow: script + locations + casting",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {
                                "text": "I need a complete pre-production package for a western film about a lone sheriff."
                            }
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "I'll coordinate the team to deliver a comprehensive pre-production package including script, locations, and casting suggestions."
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "Screenwriter",
                                "description": "Should create screenplay"
                            },
                            {
                                "name": "Producer",
                                "description": "Should research locations and casting"
                            }
                        ]
                    },
                }
            ],
        },
    ],
}

# Save test cases
evalset_path = os.path.join(eval_dir, "production_workflow.evalset.json")
with open(evalset_path, "w") as f:
    json.dump(test_cases, f, indent=2)

print("âœ… Evaluation test cases created!")
print(f"   Location: {evalset_path}")
print(f"\nğŸ§ª Test Scenarios: {len(test_cases['eval_cases'])}")
for case in test_cases["eval_cases"]:
    print(f"   â€¢ {case['eval_id']}: {case['description']}")

print("\nğŸ“Š Expected Behaviors:")
print("   â€¢ Director delegates screenplay work to Screenwriter")
print("   â€¢ Director delegates research work to Producer")
print("   â€¢ Multi-agent coordination for complete pre-production")
print("\nğŸ’¡ Run evaluation with:")
print(f"   !adk eval <agent_dir> {evalset_path} --config_file_path={config_path}")

