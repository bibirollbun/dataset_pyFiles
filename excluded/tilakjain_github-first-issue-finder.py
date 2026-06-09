!pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, AgentTool
from google.genai import types
from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext

print("ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


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


def get_good_first_issues(repo: str) -> list:
    import requests
    if not repo:
        repo_to_search = "tilakjain619/LearnX" # Placeholder for user input
    else:
        repo_to_search = repo
    GITHUB_API = "https://api.github.com"
    url = f"{GITHUB_API}/repos/{repo_to_search}/issues"
    params = {"labels": "good first issue", "state": "open"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        issues = response.json()
        return [{"title": i["title"], "url": i["html_url"]} for i in issues]
    return {"error": "Could not fetch issues"}

print("Function to get good first issues created")


# try out function for demo
get_good_first_issues('cutenode/good-first-issue')


# Research Agent: Its job is to use the google_search tool and present findings.
research_agent = Agent(
    name="ResearchAgent",
    model="gemini-2.5-flash",
    instruction="""You are a specialized research agent. Your input will be a repository name 
    in the 'owner/repo' format.
    
    **TASK:** Use the 'get_good_first_issues' tool with the repository input to find 4-5 recent good first issues.
    """,
    tools=[get_good_first_issues], 
    output_key="research_findings",
)
print("Research Agent with Fallback created.")


explainer_agent = Agent(
    name="ExplainerAgent",
    model="gemini-2.5-flash",
    instruction="""Read the provided research findings: {research_findings}
    
    If the findings are empty or indicate an error, respond by stating clearly that no issues were found for the repository. 
    Otherwise, explain each issue to a beginner in a simple and easy way.""",
    output_key="final_summary",
)

print("Explainer Agent created.")


root_agent = Agent(
    name="ResearchCoordinator",
    model="gemini-2.5-flash",
    instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
    
    1. **Input Parsing:** Analyze the user's query (e.g., 'Find issues in cutenode/good-first-issue repo'). You MUST extract ONLY the clean 'owner/repo' string ('cutenode/good-first-issue').
    
    2. **Call ResearchAgent:** You MUST call the `ResearchAgent` tool, **passing ONLY the extracted 'owner/repo' string** as the input argument.
    
    3. **Call ExplainerAgent:** After receiving the findings, you MUST call the **`ExplainerAgent`** tool to explain the issues/findings in a simple manner.
    
    4. Finally, present the final explanation clearly to the user as your response.""",
    
    tools=[
        AgentTool(research_agent),
        AgentTool(explainer_agent) 
    ],
)
print("Root Agent created with Advanced Input Handling.")


import warnings

APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session
MODEL_NAME = "gemini-2.5-flash-lite"

warnings.filterwarnings("ignore", category=UserWarning, module="google_genai")
session_service = InMemorySessionService()

runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
print("Stateful Agent Initialized")


!pip install rich


# Assuming runner and session_service are initialized as before (make sure all above cells are executed).
# Assuming run_session is defined and available.
import asyncio
import sys 

from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown

# Initialize the Rich Console once outside the loop for best performance
rich_console = Console() 

async def interactive_chat(runner_instance, session_service, session_name="live_chat"):
    """Starts a continuous, interactive chat session with the agent, rendering Markdown."""

     # 1. Initialize Session
    app_name = runner_instance.app_name
    try:
        # Create or retrieve the session for continuous context
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
        print(f"\n--- Starting Interactive Chat Session: {session_name} ---")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
    except Exception:
        # If create fails, assume it exists and retrieve it
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    # 2. Start the Continuous Loop
    while True:
        try:
            user_input = input("User > ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nConversation ended. Session context saved.")
                break

            # Assuming types.Content is available
            query_content = types.Content(role="user", parts=[types.Part(text=user_input)])
            
            # Print the Agent prefix using Rich for consistent styling
            # Print with a newline separator to keep user/agent turns distinct
            rich_console.print(Text("Agent > ", style="bold magenta")) 

            # 3. Stream the response (Raw Text)
            full_markdown_output = ""
            
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query_content
            ):
                if event.content and event.content.parts:
                    text_chunk = event.content.parts[0].text
                    
                    if text_chunk and text_chunk.strip():
                        # Append the chunk to the buffer
                        full_markdown_output += text_chunk
                        
                        # Print the text chunk raw for the "talking feel"
                        rich_console.print(
                            Markdown(text_chunk), 
                            end="", 
                        )

            # 4. Final Cleanup and Rendering
            
            # Print a final newline after the raw stream is done
            print() 
            
            # If the output contains significant Markdown (like headings, lists),
            # re-render the final, complete block for clean formatting.
            if full_markdown_output.strip().startswith(('#', '*')):
                 rich_console.rule("Formatted Output", style="dim")
                 rich_console.print(Markdown(full_markdown_output))
                 rich_console.rule()


        except KeyboardInterrupt:
            print("\nConversation interrupted.")
            break
        except Exception as e:
            print(f"\nAn error occurred during the conversation: {e}")
            break


await interactive_chat(runner, session_service)




