
# ===========================================================
#  YouTube SEO Concierge Agent 
#  1. Setup - Confiugre necessary API Keys 
# ===========================================================
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    YOUTUBE_API_KEY = UserSecretsClient().get_secret("YOUTUBE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API + YouTube API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' or 'YOUTUBE_API_KEY' to your Kaggle secrets. Details: {e}")



from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")

# Google API client for YouTube
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
print("âœ… Google API client for YouTube components imported successfully.")


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


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# ===========================================================
# 3. AGENT TOOLS
# ===========================================================

import os
import re
import json
from typing import Dict, Any  

from google.adk.tools import AgentTool
from googleapiclient.discovery import build 
from googleapiclient.errors import HttpError


def fetch_youtube_video_data(video_id: str, api_key: str) -> Dict[str, Any]:
    """
    Fetches vital metadata and top comments for a given YouTube video ID.

    Args:
        video_id: The unique 11-character identifier for the YouTube video.
        api_key: Your YouTube Data API key.

    Returns:
        A dictionary containing cleaned metadata and comments, or an error message.
    """
    print(f"ğŸ› ï¸�  [Function Running] Fetching YouTube data for video_id: {video_id}")
    if not api_key:
        return {"error": "YouTube API key was not provided."}

    # Validate the video ID format
    if not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
        return {"error": f"Invalid YouTube video_id format: '{video_id}'. It must be 11 characters long."}

    try:
        # Build the YouTube service object
        youtube = build("youtube", "v3", developerKey=api_key)

        # --- Fetch Video Metadata and Statistics ---
        video_request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        video_response = video_request.execute()

        if not video_response.get("items"):
            return {"error": f"No video found with ID: {video_id}"}

        item = video_response["items"][0]
        snippet, stats = item.get("snippet", {}), item.get("statistics", {})

        clean_metadata = {
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "tags": snippet.get("tags", []),
            "viewCount": stats.get("viewCount"),
            "likeCount": stats.get("likeCount")
        }

        # --- Fetch Top Comments ---
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=10,
            order="relevance",
            textFormat='plainText'
        )
        comment_response = comment_request.execute()

        clean_comments = [
            item['snippet']['topLevelComment']['snippet']['textDisplay']
            for item in comment_response.get("items", [])
        ]

        print("âœ… [Function Success] YouTube data fetched.")
        return {"metadata": clean_metadata, "comments": clean_comments}

    except HttpError as e:
        # Handle API errors gracefully
        try:
            error_content = json.loads(e.content.decode())
            error_message = error_content.get("error", {}).get("message", "Unknown API error")
        except (json.JSONDecodeError, AttributeError):
            error_message = str(e.content)
        
        full_error = f"An API error occurred: {error_message}"
        print(f"â�Œ [Function Error] {full_error}")
        return {"error": full_error}
    
    except Exception as e:
        # Handle other unexpected errors
        msg = f"An unexpected error occurred: {e}"
        print(f"â�Œ [Function Error] {msg}")
        return {"error": msg}

# --- Example of how to use the function ---
if __name__ == "__main__":
    example_id = "jV1vkHv4zq8"  # Example: "Introducing Gemini"
    
    # Call the function directly
    video_data = fetch_youtube_video_data(video_id=example_id, api_key=YOUTUBE_API_KEY)
    
    # Print the result nicely
    print("\n--- Function Output ---")
    print(json.dumps(video_data, indent=2))




llm = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# A) Research Agent - FIXED as per your request
research_agent = LlmAgent(
    name = "ResearchAgent", 
    model = llm,
    instruction="""You are a specialized research agent. Your job is to analyze the user's query and use the correct tool.
- If the query contains an 11-character video ID, you MUST use the 'youtube_video_data' tool.
- Otherwise, you MUST use 'google_search' to find trending topics related to the query.""",
    # Pass the tool CLASS and FUNCTION directly. The ADK handles initialization.
    tools=[
        fetch_youtube_video_data,
        google_search
          ],
    output_key="research_findings", # The result of this agent will be stored in the session state with this key.
)

print("âœ… research_agent created.")

# B) SEO Agent
seo_agent = LlmAgent(
    name="SEOAgent",
    model=llm,
    instruction="""You are a world-class YouTube SEO expert. Based on the provided research data, you MUST generate an optimized title, a compelling description, and a list of relevant tags.
Your final output must be a single, valid JSON object with three keys: "optimized_title", "optimized_description", and "suggested_tags".""",
    # This agent performs a transformation and needs no external tools.
    tools=[],
)
print("âœ… seo_agent created.")

# C) Evaluation Agent
evaluation_agent = LlmAgent(
    name="EvaluationAgent",
    model=llm,
    instruction="""You are a meticulous YouTube SEO analyst. You will evaluate the provided SEO suggestions against the original video metadata.
You MUST provide a score from 1-100 and a brief justification for the title, description, and tags.
Your final output must be a single, valid JSON object with keys: "title_score", "title_justification", "description_score", "description_justification", "tags_score", "tags_justification".""",
    tools=[],
)
print("âœ… evaluation_agent created.")



agent_test_runner = InMemoryRunner(seo_agent)
test_response = await agent_test_runner.run_debug("Research Agent", verbose=True)
show_python_code_and_result(test_response)


# D) Supervisor Agent - The Orchestrator
supervisor_agent = LlmAgent(
    name="SupervisorAgent",
    model=llm,
    instruction="""You are the supervisor. Your job is to orchestrate a workflow to optimize a YouTube video based on a user query.

You MUST follow these steps in order:
1. Call the `ResearchAgent` with the original user query.
2. Take the output from the `ResearchAgent` and pass it to the `SEOAgent`.
3. Take the outputs from both the `ResearchAgent` (for original data) and the `SEOAgent` (for new suggestions) and pass them to the `EvaluationAgent`.
4. Your final output should be a JSON object containing the results of all three steps, with keys "research_data", "seo_suggestions", and "evaluation". Do not add any other commentary.
""",
    # The supervisor's tools ARE the other agents.
    # The CodeExecutor allows the supervisor to generate and run python code.
    tools=[
        AgentTool(research_agent), 
        AgentTool(seo_agent), 
        AgentTool(evaluation_agent)
    ],
)

print("âœ… supervisor_agent created.")



# --- 4. Main Execution Block ---
"""Main function to run the agent system."""
example_video_id = "6Iq7wvc4i_c" # Example: "Introducing Gemini"
user_request = f"Please optimize the YouTube video with ID: {example_video_id}"
# Define a runner
enhanced_runner = InMemoryRunner(agent=supervisor_agent)

# Test the enhanced agent
response = await enhanced_runner.run_debug(
    user_request
)

print("\n\n--- ğŸš€ FINAL OUTPUT ğŸš€")
show_python_code_and_result(response)
print("--- END OF EXECUTION ---")

