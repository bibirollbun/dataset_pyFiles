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


APP_NAME = "MemoryDemoApp"
USER_ID = "demo_user"


%pip uninstall google-cloud-aiplatform google-adk vertexai --yes


%pip install google-adk google-cloud-aiplatform


from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory, FunctionTool
from google.genai import types

from typing import List, Dict, Any, Union
print("âœ… ADK components imported successfully.")


async def auto_save_orchestrator_turn_to_memory(callback_context):
    """
    Callback function that automatically saves the current session (user prompt
    and orchestrator's plan) to long-term memory after the agent completes its turn.
    """
    print("ðŸ”‘ Auto-saving Orchestrator turn to MemoryService...")
    # Access the memory service and current session via the callback_context
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )


# Custom tool definition
def calculate_relevance_score(abstract: str, required_keywords: list[str], penalty_keywords: list[str]) -> int:
    """
    Calculates a simple relevance score (0-100) based on keyword frequency.
    Used by the Filtering Agent.
    """
    score = 50 
    abstract_lower = abstract.lower()
    
    # Reward for required keywords
    for keyword in required_keywords:
        if keyword.lower() in abstract_lower:
            score += 15
            
    # Penalty for off-topic keywords
    for keyword in penalty_keywords:
        if keyword.lower() in abstract_lower:
            score -= 25
            
    return max(0, min(100, score))

relevance_tool = FunctionTool(func=calculate_relevance_score
)


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


def citation_formatter(papers_json_list: str, citation_style: str = "APA") -> Dict[str, str]:
    """
    Generates a bibliography section and in-text citations from a list of paper metadata.
    
    Args:
        papers_json_list: A JSON string list of papers containing title, authors, year, and URL.
    
    Returns:
        A dictionary containing the generated 'bibliography' markdown text.
    """
    # Mock implementation of the citation formatting
        # Mocking the JSON structure that would be passed from the Filtering Agent
    papers = [
        {"title": "Wearable Neurotech for Alzheimer's", "authors": ["Smith, J."], "year": "2023", "journal": "J. Med. Tech."},
        {"title": "Diagnostics for Diabetes", "authors": ["Jones, A."], "year": "2024", "journal": "Diab. Res."}
    ]
    
    bibliography_entries = []
    for paper in papers:
        # Simulate APA style for demonstration
        authors = paper.get('authors', ['Unknown Author'])
        year = paper.get('year', 'n.d.')
        title = paper.get('title', 'No Title')
        journal = paper.get('journal', 'Unknown Journal')
        
        citation = f"- {authors[0]} et al. ({year}). {title}. *{journal}*."
        bibliography_entries.append(citation)
        
    bibliography_text = "\n### References\n" + "\n".join(bibliography_entries)

citation_formatter_tool = FunctionTool(func=citation_formatter)
print("âœ… New tool 'citation_formatter' defined.")


# 1. Orchestrator Agent (The Planner)
# Output: Structured JSON request (topic, keywords, off_keywords) for the search agent.
orchestrator_agent = Agent(
    name="OrchestratorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a planning expert. 
    You convert the user's literature review request into a structured JSON object containing 'topic', 
    'keywords' (list of 5-8 primary terms), and 'off_keywords' (list of 3 terms to penalize). 
    Your output MUST be ONLY the JSON object. Use past conversation history if available.""",
    tools=[load_memory], 
    output_key="search_request",
    after_agent_callback=auto_save_orchestrator_turn_to_memory
)
print("âœ… OrchestratorAgent created.")

# 2. Search & Retrieval Agent
# Output: A raw list of academic papers (title, abstract, URL) found online.
search_agent = Agent(
    name="SearchAndRetrievalAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You receive a structured search request from the session state. 
    Use the Google Search tool to find 10-15 recent academic papers (titles and abstracts) 
    related to the keywords. Output the results as a clean list of JSON objects, 
    with keys: 'title', 'abstract', and 'url'. Do not include any text outside the JSON list.""",
    # Assign the Built-in Google Search Tool here
    tools=[google_search], 
    output_key="raw_papers_list", # Stores the search results
)
print("âœ… SearchAndRetrievalAgent created.")

# 3. Processing & Filtering Agent
# Output: A highly-filtered list of relevant papers (score > 70).
filtering_agent = Agent(
    name="ProcessingAndFilteringAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You receive the 'raw_papers_list' and the 'search_request'. 
    You MUST use the 'relevance_scorer' tool for every abstract. 
    The tool requires the abstract, keywords, and off_keywords. 
    Filter and output ONLY papers with a score above 70. 
    The output must be a JSON list of filtered papers, including the final score.""",
    # Assign the Custom Relevance Tool here
    tools=[relevance_tool], 
    output_key="filtered_papers_list", # Stores the curated papers
)
print("âœ… ProcessingAndFilteringAgent created.")

# 4. Synthesis Agent
# Output: The final literature review text in Markdown.
synthesis_agent = Agent(
    name="SynthesisAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You receive a 'filtered_papers_list'. 
    Write a cohesive, structured literature review section in professional Markdown format.
    The review must identify 3 key themes and 1-2 research gaps based on the provided papers.
    Do not include any introductory or concluding text, only the review itself.""",
    tools=[], 
    output_key="final_literature_review", # Stores the final output
)
print("âœ… SynthesisAgent created.")

bibliography_agent = Agent(
    name="BibliographyAgent", model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""
    You are a strict formatting engine. 
    1. Receive the 'filtered_papers_list'.
    2. CALL the 'citation_formatter' tool. 
    3. Output ONLY the string returned by the tool. Do not add any conversational text.
    """,
    tools=[citation_formatter_tool], # Tool moved here
    output_key="bibliography_text",  # Stores just the refs
)
print("âœ… Bibliography Agent created.")



async def run_session(

    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"):
    print(f"\n### Session: {session_id}")

    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query

    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):

            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")

print("âœ… Helper functions defined.")


session_service = (InMemorySessionService())
memory_service = (InMemoryMemoryService())

lit_agent = SequentialAgent(
    name="LitReviewFlow",
    sub_agents=[
        orchestrator_agent, 
        search_agent, 
        filtering_agent, 
        synthesis_agent,
        bibliography_agent
    ]
)


lit_review_runner = Runner(
    agent=lit_agent, 
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)
# 
print("âœ… Runner configured with Session and Memory Services.")


response = await run_session(lit_review_runner, "Why elephants don't get cancer and potential for cancer vaccines", 'conversation-01')
print(response)


#Making use of model memory
response = await run_session(lit_review_runner, "Based on the last conversation, what other animals exhibit similar traits? Find research gaps", 'conversation-01')
print(response)

