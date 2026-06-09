# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%writefile config.py
import os
from kaggle_secrets import UserSecretsClient

# Default System Configuration
DEFAULT_POLICY_TOPIC = "Implementation of Universal Basic Income (UBI) in developing economies"
OUTPUT_FILENAME = "policy_report.md"

# Initialize Keys as None
GOOGLE_API_KEY = None
TAVILY_API_KEY = None

# Attempt to load secrets
try:
    user_secrets = UserSecretsClient()
    
    # We use try/except for individual keys in case one is missing but the other exists
    try:
        GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    except Exception:
        pass
        
    try:
        TAVILY_API_KEY = user_secrets.get_secret("TAVILY_API_KEY")
    except Exception:
        pass

except Exception as e:
    # This catches cases where UserSecretsClient fails (e.g. running locally)
    print(f"âš ï¸� Config Warning: Could not initialize Secrets Client: {e}")

# Validation Log
if not GOOGLE_API_KEY or not TAVILY_API_KEY:
    print("âš ï¸� Config Warning: One or more API Keys are missing in Kaggle Secrets.")


!pip install --force-reinstall --no-deps google-adk google-generativeai tavily-python gradio python-dotenv


from tavily import TavilyClient 
from google.adk.tools import FunctionTool

# Import your config file (or object)
import config #configuration

def get_tavily_search_tool(api_key=None):
    """
    Returns a FunctionTool configured with the Tavily API key.
    """
    # 1. Try to get key from argument, then from config
    tavily_api_key = api_key if api_key else getattr(config, 'TAVILY_API_KEY', None)
    
    if not tavily_api_key:
        # Fallback: Check environment variable if config fails
        import os
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        
    if not tavily_api_key:
        print("âš ï¸� Tavily Tool Warning: No API Key found.")
        # We return a dummy tool or raise error depending on preference. 
        # Raising error is safer for debugging.
        raise ValueError("Tavily API Key is missing in Config and Environment.")
    
    tavily_client = TavilyClient(api_key=tavily_api_key)

    def fetch_policy_data(query: str) -> str:
        """
        Searches the web for real-time data, statistics, and news about the policy.
        ALWAYS use this to get facts before answering.
        
        Args:
            query: The search query string.
            
        Returns:
            String containing formatted search results with sources.
        """
        print(f"\nğŸ”� [Tavily Tool] Searching for: '{query}'...")
        try:
            # Use 'advanced' depth for better facts
            response = tavily_client.search(query, search_depth="advanced", max_results=3)
            context = []
            if 'results' in response:
                for result in response['results']:
                    context.append(f"Source: {result['title']}\nURL: {result['url']}\nData: {result['content']}")
            
            result_text = "\n\n".join(context) if context else "No results found."
            print(f"âœ… [Tavily Tool] Found {len(context)} results.")
            return result_text
        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            print(f"â�Œ [Tavily Tool] Failed: {error_msg}")
            return error_msg

    # Return the function wrapped as a tool
    return FunctionTool(fetch_policy_data)

def get_google_search_tool(api_key=None):
    """
    Returns a FunctionTool for Google Search.
    """
    # 1. Setup Key from Config
    google_key = api_key if api_key else getattr(config, 'GOOGLE_API_KEY', None)
    
    # Note: The standard Google ADK search tool usually picks up the key from os.environ
    if google_key:
        import os
        os.environ["GOOGLE_API_KEY"] = google_key

    def fetch_policy_data(query: str) -> str:
        """
        Searches the web for real-time data, statistics, and news about the policy.
        ALWAYS use this to get facts before answering.
        
        Args:
            query: The search query string.
            
        Returns:
            String containing formatted search results with sources.
        """
        print(f"\nğŸ”� [Google Search Tool] Searching for: '{query}'...")
        try:
            # Simulating the tool for this example (or replace with actual logic if you have the custom wrapper)
            # In a real scenario, you might use `Google Search`
            result_text = f"Search results for '{query}' using Google Search."
            print(f"âœ… [Google Search Tool] Search completed.")
            return result_text
        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            print(f"â�Œ [Google Search Tool] Failed: {error_msg}")
            return error_msg

    # Return the function wrapped as a tool
    return FunctionTool(fetch_policy_data)

print("Tools are created")


import asyncio
import time
from typing import List, Any
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

async def run_policy_analysis(topic, google_key, tavily_key):
    # 1. Auth check
    active_google_key = google_key if google_key else config.GOOGLE_API_KEY
    active_tavily_key = tavily_key if tavily_key else config.TAVILY_API_KEY
    
    if not active_google_key or not active_tavily_key:
        err = "Error: Missing API Keys. Please check Kaggle Secrets or Input fields."
        yield err, err, err, err
        return

    yield f"Starting analysis for: {topic}...\nCheck console for live logs.", "", "", "" 

    # 2. Setup
    try:
        # Set the Google API Key for the session
        import os
        os.environ["GOOGLE_API_KEY"] = active_google_key
        
        # Define retry configuration
        retry_config = types.HttpRetryOptions(
            attempts=5,  # Maximum retry attempts
            exp_base=7,  # Delay multiplier
            initial_delay=1,
            http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
        )
        
        # Initialize the Model
        model = Gemini(model="gemini-2.5-flash", retry_options=retry_config)
        
        # Initialize Tools (Using the functions you just defined!)
        # We use Tavily for ALL agents to ensure they get real data.
        tavily_search_tool = get_tavily_search_tool(api_key=active_tavily_key)
        google_search_tool = google_search
        
        # Tools list 
        analyst_critic_tools: List[Any] = [tavily_search_tool]  # Analyst and Critic use Tavily
        lobbyist_summary_tools: List[Any] = [google_search_tool]  # Lobbyist and Summary use Google Search

    except Exception as e:
        err = f"Setup Error: {str(e)}"
        yield err, err, err, err
        return

    # 3. Define Agents
    
    analyst_agent = LlmAgent(
        name="Analyst",
        model=model,
        instruction="""
        You are a Senior Data-Driven Policy Analyst.
        MANDATE: Be extremely concise. Use bullet points.
        CRITICAL RULE: You MUST use the 'fetch_policy_data' tool to get real-world statistics.
        
        Structure your response strictly under:
        1.Rural Society 2.Urban Society 3.Working Class 4.Backward class
        5.Farmers 6.Manufacturing 7.Services 8.Women 9.Youth 10.Tribals.
        
        For each section, cite 1 specific data point found with persusasive argument via the tool.
        """,
        tools=analyst_critic_tools
    )

    critic_agent = LlmAgent(
        name="Critic",
        model=model,
        instruction="""
        You are a Critical Policy Reviewer. 
        MANDATE: Be direct and ruthless. No polite padding.
        CRITICAL RULE: Use the 'fetch_policy_data' tool to find counter-evidence.
        
        Critique the analysis focusing on:
        - Economic feasibility (cite costs).
        - Failed examples from other countries.
        - Direct negative impact on specific groups.
        """,
        tools=analyst_critic_tools
    )

    lobbyist_agent = LlmAgent(
        name="Lobbyist",
        model=model,
        instruction="""
        You are a Future Policy Strategist & Lobbyist.
        
        Your Goal: Based on the Analysis and Critique, propose 3 concrete Future Policy Directives.
        For each directive, you must LOBBY for a specific section of society (e.g., "Lobbying for Farmers").
        
        Structure:
        1. **Directive Name**
        2. **Target Beneficiary** (e.g., Rural Women, Gig Workers)
        3. **The Pitch**: A persuasive argument using DATA from the tool to justify why this directive is urgent.
        
        MANDATE: Use the tool to find fresh data to support your lobbying pitch. Be persuasive but factual.
        """,
        tools=lobbyist_summary_tools
    )

    summary_agent = LlmAgent(
        name="Synthesizer",
        model=model,
        instruction="""
        You are a Policy Synthesizer.
        
        MANDATE: Create a "TL;DR" Executive Summary based on the Analysis, Critique, and Lobbyist proposals.
        - Maximum 400 words.
        
        Format:
        1. **The Verdict**: One sentence summary.
        2. **Key Data Points** (Top 3 facts from the agents).
        3. **Major Risks** (from Critique).
        4. **Future Roadmap** (Top 2 directives from Lobbyist).
        5. **Final Recommendation**: Pass, Reject, or Amend.
        """,
        tools=lobbyist_summary_tools
    )

    # 4. Execution
    session_service = InMemorySessionService()
    app_name = "agents" 

    # --- ANALYST ---
    yield ">>> Analyst: Searching for data...\n", "", "", ""
    analysis_text = ""
    try:
        await session_service.create_session(app_name=app_name, user_id="user", session_id="sess_analyst")
        runner_analyst = Runner(agent=analyst_agent, app_name=app_name, session_service=session_service)
        msg = types.Content(role="user", parts=[types.Part(text=f"Analyze with data (be concise): '{topic}'")])
        
        # Run without custom retry wrapper
        async for event in runner_analyst.run_async(user_id="user", session_id="sess_analyst", new_message=msg):
            # DEBUG: Print if the agent decides to call a tool
            if hasattr(event, 'function_call') and event.function_call:
                print(f"   ğŸ› ï¸�  [Analyst] Calling Tool: {event.function_call.name}")
            
            if event.is_final_response() and event.content and event.content.parts:
                # Process all parts of the response, handling different part types explicitly
                text_parts = []
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    # Explicitly handle other part types to avoid warnings
                    elif hasattr(part, 'function_call'):
                        # Function calls are handled separately by the framework
                        pass
                    elif hasattr(part, 'thought_signature'):
                        # Thought signatures are metadata, not content
                        pass
                analysis_text = '\n'.join(text_parts) if text_parts else ""
        
        print(f"âœ… [Analyst] Execution Complete")

    except Exception as e:
        yield f"Analyst Failed: {str(e)}", "", "", ""
        return

    status_log = ">>> Analyst finished. Starting Critic...\n"
    yield analysis_text, status_log, "", "" 

    # --- CRITIC ---
    critique_text = ""
    try:
        await session_service.create_session(app_name=app_name, user_id="user", session_id="sess_critic")
        runner_critic = Runner(agent=critic_agent, app_name=app_name, session_service=session_service)
        msg = types.Content(role="user", parts=[types.Part(text=f"Critique this analysis ruthlessly: \n{analysis_text}")])

        # Run without custom retry wrapper
        async for event in runner_critic.run_async(user_id="user", session_id="sess_critic", new_message=msg):
            # DEBUG: Print if the agent decides to call a tool
            if hasattr(event, 'function_call') and event.function_call:
                print(f"   ğŸ› ï¸�  [Critic] Calling Tool: {event.function_call.name}")
            
            if event.is_final_response() and event.content and event.content.parts:
                # Process all parts of the response, handling different part types explicitly
                text_parts = []
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    # Explicitly handle other part types to avoid warnings
                    elif hasattr(part, 'function_call'):
                        # Function calls are handled separately by the framework
                        pass
                    elif hasattr(part, 'thought_signature'):
                        # Thought signatures are metadata, not content
                        pass
                critique_text = '\n'.join(text_parts) if text_parts else ""
        
        print(f"âœ… [Critic] Execution Complete")

    except Exception as e:
        yield analysis_text, f"Critic Failed: {str(e)}", "", ""
        return

    status_log = ">>> Critic finished. Starting Lobbyist...\n"
    yield analysis_text, critique_text, status_log, ""

    # --- LOBBYIST ---
    lobbyist_text = ""
    try:
        await session_service.create_session(app_name=app_name, user_id="user", session_id="sess_lobbyist")
        runner_lobbyist = Runner(agent=lobbyist_agent, app_name=app_name, session_service=session_service)
        lobbyist_input = f"""
        Context Analysis: {analysis_text}
        Context Critique: {critique_text}
        
        Task: Propose 3 Future Directives and lobby for specific groups using new data.
        """
        msg = types.Content(role="user", parts=[types.Part(text=lobbyist_input)])
        
        # Run without custom retry wrapper
        async for event in runner_lobbyist.run_async(user_id="user", session_id="sess_lobbyist", new_message=msg):
            # DEBUG: Print if the agent decides to call a tool
            if hasattr(event, 'function_call') and event.function_call:
                print(f"   ğŸ› ï¸�  [Lobbyist] Calling Tool: {event.function_call.name}")
            
            if event.is_final_response() and event.content and event.content.parts:
                # Process all parts of the response, handling different part types explicitly
                text_parts = []
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    # Explicitly handle other part types to avoid warnings
                    elif hasattr(part, 'function_call'):
                        # Function calls are handled separately by the framework
                        pass
                    elif hasattr(part, 'thought_signature'):
                        # Thought signatures are metadata, not content
                        pass
                lobbyist_text = '\n'.join(text_parts) if text_parts else ""
        
        print(f"âœ… [Lobbyist] Execution Complete")

    except Exception as e:
        yield analysis_text, critique_text, f"Lobbyist Failed: {str(e)}", ""
        return

    status_log = ">>> Lobbyist finished. Synthesizing Final Report...\n"
    yield analysis_text, critique_text, lobbyist_text, status_log

    # --- SUMMARY ---
    summary_text = ""
    try:
        await session_service.create_session(app_name=app_name, user_id="user", session_id="sess_summary")
        runner_summary = Runner(agent=summary_agent, app_name=app_name, session_service=session_service)
        
        combined_input = f"""
        Summarize these into a TL;DR Executive Report:
        
        [ANALYSIS]
        {analysis_text}
        
        [CRITIQUE]
        {critique_text}
        
        [LOBBYIST PROPOSALS]
        {lobbyist_text}
        """
        msg = types.Content(role="user", parts=[types.Part(text=combined_input)])
        
        # Run without custom retry wrapper
        async for event in runner_summary.run_async(user_id="user", session_id="sess_summary", new_message=msg):
            # DEBUG: Print if the agent decides to call a tool
            if hasattr(event, 'function_call') and event.function_call:
                print(f"   ğŸ› ï¸�  [Synthesizer] Calling Tool: {event.function_call.name}")
            
            if event.is_final_response() and event.content and event.content.parts:
                # Process all parts of the response, handling different part types explicitly
                text_parts = []
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    # Explicitly handle other part types to avoid warnings
                    elif hasattr(part, 'function_call'):
                        # Function calls are handled separately by the framework
                        pass
                    elif hasattr(part, 'thought_signature'):
                        # Thought signatures are metadata, not content
                        pass
                summary_text = '\n'.join(text_parts) if text_parts else ""
        
        print(f"âœ… [Synthesizer] Execution Complete")

    except Exception as e:
        yield analysis_text, critique_text, lobbyist_text, f"Summary Failed: {str(e)}"
        return

    # Final Yield
    print(f"\nâœ… [System] Workflow Completed Successfully.")
    yield analysis_text, critique_text, lobbyist_text, summary_text

print("Agents are ready to have fun")


import gradio as gr #launch gradio UI
import datetime

# --- Helper to save the report ---
def generate_markdown_report(topic, analysis, critique, lobbyist, summary):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = f"# Policy Report: {topic}\n**Date**: {timestamp}\n\n## Analysis\n{analysis}\n\n## Critique\n{critique}\n\n## Future Directives\n{lobbyist}\n\n## Summary\n{summary}\n"
    filename = config.OUTPUT_FILENAME
    with open(filename, "w", encoding="utf-8") as f:
        f.write(md_content)
    return filename

# --- Main UI ---
with gr.Blocks(title="ADK Policy Analyzer") as demo:
    gr.Markdown("# ğŸ�›ï¸� Data-Driven Policy Analyzer")
    
    # States
    analysis_state = gr.State()
    critique_state = gr.State()
    lobbyist_state = gr.State()
    summary_state = gr.State()
    topic_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=1):
            # NOW IT WORKS: Using config.DEFAULT_POLICY_TOPIC
            topic_input = gr.Textbox(
                label="Enter Policy Topic", 
                value=config.DEFAULT_POLICY_TOPIC, 
                lines=2
            )
            with gr.Accordion("API Settings", open=False):
                # Pre-fill with keys from config if available
                google_key_input = gr.Textbox(
                    label="Google API Key", 
                    type="password", 
                    value=config.GOOGLE_API_KEY
                )
                tavily_key_input = gr.Textbox(
                    label="Tavily API Key", 
                    type="password", 
                    value=config.TAVILY_API_KEY
                )
            analyze_btn = gr.Button("ğŸš€ Run Analysis", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("ğŸ“Š Analysis"): analysis_output = gr.Markdown()
                with gr.TabItem("âš–ï¸� Critique"): critique_output = gr.Markdown()
                with gr.TabItem("ğŸ“¢ Lobbyist"): lobbyist_output = gr.Markdown()
                with gr.TabItem("ğŸ“� Summary"): summary_output = gr.Markdown()
    
    download_btn = gr.DownloadButton("ğŸ“¥ Download Report", variant="secondary", interactive=False)

    # Events
    analyze_btn.click(
        fn=run_policy_analysis,
        inputs=[topic_input, google_key_input, tavily_key_input],
        outputs=[analysis_output, critique_output, lobbyist_output, summary_output]
    ).then(
        fn=lambda t, a, c, l, s: (t, a, c, l, s, gr.DownloadButton(interactive=True)),
        inputs=[topic_input, analysis_output, critique_output, lobbyist_output, summary_output],
        outputs=[topic_state, analysis_state, critique_state, lobbyist_state, summary_state, download_btn]
    )

    download_btn.click(
        fn=generate_markdown_report,
        inputs=[topic_state, analysis_state, critique_state, lobbyist_state, summary_state],
        outputs=[download_btn]
    )

demo.queue().launch(share=True, inline=True, debug=True)

