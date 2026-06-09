import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor
import json
from typing import List, Dict

print("âœ… ADK components imported successfully.")


# Use the provided Retry Configuration
retry_config = types.HttpRetryOptions(
    attempts=5, 
    exp_base=7, 
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


def knowledge_base_lookup(key_terms: List[str]) -> str:
    """
    CUSTOM TOOL: Queries a simulated internal knowledge base (Vector Store) 
    for proprietary, non-public information relevant to the market.
    
    Args:
        key_terms: A list of relevant entities or concepts (e.g., competitor names).
    Returns:
        A string containing proprietary data snippets.
    """
    print(f"    [TOOL CALL] Querying internal knowledge base for: '{key_terms}'")
    
    # Mock proprietary data for a high-scoring demo
    if "PetSense" in key_terms or "wearables" in key_terms:
        return (
            "INTERNAL REPORT: PetSense Q3 2025 revenue showed 30% growth in the EU market, "
            "attributed to a key partnership with a major veterinary chain. Their proprietary "
            "diagnostic algorithm is protected by 5 patents. This indicates a strong IP barrier."
        )
    return "No highly relevant internal documents found for these specific terms."


# --- 2A. EVALUATION AGENT (LLM Judge) (Required Feature: Agent Evaluation) ---
# Used as a tool by the Strategy Agent for quality control.
evaluation_agent = LlmAgent(
    name="EvaluationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config), 
    instruction="""You are an LLM Judge. Your sole task is to assess the quality of the provided strategic report 
    against these criteria: Clarity, Insightfulness, and Completeness.
    You must ONLY respond with a quality score out of 100 and brief, constructive feedback.
    Example: 'EVALUATION: Score 95/100. Report is clear and data-backed. Passes quality check.'""",
)
evaluation_tool = AgentTool(agent=evaluation_agent) # Wrap as a tool for the Strategy Agent


# --- 2B. RESEARCHER AGENT (The Data Collector) ---
# Fulfills: Built-in Tool (google_search) and Custom Tool (knowledge_base_lookup)
researcher_agent = LlmAgent(
    name="ResearcherAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are the RESEARCHER AGENT. Your sole task is to gather facts and raw data 
    for the given market analysis objective. You MUST use 'google_search' for real-time market data 
    and 'knowledge_base_lookup' for internal context, the 'knowledge_base_lookup' need key_terms as List[str]. Do not summarize or synthesize. 
    Provide the raw, complete data snippets and source links found, ready for the SynthesizerAgent.
    Print the response and status from each tool usage.""",
    tools=[google_search, knowledge_base_lookup], # Built-in and Custom Tools
)

researcher_tool = AgentTool(agent=researcher_agent)


# --- 2C. SYNTHESIZER AGENT (The Editor) (Required Feature: Context Engineering) ---
synthesizer_agent = LlmAgent(
    name="SynthesizerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are the SYNTHESIZER AGENT. Your mission is to structure and compact 
    the raw research data provided into a coherent, executive-ready draft report.
    
    You must apply **Context Engineering** to filter and summarize the raw text into the 
    following structured sections: Market Overview (Size, CAGR), Key Competitors, 
    and Consumer Pain Points. Your output must be a clean, structured draft report.""",
    tools=[],
)

synthesizer_tool = AgentTool(agent=synthesizer_agent)


# --- 2D. STRATEGY AGENT (The Consultant) ---
strategy_agent = LlmAgent(
    name="StrategyAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are the STRATEGY AGENT. Your mission is to take the structured draft report 
    and add a section titled 'Strategic Recommendations'. 
    
    1. Analysis: Generate 3-5 actionable business recommendations (e.g., pricing, features).
    2. Quality Check: You MUST use the 'EvaluationAgent' tool to self-assess the quality of 
       the final report before returning the result. Combine the report and the evaluation 
       score into a single final output.""",
    tools=[evaluation_tool], # Using the LLM Judge as a tool
)

strategy_tool = AgentTool(agent=strategy_agent)


due_diligence_agent_orchestrator = LlmAgent(
    name="DueDiligenceAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are the Due Diligence & Strategy Orchestrator. Your role is to manage the 
    sequential execution of the specialist analyst team to generate a full strategy report.
    
    **EXECUTION FLOW (Sequential Agents):**
    1. **Research**: Call the 'researcher_tool' with the user's market query.
    2. **Synthesize**: Pass the raw output from the ResearcherAgent to the 'synthesizer_tool'.
    3. **Strategize**: Pass the synthesized report from the SynthesizerAgent to the 'strategy_tool', 
       which will also run the final quality check.
    4. **Final Output**: Present the complete, evaluated report from the StrategyAgent to the user.
    Print the status update on each stage of the process.""",
    tools=[
        researcher_tool,
        synthesizer_tool,
        strategy_tool,
    ],
)


# Define the main runner for the Orchestrator
orchestrator_runner = InMemoryRunner(agent=due_diligence_agent_orchestrator)

USER_QUERY = "Analyze the market for AI-powered pet wearables, including size, key competitors, and consumer pain points."

print("###########################################")
print("# STARTING LEVEL 3 MULTI-AGENT EXECUTION")
print(f"# USER REQUEST: {USER_QUERY}")
print("###########################################")

# The Orchestrator's debug run handles the entire sequential chain of tool calls (Agents)
# This command simulates the Orchestrator executing its instruction to call the chain of Agents
final_report = await orchestrator_runner.run_debug(USER_QUERY)

print("###########################################")
print("# COMPLETED EXECUTION")
print("###########################################")

