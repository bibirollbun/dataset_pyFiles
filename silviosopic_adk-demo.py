!pip install -q -U google-adk neo4j gradio litellm








import os
import json
import sys
import asyncio
import gradio as gr
from typing import Dict, Any, List
from neo4j import GraphDatabase
from kaggle_secrets import UserSecretsClient
import litellm 

# Import ADK components
from google.adk.agents.llm_agent import Agent

# --- 1. CONFIGURATION & AUTH ---

def get_secret(name):
    try:
        return UserSecretsClient().get_secret(name)
    except:
        return os.environ.get(name)

NEO4J_URI = get_secret("NEO4J_URI")
NEO4J_USER = get_secret("NEO4J_USERNAME")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD")

# MISTRAL CONFIGURATION (Via LiteLLM)
MISTRAL_API_KEY = get_secret("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    MISTRAL_API_KEY = get_secret("mistral_key")

if not MISTRAL_API_KEY:
    raise ValueError("Missing MISTRAL_API_KEY. Please add it to Kaggle Secrets.")

os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# --- 2. CONTEXT ENGINEERING (SCHEMA & PROMPTS) ---

GRAPH_SCHEMA = """
**GRAPH SCHEMA (Evidence Pattern):**
1. **Nodes:**
   - `:PERSON` (name, entity_id)
   - `:ORGANIZATION` (name, entity_id)
   - `:Document` (doc_id, date, subject, content)

2. **Relationships:**
   - `(:PERSON)-[:mentioned_in]->(:Document)`
     * Contains FACTS about the person.
     * Property `attributes` (JSON String) stores details (e.g. role, sentiment).
     * USE `apoc.convert.fromJsonMap(r.attributes)` to read.
   
   - `(:PERSON)-[:INTERACTS_WITH]->(:PERSON)` (or specific types like :FINANCIAL_TRANSACTION)
     * Represents interactions.
     * Property `raw_verbs` (List): Original text (e.g. "wired").
     * Property `source_pks` (List): IDs of Documents verifying this link.

**QUERY RULES:**
- Always use `toLower(n.name) CONTAINS "..."` for flexible matching.
- To find FACTS, check `:mentioned_in` edges.
- To find PROOF, return `d.doc_id` or `r.source_pks`.
- To read relationship attributes, parse `r.evidence_json`.
"""

# SYSTEM PROMPTS (Strings for injection)

WRITER_PROMPT = f"""
You are an expert Neo4j Developer.
Your goal is to translate user questions into a PRECISE Cypher query based on the Schema.

{GRAPH_SCHEMA}

**INSTRUCTIONS:**
1. Analyze the user request.
2. Write the correct Cypher query.
3. Output ONLY the query string. Do not execute it.
"""

GUARDIAN_PROMPT = """
You are the Database Security Guardian.
You receive a Cypher query from the Architect.

**YOUR JOB:**
1. **Security Scan:** Check if the query tries to modify data (DELETE, SET, CREATE, MERGE).
   - IF UNSAFE: Reply "SECURITY BLOCK: Action denied."
   - IF SAFE: Execute the query using the `query_neo4j` tool.

2. Pass the raw tool output to the Investigator.
"""

ANALYST_PROMPT = """
You are a Senior Investigator.
You will receive raw data (JSON) from the Guardian.

**YOUR JOB:**
1. Interpret the data.
2. Answer the user's original question clearly.
3. Cite your sources! If the data contains `doc_id` or `source_pks`, list them as "Evidence".
4. If the data is empty or an error occurred, explain it politely.
"""

# --- 3. CUSTOM TOOL (NEO4J) ---

def query_neo4j_tool(cypher_query: str) -> str:
    """
    Executes a read-only Cypher query on the Neo4j Graph.
    
    Args:
        cypher_query: The Cypher query string to execute.
    
    Returns:
        JSON string of results.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Hard Guardrails
    dangerous_keywords = ["DELETE", "DETACH", "SET", "CREATE", "MERGE", "DROP", "REMOVE"]
    if any(keyword in cypher_query.upper() for keyword in dangerous_keywords):
        return "Error: Security Violation. Modification queries are strictly blocked."
    
    # Auto-limit
    if "LIMIT" not in cypher_query.upper() and "count" not in cypher_query.lower():
        cypher_query += " LIMIT 20"
        
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [r.data() for r in result]
            if not records:
                return "No results found in the database for this query."
            return json.dumps(records)
    except Exception as e:
        return f"Cypher Execution Error: {str(e)}"
    finally:
        driver.close()

# --- 4. AGENT DEFINITIONS ---

# Agent 1: The Architect
writer_agent = Agent(
    name="CypherArchitect",
    model="mistral/mistral-large-latest"
)

# Agent 2: The Guardian
guardian_agent = Agent(
    name="Guardian",
    model="mistral/mistral-large-latest",
    tools=[query_neo4j_tool] 
)

# Agent 3: The Investigator
analyst_agent = Agent(
    name="Investigator",
    model="mistral/mistral-small-latest"
)

# --- 5. WORKFLOW & MEMORY ---

# Global History Storage
session_history = []

async def execute_adk_agent(agent, input_text):
    """
    Executes an ADK agent using the explicit run_async method and 
    correctly extracts the final response from the async generator.
    """
    try:
        # Check if the agent has the run_async method (it should)
        if not hasattr(agent, 'run_async'):
             raise NotImplementedError(f"Agent {agent.name} does not have a 'run_async' method.")
        
        last_response = None
        # Use a list comprehension to force iteration and collect all yielded items
        # then take the last one, which should be the final response/output.
        # This reliably drains the async generator.
        async for response in agent.run_async(input_text):
            last_response = response
            
        if last_response is None:
            return "Error: Agent executed but returned no response."

        return last_response # Return the final ADK Response object
    
    except Exception as e:
        methods = [m for m in dir(agent) if not m.startswith('_')]
        # This re-raises the error with context for debugging
        raise SystemError(f"Could not execute agent '{agent.name}'. Error: {e}. Available attributes/methods: {methods}")
def get_resp_text(response):
    """Extracts text from response object safely"""
    if hasattr(response, 'output'): return str(response.output)
    if hasattr(response, 'text'): return str(response.text)
    if hasattr(response, 'get_text'): return response.get_text()
    if isinstance(response, dict): return str(response.get('output', response))
    return str(response)

    
async def run_sequential_chain(user_input, session_id):
    """
    Manually orchestrates the Writer -> Guardian -> Analyst chain.
    ASYNC version to support run_async.
    """
    # 0. Context Injection
    recent_history = session_history[-2:] if session_history else []
    history_text = "\n".join([f"User: {h['user']}\nAgent: {h['agent']}" for h in recent_history])
    
    # 1. Writer (Drafts Query)
    #writer_input = f"{WRITER_PROMPT}\n\nPREVIOUS CONTEXT:\n{history_text}\n\nCURRENT REQUEST:\n{user_input}"
    writer_input = (
    f"{WRITER_PROMPT}\n\n"  # This is the massive system prompt
    f"PREVIOUS CONTEXT:\n{history_text}\n\n"
    f"CURRENT REQUEST:\n{user_input}"
        )
    writer_resp = await execute_adk_agent(writer_agent, writer_input)
    draft_query = get_resp_text(writer_resp)
    
    # 2. Guardian (Executes Query)
    guardian_input = f"{GUARDIAN_PROMPT}\n\nTASK: Execute this query if safe: {draft_query}"
    guardian_resp = await execute_adk_agent(guardian_agent, guardian_input)
    raw_data = get_resp_text(guardian_resp)
    
    # 3. Analyst (Summarizes Data)
    analyst_input = f"{ANALYST_PROMPT}\n\nDATABASE RESULT:\n{raw_data}"
    analyst_resp = await execute_adk_agent(analyst_agent, analyst_input)
    final_answer = get_resp_text(analyst_resp)
    
    # 4. Save to Memory
    session_history.append({"user": user_input, "agent": final_answer})
    
    return final_answer

# --- 6. UI (GRADIO) ---

async def interact_with_agent(user_input, history):
    try:
        # Run the workflow
        return await run_sequential_chain(user_input, session_id="user_session_1")
    except Exception as e:
        return f"System Error: {str(e)}"

# Clean UI without complex theming that caused errors
with gr.Blocks() as demo:
    gr.Markdown("# ğŸ•µï¸�â€�â™‚ï¸� Graph RAG Investigator (Secured + Mistral)")
    gr.Markdown("Ask questions about the **Epstein/Maxwell Email Corpus**. Powered by **Google ADK, LiteLLM & Mistral**.")
    
    chatbot = gr.ChatInterface(
        fn=interact_with_agent,
        examples=[
            "Who did Jeffrey Epstein transact with?",
            "What documents mention 'pedophile'?",
            "Show me the shortest path between Clinton and Epstein.",
            "List all organizations linked to 'Financial' activities."
        ]
    )

# commented out because it will make saving take forever
# if __name__ == "__main__":
#     try:
#         demo.launch(share=True, debug=True)
#     except ImportError as e:
#         if "ServerReloader" in str(e):
#             print("\n" + "="*60)
#             print("CRITICAL ERROR: KERNEL RESTART REQUIRED")
#             print("="*60)
#             print("You updated Gradio/ADK but the old version is stuck in memory.")
#             print("Please click 'Restart Session' (Circular Arrow icon) and run this cell again.")
#             print("="*60 + "\n")





