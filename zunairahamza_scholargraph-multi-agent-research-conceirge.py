# %% [code]
# 1. Install Libraries (Quiet mode)
!pip install -q -U langchain langchain-google-genai langgraph arxiv

import os
import warnings
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# 2. THE KAGGLE COMPATIBILITY PATCH (CRITICAL)
# This fixes the "AttributeError: type object 'GenerationConfig' has no attribute 'MediaResolution'"
# error caused by stale dependencies in the Kaggle environment.
if not hasattr(genai.GenerationConfig, 'MediaResolution'):
    class MediaResolution:
        MEDIA_RESOLUTION_UNSPECIFIED = "MEDIA_RESOLUTION_UNSPECIFIED"
        MEDIA_RESOLUTION_LOW = "MEDIA_RESOLUTION_LOW"
        MEDIA_RESOLUTION_MEDIUM = "MEDIA_RESOLUTION_MEDIUM"
        MEDIA_RESOLUTION_HIGH = "MEDIA_RESOLUTION_HIGH"
    genai.GenerationConfig.MediaResolution = MediaResolution
    print("âœ… Compatibility Patch Applied: 'MediaResolution' fixed.")


# 3. Securely Load API Key
try:
    user_secrets = UserSecretsClient()
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
    print("âœ… Google API Key Loaded")
except:
    print("âš ï¸� WARNING: Please set 'GOOGLE_API_KEY' in Kaggle Secrets (Add-ons -> Secrets).")



genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# UPDATED CANDIDATES: Prioritize newer models, REMOVE deprecated 'gemini-pro'
candidate_models = [
    "gemini-2.5-flash-lite", 
    "gemini-2.0-flash-exp", 
    "gemini-1.5-flash", 
    "gemini-1.5-pro"
]
MODEL_NAME = "gemini-1.5-flash" # Safe Fallback

print("ğŸ”„ Testing model availability...")
for model in candidate_models:
    try:
        # Try to generate a single token to verify access
        test_model = genai.GenerativeModel(model)
        # Short timeout to fail fast if not working
        response = test_model.generate_content("Test", request_options={"timeout": 5})
        MODEL_NAME = model
        print(f"âœ… Success: '{model}' is working.")
        break # Stop at the first working model
    except Exception as e:
        print(f"âš ï¸� Skipping '{model}': {str(e)[:50]}...")

print(f"ğŸ�¯ Agent Brain Selected: {MODEL_NAME}")



import arxiv
from typing import TypedDict, List, Annotated
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Initialize the LLM ---
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)

# --- 2. Define the Schema (The Memory) ---
class Paper(TypedDict):
    title: str
    url: str
    doi: str
    abstract: str
    summary: str
    verification_status: str # "Verified" or "Rejected"

class AgentState(TypedDict):
    query: str                  # User's research topic
    raw_papers: List[Paper]     # Initial search results
    verified_papers: List[Paper]# Papers that passed the Verifier
    final_report: str           # The final output text

# --- 3. Define Tools ---
def search_arxiv_tool(query: str) -> List[dict]:
    """
    Real tool that searches Arxiv.org for papers.
    Returns raw paper data including Title, DOI, and Abstract.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=5, # Fetch top 5 relevant
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for r in client.results(search):
        # We assume if it has an entry_id, it's valid, but we verify DOI later
        results.append({
            "title": r.title,
            "url": r.entry_id,
            "doi": r.doi or "No DOI Found",
            "abstract": r.summary.replace("\n", " "),
            "summary": "",
            "verification_status": "Pending"
        })
    return results



from langchain_core.messages import SystemMessage, HumanMessage

# --- AGENT 1: RESEARCHER ---
def research_node(state: AgentState):
    print(f"ğŸ•µï¸� [RESEARCHER]: Searching Arxiv for '{state['query']}'...")
    
    # Execute the tool
    papers_data = search_arxiv_tool(state['query'])
    
    # Return updates to the state
    return {"raw_papers": papers_data}

# --- AGENT 2: VERIFIER ---
def verifier_node(state: AgentState):
    print("âš–ï¸� [VERIFIER]: Validating sources...")
    raw_papers = state.get('raw_papers', [])
    verified = []
    
    for paper in raw_papers:
        # Verification Rule: Must have a DOI and a substantial abstract
        has_doi = paper['doi'] != "No DOI Found"
        has_content = len(paper['abstract']) > 50
        
        if has_doi and has_content:
            paper['verification_status'] = "Verified"
            verified.append(paper)
            print(f"   âœ… Verified: {paper['title'][:40]}...")
        else:
            paper['verification_status'] = "Rejected"
            print(f"   â�Œ Rejected: {paper['title'][:40]}...")
            
    return {"verified_papers": verified}

# --- AGENT 3: SUMMARIZER ---
def summarizer_node(state: AgentState):
    print("âœ�ï¸� [SUMMARIZER]: Synthesizing abstracts...")
    verified_papers = state.get('verified_papers', [])
    
    # Loop through each paper and summarize using LLM
    for paper in verified_papers:
        prompt = f"""
        Task: Summarize this academic abstract in 2 concise sentences.
        Highlight the methodology and the main result.
        
        Abstract: {paper['abstract']}
        """
        response = llm.invoke(prompt)
        paper['summary'] = response.content
        
    return {"verified_papers": verified_papers}

# --- AGENT 4: META-WRITER ---
def writer_node(state: AgentState):
    print("ğŸ“� [WRITER]: Compiling final report...")
    papers = state.get('verified_papers', [])
    
    if not papers:
        return {"final_report": "No verified papers were found to generate a report."}
    
    # Context Engineering: Compact the data for the final prompt
    context_str = ""
    for idx, p in enumerate(papers, 1):
        context_str += f"{idx}. {p['title']}\n   DOI: {p['doi']}\n   Summary: {p['summary']}\n\n"
        
    prompt = f"""
    You are a Senior Research Editor.
    Produce a structured Literature Review based on the following verified papers.
    
    Topic: {state['query']}
    
    Papers:
    {context_str}
    
    Format Requirement:
    # Executive Summary
    [Synthesis of the state of the art]
    
    # Key Findings
    - [Bullet points citing specific papers]
    
    # References
    [List title and DOI links]
    """
    
    response = llm.invoke(prompt)
    return {"final_report": response.content}



from langgraph.graph import StateGraph, END

# 1. Initialize Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("researcher", research_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

# 3. Define Logic
def check_results(state: AgentState):
    """
    If Verifier rejects all papers, end the process early.
    """
    if len(state['verified_papers']) > 0:
        return "continue"
    else:
        print("ğŸ›‘ [SYSTEM]: No papers passed verification. Stopping.")
        return "stop"

# 4. Connect Edges
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "verifier")

workflow.add_conditional_edges(
    "verifier",
    check_results,
    {
        "continue": "summarizer",
        "stop": END
    }
)

workflow.add_edge("summarizer", "writer")
workflow.add_edge("writer", END)

# 5. Compile
app = workflow.compile()
print("âœ… Agent Workflow Compiled Successfully")



from IPython.display import Markdown, display

# --- CONFIGURATION ---
TOPIC = "Prompt Engineering"

# --- EXECUTION ---
print(f"ğŸš€ Launching Research Concierge for: '{TOPIC}'")
print("="*60)

inputs = {"query": TOPIC}
result = app.invoke(inputs)

print("="*60)
print("âœ… Process Complete. Generating Output...")

# --- OUTPUT ---
if result.get("final_report"):
    display(Markdown(result["final_report"]))
else:
    print("No report generated.")


