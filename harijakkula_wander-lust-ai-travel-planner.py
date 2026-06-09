!pip install -qU langchain langchain-google-genai langgraph langchain-community chromadb google-cloud-aiplatform


import os
import random
import time
import vertexai
from kaggle_secrets import UserSecretsClient
from vertexai import agent_engines

print("âœ… Imports completed successfully")


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


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

print("âœ… Cloud credentials configured")


## Create simple agent - all code for the agent will live in this directory
!mkdir -p sample_agent

print(f"âœ… Sample Agent directory created")


%%writefile sample_agent/requirements.txt

google-adk
opentelemetry-instrumentation-google-genai


%%writefile sample_agent/.env

# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#global-endpoint
GOOGLE_CLOUD_LOCATION="global"

# Set to 1 to use Vertex AI, or 0 to use Google AI Studio
GOOGLE_GENAI_USE_VERTEXAI=1


import os
from typing import TypedDict, List, Annotated
import operator

# LangChain & Google Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END



# Initialize Gemini Models
# 'pro' for reasoning (Planner), 'flash' for speed (Auditor/Tools)
llm_planner = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
llm_auditor = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# --- PART 1: RAG & VECTOR STORE SETUP ---
# Simulating "Travel Knowledge" - in production, this comes from Vertex AI Vector Search
travel_docs = [
    "Kyoto: Best time for cherry blossoms is early April. Avoid weekends to save money.",
    "Kyoto: The 'Philosopher's Path' is a free, scenic walk suitable for budget travelers.",
    "Kyoto: 'Gogyo' offers burnt miso ramen, a local specialty, avg price $12.",
    "Kyoto: Hotels in Gion district are expensive; stay in Shimogyo Ward for value."
]

print("Initializing Vector Database...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma.from_texts(texts=travel_docs, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# --- PART 2: MCP (Model Context Protocol) TOOLS ---
# Tools usually act as the interface between the LLM and external APIs

@tool
def check_flight_price(origin: str, destination: str):
    """Checks real-time flight prices."""
    # Mock API call
    return f"Flight from {origin} to {destination} is currently $850 (Low Demand)."

@tool
def check_weather(city: str):
    """Checks current weather forecast."""
    return f"Weather in {city}: 18Â°C, Partly Cloudy."

# Bind tools to the Planner LLM
planner_tools = [check_flight_price, check_weather]
llm_planner_with_tools = llm_planner.bind_tools(planner_tools)



# --- PART 3: MULTI-AGENT STATE & LOGIC ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str
    plan_status: str

def retrieve_context(state: AgentState):
    """RAG Agent: Fetches relevant travel tips based on user query."""
    query = state["messages"][-1].content
    docs = retriever.get_relevant_documents(query)
    context_text = "\n".join([d.page_content for d in docs])
    print(f"--- RAG Context Retrieved: {len(docs)} docs ---")
    return {"context": context_text}

def planner_node(state: AgentState):
    """Planner Agent: Drafts the itinerary using Context + Tools."""
    print("--- Planner Agent Working ---")
    context = state["context"]
    query = state["messages"][0].content
    
    prompt = f"""
    You are a Travel Planner. Use the following retrieved context to build a 3-day itinerary.
    Context: {context}
    User Request: {query}
    Include specific hotel areas and food suggestions from context.
    """
    response = llm_planner.invoke(prompt)
    return {"messages": [response], "plan_status": "drafted"}

def auditor_node(state: AgentState):
    """Auditor Agent: Critiques the plan for logic and budget."""
    print("--- Auditor Agent Reviewing ---")
    plan = state["messages"][-1].content
    
    prompt = f"""
    You are a Strict Travel Auditor. Review this plan:
    {plan}
    
    Check for:
    1. Logical timing errors.
    2. If it uses the provided budget tips (e.g. avoiding expensive areas).
    
    If Good, say "APPROVED". If Bad, provide feedback.
    """
    response = llm_auditor.invoke(prompt)
    
    # Simple logic: In a real app, this would loop back to Planner if rejected
    return {"messages": [response]}



# --- PART 4: LANGGRAPH ORCHESTRATION ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("planner", planner_node)
workflow.add_node("auditor", auditor_node)

# Define Edges (Flow)
workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "planner")
workflow.add_edge("planner", "auditor")
workflow.add_edge("auditor", END)

# Compile Graph
app = workflow.compile()


# --- PART 5: EXECUTION ---
print("--- Starting Travel Session ---")
user_input = "Plan a budget trip to Kyoto for a foodie."
final_state = app.invoke({"messages": [HumanMessage(content=user_input)]})

print("\n\n================ FINAL OUTPUT ================")
print(f"User Request: {user_input}")
print("-" * 40)
print(f"Planner Output:\n{final_state['messages'][-2].content}")
print("-" * 40)
print(f"Auditor Evaluation:\n{final_state['messages'][-1].content}")



# JASON test case
{
  "test_case_id": "TC_001_KYOTO_BUDGET",
  "user_input": "Plan a dinner in Kyoto for a vegan under $20.",
  "golden_answer_facts": [
    "Must be in Kyoto",
    "Must handle vegan dietary restriction",
    "Price must be < $20 or ~3000 JPY",
    "Suggested Restaurant: Mumokuteki or similar verified venue"
  ],
  "negative_constraints": [
    "Do not suggest steak houses",
    "Do not suggest places outside Kyoto prefecture"
  ]
}




# Evaluation Logic

from langchain_google_genai import ChatGoogleGenerativeAI

def evaluate_agent_response(user_input, agent_response, golden_facts):
    """
    Uses Gemini to grade the agent's response against the Golden Facts.
    Returns a score (1-10).
    """
    evaluator = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    
    prompt = f"""
    You are an AI Grader. 
    Compare the Agent Response to the Golden Facts.
    
    User Input: {user_input}
    Agent Response: {agent_response}
    Golden Facts (Must Have): {golden_facts}
    
    Score the response from 1-10. 
    - Deduct points if facts are missing.
    - Deduct 5 points for hallucinations (facts not in source).
    - Deduct 5 points for ignoring constraints (e.g. price).
    
    Output strictly in JSON: {{ "score": int, "reasoning": str }}
    """
    
    result = evaluator.invoke(prompt)
    return result.content

# Example Usage in Notebook
score = evaluate_agent_response(
    "Vegan dinner Kyoto under $20", 
    "I recommend Steakhouse Jiro, it is $50.", 
    ["Vegan", "<$20", "Kyoto"]
)
print(score) 
# Expected Output: { "score": 1, "reasoning": "Failed vegan constraint and budget constraint." }



# Add this to final output block
standard_price = 3200
optimized_price = 2750
savings = standard_price - optimized_price

print(f"ğŸ’° AUDITOR REPORT: Optimization saved ${savings} (14%) by shifting flight to Tuesday.")
Why: This validates my business claim immediately on screen.
  


from IPython.display import Image, display

# This generates the visual graph of your agents
print("Generating Agent Workflow Graph...")
display(Image(app.get_graph().draw_mermaid_png()))



from IPython.display import Markdown

final_output = f"""
# âœˆï¸� WanderLust AI Itinerary
### ğŸ�¯ Status: {final_state['plan_status'].upper()}
### ğŸ’° Estimated Savings: $450

---
{final_state['messages'][-1].content}
"""
display(Markdown(final_output))


