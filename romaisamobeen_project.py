# --- 0. INSTALLATIONS ---
# Run this cell first. Restart kernel afterwards if necessary.
! pip install -qU \
    'langgraph' \
    'langchain-google-genai' \
    'langchain-community' \
    'pydantic' 

# ğŸ”‘ Imports
import os
import json
import logging
from typing import TypedDict, List
from operator import itemgetter

# LangChain/LangGraph Components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

# --- CRITICAL FIX 1: Pydantic Import ---
from pydantic import BaseModel, Field 

# Kaggle Secrets for API Key
from kaggle_secrets import UserSecretsClient

# --- 1. Authentication and Observability Setup ---

# Set up API Keys for Gemini from Kaggle Secrets
try:
    os.environ["GOOGLE_API_KEY"] = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    print("âœ… Gemini API Key loaded from Kaggle Secrets.")
except Exception as e:
    # We only need the Gemini key now!
    print(f"â�Œ Error loading API Key: {e}. Please ensure GOOGLE_API_KEY is set in Kaggle Secrets.")

# Observability Setup (Logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger("CDSA_Agent")
LOGGER.info("Observability (Logging) initialized.")

# --- 2. Initializing LLM and Tools ---
# Use a strong LLM for reasoning since it won't be using an external search tool
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro") # Using 'pro' model for better complex reasoning

# NOTE: No external tools are initialized, satisfying the request to remove the source of error.


## Sessions & Memory: State Definition
from typing import TypedDict, List
import json
import logging

# Ensure LOGGER is initialized (it was in your previous output)
LOGGER = logging.getLogger("CDSA_Agent")

class AgentState(TypedDict):
    """Represents the state of our multi-agent workflow."""
    patient_summary: str  # Initial user input
    normalized_data: dict  # JSON output from Agent 1
    evidence_base: str    # New: This is now LLM-generated synthetic research
    final_report: str     # Final output from Agent 2
    
# --- Long Term Memory (Simple In-Memory Mock) ---
# Long Term Memory (Memory Bank)
TREATMENT_MEMORY_BANK = [
    {"disease": "Rheumatoid Arthritis", "comorbidities": ["Hypertension"], "drug": "Methotrexate", "outcome": "Success", "note": "Rapid symptom control."},
    {"disease": "Rheumatoid Arthritis", "comorbidities": ["Obesity"], "drug": "Hydroxychloroquine", "outcome": "Failure", "note": "No improvement after 6 months."},
    {"disease": "Type 2 Diabetes", "comorbidities": ["Obesity"], "drug": "Metformin", "outcome": "Success", "note": "A1C reduced by 1.5%"},
]
LOGGER.info(f"Memory Bank loaded with {len(TREATMENT_MEMORY_BANK)} past records.")

def retrieve_from_memory(disease: str) -> str:
    """Simulates querying the Long Term Memory Bank for past treatment outcomes."""
    relevant_records = [
        record for record in TREATMENT_MEMORY_BANK 
        if record['disease'] == disease
    ]
    if relevant_records:
        return json.dumps(relevant_records, indent=2)
    return "No relevant past treatment records found in the memory bank."

# Initializing LLM from the previous cell again, just to be safe if kernel reset occurred.
from langchain_google_genai import ChatGoogleGenerativeAI
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
print("âœ… AgentState class, Memory Bank, and retrieval function defined.")


# --- Agent 1: Clinical Data Normalizer ---
from pydantic import BaseModel, Field 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List

# (AgentState, retrieve_from_memory, and LLM are assumed to be defined)

class NormalizedData(BaseModel):
    disease: str = Field(description="The primary disease/condition.")
    comorbidities: List[str] = Field(description="List of significant co-existing medical conditions.")
    age_group: str = Field(description="Patient age classification: 'Pediatric', 'Adult', or 'Geriatric'.")

def clinical_data_normalizer(state: AgentState) -> AgentState:
    """Agent 1: Normalizes text input into structured JSON and generates synthetic evidence."""
    LOGGER.info("Agent 1: Clinical Data Normalizer started.")
    
    # 1. Normalization (Structured Output)
    normalizer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Clinical Data Normalizer. ..."),
        ("human", f"Patient Summary: {state['patient_summary']}")
    ])
    
    normalizer_chain = normalizer_prompt | LLM.with_structured_output(NormalizedData)
    normalized_output = normalizer_chain.invoke({'patient_summary': state['patient_summary']})
    
    new_state = state.copy()
    # Pydantic V2 fix: Use model_dump()
    new_state['normalized_data'] = normalized_output.model_dump()
    
    # 2. Synthetic Evidence Generation (Internal Research)
    synthetic_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Clinical Knowledge Engine. Based ONLY on your internal, up-to-date knowledge, "
            "provide 3 concise, highly relevant findings (citing sources like 'Recent NEJM Study' or 'FDA Guidance') "
            "for treating the following condition. Focus on latest guidelines."
        )),
        ("human", 
         f"Condition: {normalized_output.disease}, Comorbidities: {', '.join(normalized_output.comorbidities)}, Age Group: {normalized_output.age_group}"
        )
    ])
    synthetic_evidence = (synthetic_prompt | LLM).invoke({}).content
    new_state['evidence_base'] = synthetic_evidence
    
    LOGGER.info(f"Agent 1 finished. Synthetic Evidence generated based on internal knowledge.")
    return new_state

# --- Agent 2: Treatment Synthesizer ---
def treatment_synthesizer(state: AgentState) -> AgentState:
    """Agent 2: Synthesizes evidence, memory, and patient data into a final report."""
    LOGGER.info("Agent 2: Treatment Synthesizer started. Performing synthesis.")
    
    disease = state['normalized_data']['disease']
    comorbidities = ", ".join(state['normalized_data']['comorbidities'])
    evidence = state['evidence_base']
    
    # Sessions & Memory: Long Term Memory Retrieval
    memory_results = retrieve_from_memory(disease)
    
    synthesizer_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Final Treatment Synthesizer. Generate a structured treatment report using the provided 'Clinical Evidence' (from your knowledge base) "
            "and 'Past Treatment Memory' (from the memory bank). The report must be professional and support recommendations with citations/references. "
            "Your output must be a professional medical document, not conversational."
        )),
        # NOTE: We define placeholders (input variables) in the prompt template itself:
        ("human", 
            "**Patient Disease:** {disease}\n"
            "**Comorbidities:** {comorbidities}\n\n"
            "**Clinical Evidence (Internal Knowledge):**\n---\n{evidence}\n---\n\n"
            "**Past Treatment Memory (Internal Bank):**\n---\n{memory_results}\n---\n\n"
            "**TASK:** Generate a final report with the following sections:\n"
            "1. **Summary of Patient Profile & Risk Factors**\n"
            "2. **Ranked Treatment Recommendations (3-5 options)**, with Rationale based on evidence/memory.\n"
            "3. **Conclusion & Next Steps**"
        )
    ])
    
    # --- FIX APPLIED HERE: Pass variables to the final invoke call ---
    final_report = (synthesizer_prompt | LLM).invoke(
        {
            "disease": disease,
            "comorbidities": comorbidities,
            "evidence": evidence,
            "memory_results": memory_results,
        }
    ).content
    # -----------------------------------------------------------------
    
    new_state = state.copy()
    new_state['final_report'] = final_report
    LOGGER.info("Agent 2 finished. Final Report generated.")
    return new_state

print("âœ… Agent Functions (Normalizer and Synthesizer) redefined and fixed successfully.")


# --- 4. Define the Sequential LangGraph Workflow (Multi-Agent System) ---
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Re-define AgentState for local scope clarity (it should be globally available but good practice)
class AgentState(TypedDict):
    patient_summary: str
    normalized_data: dict
    evidence_base: str
    final_report: str

workflow = StateGraph(AgentState)

# Add Nodes (Agents)
# Note: These node names match the function names you defined in the previous cell.
workflow.add_node("Normalizer", clinical_data_normalizer)
workflow.add_node("Synthesizer", treatment_synthesizer)

# Set the Entry Point and Edges (Sequential Flow)
workflow.set_entry_point("Normalizer")
workflow.add_edge("Normalizer", "Synthesizer")

# Set the End Point
workflow.add_edge("Synthesizer", END)

# Compile the graph into a runnable App
app = workflow.compile()
LOGGER.info("Multi-Agent Workflow (LangGraph) compiled successfully.")

# --- 5. Execution Example ---
# The specific patient case you chose:
patient_case = (
    "Patient is a 55-year-old male recently diagnosed with Rheumatoid Arthritis. "
    "He has a history of Hepatitis B infection and severe Gastroesophageal Reflux Disease (GERD). "
    "The physician needs a recommendation for an initial disease-modifying antirheumatic drug (DMARD)."
)

print("\n" + "="*80)
print(f"ğŸ�¥ Starting Personalized Treatment Plan Recommender Agent (CDSA) for Patient: {patient_case[:300]}...")
print("="*80 + "\n")

# Run the entire multi-agent system
# This initiates the flow: Normalizer runs, then Synthesizer runs.
final_state = app.invoke({"patient_summary": patient_case})

# Final Output
print("\n" + "="*80)
print("âœ… FINAL TREATMENT RECOMMENDATION REPORT GENERATED")
print("="*80)
print(final_state['final_report'])

