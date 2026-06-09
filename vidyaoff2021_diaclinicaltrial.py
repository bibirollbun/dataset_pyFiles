# =============================================================================
# 1. Install Dependencies
# =============================================================================
import asyncio
import os
import json
import logging # Added for standard logging
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- OBSERVABILITY IMPORTS ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


# =============================================================================
# 2. Configure API Key
# =============================================================================
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# =============================================================================
# Intialization
# ============================================================================
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.0-flash"
APP_NAME = "clinical_trial_app"
USER_ID = "demo_user"
SESSION_ID = "session_single_001"


# =============================================================================
# 3. Extrat Patient Data from mock-patientdata.json (Simplified Clinical Records)
# =============================================================================
import json
import os

def load_patient_database():
    """
    Loads patient data from the Kaggle input directory.
    """
    file_path = "/kaggle/input/mock-patientdata/patients.json"    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            print(f"âœ… Successfully loaded {len(data)} records from {file_path}")
            return data     
    except FileNotFoundError:
        print(f"â�Œ Error: File not found at {file_path}")
        print("Contents of /kaggle/input/mock-patientdata:")
        try:
            print(os.listdir("/kaggle/input/mock-patientdata"))
        except:
            print("Could not list directory.")
        return {}      
    except json.JSONDecodeError:
        print(f"â�Œ Error: Failed to parse JSON at {file_path}")
        return {}

# Load the data when this module is imported
PATIENT_DB = load_patient_database()


# =============================================================================
# 4. CUSTOM TOOLS
# =============================================================================

def list_patient_ids() -> str:
    """Returns all patient IDs from the database."""
    # Tracing will automatically capture that this tool was called
    all_ids = list(PATIENT_DB.keys())
    print(f"\n[Tool: list_patient_ids] Found {len(all_ids)} patients: {all_ids}")
    return json.dumps({"patient_ids": all_ids})

def get_patient_data(patient_id: str) -> str:
    """Retrieves patient data from the database."""
    print(f"\n[Tool: get_patient_data] Fetching data for {patient_id}...")
    p_data = PATIENT_DB.get(patient_id)
    if not p_data:
        return json.dumps({"error": f"Patient {patient_id} not found"})
    
    print(f"   Age: {p_data['age']}, HbA1c: {p_data['labs']['hba1c']}, Meds: {p_data['medications']}")
    return json.dumps({"patient_id": patient_id, "data": p_data})

def assess_patient_eligibility(patient_id: str, age: int, hba1c: float, medications: list[str], conditions: list[str]) -> str:
    """Checks eligibility based on trial criteria."""
    print(f"\n[Tool: assess_patient_eligibility] Analyzing {patient_id}...")
    reasons = []
    
    # Check Age
    if not (30 <= age <= 80):
        print(f"   â�Œ Failed Age Criteria ({age})")
        reasons.append(f"Age {age} outside range 30-80")
        return json.dumps({"patient_id": patient_id, "eligible": False, "reasons": reasons})
    else:
        print(f"   âœ… Age Check Passed ({age})")
    
    # Check HbA1c
    if hba1c <= 8.0:
        print(f"   â�Œ Failed HbA1c Criteria ({hba1c})")
        reasons.append(f"HbA1c {hba1c} not > 8.0")
        return json.dumps({"patient_id": patient_id, "eligible": False, "reasons": reasons})
    else:
        print(f"   âœ… HbA1c Check Passed ({hba1c})")
    
    # Check for Metformin
    if any("metformin" in med.lower() for med in medications):
        print("   â�Œ Failed Medication Exclusion (Metformin)")
        reasons.append("Exclusion: Currently on Metformin")
        return json.dumps({"patient_id": patient_id, "eligible": False, "reasons": reasons})
    else:
        print("   âœ… Medication Check Passed")
    
    # Check for ESRD
    if any("end-stage renal disease" in cond.lower() or "esrd" in cond.lower() for cond in conditions):
        print("   â�Œ Failed Condition Exclusion (ESRD)")
        reasons.append("Exclusion: Has End-Stage Renal Disease")
        return json.dumps({"patient_id": patient_id, "eligible": False, "reasons": reasons})
    else:
        print("   âœ… Condition Check Passed")
    
    # All checks passed
    print("   âœ… PATIENT IS ELIGIBLE")
    return json.dumps({"patient_id": patient_id, "eligible": True, "reasons": ["Passed all inclusion and exclusion criteria"]})



# Wrap tools
tool_list_patients = FunctionTool(func=list_patient_ids)
tool_get_data = FunctionTool(func=get_patient_data)
tool_assess = FunctionTool(func=assess_patient_eligibility)


# Setting up Observability
def setup_observability(log_file="trace_output.json"):
    """
    Configures OpenTelemetry to capture agent traces.
    Safely handles re-runs in Jupyter Notebooks.
    """
    # 1. Define where the traces go
    trace_file = open(log_file, "a") 
    
    # 2. Check if a TracerProvider is already registered
    current_provider = trace.get_tracer_provider()
    
    # If it's already a real TracerProvider (not the default Proxy), just reuse it.
    if isinstance(current_provider, TracerProvider):
        print(f"â„¹ï¸� Observability already configured. Appending to: {log_file}")
        return current_provider, trace_file

    # 3. If not set, create and register a new one
    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter(out=trace_file))
    provider.add_span_processor(processor)
    
    trace.set_tracer_provider(provider)
    
    print(f"ğŸ”� Observability enabled. Detailed traces will be written to: {log_file}")
    return provider, trace_file


# =============================================================================
# 5. AGENTS
# =============================================================================

# Agent 1: Cohort Manager
cohort_agent = LlmAgent(
    name="Cohort_Manager",
    model=MODEL_ID,
    instruction="""
    You are the Cohort Manager agent.
    Your job is to identify which patient(s) need to be screened.
    Steps:
    1. Call the `list_patient_ids` tool to get patient IDs
    2. Return the patient ID(s) to pass to the next agent
    Output format: "Patient IDs to screen: [list of IDs]"
    """,
    tools=[tool_list_patients]
)

# Agent 2: Data Collector
collector_agent = LlmAgent(
    name="Data_Collector",
    model=MODEL_ID,
    instruction="""
    You are the Data Collector agent.
    You receive patient ID(s) from the previous agent.
    Steps:
    1. Extract ALL patient IDs from the previous agent's output
    2. Call `get_patient_data` for EACH patient ID (call the tool multiple times if needed)
    3. Collect all patient data and pass it to the next agent
    Output format: "Patient data retrieved for [count] patients: [list IDs and key data]"
    """,
    tools=[tool_get_data]
)

# Agent 3: Medical Evaluator
evaluator_agent = LlmAgent(
    name="Medical_Evaluator",
    model=MODEL_ID,
    instruction="""
    You are the Medical Evaluator agent.
    You receive patient data from the previous agent.
    Steps:
    1. For EACH patient in the data received:
       - Extract the patient_id, age, hba1c, medications, and conditions
       - Call `assess_patient_eligibility` with these parameters
    2. Collect all eligibility assessments
    3. Report all results to the next agent
    Output format: "Eligibility assessment complete for [count] patients: [list each patient with eligible/not eligible status]"
    """,
    tools=[tool_assess]
)

# Agent 4: Final Reviewer
reviewer_agent = LlmAgent(
    name="Final_Reviewer",
    model=MODEL_ID,
    instruction="""
    You are the Final Reviewer agent.
    Review all the information from previous agents and create a final summary.
    Output a clear, concise summary that includes:
    - Total number of patients screened
    - For EACH patient:
      * Patient ID
      * Eligibility status (ELIGIBLE or NOT ELIGIBLE)
      * Key reasons for the decision
    - Summary statistics (how many eligible vs not eligible)
    Format it as a professional clinical trial screening report.
    """,
    tools=[]
)


# Create Sequential Pipeline
pipeline_agent = SequentialAgent(
    name="Patient_Screening_Pipeline",
    sub_agents=[cohort_agent, collector_agent, evaluator_agent, reviewer_agent],
    description="Multi-agent pipeline for clinical trial patient screening."
)

root_agent = pipeline_agent


# =============================================================================
# 6. EXECUTION
# =============================================================================

PROTOCOL_TEXT = """
Find patients for the 'Dia-Beta-Care Trial'.
Inclusion Criteria:
1. Patient must be between 30 and 80 years old.
2. Must have a "Hemoglobin A1c" lab result > 8.0.
Exclusion Criteria:
1. Patient must NOT have "End-Stage Renal Disease".
2. Patient must NOT be on "Metformin".
"""
async def main():
    print("="*70)
    print("CLINICAL TRIAL PATIENT SCREENING - MULTI-AGENT SYSTEM")
    print("="*70)
    

    # --- START OBSERVABILITY ---
    trace_provider, trace_file_handle = setup_observability()
    
    try:
        # Initialize session
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id=SESSION_ID
        )

        # Create runner
        runner = Runner(
            agent=root_agent, 
            app_name=APP_NAME, 
            session_service=session_service
        )
        
        # Create message with protocol
        content = types.Content(
            role='user', 
            parts=[types.Part(text=PROTOCOL_TEXT)]
        )

        print(f"\nğŸ“‹ Protocol Loaded: Dia-Beta-Care Trial")
        print(f"ğŸ‘¥ Total Patients in Database: {len(PATIENT_DB)}")
        print("\n" + "="*70)
        print("AGENT EXECUTION LOG")
        print("="*70)

        # Run the pipeline
        events = runner.run(
            user_id=USER_ID, 
            session_id=SESSION_ID, 
            new_message=content
        )

        # Process events
        final_output = []
        for event in events:
            if event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        final_output.append(part.text)
        
        # --- DISPLAY & SAVE REPORT ---
        print("\n" + "="*70)
        print("FINAL SCREENING REPORT")
        print("="*70)
        
        # Open a file to write the report
        output_filename = "screening_report.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            # Write Header
            header = "="*70 + "\nFINAL SCREENING REPORT\n" + "="*70 + "\n"
            f.write(header)
            
            for output in final_output:
                # 1. Print to Console
                print(output)
                # 2. Write to File
                f.write(output + "\n")
                
            footer = "="*70 + "\n"
            print(footer)
            f.write(footer)

        print(f"\nâœ… Report saved to: {output_filename}")

    finally:
        # --- STOP OBSERVABILITY ---
        trace_provider.shutdown()
        trace_file_handle.close()
        print(f"\n[Trace] Observability data flushed to 'trace_output.json'")


if __name__ == "__main__":
    await main()

