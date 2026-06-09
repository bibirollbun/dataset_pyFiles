# 1. Install the Google AI SDK (Run this once)
!pip install -q -U google-generativeai

# 2. Import Libraries
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import json
import time # Used for Observability (Tracing/Metrics)
import textwrap
from IPython.display import display, Markdown

# --- Utility Function for Clean Display ---
def to_markdown(text):
    """Converts a string to a clean Markdown format for display in the notebook."""
    if text and (text.startswith("'#") or text.startswith('"#')):
        text = text.strip("'").strip('"').strip("# ")
    text = text.replace('â€¢', '  *')
    return Markdown(textwrap.indent(text, '', predicate=lambda _: True))

# --- API Key Configuration ---
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ… Success: Google Gemini API configured!")
except Exception as e:
    print(f"â�Œ Error during API configuration: {e}")
    print("Please ensure you added the secret with the Label 'GOOGLE_API_KEY'.")

# --- Agent Class (Includes Observability) ---
class Agent:
    def __init__(self, role, system_instruction, model_name='gemini-2.5-flash'):
        self.role = role
        self.system_instruction = system_instruction
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.system_instruction,
        )
        print(f"âš™ï¸� Agent '{self.role}' initialized using {model_name}.")

    # MODIFIED: Tracks generation time for Tracing and returns token approximation for Logging
    def generate_response(self, prompt: str) -> tuple[str, float]:
        start_time = time.time() # START TRACE
        try:
            response = self.model.generate_content(prompt)
            end_time = time.time() # END TRACE
            
            # --- OBSERVABILITY: Logging ---
            print(f"   [LOG] {self.role} Input Tokens (Approx): {len(prompt)}")
            print(f"   [LOG] {self.role} Output Tokens (Approx): {len(response.text)}")
            
            # Return response text and time taken (Tracing metric)
            return response.text, (end_time - start_time)
        except Exception as e:
            print(f"   [LOG] Error running agent {self.role}: {e}")
            return f"Error running agent {self.role}: {e}", 0.0

# --- Constants for Agent System Instructions ---
ORCHESTRATOR_INSTRUCTION = "You are the Multi-Agent System Manager. Your role is to coordinate the Analyst and the Evaluator. 1. Start: Send the initial business problem to the Analyst. 2. Loop: If the Evaluator provides feedback, send a clear, concise REVISION instruction to the Analyst based on the feedback. 3. Finish: Only deliver the final output to the user after the Evaluator approves the work or after 3 revision attempts."
ANALYST_INSTRUCTION = "You are a Senior Business Analyst. Your task is to draft detailed, clear, and unambiguous business requirements and user stories for the given problem. Initial Task: Draft the full requirements document. Revision Task: Read the provided 'EVALUATOR FEEDBACK' carefully and integrate all critiques into your revised document. Do not just append; actually fix the content. Always output a clean, well-formatted Markdown document, starting with a major title like '# Business Requirements Document...'"
# UPDATED: Emphasizing the need for a numeric score to fix the 'score=0' issue
EVALUATOR_INSTRUCTION = "You are a ruthless Quality Assurance and Feasibility Expert. You must critique the Business Analyst's draft based on Clarity, Completeness, and Feasibility (CCF). YOU MUST INCLUDE A NUMERIC 'score' FROM 1 to 10. ONLY output a JSON object with the following schema: { 'score': 1-10, 'pass_status': 'PASS' or 'FAIL', 'feedback_summary': 'A brief summary of the main issues.', 'detailed_critiques': ['Critique 1 on Clarity...', 'Critique 2 on Completeness...']}. A score of 9 or above with 'PASS' status is required for final submission."

MAX_ATTEMPTS = 3


# --- The Orchestration Function (Including Loop, Context, and Observability) ---
def run_analysis_system(user_problem: str):
    
    # METRICS initialization
    total_llm_time = 0.0
    
    # Instantiate the Agents
    orchestrator = Agent("Orchestrator", ORCHESTRATOR_INSTRUCTION)
    analyst = Agent("Business Analyst", ANALYST_INSTRUCTION)
    evaluator = Agent("Evaluator", EVALUATOR_INSTRUCTION)

    start_system_time = time.time()
    print(f"\n--- ğŸš€ Starting Analysis System for: {user_problem} ---\n")
    
    current_draft = ""
    final_evaluation = {}
    
    # 1. Initial Run (Orchestrator -> Analyst)
    initial_prompt = f"USER PROBLEM: {user_problem}. Begin drafting the requirements now."
    print("ğŸ“¢ Orchestrator: Tasking Analyst for initial draft...")
    current_draft, generation_time = analyst.generate_response(initial_prompt)
    total_llm_time += generation_time
    
    
    # --- MULTI-AGENT LOOP ARCHITECTURE ---
    for attempt in range(1, MAX_ATTEMPTS + 1):
        time.sleep(1) 
        print(f"\n-- ğŸ”„ Attempt {attempt} of {MAX_ATTEMPTS} --")
        
        # 2. Evaluation Phase (Agent Evaluation)
        evaluation_prompt = f"ANALYST DRAFT:\n{current_draft}\n\nEVALUATOR: Critique this draft and output ONLY JSON."
        print(f"ğŸ”¬ Orchestrator: Sending draft to Evaluator...")
        
        eval_raw_output, generation_time = evaluator.generate_response(evaluation_prompt)
        total_llm_time += generation_time
        
        # 3. Parse Evaluation and Check Status
        try:
            # Safely extract JSON block
            json_start = eval_raw_output.find('{')
            json_end = eval_raw_output.rfind('}') + 1
            eval_json = json.loads(eval_raw_output[json_start:json_end])
            final_evaluation = eval_json
            
            # This line handles missing 'score' key by defaulting to 0
            status = eval_json.get("pass_status", "FAIL").upper()
            score = eval_json.get("score", 0) 
            
            print(f"âœ… Evaluator Result: Status={status}, Score={score}")
            
            if status == "PASS" and score >= 9:
                end_system_time = time.time()
                print("\nğŸ�‰ Analysis Passed! Submitting Final Draft.")
                
                # --- OBSERVABILITY: Metrics Report ---
                print("\n--- ğŸ“ˆ OBSERVABILITY & METRICS REPORT ---")
                print(f"Total Attempts: {attempt}")
                print(f"Total System Runtime: {end_system_time - start_system_time:.2f} seconds")
                print(f"Total LLM Generation Time: {total_llm_time:.2f} seconds")
                print("------------------------------------------")

                print("\n--- ğŸ“„ FINAL DELIVERABLE: BUSINESS REQUIREMENTS DOCUMENT ---")
                display(to_markdown(current_draft))
                print("\n--- âœ… FINAL EVALUATION CRITIQUE ---")
                print(json.dumps(final_evaluation, indent=4))
                return 
            
            # Revision logic (If FAIL)
            feedback = eval_json.get("detailed_critiques", ["No detailed feedback provided. Ensure Clarity and Completeness."])
            feedback_str = "\n".join([f"- {c}" for c in feedback])
            
            if attempt < MAX_ATTEMPTS:
                print("ğŸ“� Orchestrator: Sending revision instructions back to Analyst...")
                
                # --- CONTEXT ENGINEERING: Compaction ---
                # Only sends the critique, not the entire previous draft, to save tokens.
                revision_prompt = f"""
                Your previous draft failed evaluation (Score: {score}). You MUST revise the document based ONLY on the following detailed feedback.
                EVALUATOR FEEDBACK:
                {feedback_str}
                
                ACTION: Provide a fully revised requirements document now.
                """
                current_draft, generation_time = analyst.generate_response(revision_prompt)
                total_llm_time += generation_time
                
        except json.JSONDecodeError as e:
            # --- CRITICAL JSON DEBUG LOG ---
            print("â�Œ Error: Evaluator did not return valid JSON.")
            print(f"   [DEBUG] JSON Decode Error: {e}")
            print("   [DEBUG] RAW EVALUATOR OUTPUT START:")
            print(textwrap.indent(eval_raw_output, '   > '))
            print("   [DEBUG] RAW EVALUATOR OUTPUT END")
            
            # Forces a revision to fix the JSON error
            feedback_str = "The previous output was not valid JSON. Please ensure your revised output is clean JSON and Markdown."
            current_draft, generation_time = analyst.generate_response(f"Fix your previous draft based on this instruction: {feedback_str}")
            total_llm_time += generation_time
        
    # If Max attempts are reached without PASSING
    end_system_time = time.time()
    print("\nâš ï¸� Max attempts reached. Returning the latest draft.")
    
    # --- OBSERVABILITY: Final Metrics Report ---
    print("\n--- ğŸ“ˆ OBSERVABILITY & METRICS REPORT ---")
    print(f"Total Attempts: {MAX_ATTEMPTS}")
    print(f"Total System Runtime: {end_system_time - start_system_time:.2f} seconds")
    print(f"Total LLM Generation Time: {total_llm_time:.2f} seconds")
    print("------------------------------------------")

    print("\n--- ğŸ“„ LATEST DRAFT (FAILED EVALUATION) ---")
    display(to_markdown(current_draft))
    print("\n--- âœ… FINAL EVALUATION CRITIQUE ---")
    print(json.dumps(final_evaluation, indent=4))


PROBLEM_STATEMENT = input("Enter your problem statement :")
# Execute the multi-agent analysis system
run_analysis_system(PROBLEM_STATEMENT)




