!pip install --upgrade --quiet google-generativeai pylint pytest


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")



import google.generativeai as genai
import pathlib
import subprocess
from typing import List
from google.ai.generativelanguage_v1beta import FunctionResponse, Part


# --- 1. Define All Agent Functions ---

# helper LLMs used for agents that need to "think" This separates the "Orchestrator" brain from the "Worker" brains

genai.configure(api_key=GOOGLE_API_KEY)

llm = genai.GenerativeModel(model_name="gemini-2.5-flash")

def llm_helper(prompt: str) -> str:
    return llm.generate_content(prompt).text

# --- Agent 2: Requirements Agent ---
def requirements_agent(requirements: str, source_code: str) -> str:
    
    print("--- ğŸ¤– Requirements Analysis In Progress... ---")
    prompt = f"""
    You are a Senior Software Architect and Technical Analyst. Your job is to analyze this text from a developer's perspective and generate a single, valid JSON object as the output.
    Output a JSON object with two keys:
    "structured_requirements": ["list", "of", "criteria"]
    "gap_report": "A markdown string identifying any unclear, missing, or conflicting information. If there are no issues, just write "No gaps found."
    
    REQUIREMENTS:
    {requirements}
    
    SOURCE CODE:
    {source_code}
    """
    # Using generate_content for a simple string response
    return llm_helper(prompt)

# --- Agent 3: Test Case Agent (QA Engineer) ---
def test_case_agent(structured_requirements: List[str]) -> str:
    
    print("--- ğŸ¤– Test Case Agent working... ---")
    prompt = f"""
    You are a senior QA Engineer. rite a pytest test suite as a single string of Python code (as a string).
    based on these requirements: {structured_requirements}
    Assume functions are imported from 'app.py' (e.g., `from app import *`).
    """
    return llm_helper(prompt).replace("```python", "").replace("```", "")

# --- Agent 4: Static Analysis Agent (Code Reviewer) ---
def static_analysis_agent(source_code: str) -> str:
    
    print("--- ğŸ¤– Static Analysis Agent working... ---")
    code_path = pathlib.Path("temp_app.py")
    code_path.write_text(source_code)
    
    try:
        result = subprocess.run(
            ['pylint', str(code_path)], 
            capture_output=True, 
            text=True,
            check=False
        )
        pylint_report = result.stdout
    except FileNotFoundError:
        pylint_report = "Error: pylint not found."
    
    code_path.unlink()
    
    prompt = f"""
    You are a Senior Code Reviewer. Interpret this pylint report and 
    create a human-readable "Issues & Improvements Report".
    
    PYLINT REPORT:
    {pylint_report}
    """
    return llm_helper(prompt)

# --- Agent 5: Dynamic Analysis Agent (Runtime Tester) ---
def dynamic_analysis_agent(source_code: str, test_suite: str) -> str:
    
    print("--- ğŸ¤– Dynamic Analysis Agent working... ---")
    code_path = pathlib.Path("app.py")
    test_path = pathlib.Path("test_app.py")
    
    try:
        code_path.write_text(source_code)
        test_path.write_text(test_suite)
        
        result = subprocess.run(
            ['pytest', str(test_path)],
            capture_output=True,
            text=True,
            check=False
        )
        pytest_report = result.stdout + "\n" + result.stderr
        
    except Exception as e:
        pytest_report = f"Error running tests: {e}"
    finally:
        if code_path.exists(): code_path.unlink()
        if test_path.exists(): test_path.unlink()

    prompt = f"""
    You are a Runtime Tester. Interpret this pytest output and 
    create a "Runtime & Test Results Report".
    
    PYTEST OUTPUT:
    {pytest_report}
    """
    return llm_helper(prompt)

# --- Agent 6: Developer Agent (Auto-Remediator) ---
def developer_agent(source_code: str, all_reports: str) -> str:
   
    print("--- ğŸ¤– Developer Agent working... ---")
    prompt = f"""
    You are an expert Developer. Rewrite the source code to fix all
    issues identified in the reports.
    
    REPORTS:
    {all_reports}
    
    ORIGINAL SOURCE CODE:
    {source_code}
    
    Output *only* the new, complete, fixed Python code.
    """
    return llm_helper(prompt).replace("```python", "").replace("```", "")





# Example Inputs

#"user-provided" requirements

EXAMPLE_REQUIREMENTS = """
Objective 1.1. Create a Python function validate_hex_color that determines if a given string is a valid 3-digit or 6-digit hexadecimal color code.

Functional Requirements 
2.1. Function Signature 
    2.1.1. The function must be named validate_hex_color. 
    2.1.2. It must accept one string argument named color_code. 
    2.1.3. It must return a boolean value (True or False).

2.2. Validation Logic 
    2.2.1. The function must return True if the string is a valid 3-digit or 6-digit hex code. 
    2.2.2. A valid hex code must start with a pound/hash symbol (#). 
    2.2.3. Following the #, the string must contain exactly 3 or 6 characters. 
    2.2.4. All characters following the # must be valid hexadecimal characters (A-F, a-f, 0-9). 
    2.2.5. The validation must be case-insensitive (e.g., #FF0000 and #ff0000 are both valid).

2.3. Handling Invalid Input 2.3.1. The function must return False for any string that does not meet the criteria.
    2.3.2. This includes, but is not limited to: 
        2.3.2.1. Strings without a leading # (e.g., FF0000). 
        2.3.2.2. Strings with the wrong length (e.g., #F000, #12345). 
        2.3.2.3. Strings containing non-hexadecimal characters (e.g., #F00G00). 
        2.3.2.4. Empty strings ("") or non-string inputs.

Non-Functional Requirements 
3.1. Dependencies 
    3.1.1. The function must not require any external libraries. It should only use built-in Python modules (like the re module for regular expressions, or standard string methods).

3.2. Documentation 
    3.2.1. The function must include a concise docstring (e.g.,Checks if a string is a valid #RRGGBB or #RGB hex code.).

Acceptance Criteria (Examples) 
4.1. The function must produce the following outputs for the given inputs: 
    4.1.1. Input: "#FF0000" -> Output: True (Rationale: Valid 6-digit hex code.) 
    4.1.2. Input: "#f00" -> Output: True (Rationale: Valid 3-digit hex code.) 
    4.1.3. Input: "#faC123" -> Output: True (Rationale: Valid 6-digit, mixed-case.) 
    4.1.4. Input: "FF0000" -> Output: False (Rationale: Missing leading #.) 
    4.1.5. Input: "#1234" -> Output: False (Rationale: Invalid length (4).) 
    4.1.6. Input: "#GG0000" -> Output: False (Rationale: Contains invalid char 'G'.) 
    4.1.7. Input: "#F0" -> Output: False (Rationale: Invalid length (2).) 
    4.1.8. Input: "#" -> Output: False (Rationale: Invalid length (0).) 
    4.1.9. Input: "" -> Output: False (Rationale: Empty string.)
"""

# Our example "user-provided" source code (with intentional bugs)
EXAMPLE_CODE = """
import re

def validate_hex_color(color_code):
    
    if len(color_code) != 4 and len(color_code) != 7:
        return False
    
    code = color_code[1:]
    
    for char in code:
        if char not in '0123456789abcdef':
            return False
            
    return True
"""




# Initialize Orchestrator and Tools ---

# Create the tool list for the orchestrator

agent_tools = [
    requirements_agent,
    test_case_agent,
    static_analysis_agent,
    dynamic_analysis_agent,
    developer_agent,
]

agent_functions = {
    "requirements_agent": requirements_agent,
    "test_case_agent": test_case_agent,
    "static_analysis_agent": static_analysis_agent,
    "dynamic_analysis_agent": dynamic_analysis_agent,
    "developer_agent": developer_agent,
}

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="You are the Orchestrator Agent. Your job is to manage a team of specialist AI agents to validate user code. "
                      "You will be given requirements and source code. "
                      "You must follow this workflow: "
                      "1. Call `requirements_agent` to get a structured list and gap report. "
                      "2. Call `test_case_agent` to generate tests. "
                      "3. Call `static_analysis_agent` to find bugs. "
                      "4. Call `dynamic_analysis_agent` to run the tests. "
                      "5. Once all 4 reports are in, present a 'Master Validation Report' to the user. "
                      "6. Ask the user for permission to fix the code. "
                      "7. If they say 'yes', call `developer_agent` with all reports to get the fixed code. "
                      "8. DO NOT fix the code without explicit user permission.",
    tools=agent_tools  # <--- This is correct now
)

print("âœ… Orchestrator Model (gemini-2.5-flash) initialized with tools.")

# --- 3.5. Create a Function Lookup Dictionary ---
# This lets us call functions by their string name.
agent_functions = {
    "requirements_agent": requirements_agent,
    "test_case_agent": test_case_agent,
    "static_analysis_agent": static_analysis_agent,
    "dynamic_analysis_agent": dynamic_analysis_agent,
    "developer_agent": developer_agent,
}
print("âœ… Agent function lookup dictionary created.")





# --- 4. Start the Chat Workflow ---

chat = model.start_chat()

print("--- ğŸ‘¨â€�ğŸ’» USER ---")
print("Please validate this code.")

# Send the initial message
response = chat.send_message(
    f"Here are my files. Please start the validation workflow."
    f"\n\nREQUIREMENTS.md:\n{EXAMPLE_REQUIREMENTS}"
    f"\n\napp.py:\n{EXAMPLE_CODE}",
    tools=agent_tools  
)

print("\n--- ğŸ¤– ORCHESTRATOR (Gemini) ---")

# This loop handles the agent calls for the *analysis* phase
while response.parts and response.parts[0].function_call:
    fc = response.parts[0].function_call
    fname = fc.name
    args = {key: value for key, value in fc.args.items()}
    
    print(f"ğŸ”© Calling function: {fname}()...")
    
    function_to_call = agent_functions[fname]
    result = function_to_call(**args)
    
    # Your manual response part (this is correct for this library version)
    part = Part(function_response=FunctionResponse(name=fname, response={'result': result}))
    response = chat.send_message(content=[part])

# Once the loop finishes, the response is the final text report
master_report = response.text  
print(master_report)




# --- 5. Human-in-the-Loop step (Corrected for Old Library) ---
print("\n--- ğŸ‘¨â€�ğŸ’» USER (Decision) ---")
user_input = "Yes, please attempt the auto-fix."
print(user_input)

fix_prompt = f"""
Thank you, you are now authorized to fix the code.
Please call the `developer_agent` function.

Here is the original `source_code`:```python {EXAMPLE_CODE}"""

response = chat.send_message(user_input)

# This SECOND loop handles the call to the *developer_agent*
while response.parts and response.parts[0].function_call:
    fc = response.parts[0].function_call
    fname = fc.name
    args = {key: value for key, value in fc.args.items()}
    
    print(f"ğŸ”© Calling function: {fname}()...")
    
    function_to_call = agent_functions[fname]
    result = function_to_call(**args)
    
    # --- THIS IS THE FIX (applied again) ---
    part = Part(function_response=FunctionResponse(name=fname, response={'result': result}))
    response = chat.send_message(content=[part])

print("\n--- ğŸ¤– ORCHESTRATOR (Gemini) ---")
# This is the final output: the fixed code
print(response.text)

