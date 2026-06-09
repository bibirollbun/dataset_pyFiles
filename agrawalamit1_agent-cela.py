# 1. Setup and Authentication
!pip install --upgrade google-cloud-aiplatform google-genai -q

import google.generativeai as genai
# Import the low-level library to bypass "Part" errors
import google.ai.generativelanguage as glm 
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown

# --- AUTHENTICATION ---
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("âœ… Environment Initialized & API Key retrieved successfully.")
except Exception as e:
    print(f"â�Œ Error during authentication. Ensure 'GOOGLE_API_KEY' is set in Kaggle Secrets.\n{e}")


# Define Tools (Code Executor)

def execute_python_code(code: str):
    """
    Executes Python code and returns the standard output.
    Useful for verifying algorithms or running test cases.
    """
    import sys
    from io import StringIO
    
    # Create a buffer to capture stdout
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        # Create a safe dictionary for execution globals
        exec_globals = {}
        exec(code, exec_globals)
        output = redirected_output.getvalue()
        if not output:
            output = "Code executed successfully (No Output)."
        return output
    except Exception as e:
        return f"Execution Error: {e}"
    finally:
        sys.stdout = old_stdout

# Convert to a Tool object for Gemini
tools_list = [execute_python_code]
print("âœ… Tools Defined: execute_python_code")


# @title 3. Define the Multi-Agent System (Stable GLM Version)

class Agent:
    def __init__(self, name, model_name="gemini-2.5-pro", system_instruction="", tools=None, temperature=0.2):
        self.name = name
        self.tools = tools
        
        # Set up generation config for consistency
        generation_config = {
            "temperature": temperature, # Lower temperature = more consistent, less random
            "max_output_tokens": 8192,
        }
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            tools=tools,
            generation_config=generation_config
        )
        self.history = []

    def generate(self, prompt):
        # 1. Start a new chat session with the accumulated history
        chat = self.model.start_chat(history=self.history)
        
        # 2. Add the user prompt to the history (Manual history tracking)
        # Note: We use the simple dict format for history which the SDK converts automatically
        self.history.append({"role": "user", "parts": [prompt]})
        
        # 3. Send the prompt
        response = chat.send_message(prompt)
        
        # Helper: Extract function calls safely
        def get_function_calls(response):
            if not response.candidates: return []
            return [part.function_call for part in response.candidates[0].content.parts if part.function_call]

        function_calls = get_function_calls(response)
        
        # 4. Tool Handling Loop
        while function_calls:
            # We must construct the response parts using the low-level GLM library
            # to avoid version conflicts with the high-level SDK.
            tool_response_parts = []
            
            for fc in function_calls:
                # Find the matching tool function
                tool_function = next((t for t in self.tools if t.__name__ == fc.name), None)
                
                if tool_function:
                    try:
                        # Execute the tool
                        args = dict(fc.args)
                        tool_output = tool_function(**args)
                    except Exception as e:
                        tool_output = f"Error executing tool: {str(e)}"
                    
                    # --- THE FIX: Use glm.Part and glm.FunctionResponse ---
                    # This constructs the exact Protobuf object the API expects.
                    part = glm.Part(
                        function_response=glm.FunctionResponse(
                            name=fc.name,
                            response={"result": tool_output}
                        )
                    )
                    tool_response_parts.append(part)
                
            # Update history with the Model's request (Function Call)
            self.history.append(response.candidates[0].content)
            
            # Update history with the User's answer (Function Response)
            # We create a Content object for the history
            tool_content = glm.Content(role="user", parts=tool_response_parts)
            self.history.append(tool_content)

            # Send the tool output back to the model
            response = chat.send_message(tool_content)
            
            # Check for subsequent function calls
            function_calls = get_function_calls(response)

        # 5. Final Response
        self.history.append(response.candidates[0].content)
        return response.text

# --- AGENT INITIALIZATION ---

# 1. Formulator (No Tools)
formulator_agent = Agent(
    name="Formulator",
    model_name="gemini-2.5-flash", 
    temperature=0.2,
    system_instruction="""
    You are a strict Computer Engineering Professor. 
    Rewrite user requests into professional 'Laboratory Problem Statements'.
    1. Keep the 'Problem Statement' concise and easy to read. Use academic terminology.
    2. List Constraints and Edge Cases clearly using bullet points.
    3. Do NOT write code. Only define the problem.
    4. Avoid overly verbose academic jargon; keep it practical
    """
)

# 2. Architect (No Tools)
architect_agent = Agent(
    name="Architect",
    model_name="gemini-2.5-pro",
    temperature=0.1, # Very low temp for code consistency
    system_instruction="""
    You are a Senior Software Architect.
    Output structure:
    1. **Solution Logic**: Explain the approach. Avoid complex jargon.
    2. **Complexity Analysis**: Briefly state Time and Space Big O.
    3. **Write the Code**:
       - PRIORITIZE READABILITY over cleverness.
       - Use standard loops (for/while) instead of complex one-liners (list comprehensions) if it makes logic clearer.
       - Use meaningful variable names (e.g., 'current_node' instead of 'curr').
       - Add comments explaining *why* a step is taken.
    4. **Output**: Show what the output looks like.
    5. **Documentation**: Brief summary.

    Constraint: Do NOT use obscure libraries. Use standard approaches.
    """
)

# 3. Reviewer (HAS TOOLS)
# We need to make sure 'tools_list' is defined from Cell 2 before running this.
reviewer_agent = Agent(
    name="Reviewer",
    model_name="gemini-2.5-flash",
    tools=tools_list, 
    temperature=0.0, # Zero temp for strict validation
    system_instruction="""
    You are a Quality Assurance Engineer.
    1. If the language is Python, use the 'execute_python_code' tool to RUN the code.
    2. If not Python, generate a 'Dry Run Table' (Trace Table).
    3. Verify if the actual output matches expectations. Keep your final report short: "Pass/Fail" and the Output. 
    """
)

print("âœ… Agents Initialized: Formulator, Architect, Reviewer.")


# Run the Agent Chain

def run_lab_assistant(user_prompt, language="Python"):
    print(f"ğŸ”µ **Input Received:** {user_prompt} [{language}]\n")
    
    # --- STEP 1: FORMULATE ---
    print("...Formulator is thinking...")
    problem_statement = formulator_agent.generate(
        f"User Request: {user_prompt}\nTarget Language: {language}"
    )
    display(Markdown(f"### 1. Problem Formulation\n{problem_statement}"))
    
    # --- STEP 2: ARCHITECT ---
    print("...Architect is coding...")
    solution_package = architect_agent.generate(
        f"Based on this problem statement, generate the solution:\n{problem_statement}\n\nLanguage: {language}"
    )
    display(Markdown(f"### 2. Proposed Solution & Code\n{solution_package}"))
    
    # --- STEP 3: REVIEW & EXECUTE ---
    print("...Reviewer is validating...")
    validation = reviewer_agent.generate(
        f"Verify this code and logic:\n{solution_package}\n\nIf Python, EXECUTE it. If not, DRY RUN it."
    )
    display(Markdown(f"### 3. Verification & Output\n{validation}"))
    
    return solution_package


# Test Case
# Example 1: C# (Will trigger Tool Execution)
run_lab_assistant("Identify entered character is Vowel or Consonent", language="Python")

# Example 2: Java (Will trigger Dry Run Trace)
run_lab_assistant("Identify entered character is Vowel or Consonent", language="Java")

