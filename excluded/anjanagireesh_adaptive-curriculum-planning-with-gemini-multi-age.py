!pip install -q google-genai


# %% [markdown]
# # ğŸ�† The "EduMentor" Autonomous Multi-Agent Study Assistant
# 
# ## Track: Concierge Agents
# 
# ### Applied Concepts: Multi-Agent System (Sequential/Loop), Tools (Google Search/Function Calling, Code Execution Simulation), Long-Term Memory
# 
# **Solution:** An autonomous multi-agent system that plans, teaches, evaluates, and adapts the learning curriculum in real-time.
# 
# ---

# %% [code]
# 1. Setup & Dependencies
# (Assuming google-genai is installed via a previous cell or the notebook environment)

import os
import json
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient

# --- Configuration ---
GEMINI_MODEL = "gemini-2.5-flash" # The base model for all agents
USER_ID = "student_alpha" 
# ---------------------

# Setup API Key
try:
    # Use the UserSecretsClient to safely access the API key
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
    client = genai.Client()
    print("âœ… Gemini Client Initialized and API Key Loaded Successfully.")
except Exception as e:
    print(f"â�Œ Error initializing Gemini Client: {e}")
    client = None

# %% [code]
# 2. Tool Definitions & Long-Term Memory (LTM) Setup

# --- 2a. Long-Term Memory (LTM) Bank ---
class PerformanceMemoryBank:
    """Simulated Long-Term Memory Bank to store and retrieve user performance."""
    def __init__(self):
        self.data = {
            "performance_history": [], 
            "preferences": {"user_id": USER_ID, "last_topic": None}
        }
    
    def save_report(self, user_id: str, report: str) -> str:
        """Saves the latest evaluation report (score, weak concepts) to history."""
        try:
            parsed_report = json.loads(report)
        except json.JSONDecodeError:
            parsed_report = {"summary": report, "score": "N/A"}

        self.data["performance_history"].append(parsed_report)
        self.data["preferences"]["last_topic"] = parsed_report.get("topic")
        return f"Report successfully saved to memory for {user_id}. History length: {len(self.data['performance_history'])}"
        
    def get_weakness_summary(self, user_id: str) -> str:
        """Retrieves a summary of the user's past weak areas from the LTM."""
        if not self.data["performance_history"]:
            return "No previous performance data found. Treat this as the user's first session."
        
        weakness = ", ".join([
            item.get("weakness", "") for item in self.data["performance_history"] 
            if item.get("score", 0) and (isinstance(item["score"], int) and item["score"] < 3) 
        ])
        
        if not weakness:
             return "Previous performance was good. No specific weak topics recorded."
             
        return f"Summary of past performance: User struggled with these concepts: {weakness}. Prioritize remediation."

memory_bank = PerformanceMemoryBank()

# --- 2b. Tool Functions (Simulated) ---

def grade_quiz(quiz_topic: str, user_answers_json: str) -> str:
    """
    Grades user answers for a quiz and returns a structured report (JSON format).
    Call this function only with the required parameters: quiz_topic and user_answers_json.
    """
    if "for loop" in user_answers_json.lower() and "def function" in user_answers_json.lower():
        score = 5
        weakness = "None"
    else:
        score = 2
        weakness = "Python Syntax and Function Definition"

    report = {
        "user_id": USER_ID,
        "topic": quiz_topic,
        "score": score,
        "total": 5,
        "weakness": weakness,
        "feedback": f"Needs work on: {weakness}. Score: {score}/5."
    }
    # Call the memory tool to save the report immediately
    memory_bank.save_report(USER_ID, json.dumps(report)) 
    
    return json.dumps(report)

# --- CORRECTED TOOL ASSIGNMENTS (Fixes NameError) ---
# We assign the function references to the required variable names, wrapped in a list.

MEMORY_TOOL = [memory_bank.get_weakness_summary] # Tool for the Planner
GRADING_TOOL = [grade_quiz] # Tool for the Evaluator 
# NOTE: The `tools` argument in `run_agent` expects a list of functions.

# %% [code]
# 3. Agent Execution Logic (LLM-Powered Agents Simulation)

def run_agent(system_prompt: str, user_prompt: str, tools: list = None, print_response=False) -> str:
    """Central function to run a single agent using the Gemini Client."""
    if not client: return "Error: Client not initialized."
    
    # Configuration with the system prompt and the list of functions
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools if tools else None
    )
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_prompt],
            config=config
        )

        if print_response:
             print(f"Agent Thought/Response: {response.text}")

        # Handle Function/Tool Calls (If the agent chooses to use a tool)
        if response.function_calls:
            tool_outputs = []
            for func_call in response.function_calls:
                func_name = func_call.name
                func_args = dict(func_call.args)
                
                # --- Execute the actual function based on the name ---
                if func_name == "get_weakness_summary":
                    # Call without args, as the user_id is hardcoded in the LTM class for simplicity
                    output = memory_bank.get_weakness_summary(user_id=USER_ID) 
                elif func_name == "grade_quiz":
                    output = grade_quiz(**func_args)
                else:
                    output = f"Error: Unknown function call: {func_name}"

                tool_outputs.append(
                    types.Part.from_function_response(
                        name=func_name,
                        response={"result": output}
                    )
                )

            # Send the tool output back to the model for a final response
            final_response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[user_prompt, response.candidates[0].content, tool_outputs],
                config=config
            )
            return final_response.text

        return response.text
    
    except Exception as e:
        return f"AGENT ERROR: {e}"

# %% [code]
# 4. Multi-Agent Orchestration (Sequential/Loop Simulation)

def adaptive_study_workflow(initial_goal: str):
    """
    Orchestrates the Planner -> Teacher -> Evaluator -> Planner loop.
    Simulates the Sequential and Loop Agent concepts.
    """
    
    print(f"--- ğŸš€ Starting EduMentor Workflow for Goal: {initial_goal} ---")
    
    # ----------------------------------------------------
    # AGENT 1: PLANNER (Initialization and adaptation)
    # ----------------------------------------------------
    PLANNER_PROMPT = (
        "You are the **Planner Agent**. Your goal is to create a sequential, 2-topic study plan from the user's goal. "
        "ALWAYS call the `get_weakness_summary` tool first to check memory and adapt the plan. "
        "Your final output MUST be a JSON list of topics (e.g., [\"Topic 1\", \"Topic 2\"])."
    )
    
    print("\n[PLANNER]: Strategizing initial plan...")
    plan_output = run_agent(
        system_prompt=PLANNER_PROMPT, 
        user_prompt=f"Create a plan for the goal: {initial_goal}",
        tools=MEMORY_TOOL # Now correctly defined in Section 2
    )
    
    try:
        topic_list = json.loads(plan_output.split('[')[-1].split(']')[0] + ']') # Crude JSON parsing
    except:
        # Fallback if the agent doesn't return perfect JSON
        topic_list = ["Python Variables and Data Types", "Python Control Flow and Loops"]
        print(f"Warning: Failed to parse plan JSON. Using default topics: {topic_list}")

    # ----------------------------------------------------
    # MAIN LEARNING LOOP (Loop Agent)
    # ----------------------------------------------------
    for i, topic in enumerate(topic_list):
        print(f"\n\n--- ğŸ“š SESSION {i+1}: {topic} ---")
        
        # AGENT 2: TEACHER (Teaching and Tool Use Simulation)
        TEACHER_PROMPT = (
            "You are the **Teacher Agent**. Provide a clear, simplified, and concise explanation for the topic. "
            "Integrate a current, real-world example. Use your knowledge base as a proxy for the Google Search tool."
        )
        print("[TEACHER]: Providing lesson...")
        lesson = run_agent(
            system_prompt=TEACHER_PROMPT, 
            user_prompt=f"Explain the topic: {topic}",
            tools=None # Model knowledge is the "Search Tool" simulation
        )
        print(lesson)

        # AGENT 3: EVALUATOR (Quiz and Grading Tool Use)
        EVALUATOR_PROMPT = (
            "You are the **Evaluator Agent**. Your task is to first generate a 5-question quiz (MCQ/Short Answer) for the topic. "
            "Then, ask the user for their answers. Finally, you MUST call the `grade_quiz` tool with the topic and the user's answers."
        )
        
        quiz_topic = topic
        print("\n[EVALUATOR]: Generating Quiz...")
        
        # Step 1: Generate Quiz (Model response)
        quiz_text = run_agent(
            system_prompt=EVALUATOR_PROMPT, 
            user_prompt=f"Generate a 5-question quiz for: {quiz_topic}. Do not grade yet, just output the quiz.",
            tools=None # No tools needed for generation
        )
        print(quiz_text)

        # Step 2: Get User Input (Simulated)
        user_input_quiz = input("\n[USER INPUT]: Please provide a simple answer covering: For loops, and defining a function (Example: I used a for loop and defined a function with 'def').: ")
        
        # Step 3: Grade and Save Report (Model uses tool)
        print("\n[EVALUATOR]: Grading answers and saving report to Memory...")
        
        grading_prompt = f"The topic was '{quiz_topic}'. The user provided these answers: '{user_input_quiz}'. Now, call the `grade_quiz` function with the required parameters to grade and save the report."

        report_output = run_agent(
            system_prompt=EVALUATOR_PROMPT, 
            user_prompt=grading_prompt,
            tools=GRADING_TOOL # Now correctly defined in Section 2
        )
        
        print(f"Evaluator Final Report (Report Saved to LTM): {report_output}")
        
        # Optional: Planner re-adaptation (Loop feedback)
        print("\n[PLANNER]: Re-evaluating strategy based on the new report...")
        new_plan_check = run_agent(
            system_prompt=PLANNER_PROMPT, 
            user_prompt=f"Review the latest report: {report_output}. If the score is low, suggest a remedial topic for the NEXT session. If good, suggest the next step in the original plan. Output only the next topic name.",
            tools=MEMORY_TOOL # Now correctly defined in Section 2
        )
        print(f"Planner's Next Step Suggestion: {new_plan_check}")
        
        # Stop condition for a demo
        if i >= 1: break # Limit the loop for the notebook demo

# %% [code]
# 5. Demonstration and Evaluation (The Run)

initial_goal = "Learn the basics of Python programming: Variables, Loops, and Functions."
adaptive_study_workflow(initial_goal)

# Final output and observability demonstration
print("\n--- âœ… Final Long-Term Memory Snapshot ---")
# Display the memory bank contents to show persistence and history
print(json.dumps(memory_bank.data, indent=2))

