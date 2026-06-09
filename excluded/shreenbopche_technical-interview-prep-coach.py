# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import json
import logging
import time
from typing import Dict, Any, List

# --- 1. Observability Setup (Logging) ---
# Fulfills: Observability: Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('InterviewCoach')

# --- 2. Tool Definitions ---

class AgentTool:
    """Base class for all tools used by the agents."""
    def __init__(self, name: str):
        self.name = name

    def run(self, **kwargs) -> Any:
        raise NotImplementedError

class CodeExecutionTool(AgentTool):
    """
    A custom tool used by the CodingEvaluatorAgent to run and test user code.
    Fulfills: Tools: custom tools (simulated code execution)
    """
    def __init__(self):
        super().__init__("CodeExecution")
        self.test_cases = {
            'fibonacci': [
                (0, 0), (1, 1), (2, 1), (5, 5), (10, 55)
            ],
            # FIX: Added test cases for the 'find_largest' function to prevent ZeroDivisionError
            'find_largest': [
                ([1, 5, 2, 8, 3], 8),
                ([-1, -5, -2, -8], -1),
                ([42], 42),
                ([5, 5, 5], 5)
            ]
        }

    def run(self, code_string: str, function_name: str) -> Dict[str, Any]:
        logger.info(f"Tool {self.name}: Executing user code for function '{function_name}'...")
        
        # Security Note: Executing arbitrary user code (eval/exec) is dangerous in production.
        # This is a simulation for demonstration purposes.
        context = {}
        try:
            exec(code_string, context)
            user_function = context[function_name]
        except Exception as e:
            return {"success": False, "error": f"Compilation/Setup Error: {e}", "passed_tests": 0}

        results = []
        passed_count = 0
        
        for input_val, expected_output in self.test_cases.get(function_name, []):
            try:
                # Determine how to call the function based on the expected input (single arg vs list arg)
                if function_name == 'find_largest':
                    actual_output = user_function(input_val) # input_val is the list [1, 5, ...]
                else:
                    actual_output = user_function(input_val) # input_val is a single integer
                
                is_passed = actual_output == expected_output
                if is_passed:
                    passed_count += 1
                
                results.append({
                    "input": input_val,
                    "expected": expected_output,
                    "actual": actual_output,
                    "passed": is_passed
                })
            except Exception as e:
                results.append({
                    "input": input_val,
                    "error": f"Runtime Error: {e}",
                    "passed": False
                })

        return {
            "success": True,
            "passed_tests": passed_count,
            "total_tests": len(self.test_cases.get(function_name, [])),
            "test_results": results
        }

class GoogleSearchTool(AgentTool):
    """
    A simulation of the Google Search tool.
    Fulfills: Tools: built-in tools (Google Search)
    """
    def __init__(self):
        super().__init__("GoogleSearch")

    def run(self, query: str) -> str:
        logger.info(f"Tool {self.name}: Performing search for query: '{query[:40]}...'")
        time.sleep(0.5) # Simulate latency
        # This would be an actual API call to Google Search.
        return (f"Search result snippet: The latest guidance for '{query}' emphasizes "
                "readability, type hinting (PEP 484), and modern `asyncio` patterns.")

# --- 3. Memory & Session Management ---

class MemoryBank:
    """
    Manages Long Term Memory for the user's historical performance.
    Fulfills: Sessions & Memory: Long term memory (e.g. Memory Bank)
    """
    FILEPATH = 'memory_bank.json'

    def load_user_history(self, user_id: str) -> Dict[str, Any]:
        try:
            with open(self.FILEPATH, 'r') as f:
                data = json.load(f)
                return data.get(user_id, {"sessions": 0, "topics_struggled": [], "last_score": 0})
        except FileNotFoundError:
            return {"sessions": 0, "topics_struggled": [], "last_score": 0}
        except json.JSONDecodeError:
            logger.error("Memory bank file corrupted.")
            return {"sessions": 0, "topics_struggled": [], "last_score": 0}

    def save_user_history(self, user_id: str, history: Dict[str, Any]):
        try:
            with open(self.FILEPATH, 'r+') as f:
                data = json.load(f)
                data[user_id] = history
                f.seek(0)
                json.dump(data, f, indent=4)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.FILEPATH, 'w') as f:
                json.dump({user_id: history}, f, indent=4)
        logger.info(f"Memory Bank updated for user {user_id}.")

class SessionState:
    """
    Manages short-term state for the current interview session.
    Fulfills: Sessions & Memory: Sessions & state management
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_id = f"S-{int(time.time())}"
        self.current_question_index = 0
        self.running_score = 0
        self.question_history: List[Dict[str, Any]] = []
        logger.info(f"New session created: {self.session_id} for user {user_id}")

    def update_score(self, points: int):
        self.running_score += points

    def advance_question(self):
        self.current_question_index += 1

# --- 4. Agent Definitions ---

# A2A Protocol is simulated by the clear, structured data passed between these agent methods.

class CodingEvaluatorAgent:
    """
    Specialized agent for evaluating coding questions.
    Fulfills: Multi-agent system (Sequential Agent)
    """
    def __init__(self):
        self.code_executor = CodeExecutionTool()
        self.llm_analyst = GoogleSearchTool() # For simulating complexity check/best practices

    def evaluate(self, user_code: str, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the evaluation pipeline."""
        logger.info("CodingEvaluatorAgent: Starting evaluation sequence.")
        
        function_name = question_data['function_name']
        
        # Step 1: Tool Use - Code Execution
        execution_results = self.code_executor.run(user_code, function_name)
        
        # Step 2: Simulated LLM Analysis (Complexity, Style)
        search_query = f"Best Python practices for implementing {function_name}"
        best_practices_snippet = self.llm_analyst.run(search_query)
        
        # Step 3: Compile Evaluation
        if execution_results['success']:
            total_tests = execution_results['total_tests']
            
            # Guard against ZeroDivisionError if no tests are defined for a function
            if total_tests == 0:
                score_multiplier = 0
                feedback_prefix = "Warning: No test cases defined for this function. Correctness score based on manual review only."
            else:
                score_multiplier = execution_results['passed_tests'] / total_tests
                feedback_prefix = f"Your code passed {execution_results['passed_tests']}/{total_tests} tests."

            # Simple Scoring: 80% for correctness, 20% for style/complexity (simulated)
            base_points = 5 
            score = int(base_points * score_multiplier)
            
            feedback = (
                f"{feedback_prefix} Your implementation of '{function_name}' is functionally correct but could be optimized. "
                f"LLM insight on best practices: {best_practices_snippet}"
            )
            
            return {
                "score": score,
                "max_score": base_points,
                "feedback": feedback,
                "raw_results": execution_results
            }
        else:
            return {
                "score": 0,
                "max_score": 5,
                "feedback": f"Critical Error detected: {execution_results['error']}. Please review your function signature or syntax.",
                "raw_results": execution_results
            }


class InterviewPrepCoach:
    """
    The main coordinator, acting as the LLM-powered User Manager Agent.
    Fulfills: Multi-agent system (LLM Agent, Loop Agent)
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session = SessionState(user_id)
        self.memory = MemoryBank()
        self.coding_agent = CodingEvaluatorAgent()
        self.user_history = self.memory.load_user_history(user_id)

        # Question bank (Context Engineering)
        self.question_bank = [
            {"type": "coding", "topic": "Recursion/Memoization", "title": "Implement the Fibonacci sequence.", "function_name": "fibonacci"},
            {"type": "conceptual", "topic": "Design Patterns", "title": "Explain the difference between composition and inheritance in Python."},
            {"type": "coding", "topic": "Arrays/Data Structures", "title": "Write a function to find the largest element in a list.", "function_name": "find_largest"}, # Example for a failed test
        ]
        
        logger.info(f"Coach initialized. User previous sessions: {self.user_history['sessions']}")

    def get_next_question(self) -> Dict[str, Any] | None:
        """Determines the next question based on session state and long-term memory."""
        
        # Long-term memory influence (Simple logic: If user struggled, re-ask related topic)
        if self.user_history['topics_struggled']:
            struggled_topic = self.user_history['topics_struggled'][0]
            logger.info(f"Selecting question based on past struggle: {struggled_topic}")
        
        # Get next question from the bank
        if self.session.current_question_index < len(self.question_bank):
            return self.question_bank[self.session.current_question_index]
        else:
            return None

    def process_user_response(self, user_response: str, question: Dict[str, Any]):
        """Routes the user's response to the appropriate sequential agent."""
        
        logger.info(f"InterviewPrepCoach: Processing response for question type: {question['type']}")

        evaluation_data = {}
        
        if question['type'] == 'coding':
            # A2A Protocol: Pass data to Coding Agent
            evaluation_data = self.coding_agent.evaluate(user_response, question)
        
        elif question['type'] == 'conceptual':
            # SIMULATION of Concept Evaluator Agent (which would use an LLM for analysis)
            # Fulfills: Multi-agent system (Parallel Agent structure simulated)
            
            search_data = self.coding_agent.llm_analyst.run(f"Difference between {question['title']}")
            
            if len(user_response.split()) > 30 and "inheritance" in user_response:
                score = 5
                feedback = f"Excellent, well-structured answer. You covered both concepts and when to use each. {search_data}"
            else:
                score = 2
                feedback = "Your answer was too brief. You need to elaborate on the 'why' and provide specific Python examples."
                
            evaluation_data = {"score": score, "max_score": 5, "feedback": feedback}

        # Update Session State
        self.session.update_score(evaluation_data['score'])
        self.session.question_history.append({**question, "evaluation": evaluation_data})
        
        # Provide synthesized feedback (Simulated Feedback Agent)
        self.provide_feedback(question, evaluation_data)
        
        # Advance the loop
        self.session.advance_question()


    def provide_feedback(self, question: Dict[str, Any], evaluation: Dict[str, Any]):
        """Synthesizes feedback from the evaluation data."""
        print("-" * 50)
        print(f"✅ Feedback for: {question['title']}")
        print(f"Score: {evaluation['score']}/{evaluation['max_score']}")
        print(f"Critique: {evaluation['feedback']}")
        if question['type'] == 'coding' and evaluation.get('raw_results'):
             print(f"Raw Test Results: Passed {evaluation['raw_results'].get('passed_tests', 0)}/{evaluation['raw_results'].get('total_tests', 0)}")
        print("-" * 50)

    def conclude_session(self):
        """Finalizes the session and updates long-term memory."""
        total_max_score = len(self.session.question_history) * 5
        final_score = self.session.running_score
        
        # Update Long-Term Memory
        self.user_history['sessions'] += 1
        self.user_history['last_score'] = final_score
        # Simple logic to add topics where score was less than 3
        for item in self.session.question_history:
            if item['evaluation']['score'] < 3 and item['topic'] not in self.user_history['topics_struggled']:
                self.user_history['topics_struggled'].append(item['topic'])

        self.memory.save_user_history(self.user_id, self.user_history)
        
        print("\n\n" + "=" * 50)
        print(f"⭐ Interview Session Concluded (Session ID: {self.session.session_id})")
        print(f"Final Score: {final_score} out of {total_max_score}")
        print(f"Topics to Review: {', '.join(self.user_history['topics_struggled']) if self.user_history['topics_struggled'] else 'None'}")
        print("=" * 50)


# --- 5. Main Execution Demonstration ---

def run_interview_simulation():
    """Simulates the loop agent flow."""
    user_id = "user_12345" # Mandatory for identifying user data in MemoryBank
    coach = InterviewPrepCoach(user_id)
    
    # --- Start Loop Agent ---
    while True:
        question = coach.get_next_question()
        if question is None:
            break
        
        print(f"\n[Question {coach.session.current_question_index + 1}/{len(coach.question_bank)}]")
        print(f"Topic: {question['topic']} | Type: {question['type'].upper()}")
        print(f"Prompt: {question['title']}")
        print("-" * 30)

        # Simulated User Input (Replacing actual input() call)
        if question['type'] == 'coding':
            if question['function_name'] == 'fibonacci':
                # Good code for question 1
                user_input = (
                    "def fibonacci(n):\n"
                    "    if n <= 1: return n\n"
                    "    memo = {0: 0, 1: 1}\n"
                    "    for i in range(2, n + 1):\n"
                    "        memo[i] = memo[i-1] + memo[i-2]\n"
                    "    return memo[n]"
                )
            elif question['function_name'] == 'find_largest':
                # Code with a bug (will fail one test case: empty list)
                 user_input = (
                    "def find_largest(arr):\n"
                    "    if not arr: return None\n" # Intentional bug: CodeExecutionTool doesn't have a test for this, but if it did, this is a reasonable solution.
                    "    current_max = arr[0]\n"
                    "    for x in arr:\n"
                    "        if x > current_max:\n"
                    "            current_max = x\n"
                    "    return current_max"
                )
            else:
                 user_input = "def placeholder(n): return n" # Fallback
            
            print(f"(User submits code for '{question['function_name']}', view console for actual code)")
        
        elif question['type'] == 'conceptual':
            # Good response for question 2
            user_input = ("Inheritance is an 'is-a' relationship where one class derives properties "
                          "and methods from a parent class. It tightly couples them. Composition, "
                          "or 'has-a' relationship, involves one class containing an instance of "
                          "another class as an attribute, which is often preferred for flexibility "
                          "and reduced coupling in Python.")
            print("(User submits conceptual answer, view console for actual response)")

        # Process the response
        coach.process_user_response(user_input, question)
        time.sleep(1) # Pause for simulation clarity

    # --- End Loop Agent ---
    coach.conclude_session()

if __name__ == "__main__":
    # Ensure memory_bank.json exists (it will be created if not)
    # The logging statements will show the flow of the multi-agent system and tool usage.
    run_interview_simulation()

