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

# --- START OF FINANCEGUIDE AI CAPSTONE PROJECT CODE ---

import json
import random
from typing import Dict, Any, List

# --- SIMULATED GOOGLE AGENT DEVELOPMENT KIT (ADK) FRAMEWORK ---

class AgentTool:
    """Represents a callable function tool for the agents."""
    def __init__(self, name: str, description: str, func: callable):
        self.name = name
        self.description = description
        self.func = func

class Agent:
    """Simulated Agent class that mimics ADK functionality (Instructions, Tools, Memory, Run method)."""
    def __init__(self, name: str, model: str, instructions: str, tools: List[AgentTool] = None):
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools if tools is not None else []
        self.memory = {} # Persistent memory store
    
    def run(self, prompt: str, context: Dict[str, Any]) -> Any:
        """A public method to simulate running the agent's LLM logic."""
        return self._simulate_llm_response(prompt, context)

    def _simulate_llm_response(self, prompt: str, context: Dict[str, Any]) -> Any:
        """
        Simulates the LLM's response based on the agent's role and the input prompt.
        This function demonstrates the expected output for a real Gemini LLM call.
        """
        if self.name == "AssessmentAgent":
            # Simulate JSON Mode Output
            return {
                "user_level": "Beginner",
                "goal_focus": "Debt Reduction and Saving",
                "current_debt_amount": 15000,
                "monthly_income": 4000
            }
        
        elif self.name == "CurriculumAgent":
            if "update_plan" in context.get("action", ""):
                # Simulate adaptive plan update based on Evaluation Feedback
                feedback = context['feedback_data']
                if feedback.get("failed_concepts") == ["APR", "Simple vs Compound Interest"]:
                    return {
                        "plan_status": "Plan Adjusted",
                        "next_module": "Remedial Lesson: The Power of Compounding & Debt Cost",
                        "modules_remaining": 3
                    }
            # Simulate initial plan generation
            return {
                "plan_id": "FG-001-BEGINNER",
                "modules": [
                    {"topic": "Budgeting Fundamentals", "type": "budget_planning"},
                    {"topic": "Understanding Credit & APR", "type": "lesson"},
                    {"topic": "High-Interest Debt Strategy", "type": "lesson"},
                    {"topic": "Intro to Index Funds", "type": "simulation"}
                ]
            }

        elif self.name == "ExplainerAgent":
            level = context.get('user_level', 'Intermediate')
            if "budget_planning" in prompt:
                # Simulate FunctionTool Call for budget analysis
                if self.tools and self.tools[0].name == "BudgetAnalysisTool":
                    # Pass relevant data for the tool from the context
                    tool_output = self.tools[0].func(context.get('budget_data', 4000)) 
                    return (f"**[Explainer Agent]** Based on your {tool_output['focus']} analysis, which is crucial for a **{level}** learner:\n"
                            f"- **Insight**: Your spending is healthy, but we found a potential savings of **{tool_output['potential_savings']}** by reviewing dining expenses.\n"
                            f"- **Action**: Let's create a debt repayment schedule using this extra cash.")
            
            if "Intro to Index Funds" in prompt:
                return (f"**[Explainer Agent]** Welcome to Investing (Level: **{level}**).\n"
                        f"An **Index Fund** is like buying the whole basket of fruit (the market), not just one apple (a single stock).\n"
                        f"This strategy lowers your risk dramatically. Now, let's run a simulation.")
            
            if "Remedial Lesson" in prompt:
                 return f"**[Explainer Agent]** Welcome back to **{context['module_type']}**! We need to clarify debt cost."
                
        elif self.name == "EvaluatorAgent":
            if "simulation" in prompt:
                # Simulate FunctionTool Call for simulation
                if self.tools and self.tools[0].name == "SimulationTool":
                    # Parameters would normally come from the LLM's reasoning
                    tool_output = self.tools[0].func(1000, "VTSAX", 5)
                    return (f"**[Evaluator Agent - Simulation Result]** Investment Simulation Complete:\n"
                            f"- Initial: $1,000 | Ticker: {tool_output['symbol']}\n"
                            f"- Final Value (5 Yrs): **{tool_output['final_value']}**\n"
                            f"- Annualized Return: {tool_output['annual_return']}%")
            
            # Simulate Quiz Generation and Grading (Structured JSON Output for A2A feedback)
            if "Generate a quiz" in prompt:
                topic = prompt.split("'")[1]
                if "Understanding Credit & APR" in topic:
                     # Simulate a failure to trigger the adaptive loop
                    return {
                        "module_name": topic,
                        "quiz_score_percent": 40,
                        "feedback_summary": "User struggled with the difference between Simple and Compound Interest.",
                        "passed": False,
                        "failed_concepts": ["APR", "Simple vs Compound Interest"]
                    }
                else:
                    # Simulate passing other modules
                    score = random.choice([85, 90, 95])
                    return {
                        "module_name": topic,
                        "quiz_score_percent": score,
                        "feedback_summary": "Excellent grasp of core concepts.",
                        "passed": True,
                        "failed_concepts": []
                    }
        return f"Simulated response for: {prompt}"


# --- 2. CUSTOM FUNCTION TOOLS (Function Calling) ---

def run_investment_simulation(initial_capital: float, symbol: str, duration_years: int) -> Dict[str, Any]:
    """Simulates a 5-year investment and returns the final value and return rate."""
    # Simulation uses a fixed average annual return for demonstration
    annual_return_rate = 0.08  # 8% average
    final_value = initial_capital * ((1 + annual_return_rate) ** duration_years)
    
    return {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "annual_return": annual_return_rate * 100
    }

def analyze_user_budget(monthly_income: float) -> Dict[str, Any]:
    """Analyzes a simplified budget (simulated from income) and suggests savings."""
    # Simple 50/30/20 rule simulation: 50% Needs, 30% Wants, 20% Savings/Debt
    wants_budget = monthly_income * 0.30
    
    # Simulate a 10% potential optimization on 'Wants' spending
    potential_savings = wants_budget * 0.10
    
    return {
        "focus": "50/30/20 Budget Rule",
        "potential_savings": round(potential_savings, 2),
        "wants_spent": round(wants_budget, 2)
    }

# Register the tools
simulation_tool = AgentTool(
    name="SimulationTool",
    description="Runs investment scenarios using historical market data.",
    func=run_investment_simulation
)

budget_tool = AgentTool(
    name="BudgetAnalysisTool",
    description="Analyzes user's financial data to find optimization and savings opportunities.",
    func=analyze_user_budget
)


# --- 3. AGENT DEFINITIONS ---

assessment_agent = Agent(
    name="AssessmentAgent",
    model="gemini-2.0-flash",
    instructions="You are the initial intake specialist. Your sole job is to determine the user's current knowledge level (Beginner/Intermediate/Advanced) and primary goal (e.g., Debt Reduction, Retirement). Output the result in a strict JSON format."
)

curriculum_agent = Agent(
    name="CurriculumAgent",
    model="gemini-2.0-flash",
    instructions="You are the adaptive learning planner. You receive the user profile and evaluation feedback. Based on this, you design and adjust the sequential learning plan. Prioritize debt repayment modules for users with high debt. Use Task Decomposition to break goals into modules."
)

explainer_agent = Agent(
    name="ExplainerAgent",
    model="gemini-2.0-pro",
    instructions="You are the personalized teacher. Deliver engaging lessons. Crucially, adjust your complexity and use analogies based on the user_level stored in your memory. Use the BudgetAnalysisTool when the user enters the Budgeting module.",
    tools=[budget_tool]
)

evaluator_agent = Agent(
    name="EvaluatorAgent",
    model="gemini-2.0-flash",
    instructions="You generate 5-question quizzes for the current topic. You also execute investment simulations using the SimulationTool. After grading a quiz, output the result as a structured JSON object containing the 'passed' status and 'failed_concepts' for the Curriculum Agent.",
    tools=[simulation_tool]
)

# --- 4. ORCHESTRATOR WORKFLOW (The Main Execution Logic) ---

def run_financeguide_ai(user_starting_prompt: str) -> str:
    """
    The Orchestrator function, coordinating the A2A flow.
    """
    orchestrator_output = ["\n--- ğŸš€ FinanceGuide AI Orchestrator: Starting Session ---"]
    
    # --- 1. ASSESSMENT PHASE ---
    orchestrator_output.append(f"\n[Orchestrator] Step 1: Delegating initial assessment to **{assessment_agent.name}**...")
    user_profile = assessment_agent.run(prompt=f"User: {user_starting_prompt}", context={})
    
    # Store key info in shared memory/context (simulated here)
    session_context = {"user_profile": user_profile}
    orchestrator_output.append(f"[Assessment Agent] Result: User Level: {user_profile['user_level']}, Goal: {user_profile['goal_focus']}")

    # --- 2. INITIAL PLANNING PHASE ---
    orchestrator_output.append(f"\n[Orchestrator] Step 2: Delegating curriculum planning to **{curriculum_agent.name}**...")
    curriculum_plan = curriculum_agent.run(prompt="Generate a 4-module curriculum plan based on the assessment.", context=session_context)
    
    orchestrator_output.append(f"[Curriculum Agent] Initial Plan Generated: {len(curriculum_plan['modules'])} modules.")
    
    # Use a list to hold the modules and allow dynamic insertion
    modules_to_run = list(curriculum_plan['modules'])

    # --- 3. EXECUTION LOOP (Teacher, Tool Usage, Evaluation, Feedback Loop) ---
    orchestrator_output.append("\n[Orchestrator] Step 3: Starting Adaptive Learning Execution Loop...")

    i = 0
    while i < len(modules_to_run):
        module = modules_to_run[i]
        topic = module['topic']
        m_type = module['type']
        
        orchestrator_output.append(f"\n[Orchestrator] Sub-Step: Starting Module **'{topic}'** ({m_type})...")
        
        # A. Explainer Agent Teaches/Guides
        teaching_prompt = f"Teach the module '{topic}'. Use the level: {user_profile['user_level']}"
        teaching_context = {"user_level": user_profile['user_level'], "budget_data": user_profile['monthly_income'], "module_type": m_type}

        explainer_response = explainer_agent.run(prompt=teaching_prompt, context=teaching_context)
        orchestrator_output.append(explainer_response)

        # B. Evaluator Agent Executes Quiz/Simulation
        if m_type == 'simulation':
            orchestrator_output.append(f"[Orchestrator] Simulation Time: Delegating to **{evaluator_agent.name}**.")
            simulation_result = evaluator_agent.run(prompt="Run a 5-year index fund simulation (VTSAX $1000).", context={})
            orchestrator_output.append(simulation_result)
        
        # Only quiz on knowledge modules (lessons and remedial)
        if m_type == 'lesson' or m_type == 'remedial': 
            orchestrator_output.append(f"[Orchestrator] Quiz Time: Delegating to **{evaluator_agent.name}** for quiz generation.")
            evaluation_result = evaluator_agent.run(prompt=f"Generate a quiz on '{topic}' and grade the user's (simulated) answers.", context={})
            
            orchestrator_output.append(f"[Evaluator Agent] Quiz Score: {evaluation_result['quiz_score_percent']}%. Passed: {evaluation_result['passed']}")
            
            # C. Feedback Loop to Curriculum Agent
            if not evaluation_result['passed']:
                orchestrator_output.append(f"[Orchestrator] **Feedback Loop Triggered!** User failed. Notifying Curriculum Agent to adjust plan...")
                
                update_context = {
                    "action": "update_plan",
                    "feedback_data": evaluation_result
                }
                
                curriculum_update = curriculum_agent.run(prompt="User failed a quiz. Inject a remedial lesson now.", context=update_context)
                orchestrator_output.append(f"[Curriculum Agent] **Plan Update**: {curriculum_update['plan_status']} - Next module is now: **{curriculum_update['next_module']}**.")
                
                # Insert the remedial lesson immediately after the current failed module
                remedial_module = {"topic": curriculum_update['next_module'], "type": "remedial"}
                modules_to_run.insert(i + 1, remedial_module)
                # Note: We do *not* increment i here, as we want the loop to immediately process the newly inserted module next.
                
        i += 1 # Move to the next module in the (potentially adjusted) list
                
    orchestrator_output.append("\n--- ğŸ�� FinanceGuide AI Orchestrator: Session Complete ---")
    
    return "\n".join(orchestrator_output)

# --- 5. EXECUTION START ---
user_query = "I want to start learning how to get out of debt and eventually save for retirement."
full_result = run_financeguide_ai(user_query)

# --- 6. FINAL RESULT GENERATION ---
print(full_result)

# --- END OF FINANCEGUIDE AI CAPSTONE PROJECT CODE ---

