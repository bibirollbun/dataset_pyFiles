import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.runners import InMemoryRunner


print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


# Theory Retrieval Agent: Searches the .
theory_retrieval_agent = LlmAgent(
      name="TheoryRetrievalAgent",
      model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
      ),
    instruction="""You are a high school tutor.
Task:
- Read the provided educational content about a topic.
- Write a concise explanation in simple language suitable for high school students.
- Include 1â€“2 clear examples illustrating the concept.
- Keep the explanation under 180 words.
- Use short sentences and plain language.
- Do not include unnecessary details, long derivations, or unrelated content.""",
    tools=[google_search], #TODO: limit search only in specific educational sites
    output_key="theory_retrieved_draft", # The result will be stored with this key.
)

print("âœ… theory retrieval agent created.")


bias_cleaner_agent = LlmAgent(
    name="BiasCleanerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
      ),
    instruction="""
You are an expert in educational content and bias mitigation.

Task:
- Edit: {theory_retrieved_draft}
- Analyze the given learning material for **potential biases**.
- Focus on **10 common bias categories**: 
  1. Gender
  2. Cultural
  3. Socioeconomic
  4. Ability / Disability
  5. Idioms / Language
  6. Age
  7. Racial / Ethnic
  8. Regional / Geographic
  9. Religion / Belief
  10. Stereotypes / Role expectations

Instructions:
1. Identify any biased wording, phrasing, or examples.
2. Rewrite the original content into a **bias-free version**, keeping the educational meaning intact.
3. Keep the output **concise and clear**, suitable for high school students.
4. Output:
    "cleaned_content": str
""",
    #In this stage we could potentially track the top 10 detected biases with a short explanation
    #and make a bias report, which could contribute to a statistical analysis study regarding 
    #the biases that can be found in learning material.
    output_key="cleaned_content"
)

print("âœ… bias cleaner agent created.")


theory_agent = SequentialAgent(
    name="TheoryPipeline",
    sub_agents=[theory_retrieval_agent, bias_cleaner_agent],
)

print("âœ… Theory Sequential Agent created.")


runner = InMemoryRunner(agent=theory_agent)
user_topic = "Quadratic Functions"

events = await runner.run_debug(f"Provide a short theory with examples for the topic: {user_topic}")

# Extract text - the response is likely in the model_response attribute
theory_text = ""
for event in events:
    if hasattr(event, 'model_response') and event.model_response:
        for content_block in event.model_response.content:
            if hasattr(content_block, 'text'):
                theory_text += content_block.text

print(f"--- Theory for {user_topic} ---\n")
print(theory_text)


EXERCISE_DATABASE = {
    "quadratic_equations": {
        "easy": [
            {
                "id": "quad_easy_1",
                "topic": "quadratic_equations",
                "difficulty": "easy",
                "problem": "Solve: xÂ² - 5x + 6 = 0",
                "steps": [
                    {"id": 1, "description": "Identify a, b, c coefficients", "expected_time": 30},
                    {"id": 2, "description": "Factor the quadratic", "expected_time": 60},
                    {"id": 3, "description": "Set each factor to zero", "expected_time": 30},
                    {"id": 4, "description": "Solve for x", "expected_time": 30}
                ],
                "solution": "x = 2 or x = 3"
            }
        ],
        "medium": [
            {
                "id": "quad_med_1",
                "topic": "quadratic_equations",
                "difficulty": "medium",
                "problem": "Solve using quadratic formula: 2xÂ² + 7x + 3 = 0",
                "steps": [
                    {"id": 1, "description": "Identify a, b, c coefficients", "expected_time": 30},
                    {"id": 2, "description": "Apply quadratic formula", "expected_time": 90},
                    {"id": 3, "description": "Simplify the discriminant", "expected_time": 60},
                    {"id": 4, "description": "Calculate both solutions", "expected_time": 60}
                ],
                "solution": "x = -0.5 or x = -3"
            }
        ]
    },
    "linear_equations": {
        "easy": [
            {
                "id": "lin_easy_1",
                "topic": "linear_equations",
                "difficulty": "easy",
                "problem": "Solve: 2x + 5 = 13",
                "steps": [
                    {"id": 1, "description": "Subtract 5 from both sides", "expected_time": 30},
                    {"id": 2, "description": "Divide both sides by 2", "expected_time": 30},
                    {"id": 3, "description": "Write final answer", "expected_time": 20}
                ],
                "solution": "x = 4"
            }
        ]
    },
    "fractions": {
        "easy": [
            {
                "id": "frac_easy_1",
                "topic": "fractions",
                "difficulty": "easy",
                "problem": "Simplify: 1/2 + 1/3",
                "steps": [
                    {"id": 1, "description": "Find common denominator", "expected_time": 40},
                    {"id": 2, "description": "Convert fractions to common denominator", "expected_time": 50},
                    {"id": 3, "description": "Add numerators", "expected_time": 30},
                    {"id": 4, "description": "Simplify if needed", "expected_time": 30}
                ],
                "solution": "5/6"
            }
        ]
    }
}


class DifficultyCalculator:
    """Calculate difficulty score based on student performance metrics"""
    
    @staticmethod
    def calculate_difficulty(
        time_spent: float,
        expected_time: float,
        attempts: int,
        hints_requested: int,
        is_correct: bool
    ) -> float:
        """
        Calculate difficulty score (0-100) based on student performance.
        
        Formula:
        T = min(t / t_ref, 3)        # Time ratio (capped at 3x)
        A = min(a / 3, 2)            # Attempts ratio (capped at 2x)
        H = min(h / 4, 2)            # Hints ratio (capped at 2x)
        C = 1 - c                    # Correctness (0 if solved, 1 if not)
        
        D_raw = 0.45*T + 0.25*A + 0.20*H + 0.10*C
        D = round((D_raw / max_possible) * 100)
        """
        # Normalized components (clamped to sensible ranges)
        T = min(time_spent / expected_time if expected_time > 0 else 1, 3.0)
        A = min(attempts / 3.0, 2.0)
        H = min(hints_requested / 4.0, 2.0)
        C = 0.0 if is_correct else 1.0
        
        # Weighted difficulty score
        weights = {'T': 0.45, 'A': 0.25, 'H': 0.20, 'C': 0.10}
        D_raw = (weights['T'] * T + weights['A'] * A + 
                 weights['H'] * H + weights['C'] * C)
        
        # Normalize to 0-100 scale
        max_possible = (weights['T'] * 3 + weights['A'] * 2 + 
                       weights['H'] * 2 + weights['C'] * 1)
        D = round((D_raw / max_possible) * 100)
        
        return min(D, 100)

# Test the difficulty calculator
print("\nğŸ§® Testing Difficulty Calculator:")
print("-" * 60)

test_cases = [
    {"name": "Easy step", "time": 30, "expected": 30, "attempts": 1, "hints": 0, "correct": True},
    {"name": "Moderate difficulty", "time": 90, "expected": 60, "attempts": 2, "hints": 1, "correct": True},
    {"name": "High difficulty", "time": 180, "expected": 60, "attempts": 4, "hints": 3, "correct": True},
    {"name": "Failed step", "time": 120, "expected": 60, "attempts": 5, "hints": 4, "correct": False},
]

for test in test_cases:
    score = DifficultyCalculator.calculate_difficulty(
        test["time"], test["expected"], test["attempts"], 
        test["hints"], test["correct"]
    )
    print(f"{test['name']:20s}: Difficulty Score = {score}/100")


# Student Performance Tracker

class StepPerformance:
    """Track performance on a single step"""
    def __init__(self, step_id: int, description: str, expected_time: float):
        self.step_id = step_id
        self.description = description
        self.expected_time = expected_time
        self.time_spent = 0.0
        self.attempts = 0
        self.hints_requested = 0
        self.is_correct = False
        self.difficulty_score = 0.0
        self.start_time = None
    
    def start(self):
        """Start timing this step"""
        self.start_time = datetime.now()
    
    def end(self, is_correct: bool):
        """End timing and calculate difficulty"""
        if self.start_time:
            self.time_spent = (datetime.now() - self.start_time).total_seconds()
        self.is_correct = is_correct
        self.difficulty_score = DifficultyCalculator.calculate_difficulty(
            self.time_spent,
            self.expected_time,
            self.attempts,
            self.hints_requested,
            self.is_correct
        )
    
    def add_attempt(self):
        """Increment attempt counter"""
        self.attempts += 1
    
    def add_hint(self):
        """Increment hint counter"""
        self.hints_requested += 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting"""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "time_spent_seconds": round(self.time_spent, 1),
            "expected_time_seconds": self.expected_time,
            "attempts": self.attempts,
            "hints_requested": self.hints_requested,
            "is_correct": self.is_correct,
            "difficulty_score": self.difficulty_score
        }

print("âœ… Performance tracking system ready!")


# We need a global/class-level variable to store the state,
# as the functions are standalone tools.
from typing import List, Dict, Optional

# Global State Management 
global_step_performances: List[StepPerformance] = []
global_current_exercise: Optional[Dict] = None

# Helper function to reset the state for a new session
def reset_tutor_state():
    """Resets all global state variables for a new tutoring session."""
    global global_step_performances
    global global_current_exercise
    global_step_performances = []
    global_current_exercise = None
    return "Tutor state has been reset."


# Helper function to reset the state for a new session
def reset_tutor_state():
    """Resets all global state variables for a new tutoring session."""
    global global_step_performances
    global global_current_exercise
    global_step_performances = []
    global_current_exercise = None
    return "Tutor state has been reset."



def select_exercise_tool(topic: str, difficulty: str) -> str:
    """
    Selects randomly an exercise from the database based on topic and difficulty.
    Returns the selected exercise details as a JSON string.
    """
    global global_current_exercise
    import random
    
    available_exercises = EXERCISE_DATABASE.get(topic, {}).get(difficulty, [])
    
    if not available_exercises:
        return json.dumps({"error": f"No exercises found for {topic} at {difficulty} level"})
    
    # Randomly select an exercise and update global state
    global_current_exercise = random.choice(available_exercises)
    
    # The agent's prompt requires this specific JSON output
    return json.dumps({
        "selected_exercise_id": global_current_exercise['id'],
        "problem": global_current_exercise['problem'],
        "topic": global_current_exercise['topic'],
        "difficulty": global_current_exercise['difficulty'],
        "num_steps": len(global_current_exercise['steps'])
    })


def record_step_attempt_tool(step_index: int, is_correct: bool, requested_hint: bool = False):
    """
    Records a student's attempt on a specific step, updates time spent, attempts, 
    and calculates the difficulty score if the step is completed (is_correct=True).
    """
    global global_step_performances
    global global_current_exercise

    if not global_current_exercise:
        return "Error: No exercise is currently loaded."

    if step_index < len(global_current_exercise['steps']):
        # If starting a new step, initialize performance tracking
        if len(global_step_performances) <= step_index:
            step = global_current_exercise['steps'][step_index]
            perf = StepPerformance(
                step['id'],
                step['description'],
                step['expected_time']
            )
            perf.start()
            global_step_performances.append(perf)
        
        perf = global_step_performances[step_index]
        perf.add_attempt()
        if requested_hint:
            perf.add_hint()
        if is_correct:
            perf.end(True)
            # Return the calculated score for the orchestrator to see
            return json.dumps({
                "status": "Step Completed",
                "step_index": step_index,
                "difficulty_score": perf.difficulty_score
            })
        else:
             return "Attempt recorded. Step is not yet correct."
    return f"Error: Invalid step index {step_index} or no exercise loaded."


def get_performance_data_tool() -> str:
    """
    Returns the full, current performance data for all steps as a JSON string.
    """
    global global_step_performances
    if not global_step_performances:
        return json.dumps({"error": "No performance data recorded yet."})
    performance_data = [perf.to_dict() for perf in global_step_performances]
    return json.dumps(performance_data)


async def run_sub_agent_tool(agent_name: str, prompt: str) -> str:
    """
    Runs a specified sub-agent (Step Guide, Difficulty Analyzer, etc.) 
    with a specific prompt and returns its output text.
    """
    # Map agent name to the actual Agent object
    agent_map = {
        "step_guide": step_guide_agent,
        "difficulty_analyzer": difficulty_analyzer_agent,
        "exercise_search": exercise_search_agent,
        "teacher_report": teacher_report_agent
    }

    if agent_name not in agent_map:
        return f"Error: Unknown agent name '{agent_name}'. Must be one of: {list(agent_map.keys())}"

    runner = InMemoryRunner(agent=agent_map[agent_name])
    events = await runner.run(prompt)
    
    output_text = ""
    for event in events:
        if hasattr(event, 'model_response') and event.model_response:
            for content_block in event.model_response.content:
                if hasattr(content_block, 'text'):
                    output_text += content_block.text
    
    return output_text

print("âœ… Standalone tool functions defined.")


step_guide_prompt = """You are a Step Guide Agent for an AI tutoring system.

Your task: Guide the student through ONE step at a time in solving a problem.
Use: 
You will receive:
1. The current step description
2. The student's previous response (if any)
3. Context about the problem

Provide clear, encouraging guidance for THIS STEP ONLY. Do not solve the entire problem.

If the student is struggling:
- Offer hints (be supportive and gradual)
- Break down the step into smaller parts
- Ask guiding questions

Output format:
- If student needs more guidance: Provide a hint or ask a question
- If student completed the step correctly: Confirm and move to next step
- If student is very stuck: Suggest requesting a detailed hint

Keep responses concise and focused on the current step."""

step_guide_agent = Agent(
    name="step_guide_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
      ),
    instruction=step_guide_prompt
)

print("âœ… Step Guide Agent created")


difficulty_analyzer_prompt = """You are a Difficulty Analyzer Agent for an AI tutoring system.

Your task: Analyze student performance data and identify steps where the student struggled most.

You will receive performance metrics for each step including:
- Time spent vs expected time
- Number of attempts
- Hints requested
- Correctness
- Calculated difficulty score (0-100)

Output ONLY a JSON object with:
{
  "most_difficult_step": {
    "step_id": int,
    "description": "step description",
    "difficulty_score": float,
    "issues": ["list of observed issues"]
  },
  "overall_performance": "brief summary",
  "recommendation": "what type of remedial exercise to search for"
}

Focus on actionable insights for the teacher."""

difficulty_analyzer_agent = Agent(
    name="difficulty_analyzer_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
      ),
    instruction=difficulty_analyzer_prompt
)

print("âœ… Difficulty Analyzer Agent created")



exercise_search_prompt = """You are an Exercise Search Agent for an AI tutoring system.

Your task: Search Google for simple, beginner-friendly practice exercises for a specific learning concept.

You will receive:
- The specific concept/step the student struggled with
- The difficulty level needed (always search for "simple" or "easy" exercises)

Use web_search to find:
1. Educational websites with practice problems
2. Step-by-step tutorials
3. Interactive exercises

Output ONLY a JSON object with:
{
  "search_results": [
    {
      "title": "resource title",
      "url": "resource url",
      "description": "brief description of why this is helpful"
    }
  ],
  "summary": "brief summary of available resources"
}

Prioritize high-quality educational sources (Khan Academy, Math is Fun, educational institutions)."""
#In order not to be confused, the initial exercise is retrieved by the mock database.
#This agent is called if the user encounters great difficulty during a step in the initial exercise.
#So the system suggests very simple exercises in order to overcome the difficulty to this specific step.
exercise_search_agent = Agent(
    name="exercise_search_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
      ),
    instruction=exercise_search_prompt,
    tools=[google_search]
)

print("âœ… Exercise Search Agent created (with web search capability)")



teacher_report_prompt = """You are a Teacher Report Generator Agent.

Your task: Create a comprehensive report for the teacher about the student's performance.

You will receive:
1. Student performance data for all steps
2. Analysis of the most difficult step
3. Recommended remedial exercises from web search

Generate a clear, structured report that includes:
- Overall performance summary
- Step-by-step breakdown with difficulty scores
- Highlighted problem areas
- Recommended resources for improvement

Output a well-formatted markdown report that is easy for teachers to read and act upon."""

teacher_report_agent = Agent(
    name="teacher_report_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
      ),
    instruction=teacher_report_prompt
)

print("âœ… Teacher Report Generator Agent created")


orchestrator_prompt = """You are the **AI Tutor Orchestrator Agent**, the primary interface for the tutoring system.

Your task is to manage the entire student session, from selecting an exercise to generating a final report. You must use the provided tools and sub-agents (via `run_sub_agent_tool`) to complete the workflow.

**Workflow:**
1. **Reset State:** Always start a new session by using `reset_tutor_state`.
2. **Select Exercise:** Use `select_exercise_tool` based on the student's request (e.g., topic, difficulty).
3. **Guide Steps:** For each step in the exercise:
    a. Use `run_sub_agent_tool` with the `step_guide` agent and the student's response to get guidance.
    b. **Crucially:** After the student provides an answer, you **must** use `record_step_attempt_tool` to log the performance (attempts, hints, correctness).
    c. Continue until all steps are marked as correct/completed.
4. **Analyze Difficulty:** Use `get_performance_data_tool` to get the metrics, then pass this data to the `difficulty_analyzer` agent via `run_sub_agent_tool`.
5. **Search Remedial:** Use the concept identified by the analyzer to search for resources using the `exercise_search` agent via `run_sub_agent_tool`.
6. **Generate Report:** Compile all data (raw performance, difficulty analysis, remedial search results) and pass it to the `teacher_report` agent via `run_sub_agent_tool`.

**Your final output should be the full Teacher Report generated by the Teacher Report Agent.**"""

# Instantiate the new Orchestrator Agent with all the tools
exercise_orchestrator_agent = Agent(
    name="exercise_orchestrator_agent",
    model="gemini-2.5-flash",  # Use a powerful, tool-use optimized model
    instruction=orchestrator_prompt,
    tools=[
        reset_tutor_state,
        select_exercise_tool,
        record_step_attempt_tool,
        get_performance_data_tool,
        run_sub_agent_tool  # This tool allows it to call all the other Agents
    ]
)

print("\nâœ… AITutorOrchestratorAgent created with all necessary tools.")


ai_tutor_agent = SequentialAgent(
    name="OverallPipeline",
    sub_agents=[theory_agent, exercise_orchestrator_agent],
)

print("âœ… AI Tutor Agent created.")

