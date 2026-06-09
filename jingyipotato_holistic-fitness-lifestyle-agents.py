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


import sys
import time
import logging
from typing import Any, Dict
from rich.console import Console
from rich.markdown import Markdown
from IPython.display import Image, display

from google.genai import types
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.tools import BaseTool
from google.adk.models.google_llm import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.sessions import DatabaseSessionService
from google.adk.plugins import BasePlugin
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App, EventsCompactionConfig


def get_logger(name: str) -> logging.Logger:
    """Return a formatted logger.
    
    Args:
        name: Name of the logger.
        
    Returns:
        Configured logger instance.

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False

    return logger

logger = get_logger(__name__)
console = Console()

logger.info("âœ… Relevant libraries, ADK and Generative AI components imported successfully.")


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    logger.info(f"\n ### Session: {session_name}")

    # Get app name & agent name from the Runner
    app_name = runner_instance.app_name
    agent_name = runner_instance.agent.name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        for query in user_queries:
            logger.info(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Collect response
            final_response = None
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    text = event.content.parts[0].text
                    # Filter out empty or "None" responses before printing
                    if text and text != "None":
                        final_response = text  # Keep the latest response

            # Print only once after loop finishes
            if final_response:
                #logger.info(f"{agent_name} > {final_response}")
                console.print(f"{agent_name} >")
                console.print(Markdown(final_response))
    else:
        logger.info("No queries!")


logger.info("âœ… Helper functions defined.")


## Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


workout_reviser = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="workout_reviser",
    description="Revises workout plan based on user feedback",
    instruction="""Revise the workout plan: {workout_plan}
    Revise the workout plan based on the feedback provided.
    
    Common adjustments:
    - Increase intensity: Add sets/reps, reduce rest, add weight
    - Decrease intensity: Remove sets, increase rest
    - Change exercise: Replace with similar movement pattern
    - Adjust days: Restructure training split

    Important:
    - Make ONLY requested changes
    - Explain what you changed

    Return ONLY the revised plan. Do NOT ask questions.
    """,
    output_key="workout_plan"  # Overwrites the same key
)


# Sub-agent 1
workout_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="workout_program_agent",
    description="Creates and manages training plan based on provided user profile",
    instruction="""
    You are an expert personal trainer and strength coach.

    You will receive user info from the coordinator.

    STEP 1: Gather workout-specific info:
    1. Available training days per week
    2. Available equipment (full gym, home gym, bodyweight only)
    3. Any injuries or physical limitations
    
    STEP 2: Create a complete training program following these principles:
    1. Progressive overload
    2. Appropriate volume per muscle group (10-20 sets/week)
    3. Proper exercise selection (compounds before isolation)
    4. Adequate recovery between muscle groups
    5. Appropriate rep ranges for the goal
    
    For each workout, provide:
    - Exercise name
    - Sets x Reps
    - Rest period
    - RPE (Rate of Perceived Exertion) target

    STEP 3: Review Loop:
    Present the plan.
    Ask: "Are you happy with this workout plan, or would you like changes?"

    If user wants changes:
    - Use workout_reviser tool with their feedback
    - Present the revised plan
    - Ask again if they're happy
    - Repeat until user approves

    STEP 4: Complete:
    When user approves (for example: good/yes/approve/looks good/perfect):
    Say: "Great! Your workout plan is set!"
    Use transfer_to_agent with agent_name="fitness_concierge_agent"

    Format as a structured weekly program. Be professional.
    IMPORTANT: Always remember and reference information the user has already provided.
    """,
    tools=[AgentTool(agent=workout_reviser)],
    output_key="workout_plan"
)


# Setting up MCP Server
!git clone https://github.com/deadletterq/mcp-opennutrition.git

%cd mcp-opennutrition
!npm install
!npm run build


# MCP integration with Open Nutrition
mcp_nutrition_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="node",
            args=["/kaggle/working/mcp-opennutrition/build/index.js"
                
            ],
        ),
        timeout=30,
    )
)

logger.info("OpenNutrition MCP Tool created!")


async def get_filtered_mcp_tools():
    tools = await mcp_nutrition_server.get_tools()
    allowed = ["search-food-by-name", "get-foods"]
    filtered = [t for t in tools if t.name in allowed]
    logger.info(f"Using: {[t.name for t in filtered]}")
    return filtered

mcp_tools = await get_filtered_mcp_tools()


def calculate_daily_calories(
    weight_kg: float,
    age: int,
    sex: str,
    activity_level: str,
    goal: str
) -> dict:
    """
    Calculate daily calorie and macro needs.
    
    Args:
        weight_kg: Body weight in kg
        age: Age in years
        sex: 'male' or 'female'
        activity_level: sedentary, light, moderate, active, very_active
        goal: weight_loss, maintain, muscle_gain
        
    Returns:
        Dict with calories and macros
    """
    if sex.lower() == "male":
        bmr = 10 * weight_kg + 6.25 * 175 - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * 165 - 5 * age - 161

    # Activity multiplier
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }

    tdee = bmr * multipliers.get(activity_level.lower(), 1.55)

    # Goal adjustments
    if goal.lower() == "weight_loss":
        calories = tdee - 500
    elif goal.lower() == "muscle_gain":
        calories = tdee + 300
    else:
        calories = tdee

    # Calculate macros
    protein_g = round(weight_kg * 2.2)
    fats_g = round(calories * 0.30 / 9)
    carbs_g = round((calories - (protein_g * 4) - (fats_g * 9)) / 4)

    return {
        "daily_calories": round(calories),
        "protein_grams": protein_g,
        "carbs_grams": carbs_g,
        "fats_gram": fats_g
    }


meal_reviser = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="meal_agent_reviser",
    description="Revises meal plan based on user feedback",
    instruction="""Revise the meal plan: {meal_plan}
    Revise the workout plan based on the feedback provided.

    Common adjustments:
    - Replace foods: Use MCP tool to find similar alternatives
    - Adjust macros: Change portions to hit new targets
    - Change calories: Adjust all portions proportionally
    - Different meals/days: Redistribute foods

    Important:
    - Make ONLY requested changes
    - Use MCP nutrition tool for accurate data
    - Explain what you changed.

    Return ONLY the revised plan. Do NOT ask questions.
    """,
    tools=[*mcp_tools],
    output_key="meal_plan",  # Overwrites the same key
)


# Sub-agent 2
meal_planning_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="meal_planning_agent",
    description="Creates initial meal plan based on provided user profile",
    instruction="""
    You are a meal planning assistant.

    You will receive user info from the coordinator.

    STEP 1: Gather info
    Ask for: weight (kg), age, sex, activity level, goal

    STEP 2: Calculate calories
    Use calculate_daily_calories to determine their needs 
    Normalize inputs to: male/female, sedentary/light/moderate/active/very_active, weight_loss/maintain/muscle_gain)

    STEP 3: Create meal plan
    IMMEDIATELY after getting calories, create a meal plan using MCP nutrition tools to get accurate nutritional data.
    Include breakfast, lunch and dinner with portions and macros. Avoid allergens/dislikes

    STEP 4: Review
    Ask: "Are you happy with this meal plan, or would you like changes?"

    If user wants changes:
    - Use meal_reviser tool with their feedback
    - Present the revised plan
    - Ask again if they're happy
    - Repeat until user approves

    When user approves (for example: good/yes/approve/looks good/perfect):
    Say: "Great! Your meal plan is set!"
    Use transfer_to_agent with agent_name="fitness_concierge_agent"

    Format as a structured weekly program. Be professional.
    IMPORTANT: Always remember and reference information the user has already provided.
    """,
    tools=[calculate_daily_calories,
           *mcp_tools,
           AgentTool(agent=meal_reviser)],
    output_key="meal_plan"
)


display(Image(filename="/kaggle/input/insight-images000/mcp_tools.png"))


display(Image(filename="/kaggle/input/insight-images000/user_request.png"))


display(Image(filename="/kaggle/input/insight-images000/request_complete.png"))


display(Image(filename="/kaggle/input/insight-images000/tool_usage.png"))


logger.info(f"Tool Usage: {tool_tracker.get_summary()}")


# Tool Usage Plugin
class ToolUsagePlugin(BasePlugin):
    """
    A plugin that tracks tool usage patterns across agent conversations.
    
    It monitors all tool calls made by agent, recording:
    - No. of times each tool is called
    - Average execution time per tool
    
    Attributes:
        tool_calls: Dict storing call counts and timing for each tool.
        tool_start_times: Temporary storage for tracking call durations.
    """

    def __init__(self) -> None:
        """Initialize the ToolUsagePlugin with empty tracking dictionaries."""
        super().__init__(name="tool_usage")
        self.tool_calls: Dict[str, Dict[str, float]] = {}
        self.tool_start_times: Dict[int, float] = {}  # Track start times separately

    async def before_tool_callback(
        self, 
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: CallbackContext,
        **kwargs: Any
    ) -> None:
        """
        Callback executed before a tool is invoked.

        Records the start time for duration tracking and initializes
        the tool's entry in the tracking dictionary if needed.

        Args:
            tool: Tool object being invoked.
            tool_args: Dictionary of arguments being passed to the tool.
            tool_context: Context object containing session and state information.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        tool_name: str = getattr(tool, 'name', str(tool))

        if tool_name not in self.tool_calls:
            self.tool_calls[tool_name] = {"count": 0, "total_time": 0}

        self.tool_start_times[id(tool_context)] = time.time()
        logging.info(f"[TOOL] Starting: {tool_name}")

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: CallbackContext,
        result: Any,
        **kwargs: Any
    ) -> None:
        """
        Callback executed after a tool completes.

        Calculates the execution duration and updates the tool's
        statistics in the tracking dictionary.

        Args:
            tool: Tool object that was invoked.
            tool_args: Dictionary of arguments that were passed to the tool.
            tool_context: Context object containing session and state information.
            result: The return value from the tool execution.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        tool_name: str = getattr(tool, 'name', str(tool))

        # Calculate duration
        start_time: float = self.tool_start_times.pop(id(tool_context), time.time())
        duration: float = time.time() - start_time

        if tool_name in self.tool_calls:
            self.tool_calls[tool_name]["count"] += 1
            self.tool_calls[tool_name]["total_time"] += duration

        logger.info(f"[TOOL] Completed: {tool_name} | Duration: {duration:.2f}s")

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Generate a summary of tool usage statistics.

        Returns:
            Dictionary mapping tool names to their usage statistics.
            Each tool entry contains:
            - total_calls: No. of times the tool was invoked.
            - avg_time: Average execution time in seconds.
        """
        return {
            tool_name: {
                "total_calls": data["count"],
                "avg_time": data["total_time"] / data["count"] if data["count"] > 0 else 0
            }
            for tool_name, data in self.tool_calls.items()
        }


orchestrator = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="fitness_concierge_agent",
    instruction="""
    You are a fitness concierge agent that coordinates personalized fitness planning (workout + meal).

    STEP 1: Gather core user information
    Ask the user for:
    1. Primary fitness goals? (muscle gain, fat loss, strength, general fitness)
    2. Experience level (beginner, intermediate, advanced)

    STEP 2: Workout
    Say: "Great! Let's start with your workout plan."
    Transfer to workout_agent.

    (workout_agent will handle everything and transfer back when done)

    STEP 3: Meal
    When workout_agent transfers back to you:
    Say: "Excellent! Now let's create your meal plan."
    Transfer to meal_planning_agent.

    (meal_planning_agent will handle everything and transfer back when done)

    STEP 4: Finalize
    When meal_planning_agent transfers back to you:
    Present both plans together:
    - Show the workout_plan
    - Show the meal_plan
    Congratulate them on starting their fitness journey!
    
    Be professional, encouraging and supportive
    """,
    sub_agents=[workout_agent, meal_planning_agent]
)


# Constants
APP_NAME = "fitness_concierge"
USER_ID = "user_001"
MODEL_NAME = "gemini-2.5-flash-lite"

# Create plugins
tool_tracker = ToolUsagePlugin()

# Use DatabaseSessionService for persistent session storage
db_url = "sqlite:///fitness_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)

# Create runner
runner = Runner(agent=orchestrator, app_name=APP_NAME, session_service=session_service, plugins=[tool_tracker, LoggingPlugin()])

logger.info(f"Database: {session_service}")


# Create an object called App
fitness_app_compacting = App(
    name="fitness_app_compacting",
    root_agent=orchestrator,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=7,
        overlap_size=3
    )
)

# Use DatabaseSessionService for persistent session storage
db_url = "sqlite:///fitness_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)

# Create runner
runner = Runner(app=fitness_app_compacting, session_service=session_service)


await run_session(
    runner,
    "I want to start working out!",
    "compaction_fitness",
)

await run_session(
    runner,
    "I am looking to increase strength, and am an intermediate level.",
    "compaction_fitness",
)

await run_session(
    runner,
    "3 days, bodyweight only and no injuries.",
    "compaction_fitness",
)

await run_session(
    runner,
    "yes I am happy!",
    "compaction_fitness",
)

await run_session(
    runner,
    "I am 50kg, 22 years old, male, and light level, muscle gain",
    "compaction_fitness",
)

await run_session(
    runner,
    "I am allergic to prawns.",
    "compaction_fitness",
)

await run_session(
    runner,
    "yes I am happy!",
    "compaction_fitness",
)


# Get the final session state
final_session = await session_service.get_session(
    app_name=runner.app_name,
    user_id="user_001",
    session_id="compaction_fitness",
)

logger.info("--- Searching for Compaction Summary Event ---")
found_summary = False
for event in final_session.events:
    # Compaction events have a 'compaction' attribute
    if event.actions and event.actions.compaction:
        logger.info("\nâœ… SUCCESS! Found the Compaction Event:")
        logger.info(f"  Author: {event.author}")
        logger.info(f"\n Compacted information: {event}")
        found_summary = True
        break

if not found_summary:
    logger.info(
        "\nâ�Œ No compaction event found. Try increasing the number of turns in the demo."
    )


import os
from kaggle_secrets import UserSecretsClient
import json

user_secrets = UserSecretsClient()

try:
    user_credential = user_secrets.get_gcloud_credential()
    user_secrets.set_tensorflow_credential(user_credential)
    print("âœ… Kaggle GCP credential set!")
except Exception as e:
    print(f"Kaggle GCP auth failed: {e}")
    
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
GOOGLE_CLOUD_LOCATION = user_secrets.get_secret("GOOGLE_CLOUD_LOCATION")
GOOGLE_CLOUD_PROJECT = user_secrets.get_secret("GOOGLE_CLOUD_PROJECT")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ['GOOGLE_CLOUD_LOCATION'] = GOOGLE_CLOUD_LOCATION
os.environ['GOOGLE_CLOUD_PROJECT'] = GOOGLE_CLOUD_PROJECT

print("âœ… All configured!")


!adk create concierge_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


# Writing to agent.py


%%writefile /kaggle/working/concierge_agent/agent.py

# ==================================================
# IMPORTS
# ==================================================
from typing import Any, Dict

from google.genai import types
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.tools import BaseTool
from google.adk.models.google_llm import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.sessions import DatabaseSessionService
from google.adk.plugins import BasePlugin
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.agents.callback_context import CallbackContext

# ==================================================
# CONFIG
# ==================================================
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ==================================================
# TOOLS
# ==================================================
# MCP
mcp_nutrition_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="node",
            args=["/kaggle/working/mcp-opennutrition/build/index.js"
                
            ],
        ),
        timeout=30,
    )
)

# FUNCTION TOOLS
def calculate_daily_calories(
    weight_kg: float,
    age: int,
    sex: str,
    activity_level: str,
    goal: str
) -> dict:
    """
    Calculate daily calorie and macro needs.
    
    Args:
        weight_kg: Body weight in kg
        age: Age in years
        sex: 'male' or 'female'
        activity_level: sedentary, light, moderate, active, very_active
        goal: weight_loss, maintain, muscle_gain
        
    Returns:
        Dict with calories and macros
    """
    if sex.lower() == "male":
        bmr = 10 * weight_kg + 6.25 * 175 - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * 165 - 5 * age - 161

    # Activity multiplier
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }

    tdee = bmr * multipliers.get(activity_level.lower(), 1.55)

    # Goal adjustments
    if goal.lower() == "weight_loss":
        calories = tdee - 500
    elif goal.lower() == "muscle_gain":
        calories = tdee + 300
    else:
        calories = tdee

    # Calculate macros
    protein_g = round(weight_kg * 2.2)
    fats_g = round(calories * 0.30 / 9)
    carbs_g = round((calories - (protein_g * 4) - (fats_g * 9)) / 4)

    return {
        "daily_calories": round(calories),
        "protein_grams": protein_g,
        "carbs_grams": carbs_g,
        "fats_gram": fats_g
    }

# ==================================================
# ALL AGENTS
# ==================================================
workout_reviser = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="workout_reviser",
    description="Revises workout plan based on user feedback",
    instruction="""
    Revise the workout plan based on the feedback provided.
    
    Common adjustments:
    - Increase intensity: Add sets/reps, reduce rest, add weight
    - Decrease intensity: Remove sets, increase rest
    - Change exercise: Replace with similar movement pattern
    - Adjust days: Restructure training split

    Important:
    - Make ONLY requested changes
    - Explain what you changed

    Return ONLY the revised plan. Do NOT ask questions.
    """,
)

meal_reviser = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="meal_agent_reviser",
    description="Revises meal plan based on user feedback",
    instruction="""
    Revise the workout plan based on the feedback provided.

    Common adjustments:
    - Replace foods: Use MCP tool to find similar alternatives
    - Adjust macros: Change portions to hit new targets
    - Change calories: Adjust all portions proportionally
    - Different meals/days: Redistribute foods

    Important:
    - Make ONLY requested changes
    - Use MCP nutrition tool for accurate data
    - Explain what you changed.

    Return ONLY the revised plan. Do NOT ask questions.
    """,
    tools=[mcp_nutrition_server],
)

workout_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="workout_program_agent",
    description="Creates and manages training plan based on provided user profile",
    instruction="""
    You are an expert personal trainer and strength coach.

    You will receive user info from the coordinator.

    STEP 1: Gather workout-specific info:
    1. Available training days per week
    2. Available equipment (full gym, home gym, bodyweight only)
    3. Any injuries or physical limitations
    
    STEP 2: Create a complete training program following these principles:
    1. Progressive overload
    2. Appropriate volume per muscle group (10-20 sets/week)
    3. Proper exercise selection (compounds before isolation)
    4. Adequate recovery between muscle groups
    5. Appropriate rep ranges for the goal
    
    For each workout, provide:
    - Exercise name
    - Sets x Reps
    - Rest period
    - RPE (Rate of Perceived Exertion) target

    STEP 3: Review Loop:
    Present the plan.
    Ask: "Are you happy with this workout plan, or would you like changes?"

    If user wants changes:
    - Use workout_reviser tool with their feedback
    - Present the revised plan
    - Ask again if they're happy
    - Repeat until user approves

    STEP 4: Complete:
    When user approves (for example: good/yes/approve/looks good/perfect):
    Say: "Great! Your workout plan is set!"
    Use transfer_to_agent with agent_name="fitness_concierge_agent"

    Format as a structured weekly program. Be professional.
    IMPORTANT: Always remember and reference information the user has already provided.
    """,
    tools=[AgentTool(agent=workout_reviser)],
)

meal_planning_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="meal_planning_agent",
    description="Creates initial meal plan based on provided user profile",
    instruction="""
    You are a meal planning assistant.

    You will receive user info from the coordinator.

    STEP 1: Gather info
    Ask for: weight (kg), age, sex, activity level, goal

    STEP 2: Calculate calories
    Use calculate_daily_calories to determine their needs 
    Normalize inputs to: male/female, sedentary/light/moderate/active/very_active, weight_loss/maintain/muscle_gain)

    STEP 3: Create meal plan
    IMMEDIATELY after getting calories, create a meal plan.
    Use the MCP nutrition tools to get accurate nutritional data.
    Include breakfast, lunch and dinner with portions and macros. Avoid allergens/dislikes

    STEP 4: Review
    Ask: "Are you happy with this meal plan, or would you like changes?"

    If user wants changes:
    - Use meal_reviser tool with their feedback
    - Present the revised plan
    - Ask again if they're happy
    - Repeat until user approves

    When user approves (for example: good/yes/approve/looks good/perfect):
    Say: "Great! Your meal plan is set!"
    Use transfer_to_agent with agent_name="fitness_concierge_agent"

    Format as a structured weekly program. Be professional.
    IMPORTANT: Always remember and reference information the user has already provided.
    """,
    tools=[calculate_daily_calories,
           mcp_nutrition_server,
           AgentTool(agent=meal_reviser)],
)

orchestrator = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="fitness_concierge_agent",
    instruction="""
    You are a fitness concierge agent that coordinates personalized fitness planning (workout + meal).

    STEP 1: Gather core user information
    Ask the user for:
    1. Primary fitness goals? (muscle gain, fat loss, strength, general fitness)
    2. Experience level (beginner, intermediate, advanced)

    STEP 2: Workout
    Say: "Great! Let's start with your workout plan."
    Transfer to workout_agent.

    (workout_agent will handle everything and transfer back when done)

    STEP 3: Meal
    When workout_agent transfers back to you:
    Say: "Excellent! Now let's create your meal plan."
    Transfer to meal_planning_agent.

    (meal_planning_agent will handle everything and transfer back when done)

    STEP 4: Finalize
    When meal_planning_agent transfers back to you:
    Present both plans together:
    - Show the workout_plan
    - Show the meal_plan
    Congratulate them on starting their fitness journey!
    
    Be professional, encouraging and supportive
    """,
    sub_agents=[workout_agent, meal_planning_agent]
)

# REQUIRED EXPORT
root_agent = orchestrator


# Create conversation scenarios
convo_scenarios = {
  "scenarios": [
    {
      "starting_prompt": "I want to start getting fit!",
      "conversation_plan": "You are a beginner looking to lose weight. Provide these core information all at once. When asked about workout plan, provide: weight loss, beginner level \
      can train 3 days per week, home gym access all at once. For meal plan: provide 75kg, 32, male, moderate, no allergies all at once. Approve both plans without requesting changes."
    },
  ]
}

with open("/kaggle/working/concierge_agent/conversation_scenarios.json", "w") as f:
    json.dump(convo_scenarios, f, indent=2)


# Create session input file
session_input = {
    "app_name": "user_simulation",
    "user_id": "user_potato"
}

with open("/kaggle/working/concierge_agent/session_input.json", "w") as f:
    json.dump(session_input, f, indent=2)


# Create a new EvalSet
!adk eval_set create \
  /kaggle/working/concierge_agent/ \
  eval_set_with_scenarios

# Add conversation scenarios to the EvalSet as new eval cases
!adk eval_set add_eval_case \
  /kaggle/working/concierge_agent/ \
  eval_set_with_scenarios \
  --scenarios_file /kaggle/working/concierge_agent/conversation_scenarios.json \
  --session_input_file /kaggle/working/concierge_agent/session_input.json


# Create eval config
eval_config = {
  "criteria": {
    "hallucinations_v1": {
      "threshold": 0.5,
      "evaluate_intermediate_nl_responses": True
    },
    "safety_v1": {
      "threshold": 0.8
    }
  },
    "user_simulator_config": {
        "model": "gemini-2.5-flash",
        "model_configuration": {
            "thinking_config": {
                "include_thoughts": True,
                "thinking_budget": 4096
            }
        },
        "max_allowed_invocations": 10
    }
}

with open("/kaggle/working/concierge_agent/eval_config.json", "w") as f:
    json.dump(eval_config, f, indent=2)


!adk eval \
    /kaggle/working/concierge_agent/ \
    --config_file_path /kaggle/working/concierge_agent/eval_config.json \
    eval_set_with_scenarios \
    --print_detailed_results


import os
import random
import vertexai
from vertexai import agent_engines


!mkdir -p deploy_agent

logger.info("âœ… Deploy Agent directory created!")


%%writefile deploy_agent/requirements.txt

google-adk
opentelemetry-instrumentation-google-genai
mcp


%%writefile deploy_agent/.env

GOOGLE_CLOUD_LOCATION="global"

# 1 to use Vertex AI
GOOGLE_GENAI_USE_VERTEXAI=1


%%writefile deploy_agent/.agent_engine_config.json
{
    "min_instances": 0,
    "max_instances": 1,
    "resource_limits": {"cpu": "1", "memory": "1Gi"}
}


%%writefile deploy_agent/agent.py

# ==================================================
# IMPORTS
# ==================================================
from typing import Any, Dict
import vertexai
import os

from google.genai import types
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.tools import BaseTool
from google.adk.models.google_llm import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.agents.callback_context import CallbackContext

vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

# ==================================================
# CONFIG
# ==================================================
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ==================================================
# TOOLS
# ==================================================
# MCP
mcp_nutrition_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="node",
            args=["/kaggle/working/mcp-opennutrition/build/index.js"
                
            ],
        ),
        timeout=30,
    )
)

# FUNCTION TOOLS
def calculate_daily_calories(
    weight_kg: float,
    age: int,
    sex: str,
    activity_level: str,
    goal: str
) -> dict:
    """
    Calculate daily calorie and macro needs.
    
    Args:
        weight_kg: Body weight in kg
        age: Age in years
        sex: 'male' or 'female'
        activity_level: sedentary, light, moderate, active, very_active
        goal: weight_loss, maintain, muscle_gain
        
    Returns:
        Dict with calories and macros
    """
    if sex.lower() == "male":
        bmr = 10 * weight_kg + 6.25 * 175 - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * 165 - 5 * age - 161

    # Activity multiplier
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }

    tdee = bmr * multipliers.get(activity_level.lower(), 1.55)

    # Goal adjustments
    if goal.lower() == "weight_loss":
        calories = tdee - 500
    elif goal.lower() == "muscle_gain":
        calories = tdee + 300
    else:
        calories = tdee

    # Calculate macros
    protein_g = round(weight_kg * 2.2)
    fats_g = round(calories * 0.30 / 9)
    carbs_g = round((calories - (protein_g * 4) - (fats_g * 9)) / 4)

    return {
        "daily_calories": round(calories),
        "protein_grams": protein_g,
        "carbs_grams": carbs_g,
        "fats_gram": fats_g
    }

# ==================================================
# ALL AGENTS
# ==================================================
workout_reviser = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="workout_reviser",
    description="Revises workout plan based on user feedback",
    instruction="""
    Revise the workout plan based on the feedback provided.
    
    Common adjustments:
    - Increase intensity: Add sets/reps, reduce rest, add weight
    - Decrease intensity: Remove sets, increase rest
    - Change exercise: Replace with similar movement pattern
    - Adjust days: Restructure training split

    Important:
    - Make ONLY requested changes
    - Explain what you changed

    Return ONLY the revised plan. Do NOT ask questions.
    """,
)

meal_reviser = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="meal_agent_reviser",
    description="Revises meal plan based on user feedback",
    instruction="""
    Revise the workout plan based on the feedback provided.

    Common adjustments:
    - Replace foods: Use MCP tool to find similar alternatives
    - Adjust macros: Change portions to hit new targets
    - Change calories: Adjust all portions proportionally
    - Different meals/days: Redistribute foods

    Important:
    - Make ONLY requested changes
    - Use MCP nutrition tool for accurate data
    - Explain what you changed.

    Return ONLY the revised plan. Do NOT ask questions.
    """,
    tools=[mcp_nutrition_server],
)

workout_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="workout_program_agent",
    description="Creates and manages training plan based on provided user profile",
    instruction="""
    You are an expert personal trainer and strength coach.

    You will receive user info from the coordinator.

    STEP 1: Gather workout-specific info:
    1. Available training days per week
    2. Available equipment (full gym, home gym, bodyweight only)
    3. Any injuries or physical limitations
    
    STEP 2: Create a complete training program following these principles:
    1. Progressive overload
    2. Appropriate volume per muscle group (10-20 sets/week)
    3. Proper exercise selection (compounds before isolation)
    4. Adequate recovery between muscle groups
    5. Appropriate rep ranges for the goal
    
    For each workout, provide:
    - Exercise name
    - Sets x Reps
    - Rest period
    - RPE (Rate of Perceived Exertion) target

    STEP 3: Review Loop:
    Present the plan.
    Ask: "Are you happy with this workout plan, or would you like changes?"

    If user wants changes:
    - Use workout_reviser tool with their feedback
    - Present the revised plan
    - Ask again if they're happy
    - Repeat until user approves

    STEP 4: Complete:
    When user approves (for example: good/yes/approve/looks good/perfect):
    Say: "Great! Your workout plan is set!"
    Use transfer_to_agent with agent_name="fitness_concierge_agent"

    Format as a structured weekly program. Be professional.
    IMPORTANT: Always remember and reference information the user has already provided.
    """,
    tools=[AgentTool(agent=workout_reviser)],
)

meal_planning_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="meal_planning_agent",
    description="Creates initial meal plan based on provided user profile",
    instruction="""
    You are a meal planning assistant.

    You will receive user info from the coordinator.

    STEP 1: Gather info
    Ask for: weight (kg), age, sex, activity level, goal

    STEP 2: Calculate calories
    Use calculate_daily_calories to determine their needs 
    Normalize inputs to: male/female, sedentary/light/moderate/active/very_active, weight_loss/maintain/muscle_gain)

    STEP 3: Create meal plan
    IMMEDIATELY after getting calories, create a meal plan.
    Use the MCP nutrition tools to get accurate nutritional data.
    Include breakfast, lunch and dinner with portions and macros. Avoid allergens/dislikes

    STEP 4: Review
    Ask: "Are you happy with this meal plan, or would you like changes?"

    If user wants changes:
    - Use meal_reviser tool with their feedback
    - Present the revised plan
    - Ask again if they're happy
    - Repeat until user approves

    When user approves (for example: good/yes/approve/looks good/perfect):
    Say: "Great! Your meal plan is set!"
    Use transfer_to_agent with agent_name="fitness_concierge_agent"

    Format as a structured weekly program. Be professional.
    IMPORTANT: Always remember and reference information the user has already provided.
    """,
    tools=[calculate_daily_calories,
           mcp_nutrition_server,
           AgentTool(agent=meal_reviser)],
)

orchestrator = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="fitness_concierge_agent",
    instruction="""
    You are a fitness concierge agent that coordinates personalized fitness planning (workout + meal).

    STEP 1: Gather core user information
    Ask the user for:
    1. Primary fitness goals? (muscle gain, fat loss, strength, general fitness)
    2. Experience level (beginner, intermediate, advanced)

    STEP 2: Workout
    Say: "Great! Let's start with your workout plan."
    Transfer to workout_agent.

    (workout_agent will handle everything and transfer back when done)

    STEP 3: Meal
    When workout_agent transfers back to you:
    Say: "Excellent! Now let's create your meal plan."
    Transfer to meal_planning_agent.

    (meal_planning_agent will handle everything and transfer back when done)

    STEP 4: Finalize
    When meal_planning_agent transfers back to you:
    Present both plans together:
    - Show the workout_plan
    - Show the meal_plan
    Congratulate them on starting their fitness journey!
    
    Be professional, encouraging and supportive
    """,
    sub_agents=[workout_agent, meal_planning_agent]
)

# REQUIRED EXPORT
root_agent = orchestrator


# Select deployment region
regions_list = ["europe-west1", "europe-west4", "us-east4", "us-west1"]
deployed_region = random.choice(regions_list)

logger.info(f"âœ… Selected deployment region: {deployed_region}")


!adk deploy agent_engine --project=$GOOGLE_CLOUD_PROJECT --region=$deployed_region deploy_agent --agent_engine_config_file=deploy_agent/.agent_engine_config.json



# Retrieving the deployed agent

# Initialize Vertex AI
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=deployed_region)

# Get the most recently deployed agent
agents_list = list(agent_engines.list())
if agents_list:
    remote_agent = agents_list[0]  # Get the first (most recent) agent
    client = agent_engines
    logger.info(f"âœ… Connected to deployed agent: {remote_agent.resource_name}")
else:
    logger.info("â�Œ No agents found. Please deploy first.")


# Test the deployed agent
async for item in remote_agent.async_stream_query(
    message="I want to stay fit!",
    user_id="fitspo"
):
    logger.info(f"{item}")

