# pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    USDA_API_KEY = UserSecretsClient().get_secret("USDA_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["USDA_API_KEY"] = USDA_API_KEY
    print("âœ… Gemini and USDA API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types
from google.adk.agents import (
    Agent,
    LlmAgent,
    SequentialAgent,
    ParallelAgent,
)
from google.adk.code_executors import BuiltInCodeExecutor

from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.tools.google_search_tool import google_search
from google.adk.runners import Runner, InMemoryRunner

from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.adk.agents.callback_context import CallbackContext

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field
import sqlite3
import json
import asyncio
import os
import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*experimental.*",
    category=UserWarning
)

import logging

logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
logging.getLogger("google.adk").setLevel(logging.ERROR)


print("Libraries successfully imported!")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


class UserHealthProfile(BaseModel):
    conditions: List[str] = Field(
        default_factory=list,
        description="List of user health conditions (e.g., Type 2 Diabetes, Hypertension)."
    )
    calorie_target: Optional[int] = Field(
        default=None,
        description="Daily calorie target."
    )
    sodium_limit_mg: Optional[int] = Field(
        default=None,
        description="Daily sodium limit in milligrams."
    )
    carb_limit_g: Optional[int] = Field(
        default=None,
        description="Daily carbohydrate limit in grams."
    )
    allergies: List[str] = Field(
        default_factory=list,
        description="Food allergies (e.g., dairy, nuts)."
    )
    preferences: List[str] = Field(
        default_factory=list,
        description="Positive preferences (e.g., 'no red meat', '15-min meals')."
    )
    dislikes: List[str] = Field(
        default_factory=list,
        description="Foods the user dislikes."
    )


class NutritionRules(BaseModel):
    guidelines: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured nutrition guidelines (e.g., max_sodium_mg_per_day)."
    )
    recommended_foods: List[str] = Field(
        default_factory=list,
        description="List of recommended foods."
    )
    avoid_foods: List[str] = Field(
        default_factory=list,
        description="List of foods to avoid."
    )


class NutritionSafetyOutput(BaseModel):
    unsafe_foods: List[str] = Field(
        default_factory=list,
        description="Foods considered unsafe for the user's conditions."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Human-readable dietary warnings."
    )
    flagged_ingredients: Dict[str, Any] = Field(
        default_factory=dict,
        description="Map of ingredient -> risk info."
    )


class Meal(BaseModel):
    meal_type: Optional[str] = Field(
        default=None,
        description="Meal type (e.g., Breakfast, Lunch, Dinner)."
    )
    name: Optional[str] = Field(
        default=None,
        description="Name of the meal."
    )
    ingredients: List[str] = Field(
        default_factory=list,
        description="List of ingredients used in the meal."
    )
    nutrition: Dict[str, Any] = Field(
        default_factory=dict,
        description="Nutrition breakdown (calories, carbs, sodium, etc.)."
    )
    substitutions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map from ingredient to list of possible substitutes."
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Short preparation/cooking instructions."
    )


class MealPlan(BaseModel):
    profile: UserHealthProfile = Field(
        ...,
        description="The user's health profile used to generate this plan."
    )
    meals: List[Meal] = Field(
        default_factory=list,
        description="List of meals in the plan."
    )



# Simple nutrition lookup function 
import requests

def nutrition_lookup(ingredient: str) -> dict:
    """
    Search USDA database and return nutrition info
    
    Args:
        ingredient: Name of food to look up
    
    Returns:
        Dict with nutrition data
    """
    # Search for the food
    search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    search_params = {
        "api_key": os.environ["USDA_API_KEY"],
        "query": ingredient,
        "pageSize": 1,
        "dataType": ["Foundation", "SR Legacy"]
    }
    
    try:
        response = requests.get(search_url, params=search_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        foods = data.get("foods", [])
        if not foods:
            return {"error": f"No results found for '{ingredient}'"}
        
        food = foods[0]
        
        # Extract nutrients
        nutrients = {n["nutrientName"]: n["value"] 
                    for n in food.get("foodNutrients", [])}
        
        return {
            "ingredient": ingredient,
            "description": food.get("description"),
            "calories": nutrients.get("Energy", 0),
            "protein_g": nutrients.get("Protein", 0),
            "carbs_g": nutrients.get("Carbohydrate, by difference", 0),
            "fat_g": nutrients.get("Total lipid (fat)", 0),
            "sodium_mg": nutrients.get("Sodium, Na", 0),
            "fiber_g": nutrients.get("Fiber, total dietary", 0),
            "sugars_g": nutrients.get("Sugars, total including NLEA", 0)
        }
    except Exception as e:
        return {"error": str(e)}


# Test it
# print(nutrition_lookup("chicken breast"))

nutrition_lookup_tool = FunctionTool(func=nutrition_lookup)


# ==============================
# USER PROFILE & MEAL PLAN STORE TOOLS
# ==============================

def save_user_profile(tool_context: ToolContext, profile: dict):
    """
    Save the full UserHealthProfile returned by HealthIntakeAgent.
    Stored as JSON under session state.
    """
    tool_context.state["user:profile"] = profile
    return {"status": "saved_profile"}


def get_user_profile(tool_context: ToolContext):
    """
    Retrieve user profile from session memory.
    """
    return {
        "status": "ok",
        "profile": tool_context.state.get("user:profile", None)
    }


def save_nutrition_rules(tool_context: ToolContext, nutrition_rules: dict, safety: dict):
    """
    Save nutrition guidelines + safety output
    """
    tool_context.state["user:nutrition_rules"] = nutrition_rules
    tool_context.state["user:safety"] = safety
    return {"status": "saved_rules"}


def get_nutrition_rules(tool_context: ToolContext):
    return {
        "nutrition_rules": tool_context.state.get("user:nutrition_rules"),
        "safety": tool_context.state.get("user:safety")
    }


def save_meal_plan(tool_context: ToolContext, meal_plan: dict):
    """
    Save the current MealPlan (full json).
    """
    tool_context.state["user:meal_plan"] = meal_plan
    return {"status": "saved_meal"}


def get_meal_plan(tool_context: ToolContext):
    """
    Retrieve the stored meal plan.
    """
    return {
        "meal_plan": tool_context.state.get("user:meal_plan", None)
    }


# -----------------------------
# HEALTH INTAKE AGENT
# -----------------------------
HealthIntakeAgent = LlmAgent(
    name="HealthIntakeAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Extract health constraints from user query.",
    output_schema=UserHealthProfile,
    instruction="""
        Extract the user's dietary constraints.
        Return ONLY valid JSON following the UserHealthProfile schema.
    """
)


# -----------------------------
# NUTRITION SEARCH AGENT (google_search only)
# -----------------------------
NutritionSearchAgent = LlmAgent(
    name="NutritionSearchAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Search for nutrition guidelines and compile condition-specific rules.",
    tools=[google_search],  # Only google_search
    instruction="""
        You are a dietitian agent. Use google_search to gather:
        - Recommended foods for the user's conditions
        - Foods to avoid
        - Sodium/carbohydrate limits
        - Evidence-based guidelines
        - KEEP YOUR RESPONSE CONCISE ONLY IN A JSON SCHEMA, DO NOT GENERATE ANY SHOPPING LIST OR MEAL PLANS.
        
        Provide detailed nutrition guidelines based on the search results.
    """
)


# -----------------------------
# NUTRITION SAFETY AGENT (google_search only)
# -----------------------------
NutritionSafetyAgent = LlmAgent(
    name="NutritionSafetyAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Identify risks, unsafe foods, or allergy conflicts.",
    tools=[google_search],  # Only google_search
    instruction="""
        Use google_search to identify:
        - Foods risky for user conditions
        - Foods that worsen user health profile
        - High sodium foods
        - High glycemic index foods
        - Allergy risk items
        - KEEP YOUR RESPONSE CONCISE ONLY IN A JSON SCHEMA, DO NOT GENERATE ANY SHOPPING LIST OR MEAL PLANS.
        
        Provide a detailed safety report.
    """
)


# -----------------------------
# MEAL GENERATION AGENT (nutrition_lookup_tool only)
# -----------------------------
MealGenerationAgent = LlmAgent(
    name="MealGenerationAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Generate a meal plan using user profile + nutrition rules.",
    tools=[nutrition_lookup_tool],  # Only nutrition_lookup_tool
    output_schema = MealPlan,
    instruction="""
    You generate meals in TWO MODES:

    ===========================
    MODE 1 â€” FULL GENERATION
    ===========================
    When mode=full:
        - Create 3 meals: Breakfast, Lunch, Dinner
        - Use nutrition_lookup_tool for each ingredient
        - Follow nutrition rules & safety rules
        - Respect allergies, dislikes, sodium/carb limits
        - Output a complete MealPlan JSON

    ===========================
    MODE 2 â€” PARTIAL GENERATION
    ===========================
    When mode=single_meal_update:
        - Only regenerate the meal_type provided (Breakfast/Lunch/Dinner)
        - You will receive the previous MealPlan
        - Replace ONLY that meal with a new one
        - Keep all other meals unchanged
        - Still use nutrition_lookup_tool for verification
        - Output a COMPLETE MealPlan with the updated meal included

    ===========================
    INPUT FORMAT YOU WILL RECEIVE
    ===========================
    {
        "mode": "full" | "single_meal_update",
        "meal_type": "...",          # Only in partial mode
        "profile": {...},
        "nutrition_rules": {...},
        "safety": {...},
        "previous_meal_plan": {...}  # Only in partial mode
    }

    Output ONLY valid JSON following the MealPlan schema.
"""
)


# -----------------------------
# SUMMARIZER AGENT
# -----------------------------
SummarizerAgent = LlmAgent(
    name="SummarizerAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Summarizes the Meal Plans and generates a Ingredients List.",
    tools=[google_search],  # Only google_search
    instruction="""
    You MUST always return output in the following EXACT two sections IN ORDER:
    
    ======================
    SECTION 1: MEAL_SUMMARY
    ======================
    For every meal in the MealPlan, output STRICTLY using this format:
    
    Meal Type: <meal_type>
    Description: <1â€“2 sentence summary of the meal>
    Procedure: <1â€“3 short steps>
    
    Repeat this block for EACH meal in the MealPlan, in the same order they appear.
    
    ======================
    SECTION 2: INGREDIENTS_LIST
    ======================
    Create a normalized ingredients list from ALL ingredients across all meals.
    
    For every item, output STRICTLY using this structure:
    
    - item: <ingredient name>
      quantity: <concise quantity string>
      tags: [<tag1>, <tag2>, ...]
    
    Where:
    â€¢ Tags may include: "low-sodium", "diabetes-friendly", "high-fiber", "heart-healthy", etc.
    â€¢ Quantities must be simple and consistent (e.g., "2x", "1 cup", "150g").
    
    ======================
    ADDITIONAL RULES
    ======================
    1. If the user reports an ingredient is out of stock:
       - Identify that ingredient.
       - Search for substitutes using google_search.
       - Replace the ingredient with the best substitute.
       - Rebuild the INGREDIENTS_LIST section completely.
    
    2. NEVER change the schema, labels, spacing, or section headers.
       The structure must remain EXACTLY as defined above.
    
    3. NEVER output natural-language commentary outside the schema.
    4. NEVER include extra sections, explanations, or text before or after the schema.
    
    5. The output must be fully deterministic and always contain:
       - SECTION 1: MEAL_SUMMARY
       - SECTION 2: INGREDIENTS_LIST
    
    Always follow this exact structure.
    """
)


# -----------------------------
# UPDATION AGENT
# -----------------------------
UpdationAgent = LlmAgent(
    name="UpdationAgent",
    model=Gemini(model="gemini-2.5-flash"),
    tools=[google_search, get_meal_plan, save_meal_plan, get_user_profile, save_user_profile],
    instruction="""
You update or recall the user's existing meal plan.

-------------------------------------
IF USER ASKS TO RECALL PREVIOUS MEAL PLAN:
-------------------------------------
- Read the meal plan from session memory.
- Return the SAME plan exactly in the SummarizerAgent format:
  (1) MEAL_SUMMARY
  (2) INGREDIENTS_LIST
- Do NOT regenerate anything.
- Do NOT alter the meals.
- Do NOT add commentary.

-------------------------------------
IF USER ASKS TO UPDATE THE PLAN:
-------------------------------------
- Modify ONLY the parts the user specifies.
- Regenerate updated MEAL_SUMMARY + INGREDIENTS_LIST.
- Use google_search for substitutions if needed.

-------------------------------------
FINAL OUTPUT RULE:
-------------------------------------
Always output the EXACT final text only in the SummarizerAgent's two-section format.
Never output additional text.
"""
)



def after_health_pipeline(callback_context: CallbackContext):
    """Save profile, rules, safety, and meal plan into session memory."""
    result = callback_context.latest_result  # full pipeline output

    # Extract parts from each sub-agent
    profile = result.from_agent(HealthIntakeAgent)
    nutrition_rules = result.from_agent(NutritionSearchAgent)
    safety = result.from_agent(NutritionSafetyAgent)
    meal_plan = result.from_agent(MealGenerationAgent)

    # Write to session state (ToolContext.state)
    ctx = callback_context.tool_context.state
    ctx["user:profile"] = profile
    ctx["user:nutrition_rules"] = nutrition_rules
    ctx["user:safety"] = safety
    ctx["user:meal_plan"] = meal_plan



# -----------------------------
# ORCHESTRATION
# -----------------------------
parallel_agents = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents = [NutritionSearchAgent, NutritionSafetyAgent]
)

HealthMealGeneration = SequentialAgent(
    name="HealthMealGenerator",
    sub_agents=[
        HealthIntakeAgent,
        parallel_agents,
        MealGenerationAgent,
        SummarizerAgent,
    ]
)

HealthMealGenerationTool = AgentTool(HealthMealGeneration)


HealthMealGenerationWrapper = LlmAgent(
    name="HealthMealGeneratorWrapper",
    model="gemini-2.5-flash",
    tools=[HealthMealGenerationTool, save_user_profile, save_nutrition_rules, save_meal_plan],
    instruction="""
    1. Call HealthMealGenerationTool.
    2. You will receive structured output containing:
       - HealthIntakeAgent (parsed JSON)
       - NutritionSearchAgent (parsed JSON)
       - NutritionSafetyAgent (parsed JSON)
       - MealGenerationAgent (parsed JSON MealPlan)
       - SummarizerAgent (final user-visible text)

    3. After receiving the tool result:
       - Call save_user_profile with HealthIntakeAgent's JSON.
       - Call save_nutrition_rules with NutritionSearchAgent + NutritionSafetyAgent.
       - Call save_meal_plan with MealGenerationAgent's MealPlan.

    4. Return ONLY the SummarizerAgent text exactly as provided.
    5. Never add commentary.
    """
)



async def auto_save_to_memory(callback_context: CallbackContext): 
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )
print("âœ… Callback created.")


MasterAgent = LlmAgent(
    model="gemini-2.5-flash",
    name="MasterAgent",
    tools=[AgentTool(HealthMealGenerationWrapper), AgentTool(UpdationAgent), load_memory],
    instruction="""
You are a routing controller for a health meal planning system.

Your job is to decide whether the user wants:
1. A NEW full meal plan â†’ call the `HealthMealGeneration` tool  
2. A CHANGE to the existing meal plan â†’ call the `UpdationAgent` tool  
- You can take previous memory context using the `load_memory` tool if needed

-------------------------------------
ROUTING RULES
-------------------------------------
- If no profile is stored in session memory (user:profile is null or missing) â†’ call HealthMealGeneration

Call **HealthMealGeneration** if:
- This is the user's first request in the session
- The user says â€œstart overâ€�, â€œbegin againâ€�, â€œnew planâ€�, â€œregenerateâ€�
- The user wants a full meal plan or shopping list from scratch
- The user drastically changed their constraints

Call **UpdationAgent** if:
- The user wants to change something in the existing plan
- The user wants ingredient substitutions
- The user reports an item out of stock
- The user wants to modify or adjust only part of the plan
- The user updates preferences like â€œless sodiumâ€�, â€œno yogurt nowâ€�

-------------------------------------
TOOL EXECUTION RULES
-------------------------------------
- ALWAYS call EXACTLY ONE tool per user query.
- NEVER answer directly.
- NEVER produce content other than a tool call.

-------------------------------------
WHEN A TOOL RETURNS A RESULT
-------------------------------------
- The tool output will be wrapped like:
  {"SomeTool_response": {"result": "<CONTENT>"}}
- Extract ONLY the value inside "result".
- Return ONLY this inner text EXACTLY as-is.
- DO NOT rewrite, expand, reduce, reformat, or wrap it in JSON or Markdown.
- DO NOT add extra commentary.

-------------------------------------
MEMORY-RELATED REQUESTS:
-------------------------------------
If the user asks anything like:
- "What was my previous meal plan?"
- "Show me my last meal plan again."
- "What did you give me before?"
- "What meals did you suggest earlier?"
- "Repeat yesterday's plan."

Then:
â†’ Call `UpdationAgent` because the user wants to RECALL the existing plan, NOT generate a new one.
â†’ DO NOT call HealthMealGeneration.

-------------------------------------
WHEN A TOOL RETURNS A RESULT
-------------------------------------
- Extract ONLY the "result" field from the tool output.
- Output ONLY that text, EXACTLY as-is.
- Never return memory objects or wrappers.

""",
    after_agent_callback=auto_save_to_memory
)


# Run the system

# runner = InMemoryRunner(agent=MasterAgent)
runner = InMemoryRunner(agent=HealthMealGeneration)
await runner.run_debug("""
Hi, I've been recently diagnosed with Type 2 diabetes and I also have high blood pressure. 
I want a healthy meal plan for the next day.
Here are my constraints:
- Target around 1800 calories.
- Prefer meals under 600mg sodium each.
- Try to keep carbs low.
- I'm allergic to peanuts and shellfish.
- I don't like mushrooms.
- I prefer high-fiber meals, Mediterranean style, easy to cook.
Please generate the full meal plan and the shopping list.
""")

# await runner.run_debug("""
# Hey, I need to start eating better. My doctor said my sugar and blood pressure are kinda high. 
# Can you just make something healthy for me to eat tomorrow?
# Nothing too fancy please.
# """)

# await runner.run_debug("""
# Iâ€™m trying to eat cleaner. Carbs seem to make me crash lately and my blood pressure is up.
# Can you put a healthy meal plan together for me for the day?
# Keep it easy to cook.
# """)

# await runner.run_debug("""
# Hey can you give me some meal ideas for tomorrow? 
# Trying to control my sugar and I think I should lower my salt too. 
# Donâ€™t like yogurt.
# """)


session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()


APP_NAME = "HealthMealApplication"  # Application
USER_ID = "user-01"  # User
SESSION = "default"  # Session


async def run_session(runner_instance: Runner, user_queries: list[str] | str = None, session_name: str = "default"):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name
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

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                agent_name = getattr(event, "author", None)
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{agent_name} > ", event.content.parts[0].text)
    else:
        print("No queries!")
print("âœ… Helper functions defined.")


health_meal_app = App(
    name="health_meal_app",
    root_agent=MasterAgent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=6,
        overlap_size=2,
    ),
)


runner = Runner(app=health_meal_app, session_service=session_service, memory_service=memory_service)


await run_session(
    runner,
    """Tomorrowâ€™s going to be busy. I want healthy meals that wonâ€™t spike my sugar. 
    Also trying to cut back on sodium. 
    Can you handle the food planning part?
    """,
    'test-00'
)


await run_session(
    runner,
    """Whats my current meal plan?
    """,
    'test-01'
)




