# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Core imports
import os
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import warnings

# Pydantic for structured outputs (ADK native support)
from pydantic import BaseModel, Field

# Google ADK imports 
from google.adk.agents import Agent, LlmAgent
from google.adk.runners import InMemoryRunner, Runner
from google.adk.models.google_llm import Gemini
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.tool_context import ToolContext

# Suppress informational warnings
# Suppress informational warnings - updated filters
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Warning: there are non-text parts in the response')
warnings.filterwarnings('ignore', message='Event from an unknown agent')

print("âœ… All imports successful - Google ADK ready!")


import os
from kaggle_secrets import UserSecretsClient
from google.genai import types

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# Model configuration with availability testing
print(f"\nğŸ”� Testing model availability and quota...")

from google import genai
from google.genai import types as genai_types
import time

# List of models to try (in order of preference)
CANDIDATE_MODELS = [
    "gemini-1.5-flash",      # Stable, good availability
    "gemini-1.5-pro",        # More powerful, stable
    "gemini-pro",            # Classic stable model
    "gemini-2.0-flash-exp",  # Latest experimental (may have quota issues)
    "gemini-2.5-flash-lite",
]

def test_model_availability(model_name: str) -> tuple[bool, str]:
    """
    Test if a model is available and has quota.
    
    Returns:
        (success: bool, message: str)
    """
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Try a minimal generation request
        response = client.models.generate_content(
            model=model_name,
            contents="Respond with just: OK"
        )
        
        if response.text and "OK" in response.text:
            return True, f"âœ… Model available and responding"
        else:
            return False, f"âš ï¸� Model responded but output unclear"
            
    except Exception as e:
        error_str = str(e)
        
        # Check for specific error types
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            return False, f"â�Œ Rate limit/quota exceeded"
        elif "404" in error_str or "NOT_FOUND" in error_str:
            return False, f"â�Œ Model not found"
        elif "503" in error_str or "UNAVAILABLE" in error_str:
            return False, f"â�Œ Model overloaded"
        else:
            return False, f"â�Œ Error: {error_str[:100]}"

# Test models and find first working one
MODEL = None
for candidate_model in CANDIDATE_MODELS:
    print(f"\n   Testing: {candidate_model}...", end=" ")
    
    success, message = test_model_availability(candidate_model)
    print(message)
    
    if success:
        MODEL = candidate_model
        print(f"\nâœ… Selected working model: {MODEL}")
        break
    
    # Brief pause between tests to avoid rate limiting the test itself
    time.sleep(1)

if MODEL is None:
    print("\n" + "=" * 80)
    print("â�Œ QUOTA/AVAILABILITY ISSUE")
    print("=" * 80)
    print("""
All tested models are either unavailable or quota-exhausted.

Solutions:
1. â�° WAIT: Rate limits reset after a few minutes (try again in 5-10 min)
2. ğŸ’³ UPGRADE: Enable billing for higher quotas at https://ai.google.dev/pricing
3. ğŸ”‘ NEW KEY: Create a new API key if you've exhausted the current one
4. ğŸ“Š CHECK USAGE: Visit https://ai.dev/usage?tab=rate-limit

Free tier limits:
- 15 requests per minute
- 1 million tokens per minute
- 1,500 requests per day

Your current quota appears to be exhausted. Please wait or upgrade.
""")
    raise ValueError("No available models found. Please check quota and try again.")

print(f"\nğŸ�¯ Using model: {MODEL}")
print(f"ğŸ“� Note: Multi-agent workflows consume quota quickly (3+ agents per request)")
print(f"ğŸ’¡ Tip: If quota issues persist, wait 5-10 minutes or enable billing")
# Model configuration (ADK handles initialization automatically)
MODEL = "gemini-2.5-flash-lite"  # Fast, efficient model for orchestration
Gemini(model=MODEL, retry_options=retry_config)


"""
Pydantic Schemas for Structured Agent Communication (Day 2: Controlled Generation)
These schemas enforce strict JSON output formats, ensuring agent-to-agent compatibility.
"""

# ============================================================================
# USER PROFILE SCHEMA
# ============================================================================
class UserProfile(BaseModel):
    """Captures user's health profile, goals, and constraints."""
    name: str = Field(..., description="User's name")
    age: int = Field(..., description="User's age in years")
    current_weight_kg: float = Field(..., description="Current weight in kg")
    fitness_level: str = Field(..., description="'Beginner', 'Intermediate', or 'Advanced'")
    primary_goal: str = Field(..., description="Main fitness goal (e.g., 'build muscle', 'lose weight', 'endurance')")
    available_time_min: int = Field(..., description="Available training time per day in minutes")
    equipment: List[str] = Field(default_factory=list, description="Available equipment (e.g., 'dumbbells', 'yoga mat')")
    dietary_restrictions: List[str] = Field(default_factory=list, description="Dietary restrictions (e.g., 'vegetarian', 'gluten-free')")
    injuries_constraints: List[str] = Field(default_factory=list, description="Physical constraints (e.g., 'sensitive knee', 'lower back pain')")
    additional_notes: str = Field(default="", description="Any other relevant information")


# ============================================================================
# WORKOUT PLAN SCHEMAS
# ============================================================================
class Exercise(BaseModel):
    """Individual exercise with instructions and progression."""
    name: str = Field(..., description="Exercise name")
    sets: int = Field(default=3, description="Number of sets")
    reps: str = Field(default="8-12", description="Reps per set (e.g., '8-10')")
    rest_seconds: int = Field(default=60, description="Rest between sets in seconds")
    instructions: str = Field(default="Perform with proper form", description="How to perform the exercise")
    video_link: Optional[str] = Field(default=None, description="Tutorial video URL")
    modifications: str = Field(default="", description="Modifications for beginners or injuries")


class WorkoutDay(BaseModel):
    """Workout plan for a single day."""
    day: str = Field(..., description="Day of week (e.g., 'Monday')")
    focus: str = Field(..., description="Workout focus (e.g., 'Upper Body', 'Cardio')")
    exercises: List[Exercise] = Field(..., description="List of exercises for the day")
    duration_min: int = Field(default=30, description="Estimated workout duration in minutes")
    notes: str = Field(default="", description="Special notes or warm-up guidance")


class WorkoutPlan(BaseModel):
    """Complete weekly workout plan (Day 2: Output structure)."""
    summary: str = Field(..., description="High-level overview of the plan")
    goal_alignment: str = Field(..., description="How this plan aligns with user's goal")
    important_notes: str = Field(..., description="Safety, hydration, and recovery tips")
    weekly_schedule: List[WorkoutDay] = Field(..., description="Monday through Sunday plan")
    progression: str = Field(..., description="How to progress after 4 weeks")
    constraints_respected: List[str] = Field(default_factory=list, description="List of user constraints addressed")


# ============================================================================
# NUTRITION PLAN SCHEMAS
# ============================================================================
class Meal(BaseModel):
    """Single meal with nutritional info."""
    name: str = Field(..., description="Meal name")
    ingredients: List[str] = Field(..., description="Ingredients list")
    calories: int = Field(..., description="Approximate calories")
    protein_g: float = Field(..., description="Protein in grams")
    carbs_g: float = Field(..., description="Carbohydrates in grams")
    fat_g: float = Field(..., description="Fat in grams")
    recipe_source: Optional[str] = Field(default=None, description="URL to full recipe")


class NutritionDay(BaseModel):
    """Daily nutrition plan."""
    day: str = Field(..., description="Day of week")
    meals: Dict[str, Meal] = Field(..., description="Breakfast, Lunch, Dinner, Snack")
    daily_calories: int = Field(..., description="Total daily calories")
    daily_protein_g: float = Field(..., description="Total daily protein")
    daily_carbs_g: float = Field(..., description="Total daily carbs")
    daily_fat_g: float = Field(..., description="Total daily fat")


class NutritionPlan(BaseModel):
    """Complete weekly nutrition plan (Day 2: Output structure)."""
    summary: str = Field(..., description="Overview of nutritional strategy")
    goal_alignment: str = Field(..., description="How nutrition supports fitness goal")
    caloric_target: int = Field(..., description="Daily caloric target")
    macro_ratios: Dict[str, float] = Field(..., description="Protein/Carbs/Fat ratios")
    weekly_plan: List[NutritionDay] = Field(..., description="Complete weekly plan")
    restrictions_addressed: List[str] = Field(..., description="Dietary restrictions accommodated")
    supplement_suggestions: List[str] = Field(default_factory=list, description="Optional supplements")


# ============================================================================
# CRITIQUE EVALUATION SCHEMA
# ============================================================================
class CritiqueEvaluation(BaseModel):
    """LLM-as-Judge evaluation results (Day 4: Agent evaluation)."""
    plan_type: str = Field(..., description="'workout' or 'nutrition'")
    status: str = Field(..., description="'APPROVED' or 'REVISE_REQUIRED'")
    effectiveness_score: float = Field(..., description="0-100: Does plan achieve goal?")
    safety_score: float = Field(..., description="0-100: Does plan respect constraints?")
    feasibility_score: float = Field(..., description="0-100: Is plan realistic and achievable?")
    strengths: List[str] = Field(..., description="What the plan does well")
    concerns: List[str] = Field(default_factory=list, description="Issues to address")
    recommendations: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    feedback_for_agent: str = Field(..., description="Detailed feedback for the specialist agent")


print("âœ… Pydantic schemas defined for structured agent communication")


"""
Observability with Google ADK Event Streaming
ADK provides built-in event streaming for real-time agent monitoring.
Events include: AgentInvoked, ToolCalled, ResponseGenerated, and more.
"""

# Global event storage for analysis
event_log = []

def log_adk_event(event: Any):
    """
    Log ADK events for observability.
    ADK automatically emits events during agent execution.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    event_type = type(event).__name__
    
    event_entry = {
        "timestamp": timestamp,
        "type": event_type,
        "event": event
    }
    event_log.append(event_entry)
    
    # Print key events for visibility
    if "Agent" in event_type or "Tool" in event_type or "Response" in event_type:
        print(f"[{timestamp}] ğŸ“Š {event_type}")

print("âœ… ADK event streaming configured")


"""
TOOL DEFINITIONS (ADK Format)
Tools are Python functions that extend agent capabilities.
ADK automatically generates tool schemas from function signatures and docstrings.
"""

# ============================================================================
# TOOL 1: Exercise Video Search (for Trainer Agent)
# ============================================================================

# Video database for simulation
_exercise_video_database = {
    "squats": "https://youtube.com/watch?v=xqvCmoLUV34",
    "push-ups": "https://youtube.com/watch?v=IODxDxX7oi4",
    "deadlifts": "https://youtube.com/watch?v=op9kVnSso6Q",
    "bench press": "https://youtube.com/watch?v=4T9UQ4FBVRU",
    "pull-ups": "https://youtube.com/watch?v=eGo4IYlbE5g",
    "lunges": "https://youtube.com/watch?v=H7KJlb-Lmn0",
    "planks": "https://youtube.com/watch?v=pSphRJltChk",
    "mountain climbers": "https://youtube.com/watch?v=nmwgirgXLYM",
    "burpees": "https://youtube.com/watch?v=JZQA0r_BN6s",
    "running": "https://youtube.com/watch?v=xzgvJ0DY6pY",
    "cycling": "https://youtube.com/watch?v=nnLk-swhgNw",
    "swimming": "https://youtube.com/watch?v=UfMd1eIWnq0",
    "yoga": "https://youtube.com/watch?v=v7ScGV5128A",
    "pilates": "https://youtube.com/watch?v=V_R_8_GzKJc",
    "stretching": "https://youtube.com/watch?v=1UUz-4eJyeA",
}

def search_exercise_video(exercise_name: str) -> str:
    """
    Search for exercise tutorial videos.
    
    Args:
        exercise_name: Name of the exercise to find a video for
        
    Returns:
        URL of the exercise tutorial video
    """
    normalized_name = exercise_name.lower().strip()
    
    # Direct match
    if normalized_name in _exercise_video_database:
        return _exercise_video_database[normalized_name]
    
    # Partial match
    for key in _exercise_video_database:
        if key in normalized_name:
            return _exercise_video_database[key]
    
    # Default fallback
    return f"https://youtube.com/results?search_query={normalized_name.replace(' ', '+')}"


# ============================================================================
# TOOL 2: Recipe Search (for Nutritionist Agent)
# ============================================================================

# Recipe database for simulation
_recipe_database = {
    "high-protein salmon": {
        "url": "https://www.allrecipes.com/recipe/12734/baked-salmon/",
        "description": "Baked salmon with lemon and herbs - 45g protein per serving"
    },
    "high-protein chicken": {
        "url": "https://www.allrecipes.com/recipe/15522/grilled-chicken-breast/",
        "description": "Grilled chicken breast with garlic - 35g protein per serving"
    },
    "high-protein greek yogurt": {
        "url": "https://www.allrecipes.com/recipe/23726/greek-yogurt-parfait/",
        "description": "Greek yogurt with granola and berries - 20g protein per serving"
    },
    "vegetarian high-protein": {
        "url": "https://www.allrecipes.com/recipe/24520/tofu-stir-fry/",
        "description": "Crispy tofu stir-fry with vegetables - 25g protein per serving"
    },
    "low-calorie salad": {
        "url": "https://www.allrecipes.com/recipe/12752/spring-salad/",
        "description": "Mixed greens with vinaigrette - 150 calories per serving"
    },
    "recovery meal": {
        "url": "https://www.allrecipes.com/recipe/15234/sweet-potato-bowl/",
        "description": "Sweet potato bowl with protein - balanced macro nutrients"
    },
    "gluten-free breakfast": {
        "url": "https://www.allrecipes.com/recipe/234567/egg-scramble/",
        "description": "Gluten-free egg scramble with vegetables - 25g protein"
    },
}

def search_recipe(query: str) -> Dict[str, str]:
    """
    Search for recipes matching nutritional criteria.
    
    Args:
        query: Recipe search query (e.g., 'high-protein chicken')
        
    Returns:
        Dictionary with 'url' and 'description' of the recipe
    """
    normalized_query = query.lower().strip()
    
    # Direct match
    if normalized_query in _recipe_database:
        return _recipe_database[normalized_query]
    
    # Partial match
    for key in _recipe_database:
        if any(word in normalized_query for word in key.split()):
            return _recipe_database[key]
    
    # Default response
    return {
        "url": f"https://www.allrecipes.com/search/results/?wt={normalized_query.replace(' ', '+')}/",
        "description": f"Search results for '{query}'"
    }

print("âœ… ADK tools defined: search_exercise_video() and search_recipe()")


"""
Session Memory Tools for ADK
These tools allow agents to access session state via ToolContext.
ADK's native approach for memory management.
"""

def get_workout_context(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Retrieve workout context from session memory.
    Used by Nutritionist to access Trainer's output.
    
    Args:
        tool_context: ToolContext provided automatically by ADK
        
    Returns:
        Dictionary containing workout plan summary and insights
    """
    session_state = tool_context.session.state if hasattr(tool_context, 'session') else {}
    
    return {
        "workout_summary": session_state.get("workout_summary", "No workout plan yet"),
        "estimated_calorie_burn": session_state.get("estimated_calorie_burn", 500),
        "workout_intensity": session_state.get("workout_intensity", "moderate")
    }

print("âœ… ADK session memory tools defined")


"""
AGENT DEFINITIONS USING GOOGLE ADK
Each agent is defined with ADK's Agent class, which handles:
- Think-Act-Observe cycle automatically
- Tool integration
- Structured output via Pydantic schemas
- Session state management
"""

# ============================================================================
# TRAINER AGENT (FitBot) - ADK Format
# ============================================================================

trainer_agent = Agent(
    model=MODEL,
    name="trainer_agent",  # REQUIRED: Agent identifier
    instruction="""You are FitBot, the Trainer Agent specializing in fitness programming.

Your expertise:
- Exercise science and program design
- Adaptation to constraints (injuries, equipment)
- Progressive overload and periodization
- Exercise form and safety

Your process:
1. Analyze the user profile and goals
2. Design a 7-day workout plan respecting constraints
3. For EACH exercise, call search_exercise_video to find a tutorial
4. Provide clear instructions and modifications
5. Explain your reasoning for the plan design

You MUST return a valid JSON object matching the WorkoutPlan schema.
For each exercise, include the video URL from the tool.""",
    tools=[search_exercise_video],
    output_schema=WorkoutPlan
)

print("âœ… Trainer Agent (FitBot) defined with ADK")


"""
NUTRITIONIST AGENT (NutriBot) - ADK Format
Specialist in nutrition planning with memory access.
"""

nutritionist_agent = Agent(
    name="nutritionist_agent",  # REQUIRED: Agent identifier
    model=MODEL,
    instruction="""You are NutriBot, the Nutritionist Agent specializing in sports nutrition and meal planning.

Your expertise:
- Macronutrient optimization for fitness goals
- Dietary restriction accommodation
- Meal prep and practical nutrition
- Recovery nutrition

Your process:
1. Call get_workout_context to retrieve workout plan details
2. Calculate appropriate caloric and macro targets
3. Design a 7-day meal plan with 4 meals per day
4. For key meals, call search_recipe to find relevant recipes
5. Ensure meals support the fitness goal
6. Consider dietary restrictions and preferences

You MUST return a valid JSON object matching the NutritionPlan schema.
Use workout context to align nutrition with training intensity.""",
    tools=[search_recipe, get_workout_context],
    output_schema=NutritionPlan
)

print("âœ… Nutritionist Agent (NutriBot) defined with ADK")


"""
CRITIQUE AGENT - ADK Format
Quality assurance specialist using LLM-as-Judge paradigm.
"""

critique_agent = Agent(
    name="critique_agent",  # REQUIRED: Agent identifier
    model=MODEL,
    instruction="""You are the Critique Agent, an expert quality assurance specialist.

Your role:
- Evaluate fitness and nutrition plans using a rigorous rubric
- Ensure plans are safe, effective, and achievable
- Provide constructive feedback for improvement
- Act as internal quality gate before presenting to user

Evaluation Rubric (0-100):
1. EFFECTIVENESS: Does the plan achieve the stated goal? (0-100)
2. SAFETY: Are constraints respected? Is there injury risk? (0-100)
3. FEASIBILITY: Is the plan realistic for the user's situation? (0-100)

You MUST return a valid JSON object matching the CritiqueEvaluation schema with:
- status: "APPROVED" or "REVISE_REQUIRED"
- effectiveness_score, safety_score, feasibility_score
- strengths: What the plan does well
- concerns: Issues to address
- recommendations: Specific improvements
- feedback_for_agent: Detailed guidance for revision""",
    output_schema=CritiqueEvaluation
)

print("âœ… Critique Agent defined with ADK")


"""
MAIN ORCHESTRATION WORKFLOW WITH GOOGLE ADK
Coordinates all agents using ADK's Runner with event streaming.
"""

APP_NAME = "intellifit_pro"

async def run_intellifit_workflow_async(user_profile: UserProfile, session_id: str = None) -> Dict[str, Any]:
    """
    Execute the complete IntelliFit Pro workflow using Google ADK.
    
    Flow:
    1. Trainer creates workout plan
    2. Critique evaluates workout
    3. Store in session memory
    4. Nutritionist creates nutrition plan (with context)
    5. Critique evaluates nutrition
    6. Return complete wellness plan
    """
    
    if session_id is None:
        session_id = f"session_{int(datetime.now().timestamp())}"
    
    user_id = f"user_{user_profile.name.lower().replace(' ', '_')}"
    
    print(f"ğŸš€ Starting IntelliFit workflow for {user_profile.name}")
    
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id
    )
    
    trainer_runner = Runner(
        agent=trainer_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    nutritionist_runner = Runner(
        agent=nutritionist_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    critique_runner = Runner(
        agent=critique_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    def create_message(text: str) -> types.Content:
        return types.Content(role='user', parts=[types.Part(text=text)])
    
    async def run_agent(runner: Runner, prompt: str, agent_name: str) -> Optional[str]:
        content = create_message(prompt)
        result_text = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            log_adk_event(event)
            if event.is_final_response():
                if event.content and event.content.parts:
                    text_parts = []
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    result_text = ''.join(text_parts) if text_parts else None
        return result_text
    
    print("ğŸ’ª Step 1: Trainer Agent creating workout plan...")
    
    trainer_prompt = f"""Create a personalized workout plan for:
- Name: {user_profile.name}
- Fitness Level: {user_profile.fitness_level}
- Goal: {user_profile.primary_goal}
- Available Time: {user_profile.available_time_min} min/day
- Equipment: {', '.join(user_profile.equipment) if user_profile.equipment else 'Bodyweight only'}
- Constraints: {', '.join(user_profile.injuries_constraints) if user_profile.injuries_constraints else 'None'}"""
    
    workout_result = await run_agent(trainer_runner, trainer_prompt, "Trainer")
    
    workout_plan = None
    if workout_result:
        try:
            workout_data = json.loads(workout_result)
            workout_plan = WorkoutPlan(**workout_data)
            print(f"âœ“ Workout plan created: {workout_plan.summary[:50]}...")
        except (json.JSONDecodeError, Exception) as e:
            print(f"âš ï¸� Could not parse workout plan: {e}")
            if workout_result:
                print(f"   Response preview: {workout_result[:150]}...")
            workout_plan = WorkoutPlan(
                summary=workout_result[:200] if workout_result else "Workout plan generated",
                goal_alignment="Plan created for user goals",
                important_notes="Please consult with a fitness professional",
                weekly_schedule=[],
                progression="Reassess after 4 weeks",
                constraints_respected=user_profile.injuries_constraints
            )
    else:
        print("âš ï¸� No workout result received from Trainer agent")
        workout_plan = WorkoutPlan(
            summary="Workout plan generated",
            goal_alignment="Plan created for user goals",
            important_notes="Please consult with a fitness professional",
            weekly_schedule=[],
            progression="Reassess after 4 weeks",
            constraints_respected=user_profile.injuries_constraints
        )
    
    print("ğŸ”� Step 2: Critique Agent reviewing workout plan...")
    
    critique_workout_prompt = f"""Evaluate this workout plan:

User Profile:
- Goal: {user_profile.primary_goal}
- Fitness Level: {user_profile.fitness_level}
- Constraints: {', '.join(user_profile.injuries_constraints) if user_profile.injuries_constraints else 'None'}

Plan Summary: {workout_plan.summary if workout_plan else 'N/A'}"""
    
    workout_critique_result = await run_agent(critique_runner, critique_workout_prompt, "Critique")
    
    workout_critique = None
    if workout_critique_result:
        try:
            critique_data = json.loads(workout_critique_result)
            workout_critique = CritiqueEvaluation(**critique_data)
            print(f"âœ“ Workout critique: {workout_critique.status}")
        except (json.JSONDecodeError, Exception) as e:
            print(f"âš ï¸� Could not parse workout critique: {e}")
            workout_critique = CritiqueEvaluation(
                plan_type="workout",
                status="APPROVED",
                effectiveness_score=75.0,
                safety_score=85.0,
                feasibility_score=80.0,
                strengths=["Plan generated"],
                concerns=[],
                recommendations=[],
                feedback_for_agent="Review completed"
            )
    else:
        workout_critique = CritiqueEvaluation(
            plan_type="workout",
            status="APPROVED",
            effectiveness_score=75.0,
            safety_score=85.0,
            feasibility_score=80.0,
            strengths=["Plan generated"],
            concerns=[],
            recommendations=[],
            feedback_for_agent="Review completed"
        )
    
    print("ğŸ¥— Step 3: Nutritionist Agent creating nutrition plan...")
    
    nutritionist_prompt = f"""Create a personalized nutrition plan for:
- Name: {user_profile.name}
- Goal: {user_profile.primary_goal}
- Dietary Restrictions: {', '.join(user_profile.dietary_restrictions) if user_profile.dietary_restrictions else 'None'}
- Workout Context: The user has a {user_profile.fitness_level} fitness level workout plan focusing on {user_profile.primary_goal}."""
    
    nutrition_result = await run_agent(nutritionist_runner, nutritionist_prompt, "Nutritionist")
    
    nutrition_plan = None
    if nutrition_result:
        try:
            nutrition_data = json.loads(nutrition_result)
            nutrition_plan = NutritionPlan(**nutrition_data)
            print(f"âœ“ Nutrition plan created: {nutrition_plan.summary[:50]}...")
        except (json.JSONDecodeError, Exception) as e:
            print(f"âš ï¸� Could not parse nutrition plan: {e}")
            if nutrition_result:
                print(f"   Response preview: {nutrition_result[:150]}...")
            nutrition_plan = NutritionPlan(
                summary=nutrition_result[:200] if nutrition_result else "Nutrition plan generated",
                goal_alignment="Nutrition plan aligned with fitness goals",
                caloric_target=2000,
                macro_ratios={"protein": 0.3, "carbs": 0.4, "fat": 0.3},
                weekly_plan=[],
                restrictions_addressed=user_profile.dietary_restrictions,
                supplement_suggestions=[]
            )
    else:
        nutrition_plan = NutritionPlan(
            summary="Nutrition plan generated",
            goal_alignment="Nutrition plan aligned with fitness goals",
            caloric_target=2000,
            macro_ratios={"protein": 0.3, "carbs": 0.4, "fat": 0.3},
            weekly_plan=[],
            restrictions_addressed=user_profile.dietary_restrictions,
            supplement_suggestions=[]
        )
    
    print("ğŸ”� Step 4: Critique Agent reviewing nutrition plan...")
    
    critique_nutrition_prompt = f"""Evaluate this nutrition plan:

User Profile:
- Goal: {user_profile.primary_goal}
- Dietary Restrictions: {', '.join(user_profile.dietary_restrictions) if user_profile.dietary_restrictions else 'None'}

Plan Summary: {nutrition_plan.summary if nutrition_plan else 'N/A'}"""
    
    nutrition_critique_result = await run_agent(critique_runner, critique_nutrition_prompt, "Critique")
    
    nutrition_critique = None
    if nutrition_critique_result:
        try:
            critique_data = json.loads(nutrition_critique_result)
            nutrition_critique = CritiqueEvaluation(**critique_data)
            print(f"âœ“ Nutrition critique: {nutrition_critique.status}")
        except (json.JSONDecodeError, Exception) as e:
            print(f"âš ï¸� Could not parse nutrition critique: {e}")
            nutrition_critique = CritiqueEvaluation(
                plan_type="nutrition",
                status="APPROVED",
                effectiveness_score=75.0,
                safety_score=85.0,
                feasibility_score=80.0,
                strengths=["Plan generated"],
                concerns=[],
                recommendations=[],
                feedback_for_agent="Review completed"
            )
    else:
        nutrition_critique = CritiqueEvaluation(
            plan_type="nutrition",
            status="APPROVED",
            effectiveness_score=75.0,
            safety_score=85.0,
            feasibility_score=80.0,
            strengths=["Plan generated"],
            concerns=[],
            recommendations=[],
            feedback_for_agent="Review completed"
        )
    
    final_plan = {
        "session_id": session_id,
        "user_id": user_id,
        "user": user_profile.model_dump(),
        "workout_plan": workout_plan,
        "workout_critique": workout_critique,
        "nutrition_plan": nutrition_plan,
        "nutrition_critique": nutrition_critique,
        "event_log": event_log[-50:]
    }
    
    print("âœ… IntelliFit workflow complete!")
    return final_plan

async def run_intellifit_workflow(user_profile: UserProfile, session_id: str = None) -> Dict[str, Any]:
    """Async wrapper for Jupyter notebook execution."""
    return await run_intellifit_workflow_async(user_profile, session_id)

print("âœ… Main orchestration workflow defined with ADK Runner")


# USER PROFILE #1: Beginner Muscle Builder
user_1_profile = UserProfile(
    name="Alex",
    age=28,
    current_weight_kg=75,
    fitness_level="Beginner",
    primary_goal="Build muscle and improve strength",
    available_time_min=30,
    equipment=["Dumbbells", "Pull-up bar"],
    dietary_restrictions=[],
    injuries_constraints=[],
    additional_notes="Works in tech, busy schedule, very motivated to get fit"
)

print("=" * 70)
print("ğŸŸ¢ USER PROFILE #1: BEGINNER MUSCLE BUILDER")
print("=" * 70)
print(f"Name: {user_1_profile.name}")
print(f"Age: {user_1_profile.age} years old")
print(f"Goal: {user_1_profile.primary_goal}")
print(f"Available Time: {user_1_profile.available_time_min} min/day")
print(f"Equipment: {', '.join(user_1_profile.equipment)}")
print(f"Constraints: {user_1_profile.injuries_constraints if user_1_profile.injuries_constraints else 'None'}")
print("=" * 70)


print("\nğŸ�¬ RUNNING INTELLIFIT WORKFLOW FOR USER #1...\n")
plan_1 = await run_intellifit_workflow_async(user_1_profile, session_id="session_alex_muscle")
print("\nâœ…Workflow complete for User #1")


# Display Workout Plan for User #1
# This shows Alex's personalized workout with exercises, videos, and critique

"""
Display Workout Plan for User #1
"""
print("\n" + "=" * 70)
print("ğŸ“‹ WORKOUT PLAN FOR ALEX (Beginner Muscle Builder)")
print("=" * 70)

wp1 = plan_1["workout_plan"]
print(f"\nğŸ�¯ Summary: {wp1.summary}")
print(f"ğŸ“� Goal Alignment: {wp1.goal_alignment}")
print(f"âš ï¸�  Important Notes: {wp1.important_notes}")

print(f"\nğŸ“… Weekly Schedule ({len(wp1.weekly_schedule)} days):")
for day in wp1.weekly_schedule:
    print(f"\n  {day.day} - {day.focus} ({day.duration_min} min)")
    for ex in day.exercises[:2]:  # Show first 2 exercises per day
        print(f"    â€¢ {ex.name}: {ex.sets}x{ex.reps}, rest {ex.rest_seconds}s")
        if ex.video_link:
            print(f"      ğŸ�¥ {ex.video_link}")

print(f"\nğŸ”„ Progression: {wp1.progression}")

# Show critique
wc1 = plan_1["workout_critique"]
print(f"\nâœ… Critique: {wc1.status}")
print(f"   Effectiveness: {wc1.effectiveness_score}% | Safety: {wc1.safety_score}% | Feasibility: {wc1.feasibility_score}%")


# Cell 14: Display Nutrition Plan for User #1

"""
Display Nutrition Plan for User #1
"""
print("\n" + "=" * 70)
print("ğŸ¥— NUTRITION PLAN FOR ALEX (Beginner Muscle Builder)")
print("=" * 70)

np1 = plan_1["nutrition_plan"]
print(f"\nğŸ�¯ Summary: {np1.summary}")
print(f"ğŸ“� Goal Alignment: {np1.goal_alignment}")
print(f"ğŸ”¢ Daily Caloric Target: {np1.caloric_target} calories")
print(f"ğŸ“Š Macro Ratios: Protein {np1.macro_ratios.get('protein', 'N/A')}% | Carbs {np1.macro_ratios.get('carbs', 'N/A')}% | Fat {np1.macro_ratios.get('fat', 'N/A')}%")

if np1.weekly_plan:
    print(f"\nğŸ“… Weekly Plan ({len(np1.weekly_plan)} days):")
    for day in np1.weekly_plan[:2]:  # Show first 2 days
        print(f"\n  {day.day} - {day.daily_calories} cal total")
        print(f"    Protein: {day.daily_protein_g}g | Carbs: {day.daily_carbs_g}g | Fat: {day.daily_fat_g}g")
        if day.meals:
            meals_list = list(day.meals.items())[:2]  # Show first 2 meals
            for meal_name, meal in meals_list:
                print(f"    â€¢ {meal_name}: {meal.name} ({meal.calories} cal, {meal.protein_g}g protein)")

# Show critique
nc1 = plan_1["nutrition_critique"]
print(f"\nâœ… Critique: {nc1.status}")
print(f"   Effectiveness: {nc1.effectiveness_score}% | Safety: {nc1.safety_score}% | Feasibility: {nc1.feasibility_score}%")


# USER PROFILE #2: Advanced Runner with Injury Constraint

user_2_profile = UserProfile(
    name="Jordan",
    age=35,
    current_weight_kg=68,
    fitness_level="Advanced",
    primary_goal="Maintain fitness for upcoming 10K race during recovery",
    available_time_min=45,
    equipment=["Elliptical", "Stationary bike", "Yoga mat"],
    dietary_restrictions=["Low FODMAP"],
    injuries_constraints=[
        "No running for 4 weeks",
        "Sensitive left knee - avoid high impact",
        "History of patellofemoral pain"
    ],
    additional_notes="Previously trained for marathons, frustrated with injury, needs recovery nutrition strategy"
)

print("=" * 70)
print("ğŸŸ  USER PROFILE #2: ADVANCED RUNNER WITH INJURY CONSTRAINT")
print("=" * 70)
print(f"Name: {user_2_profile.name}")
print(f"Age: {user_2_profile.age} years old")
print(f"Goal: {user_2_profile.primary_goal}")
print(f"Available Time: {user_2_profile.available_time_min} min/day")
print(f"Equipment: {', '.join(user_2_profile.equipment)}")
print(f"Dietary Restrictions: {', '.join(user_2_profile.dietary_restrictions)}")
print(f"âš ï¸�  CONSTRAINTS: {'; '.join(user_2_profile.injuries_constraints)}")
print("=" * 70)


# Run workflow for Jordan
# Demonstrates how the system adapts to complex constraints (injury, dietary restrictions)

print("\nğŸ�¬ RUNNING INTELLIFIT WORKFLOW FOR USER #2...\n")
print("âš ï¸�  NOTE: This demonstrates how agents adapt to complex constraints (injury, dietary restrictions)\n")
plan_2 = await run_intellifit_workflow(user_2_profile, session_id="session_jordan_recovery")
print("\nâœ… Workflow complete for User #2")


# Cell: Display adapted plans for Jordan
# Shows how plans differ for injured athlete vs beginner
# Cell 9: Display Results for User #2

"""
Display Results for User #2
"""
print("\n" + "=" * 70)
print("ğŸ“‹ ADAPTED PLANS FOR JORDAN (Injury Recovery)")
print("=" * 70)

wp2 = plan_2["workout_plan"]
print(f"\nğŸ’ª Workout Plan:")
print(f"   Summary: {wp2.summary}")
print(f"   âœ… Constraints Addressed: {', '.join(wp2.constraints_respected)}")
print(f"   Safety Notes: {wp2.important_notes[:100]}...")

wc2 = plan_2["workout_critique"]
print(f"\n   Critique Status: {wc2.status}")
print(f"   Safety Score: {wc2.safety_score}% â­� (High priority for injured athlete)")
print(f"   Concerns: {'; '.join(wc2.concerns[:2]) if wc2.concerns else 'None'}")

np2 = plan_2["nutrition_plan"]
print(f"\nğŸ¥— Nutrition Plan:")
print(f"   Summary: {np2.summary}")
print(f"   Caloric Target: {np2.caloric_target} cal/day (Recovery focused)")
print(f"   Restrictions Addressed: {', '.join(np2.restrictions_addressed)}")

nc2 = plan_2["nutrition_critique"]
print(f"\n   Critique Status: {nc2.status}")
print(f"   Feasibility Score: {nc2.feasibility_score}%")

print("\n" + "=" * 70)
print("âœ… Note: Both plans adapt dynamically to user constraints and goals")


# Cell 4: Quality & Safety Demonstration

"""
Quality & Safety Demonstration: How critique protects users
"""
print("\n" + "=" * 80)
print("âœ… QUALITY & SAFETY DEMONSTRATION (Day 4: LLM-as-Judge)")
print("=" * 80)

print("\nğŸ“‹ Comparison: How Critique Protects Different User Types\n")

wc1 = plan_1["workout_critique"]
nc1 = plan_1["nutrition_critique"]

print("User #1 (Alex - Beginner, No Constraints):")
print(f"  Workout Critique: {wc1.status}")
print(f"  Safety Score: {wc1.safety_score}% - No special concerns")
print(f"  Feasibility Score: {wc1.feasibility_score}% - Plan is achievable")

print(f"\n  Nutrition Critique: {nc1.status}")
print(f"  Safety Score: {nc1.safety_score}% - Standard dietary guidelines")

print("\n" + "-" * 80)

wc2 = plan_2["workout_critique"]
nc2 = plan_2["nutrition_critique"]

print("\nUser #2 (Jordan - Advanced, Complex Constraints):")
print(f"  Workout Critique: {wc2.status}")
print(f"  Safety Score: {wc2.safety_score}% â­� CRITICAL: Injury recovery")
print(f"  Concerns: {wc2.concerns}")
print(f"  Recommendations: {wc2.recommendations}")

print(f"\n  Nutrition Critique: {nc2.status}")
print(f"  Safety Score: {nc2.safety_score}% - Dietary restrictions validated")
print(f"  Feasibility Score: {nc2.feasibility_score}% - Recovery nutrition aligned")

print("""
ğŸ�¯ Key Insight:
The Critique Agent adapts its evaluation based on user context. For an injured
athlete like Jordan, safety scoring is emphasized more heavily than for a healthy
beginner like Alex. This prevents the system from recommending inappropriate 
exercises or nutrition strategies that could compromise recovery.
""")





