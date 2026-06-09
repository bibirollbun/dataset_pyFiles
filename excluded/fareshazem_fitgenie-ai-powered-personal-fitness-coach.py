# Core ADK agent classes
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent, Agent

# Gemini LLM model
from google.adk.models.google_llm import Gemini

# Agent execution and state management
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

# Content and message types
from google.genai.types import Content, Part
from google.genai import types

# Agent tools and capabilities
from google.adk.tools import google_search, FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.events import Event

# Standard libraries
import asyncio
import requests
import time
import os
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

# Display libraries
from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

# Google OAuth and authentication
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2 import id_token
from google.auth.transport import requests

# Kaggle secrets for API keys
from kaggle_secrets import UserSecretsClient

print("âœ… ADK components imported successfully.")


secrets = UserSecretsClient()

# GOOGLE_API_KEY
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )
# GOOGLE_PROJECT_ID
try:
    GOOGLE_PROJECT_ID = secrets.get_secret("GOOGLE_PROJECT_ID")
    os.environ["GOOGLE_PROJECT_ID"] = GOOGLE_PROJECT_ID
    print("âœ… GOOGLE_PROJECT_ID setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please add 'GOOGLE_PROJECT_ID' to your Kaggle secrets. Details: {e}")

# GOOGLE_OAUTH_CLIENT_ID
try:
    GOOGLE_OAUTH_CLIENT_ID = secrets.get_secret("GOOGLE_OAUTH_CLIENT_ID")
    os.environ["GOOGLE_OAUTH_CLIENT_ID"] = GOOGLE_OAUTH_CLIENT_ID
    print("âœ… GOOGLE_OAUTH_CLIENT_ID setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please add 'GOOGLE_OAUTH_CLIENT_ID' to your Kaggle secrets. Details: {e}")

# GOOGLE_OAUTH_CLIENT_SECRET
try:
    GOOGLE_OAUTH_CLIENT_SECRET = secrets.get_secret("GOOGLE_OAUTH_CLIENT_SECRET")
    os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = GOOGLE_OAUTH_CLIENT_SECRET
    print("âœ… GOOGLE_OAUTH_CLIENT_SECRET setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please add 'GOOGLE_OAUTH_CLIENT_SECRET' to your Kaggle secrets. Details: {e}")

# YOUTUBE_API_KEY
try:
    YOUTUBE_API_KEY = secrets.get_secret("YOUTUBE_API_KEY")
    os.environ["YOUTUBE_API_KEY"] = YOUTUBE_API_KEY
    print("âœ… YOUTUBE_API_KEY setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please add 'YOUTUBE_API_KEY' to your Kaggle secrets. Details: {e}")


# Global shared services for session and memory management
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# Factory function to create agent runners with shared services
def make_runner(agent, app_name="fitgenie_app"):
    return Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
        memory_service=memory_service
    )


# Retry configuration for API requests with exponential backoff
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


# Agent responsible for generating personalized workout plans
plan_agent = Agent(
    name="WeeklyPlannerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
        You are an expert, evidence-based Fitness Planner. Your sole task is to generate a highly customized workout routine using the user's existing profile (weight, height, age, training status, sex, goal, weekly frequency/availability, and health conditions).
        Follow these mandatory guidelines:
        
        1. **Data Retrieval & Verification:**
            - Assume the user profile data is provided and accurate.
            - Use the **google_search** tool to verify exercise safety if the user has specific injuries (e.g., "exercises to avoid with meniscus tear").
            - **MANDATORY:** You must use the **youtube_search** tool for every distinct exercise in the plan to fetch a high-quality demonstration video. Search query format: "[Exercise Name] proper form" or "[Exercise Name] tutorial".
        
        2. **Safety & Customization (CRITICAL):**
            - Exclude any exercise that could worsen listed injuries or conditions.
            - If the user is a beginner or returning after a break, the first 1â€“2 sessions MUST be low-intensity, focusing on form, stability, and mobility.
            - Adapt intensity (RPE) and volume (sets/reps) to their specific goal.
        
        3. **Plan Frequency Logic:**
            - If the user provided a number (e.g., "3 days"), follow it precisely.
            - If the user asked for the "optimal plan" or "best results," determine the ideal frequency based on their goal/status and explain your reasoning.
        
        4. **Output Format:**
            - The output must be clean Markdown.
            - Structure the routine by Day/Session.
            - For every exercise, follow this EXACT format:
              `* **[Exercise Name]** [[Watch Demo](URL_FROM_TOOL)] - [Sets] x [Reps] (Rest: [Time])`
            - *Example:* `* **Goblet Squat** [[Watch Demo](https://youtube.com/...)] - 3 x 12 (Rest: 60s)`
            - Add a brief 1-sentence cue for form under complex lifts.
        
        5. **Constraints:**
            - Do NOT ask for profile data again if it is already present.
            - If the profile data is completely missing (no inputs provided at all), output a specific error asking for: Biometrics, Training Status, Goal, Availability, Injuries, and Equipment.
        
        6. **Ending:**
            - After the complete plan, ask:
              **"Do you approve this plan or would you like to request specific changes?"**
    """,
    tools=[google_search],
    output_key="current_plan"
)


# OpenFoodFacts API base URL
OFF_API_BASE = "https://world.openfoodfacts.org"

# Search OpenFoodFacts database for product nutrition info
def query_off_by_name(food_name: str) -> Optional[Dict[str, Any]]:
    """
    Search OpenFoodFacts for products matching food_name; return first result's nutrition info (per 100g / typical serving).
    """
    url = f"{OFF_API_BASE}/cgi/search.pl"
    params = {
        "search_terms": food_name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 5,
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return None
    data = resp.json()
    products = data.get("products", [])
    if not products:
        return None
    prod = products[0]
    nutr = prod.get("nutriments", None)
    if not nutr:
        return None
    return {
        "product_name": prod.get("product_name", food_name),
        "serving_size": prod.get("serving_size"),
        "serving_quantity": prod.get("serving_quantity"),
        "nutriments": nutr
    }

# Extract macros (protein, carbs, fat, calories) from food item
def get_nutrition_of(food_name: str) -> Optional[Dict[str, float]]:
    """
    Returns a dict with approximate macros/calories for food_name.
    If OFF lookup fails, returns None.
    """
    info = query_off_by_name(food_name)
    if not info:
        return None
    nutr = info["nutriments"]
    calories = nutr.get("energy-kcal_100g") or nutr.get("energy_100g") or nutr.get("energy-kcal_serving")
    protein = nutr.get("proteins_100g") or nutr.get("proteins_serving")
    carbs   = nutr.get("carbohydrates_100g") or nutr.get("carbohydrates_serving")
    fat     = nutr.get("fat_100g") or nutr.get("fat_serving")
    if calories is None or protein is None or carbs is None or fat is None:
        return None
    return {
        "calories_100g": calories,
        "protein_100g": protein,
        "carbs_100g": carbs,
        "fat_100g": fat
    }


# def compute_nutrition_by_profile(
#     weight_kg: float,
#     height_cm: float,
#     age: int,
#     sex: str,
#     activity_multiplier: float,
#     goal: str = "maintenance",
#     surplus_pct: float = 0.15,
#     deficit_pct: float = 0.15,
#     macro_ratio: Dict[str, float] | None = None,
# ) -> Dict[str, float]:
#     """
#     Fallback estimator (if food DB data not used) â€” returns daily calories and macros targets.
#     """
#     sex_l = sex.lower()
#     if sex_l == "male":
#         bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
#     else:
#         bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

#     maintenance = bmr * activity_multiplier

#     if goal == "muscle_gain":
#         target_cal = maintenance * (1 + surplus_pct)
#     elif goal == "fat_loss":
#         target_cal = maintenance * (1 - deficit_pct)
#     else:
#         target_cal = maintenance

#     if macro_ratio is None:
#         macro_ratio = {"protein": 0.30, "carbs": 0.50, "fat": 0.20}

#     protein = (target_cal * macro_ratio["protein"]) / 4
#     carbs   = (target_cal * macro_ratio["carbs"  ]) / 4
#     fat     = (target_cal * macro_ratio["fat"    ]) / 9

#     return {
#         "calories_per_day": round(target_cal),
#         "protein_g_day": round(protein),
#         "carbs_g_day": round(carbs),
#         "fat_g_day":   round(fat),
#     }
# food = "rice"
# info = get_nutrition_of(food)
# if info:
#     print("Nutrition per 100g of", food, ":", info)
# else:
#     print("No OFF data found for:", food)

# profile_plan = compute_nutrition_by_profile(
#     weight_kg=83, height_cm=183, age=21, sex="male",
#     activity_multiplier=1.55, goal="muscle_gain"
# )
# print("Estimated daily macros:", profile_plan)


# Wrap nutrition lookup function as an ADK tool
nutrition_tool = FunctionTool(get_nutrition_of)


# Agent responsible for creating personalized nutrition plans
nutrition_agent = Agent(
    name="NutritionAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
        You are FitGenie Nutrition Agent.
        You receive the user's fitness profile **and** the user's workout plan (from the root orchestrator; referred to as "current_plan").  
        Use these inputs to compute a nutrition plan tailored to both:  
        - Daily caloric intake (based on BMR, activity level, and training load from current_plan),  
        - Macro distribution (protein, carbs, fat) suitable for goal and training volume,  
        - Optionally â€” suggest how macros/calories may vary on training vs rest days according to the workout schedule in current_plan.  
        - Do NOT ask the user for the workout plan. Assume the orchestrator always passes it as 'current_plan'. 

        Return the nutrition plan in structured JSON or markdown. Do not ask the user for the workout plan again â€” assume the orchestrator already passed it. If the workout plan is missing or invalid, respond with an error message asking the orchestrator to provide the missing plan.  
        """,
    tools=[nutrition_tool],
    description="Fetches nutrition info for foods or computes daily macros."
)


# # --- dependencies ---
# !pip install --quiet google-api-python-client google-auth-httplib2 google-auth-oauthlib

# from googleapiclient.discovery import build
# from IPython.display import YouTubeVideo, display, Markdown

# # --- configure YouTube API ---
# YOUTUBE_API_SERVICE_NAME = "youtube"
# YOUTUBE_API_VERSION = "v3"

# def search_youtube_video_id(query, max_results=5):
#     """
#     Search YouTube for a given query (e.g. exercise name + 'tutorial') and
#     return the first videoId (if any).
#     """
#     youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
#                     developerKey=YOUTUBE_API_KEY)
#     request = youtube.search().list(
#         q=query,
#         part="snippet",
#         type="video",
#         maxResults=max_results
#     )
#     response = request.execute()
#     items = response.get("items", [])
#     for item in items:
#         if item["id"]["kind"] == "youtube#video":
#             return item["id"]["videoId"]
#     return None

# def embed_video(video_id, width=480, height=270):
#     """Display a YouTube video in the notebook via its video_id."""
#     if video_id:
#         display(YouTubeVideo(video_id, width=width, height=height))

# def generate_and_display_plan_with_videos(user_profile):
#     """
#     Example stub: Replace with your actual agent call / plan generation.
#     Suppose this returns a dict: 
#       { "Day 1": [ {"exercise": "...", "sets": 3, "reps": 8}, ... ],
#         "Day 2": [ ... ],
#         ... }
#     """
#     # Example dummy plan (replace with call to WeeklyPlannerAgent)
#     plan = {
#         "Day 1": [
#             {"exercise": "Bodyweight Squat", "sets": 3, "reps": 8},
#             {"exercise": "Push-up", "sets": 3, "reps": 8},
#             {"exercise": "Plank", "sets": 2, "duration_sec": 30},
#         ],
#         "Day 2": [
#             {"exercise": "Deadlift with light dumbbells", "sets": 3, "reps": 6},
#             {"exercise": "Bent-over Row with dumbbells", "sets": 3, "reps": 8},
#             {"exercise": "Bird Dog", "sets": 2, "reps": 10},
#         ]
#     }
#     # --- display plan + videos ---
#     for day, exercises in plan.items():
#         display(Markdown(f"## {day}"))
#         for ex in exercises:
#             ex_name = ex["exercise"]
#             sets = ex.get("sets")
#             reps = ex.get("reps")
#             duration = ex.get("duration_sec")
#             if sets and reps:
#                 display(Markdown(f"**{ex_name} â€” {sets} Ã— {reps}**"))
#             elif duration:
#                 display(Markdown(f"**{ex_name} â€” hold for {duration}s**"))
#             else:
#                 display(Markdown(f"**{ex_name}**"))
#             # search for tutorial video
#             q = f"{ex_name} exercise tutorial"
#             vid = search_youtube_video_id(q)
#             if vid:
#                 embed_video(vid)
#             else:
#                 display(Markdown("_No tutorial video found automatically â€” you may search manually_"))
#         display(Markdown("---\n"))

# # --- Example usage ---
# user_profile = {
#     "Weight: 83 kg\n"
#     "Height: 184 cm\n"
#     "Age: 21\n"
#     "Sex: Male\n"
#     "Training status: Returning after a break (rusty)\n"
#     "Goal: Gain muscle\n"
#     "Workout frequency: Optimal plan\n"
#     "Health conditions: None\n"
# }
# generate_and_display_plan_with_videos(user_profile)


# Check if YouTube video URL is accessible
def check_video_exists(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=3)
        return r.status_code == 200
    except:
        return False

# Search YouTube API for exercise tutorial videos
def search_youtube_video_id(query: str) -> str:
    """
    Searches YouTube for a tutorial video and returns the first videoId.
    Returns None if there is no valid result.
    """
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": 5,
        "key": YOUTUBE_API_KEY,
        "safeSearch": "moderate",
        "videoEmbeddable": "true",
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        if not items:
            return None
        return items[0]["id"]["videoId"]
    except Exception as e:
        print("YouTube API Error:", e)
        return None

# Fallback video URLs for common exercises
FALLBACK_VIDEOS = {
    "squat": "https://www.youtube.com/watch?v=aclHkVaku9U",
    "deadlift": "https://www.youtube.com/watch?v=1ZXobu7JvvE",
    "push up": "https://www.youtube.com/watch?v=_l3ySVKYVJ8",
    "bench press": "https://www.youtube.com/watch?v=rT7DgCr-3pg",
    "row": "https://www.youtube.com/watch?v=kBWAon7ItDw",
    "shoulder press": "https://www.youtube.com/watch?v=B-aVuyhvLHU",
    "plank": "https://www.youtube.com/watch?v=pSHjTRCQxIw",
    "bicep curl": "https://www.youtube.com/watch?v=ykJmrZ5v0Oo",
    "triceps extension": "https://www.youtube.com/watch?v=2-LAMcpzODU"
}

# Batch search for multiple exercises with fallback logic
def youtube_search_tool(exercises: List[str]) -> Dict[str, str]:
    """
    Returns: { "exercise_name": "verified_url" }
    """
    result = {}
    for name in exercises:
        query = f"{name} exercise tutorial"
        video_id = search_youtube_video_id(query)
        url = f"https://www.youtube.com/watch?v={video_id}" if video_id else None
        
        # Validate video URL
        if url and check_video_exists(url):
            result[name] = url
            continue
        
        # Try fallback for simplified exercise name
        simplified = name.split()[0]
        fallback = FALLBACK_VIDEOS.get(simplified.lower())
        if fallback:
            result[name] = fallback
            continue
        
        result[name] = "No video available"
    return result


youtube_tool = FunctionTool(youtube_search_tool)


youtube_agent = Agent(
    name="youtube_search_agent",
    model="gemini-2.5-flash",
    instruction="""
        You return demonstration video links for exercises.

        RULES:
        - ONLY call the youtube_search_tool with the list of exercise names.
        - NEVER guess or generate a URL manually.
        - NEVER search by yourself.
        - NEVER output text except the JSON result.

        Expected output format:
        {
            "videos": {
                "Exercise Name": "url",
                ...
            }
        }
    """,
    tools=[youtube_tool],
    description="Provides validated tutorial links for exercises."
)


# Timer for rest periods between exercises
def break_timer(seconds: int, exercise: str):
    """
    Break timer that streams countdown updates to the ADK interface.
    Replaces print() with yield events so the UI displays live updates.
    """
    for remaining in range(seconds, 0, -1):

        # Only send updates at 10s intervals OR the last 5 seconds
        if remaining % 10 == 0 or remaining <= 5:
            yield {
                "event": "break_timer_tick",
                "exercise": exercise,
                "seconds_left": remaining,
                "message": f"â�³ {exercise}: {remaining} seconds remaining..."
            }

        time.sleep(1)

    # Final message
    yield {
        "event": "break_timer_done",
        "exercise": exercise,
        "message": f"ğŸ”¥ {exercise} break finished!"
    }

    return {"status": "finished", "exercise": exercise}

break_timer_tool = FunctionTool(break_timer)

# Interactive step-by-step workout execution with timer
def step_by_step_workout_tool(workout_plan: dict, video_urls: dict):
    """
    Executes a single day's workout step-by-step.
    Handles warm-ups first, then main exercises.
    Calls break timer after each exercise.
    """
    results = []
    all_exercises = workout_plan.get("warmups", []) + workout_plan.get("exercises", [])
    
    for exercise in all_exercises:
        name = exercise["name"]
        sets = exercise["sets"]
        reps = exercise["reps"]
        rest = exercise.get("rest_seconds", 60)
        video = video_urls.get(name, "Video not found")
        
        print(f"\nExercise: {name}")
        print(f"Sets x Reps: {sets} x {reps}")
        print(f"Video Demo: {video}")
        
        user_input = input("Type 'done' when completed, or 'skip' to skip: ").strip().lower()
        
        if user_input == "skip":
            results.append({"exercise": name, "status": "skipped"})
            continue
        
        break_timer(rest, name)
        results.append({"exercise": name, "status": "completed"})
    
    return {"day": workout_plan.get("day"), "results": results}

step_guide_tool = FunctionTool(step_by_step_workout_tool)


MEMORY_FILE = "fitness_memory.json"

# Load memory from JSON file
def _load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"entries": []}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

# Save memory to JSON file
def _save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=4)

# Write new memory entry with timestamp
def memory_bank_write(category: str, data: dict) -> dict:
    """
    Store a memory event under 'category', with arbitrary data (dict).
    Returns status and stored entry.
    """
    mem = _load_memory()
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "category": category,
        "data": data
    }
    mem.setdefault("entries", []).append(entry)
    _save_memory(mem)
    return {"status": "success", "entry": entry}

# Read memory entries with optional category filter
def memory_bank_read(category: Optional[str] = None) -> dict:
    """
    Read memory entries. If category given, filters by that category.
    """
    mem = _load_memory()
    entries = mem.get("entries", [])
    if category:
        entries = [e for e in entries if e.get("category") == category]
    return {"entries": entries}

# Compact memory by keeping only recent entries
def memory_bank_compact(max_entries: int = 100) -> dict:
    """
    Optionally trim memory to keep only the most recent max_entries entries.
    """
    mem = _load_memory()
    entries = mem.get("entries", [])
    if len(entries) <= max_entries:
        return {"status": "unchanged", "count": len(entries)}
    new_entries = entries[-max_entries:]
    mem["entries"] = new_entries
    _save_memory(mem)
    return {"status": "compacted", "remaining": len(new_entries)}

# Wrap memory functions as ADK tools
memory_bank_write_tool = FunctionTool(memory_bank_write)
memory_bank_read_tool  = FunctionTool(memory_bank_read)
memory_bank_compact_tool = FunctionTool(memory_bank_compact)


# Agent responsible for tracking and analyzing long-term fitness progress
progress_memory_agent = Agent(
    name="ProgressMemoryAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
        You are the FitGenie Progress Memory Agent. You receive data each week and update the user's long-term fitness memory.  
        
        Inputs (via orchestration context):
        - completed_workouts: list describing which exercises were done/skipped
        - body_weight: numeric or string (e.g. kg)
        - performance_data: dict mapping exercise names â†’ performance notes (reps, difficulty, changes)
        - pain_reports: optional list describing pain/injury notes
        - current_plan: the workout plan JSON used this week
        
        Tasks:
        1. Store a memory entry summarizing the week:
           - adherence (percentage of workout days completed)
           - body weight
           - performance_changes (if any)
           - pain_reports (if any)
           - timestamp
        2. Analyze trends: compare with past weeks (if available) and note improvements or issues.
        3. Suggest modifications to next week's plan: for example, if pain was reported, mark for lighter load or alternative exercises. If performance improved consistently, suggest small progressive overload.  
        4. Return JSON with:
           - stored_entry (what was written)
           - trend_analysis (brief summary: improving / needs deload / pain alert)
           - plan_modifications (if any, list of suggested changes)
        
        You must use the memory_bank tools (write/read/compact) to persist and retrieve history.
        """,
    tools=[memory_bank_write_tool, memory_bank_read_tool, memory_bank_compact_tool],
)


# # Create runner instance for the root agent
# runner = make_runner(agent=progress_memory_agent)

# # Main async function to run the conversational interface
# async def chat():
#     user_id = "user1"; session_id = "session41"
#     await session_service.create_session(app_name="fitgenie_app", user_id=user_id, session_id=session_id)
#     while True:
#         user_text = input("You: ")
#         if user_text.lower() in ["quit", "exit"]:
#             print("FitGenie: Goodbye ğŸ‘‹")
#             break
#         content = types.Content(role="user", parts=[types.Part(text=user_text)])
#         async for ev in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        
#             # If the model requested a tool call
#             calls = ev.get_function_calls()
#             if calls:
#                 for call in calls:
#                     print(f"[TOOL CALL] â†’ {call.name}({call.args})")
        
#             # If a tool responded
#             responses = ev.get_function_responses()
#             if responses:
#                 for resp in responses:
#                     print(f"[TOOL RESULT] â†� {resp.name}: {resp.response}")
        
#             # If there's a normal assistant (or user) message
#             if ev.content and ev.content.parts:
#                 for part in ev.content.parts:
#                     if part.text:
#                         print("FitGenie:", part.text)



# # Start the interactive chat session
# await chat()

## Input Test
# {
#   "completed_workouts": [
#     {"day": "Monday", "completed": true},
#     {"day": "Wednesday", "completed": false},
#     {"day": "Friday", "completed": true}
#   ],
#   "body_weight": 78.5,
#   "performance_data": {
#     "Bench Press": {"reps": 8, "note": "felt easier"},
#     "Squat": {"reps": 6, "note": "struggled with depth"}
#   },
#   "pain_reports": ["Left knee soreness"],
#   "current_plan": {
#     "week": 1,
#     "days": [
#       {"day": "Monday", "exercises": ["Bench Press", "Lat Pulldown"]},
#       {"day": "Wednesday", "exercises": ["Squat", "Lunge"]},
#       {"day": "Friday", "exercises": ["Deadlift", "Row"]}
#     ]
#   }
# }


# # Cell A â€” set up path and remove any existing test file
# import os

# # If your functions are defined in a module called `fitness_memory_module`:
# # import fitness_memory_module as fm
# # fm.MEMORY_FILE = "/kaggle/working/fitness_memory.json"

# # If functions are in this same notebook, reassign the global:
# MEMORY_FILE = "/kaggle/working/fitness_memory.json"

# # Remove file if exists (start clean)
# if os.path.exists(MEMORY_FILE):
#     os.remove(MEMORY_FILE)
#     print("Old memory file removed.")
# else:
#     print("No existing memory file; starting fresh.")

# # Cell B â€” single write/read test
# from datetime import datetime
# # If functions are in a module, import them:
# # from fitness_memory_module import memory_bank_write, memory_bank_read, MEMORY_FILE

# # If functions were defined in the same notebook, they should already be available.
# print("MEMORY_FILE:", MEMORY_FILE)

# # Write one weekly summary
# res = memory_bank_write(
#     "weekly_summary",
#     {
#         "adherence_pct": 80,
#         "body_weight_kg": 83,
#         "performance_changes": {"squat": "+2 reps"},
#         "pain_reports": []
#     }
# )
# print("write result:", res)

# # Read back entries
# read_res = memory_bank_read("weekly_summary")
# print("read result keys:", read_res.keys())
# print("entries count:", len(read_res["entries"]))
# print("first entry (pretty):")
# import json
# print(json.dumps(read_res["entries"][0], indent=2))

# # Cell C â€” create history to test trends & compact
# import random, time
# # Create 8 weeks of synthetic summaries
# for week in range(1, 9):
#     entry = {
#         "week_number": week,
#         "adherence_pct": max(40, min(100, 60 + (week-4)*5)),  # vary adherence
#         "body_weight_kg": 84 - 0.3 * week,                    # slight weight loss
#         "performance_changes": {"bench_press": f"+{week%3} reps"},
#         "pain_reports": [] if week % 6 != 0 else ["mild knee pain"]
#     }
#     memory_bank_write("weekly_summary", entry)
#     # tiny sleep so timestamps differ (optional)
#     time.sleep(0.1)

# all_entries = memory_bank_read("weekly_summary")["entries"]
# print("Total weekly_summary entries:", len(all_entries))
# print("First timestamp:", all_entries[0]["timestamp"])
# print("Last timestamp: ", all_entries[-1]["timestamp"])
# print("Last entry data:", all_entries[-1]["data"])

# # Cell D â€” compact test (keep only last 5)
# res = memory_bank_compact(max_entries=5)
# print("compact result:", res)

# entries_after = memory_bank_read("weekly_summary")["entries"]
# print("count after compact:", len(entries_after))
# # show earliest remaining and latest remaining
# import json
# print("earliest remaining:", json.dumps(entries_after[0], indent=2))
# print("latest remaining: ", json.dumps(entries_after[-1], indent=2))

# # Cell E â€” print the raw file text
# print("File exists?", os.path.exists(MEMORY_FILE))
# with open(MEMORY_FILE, "r") as f:
#     raw = f.read()
# print(raw[:800])   # print first 800 chars

# # Cell F â€” assertions
# e = memory_bank_read("weekly_summary")["entries"]
# assert len(e) > 0, "No entries found â€” write failed"
# # check keys exist in entry
# sample = e[-1]
# assert "timestamp" in sample, "timestamp missing"
# assert "category" in sample, "category missing"
# assert "data" in sample, "data missing"
# # compact behavior test (already compacted to 5 above)
# assert len(e) <= 5, "Compact did not trim to <=5 entries"
# print("All basic checks passed âœ…")



# Agent responsible for weekly review and plan adjustment
review_agent = Agent(
    name="ReviewAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
        You are FitGenie Review Agent. Every 7 days you collect user feedback to adjust next week's training.
        Inputs (via orchestration context):
        - past_progress: JSON from ProgressMemoryAgent (last 7 days)
        - current_plan: JSON for the upcoming week
        - user_feedback: optional input (weight, energy, soreness)
        Tasks:
        1. Ask the user for:
           - weight
           - energy level (1-10)
           - muscle soreness (body parts, 1-10)
        2. Compare this feedback with past week:
           - Identify if workouts were too easy or too hard
           - Suggest a deload week if soreness is high or performance declined
        3. Adjust next week's difficulty:
           - Increase load for improved performance
           - Reduce load or swap exercises for soreness/pain
        4. Update nutrition plan if needed based on feedback and adjusted difficulty
        5. Return JSON with:
           - adjusted_plan
           - deload_recommendation (bool)
           - nutrition_update (if applicable)
           - feedback_summary
        6. Use memory_bank tools to log this review for long-term tracking.
        """,
    tools=[memory_bank_write_tool, memory_bank_read_tool, memory_bank_compact_tool]
)


# Define root orchestrator agent
root_agent = Agent(
    name="FitGenieRootAgent",
    model="gemini-2.5-flash-lite",
    instruction=
    """
        You are FitGenie â€” the Master Fitness Coach and Orchestrator.
        You coordinate all agents, control the workflow, pass data, and guarantee the user's fitness system runs smoothly and consistently.
        
        GLOBAL RULES:
            - Always ask only ONE question at a time.
            - Always output in clean, user-friendly Markdown.
            - Never wait for the user to ask for videos.
            - Never ask the user for data already stored.
            - Every exercise must include a YouTube tutorial link.
            - Never use or trust video links created by WeeklyPlannerAgent.
            - Only use video URLs returned by youtube_search_agent after validation.
            - If the workout plan contains any video URL that does not match the validated URL, overwrite it automatically.
            - Final workout plan MUST only include verified working URLs.
            - All break timers MUST stream countdown updates to the user interface.
            - Whenever break_timer_tool yields a countdown event, you MUST immediately forward (yield) it to the user so the UI shows it in real time.
            - Never hide or delay countdown events.
        
        WORKFLOW:
        
        STEP 1 â€” Profile Intake
            - Ask for one field at a time:
                weight â†’ height â†’ age â†’ sex â†’ training status â†’ goal â†’ weekly frequency â†’ injuries
            - After collecting all fields, show a profile summary and ask:
                "Is this correct?"
            - If user says â€œnoâ€�, request ONLY the incorrect fields (one at a time).
        
        STEP 2 â€” Workout Plan Generation
            - Call WeeklyPlannerAgent with full profile.
            - WeeklyPlannerAgent must return clean JSON containing:
                â€¢ weekly split
                â€¢ exercises per day
                â€¢ sets, reps, rest_seconds
                â€¢ video placeholder: "video": ""
            - After receiving the plan:
                **Immediately display the full weekly workout plan to the user in Markdown.**
                - Do not say â€œI generated your planâ€� without showing it.
                - Never wait for user request.
        
        STEP 3 â€” Automatic Video Fetch
            - Extract all unique exercise names.
            - Call youtube_search_agent immediately.
            - Validate all YouTube URLs:
                â€¢ API search
                â€¢ availability check
                â€¢ fallback if unavailable
            - Replace ALL placeholder video URLs in the workout plan with validated URLs.
            - Never display broken or missing URLs.
            - After updating with valid URLs, re-display the final complete plan.
            - Store validated tutorial URLs for StepGuideAgent.
        
        STEP 4 â€” Nutrition Plan (Optional)
            - Ask a single question: â€œWould you like a nutrition plan?â€�
            - If yes:
                - Call nutrition_agent with:
                    â€¢ full profile
                    â€¢ full workout plan JSON
                - If plan is missing, return an error to the orchestrator (never to the user).
        
        STEP 5 â€” Step-by-Step Workout Execution
            - When the user requests to train:
                - Call StepGuideAgent with:
                    â€¢ chosen day
                    â€¢ exercise list
                    â€¢ video URLs
            - StepGuideAgent:
                â€¢ Shows exercises one at a time with video.
                â€¢ Waits for user: â€œdoneâ€� or â€œskipâ€�.
                â€¢ After each exercise:
                    - MUST call break_timer_tool with rest_seconds.
                    - The break_timer_tool yields:
                        â€¢ countdown events (e.g., {"seconds_left": 30, ...})
                        â€¢ final "break finished" event
                    - You (FitGenie Root) MUST immediately forward every countdown tick to the user.
                    - Do not buffer, hide, or wait until the timer finishes.
                â€¢ Only move to the next exercise when the break timer is fully completed.
            - Record:
                â€¢ each exercise completion status
                â€¢ break-timer completion
        
        STEP 6 â€” Weekly Progress
            - ProgressAgent stores:
                â€¢ completed exercises
                â€¢ sets performed
                â€¢ consistency
                â€¢ soreness
                â€¢ weight changes
                â€¢ pain/injury flags
                â€¢ performance improvements
            - Orchestrator only forwards updates.
        
        STEP 7 â€” Weekly Review
            - Every 7 days:
                - Call ReviewAgent with:
                    â€¢ last weekâ€™s progress
                    â€¢ current plan JSON
            - ReviewAgent asks (one at a time):
                â€¢ weight
                â€¢ energy
                â€¢ soreness
            - It returns adjustments.
            - Orchestrator forwards adjustments to:
                â€¢ WeeklyPlannerAgent (plan updates)
                â€¢ nutrition_agent (calorie/macro changes)
        
        STEP 8 â€” Final Output Assembly
            - Always display:
                â€¢ Profile summary
                â€¢ Full workout plan (with verified video links only)
                â€¢ Nutrition plan (if selected)
                â€¢ Weekly review changes (when applicable)
                â€¢ Progress and insights
            - Never hide or delay the workout plan.
        """
        ,
    tools=[
        AgentTool(plan_agent),          
        AgentTool(nutrition_agent),
        AgentTool(youtube_agent),
        AgentTool(progress_memory_agent),
        AgentTool(review_agent),
        step_guide_tool,                
        break_timer_tool 
    ],
    description="Master orchestrator of FitGenie workflow."
)


# Create runner instance for the root agent
runner = make_runner(agent=root_agent)


# Main async function to run the conversational interface
async def chat():
    user_id = "user1"; session_id = "session"
    await session_service.create_session(app_name="fitgenie_app", user_id=user_id, session_id=session_id)
    while True:
        user_text = input("You: ")
        if user_text.lower() in ["quit", "exit"]:
            print("FitGenie: Goodbye ğŸ‘‹")
            break
        content = types.Content(role="user", parts=[types.Part(text=user_text)])
        async for ev in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if ev.content and ev.content.parts:
                text = ev.content.parts[0].text
                if text:  # Only print if text is not None or empty
                    print("FitGenie:", text)


# Start the interactive chat session
# await chat()


# Generate the proxied URL for ADK web UI in Kaggle environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"
    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")
    baseURL = servers[0]["base_url"]
    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")
    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"
    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """
    display(HTML(styled_html))
    return url_prefix
print("âœ… Helper functions defined.")


# Create a new ADK agent named 'fitgenie_agent' using Gemini model
!adk create fitgenie_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


# %%writefile fitgenie_agent/agent.py
# # Save complete agent configuration to agent.py file

# # Core ADK agent classes
# from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent, Agent

# # Gemini LLM model
# from google.adk.models.google_llm import Gemini

# # Agent execution and state management
# from google.adk.runners import Runner
# from google.adk.sessions import InMemorySessionService
# from google.adk.memory import InMemoryMemoryService

# # Content and message types
# from google.genai.types import Content, Part
# from google.genai import types

# # Agent tools and capabilities
# from google.adk.tools import google_search, FunctionTool
# from google.adk.tools.agent_tool import AgentTool
# from google.adk.events import Event

# # Standard libraries
# import asyncio
# import requests
# import time
# import os
# import json
# from typing import Optional, Dict, Any, List
# from datetime import datetime
# import logging

# # Display libraries
# from IPython.core.display import display, HTML
# from jupyter_server.serverapp import list_running_servers

# # Google OAuth and authentication
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.oauth2 import id_token
# from google.auth.transport import requests

# # Kaggle secrets for API keys
# from kaggle_secrets import UserSecretsClient

# secrets = UserSecretsClient()

# # GOOGLE_API_KEY
# try:
#     GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
#     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#     print("âœ… Gemini API key setup complete.")
# except Exception as e:
#     print(
#         f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
#     )
# # GOOGLE_PROJECT_ID
# try:
#     GOOGLE_PROJECT_ID = secrets.get_secret("GOOGLE_PROJECT_ID")
#     os.environ["GOOGLE_PROJECT_ID"] = GOOGLE_PROJECT_ID
#     print("âœ… GOOGLE_PROJECT_ID setup complete.")
# except Exception as e:
#     print(f"ğŸ”‘ Authentication Error: Please add 'GOOGLE_PROJECT_ID' to your Kaggle secrets. Details: {e}")

# # GOOGLE_OAUTH_CLIENT_ID
# try:
#     GOOGLE_OAUTH_CLIENT_ID = secrets.get_secret("GOOGLE_OAUTH_CLIENT_ID")
#     os.environ["GOOGLE_OAUTH_CLIENT_ID"] = GOOGLE_OAUTH_CLIENT_ID
#     print("âœ… GOOGLE_OAUTH_CLIENT_ID setup complete.")
# except Exception as e:
#     print(f"ğŸ”‘ Authentication Error: Please add 'GOOGLE_OAUTH_CLIENT_ID' to your Kaggle secrets. Details: {e}")

# # GOOGLE_OAUTH_CLIENT_SECRET
# try:
#     GOOGLE_OAUTH_CLIENT_SECRET = secrets.get_secret("GOOGLE_OAUTH_CLIENT_SECRET")
#     os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = GOOGLE_OAUTH_CLIENT_SECRET
#     print("âœ… GOOGLE_OAUTH_CLIENT_SECRET setup complete.")
# except Exception as e:
#     print(f"ğŸ”‘ Authentication Error: Please add 'GOOGLE_OAUTH_CLIENT_SECRET' to your Kaggle secrets. Details: {e}")

# # YOUTUBE_API_KEY
# try:
#     YOUTUBE_API_KEY = secrets.get_secret("YOUTUBE_API_KEY")
#     os.environ["YOUTUBE_API_KEY"] = YOUTUBE_API_KEY
#     print("âœ… YOUTUBE_API_KEY setup complete.")
# except Exception as e:
#     print(f"ğŸ”‘ Authentication Error: Please add 'YOUTUBE_API_KEY' to your Kaggle secrets. Details: {e}")
    
# # Global shared services
# session_service = InMemorySessionService()
# memory_service = InMemoryMemoryService()

# def make_runner(agent, app_name="fitgenie_app"):
#     return Runner(
#         agent=agent,
#         app_name=app_name,
#         session_service=session_service,
#         memory_service=memory_service
#     )

# retry_config=types.HttpRetryOptions(
#     attempts=5,
#     exp_base=7,
#     initial_delay=1,
#     http_status_codes=[429, 500, 503, 504]
# )

# # Agent responsible for generating personalized workout plans
# plan_agent = Agent(
#     name="WeeklyPlannerAgent",
#     model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
#     instruction="""
#         You are an expert, evidence-based Fitness Planner. Your sole task is to generate a highly customized workout routine using the user's existing profile (weight, height, age, training status, sex, goal, weekly frequency/availability, and health conditions).
#         Follow these mandatory guidelines:
        
#         1. **Data Retrieval & Verification:**
#             - Assume the user profile data is provided and accurate.
#             - Use the **google_search** tool to verify exercise safety if the user has specific injuries (e.g., "exercises to avoid with meniscus tear").
#             - **MANDATORY:** You must use the **youtube_search** tool for every distinct exercise in the plan to fetch a high-quality demonstration video. Search query format: "[Exercise Name] proper form" or "[Exercise Name] tutorial".
        
#         2. **Safety & Customization (CRITICAL):**
#             - Exclude any exercise that could worsen listed injuries or conditions.
#             - If the user is a beginner or returning after a break, the first 1â€“2 sessions MUST be low-intensity, focusing on form, stability, and mobility.
#             - Adapt intensity (RPE) and volume (sets/reps) to their specific goal.
        
#         3. **Plan Frequency Logic:**
#             - If the user provided a number (e.g., "3 days"), follow it precisely.
#             - If the user asked for the "optimal plan" or "best results," determine the ideal frequency based on their goal/status and explain your reasoning.
        
#         4. **Output Format:**
#             - The output must be clean Markdown.
#             - Structure the routine by Day/Session.
#             - For every exercise, follow this EXACT format:
#               `* **[Exercise Name]** [[Watch Demo](URL_FROM_TOOL)] - [Sets] x [Reps] (Rest: [Time])`
#             - *Example:* `* **Goblet Squat** [[Watch Demo](https://youtube.com/...)] - 3 x 12 (Rest: 60s)`
#             - Add a brief 1-sentence cue for form under complex lifts.
        
#         5. **Constraints:**
#             - Do NOT ask for profile data again if it is already present.
#             - If the profile data is completely missing (no inputs provided at all), output a specific error asking for: Biometrics, Training Status, Goal, Availability, Injuries, and Equipment.
        
#         6. **Ending:**
#             - After the complete plan, ask:
#               **"Do you approve this plan or would you like to request specific changes?"**
#     """,
#     tools=[google_search],
#     output_key="current_plan"
# )

# # OpenFoodFacts integration for nutrition lookup
# OFF_API_BASE = "https://world.openfoodfacts.org"

# def query_off_by_name(food_name: str) -> Optional[Dict[str, Any]]:
#     """
#     Search OpenFoodFacts for products matching food_name; return first result's nutrition info (per 100g / typical serving).
#     """
#     url = f"{OFF_API_BASE}/cgi/search.pl"
#     params = {
#         "search_terms": food_name,
#         "search_simple": 1,
#         "action": "process",
#         "json": 1,
#         "page_size": 5,
#     }
#     resp = requests.get(url, params=params)
#     if resp.status_code != 200:
#         return None
#     data = resp.json()
#     products = data.get("products", [])
#     if not products:
#         return None
#     prod = products[0]
#     nutr = prod.get("nutriments", None)
#     if not nutr:
#         return None
#     return {
#         "product_name": prod.get("product_name", food_name),
#         "serving_size": prod.get("serving_size"),
#         "serving_quantity": prod.get("serving_quantity"),
#         "nutriments": nutr
#     }

# def get_nutrition_of(food_name: str) -> Optional[Dict[str, float]]:
#     """
#     Returns a dict with approximate macros/calories for food_name.
#     If OFF lookup fails, returns None.
#     """
#     info = query_off_by_name(food_name)
#     if not info:
#         return None

#     nutr = info["nutriments"]
#     calories = nutr.get("energy-kcal_100g") or nutr.get("energy_100g") or nutr.get("energy-kcal_serving")
#     protein = nutr.get("proteins_100g") or nutr.get("proteins_serving")
#     carbs   = nutr.get("carbohydrates_100g") or nutr.get("carbohydrates_serving")
#     fat     = nutr.get("fat_100g") or nutr.get("fat_serving")

#     if calories is None or protein is None or carbs is None or fat is None:
#         return None

#     return {
#         "calories_100g": calories,
#         "protein_100g": protein,
#         "carbs_100g": carbs,
#         "fat_100g": fat
#     }

# nutrition_tool = FunctionTool(get_nutrition_of)

# # Define nutrition_agent for meal planning
# nutrition_agent = Agent(
#     name="NutritionAgent",
#     model="gemini-2.5-flash-lite",
#     instruction="""
#         You are FitGenie Nutrition Agent.
#         You receive the userâ€™s fitness profile **and** the userâ€™s workout plan (from the root orchestrator; referred to as â€œcurrent_planâ€�).  
#         Use these inputs to compute a nutrition plan tailored to both:  
#         - Daily caloric intake (based on BMR, activity level, and training load from current_plan),  
#         - Macro distribution (protein, carbs, fat) suitable for goal and training volume,  
#         - Optionally â€” suggest how macros/calories may vary on training vs rest days according to the workout schedule in current_plan.  
        
#         Return the nutrition plan in structured JSON or markdown. Do not ask the user for the workout plan again â€” assume the orchestrator already passed it. If the workout plan is missing or invalid, respond with an error message asking the orchestrator to provide the missing plan.  
#         """,
#     tools=[nutrition_tool],
#     description="Fetches nutrition info for foods or computes daily macros."
# )

# # Check if YouTube video URL is accessible
# def check_video_exists(url: str) -> bool:
#     try:
#         r = requests.head(url, allow_redirects=True, timeout=3)
#         return r.status_code == 200
#     except:
#         return False

# # Search YouTube API for exercise tutorial videos
# def search_youtube_video_id(query: str) -> str:
#     """
#     Searches YouTube for a tutorial video and returns the first videoId.
#     Returns None if there is no valid result.
#     """
#     url = "https://www.googleapis.com/youtube/v3/search"
#     params = {
#         "part": "snippet",
#         "q": query,
#         "type": "video",
#         "maxResults": 5,
#         "key": YOUTUBE_API_KEY,
#         "safeSearch": "moderate",
#         "videoEmbeddable": "true",
#     }
#     try:
#         r = requests.get(url, params=params, timeout=5)
#         r.raise_for_status()
#         data = r.json()
#         items = data.get("items", [])
#         if not items:
#             return None
#         return items[0]["id"]["videoId"]
#     except Exception as e:
#         print("YouTube API Error:", e)
#         return None

# # Fallback video URLs for common exercises
# FALLBACK_VIDEOS = {
#     "squat": "https://www.youtube.com/watch?v=aclHkVaku9U",
#     "deadlift": "https://www.youtube.com/watch?v=1ZXobu7JvvE",
#     "push up": "https://www.youtube.com/watch?v=_l3ySVKYVJ8",
#     "bench press": "https://www.youtube.com/watch?v=rT7DgCr-3pg",
#     "row": "https://www.youtube.com/watch?v=kBWAon7ItDw",
#     "shoulder press": "https://www.youtube.com/watch?v=B-aVuyhvLHU",
#     "plank": "https://www.youtube.com/watch?v=pSHjTRCQxIw",
#     "bicep curl": "https://www.youtube.com/watch?v=ykJmrZ5v0Oo",
#     "triceps extension": "https://www.youtube.com/watch?v=2-LAMcpzODU"
# }

# # Batch search for multiple exercises with fallback logic
# def youtube_search_tool(exercises: List[str]) -> Dict[str, str]:
#     """
#     Returns: { "exercise_name": "verified_url" }
#     """
#     result = {}
#     for name in exercises:
#         query = f"{name} exercise tutorial"
#         video_id = search_youtube_video_id(query)
#         url = f"https://www.youtube.com/watch?v={video_id}" if video_id else None
        
#         # Validate video URL
#         if url and check_video_exists(url):
#             result[name] = url
#             continue
        
#         # Try fallback for simplified exercise name
#         simplified = name.split()[0]
#         fallback = FALLBACK_VIDEOS.get(simplified.lower())
#         if fallback:
#             result[name] = fallback
#             continue
        
#         result[name] = "No video available"
#     return result

# youtube_tool = FunctionTool(youtube_search_tool)

# youtube_agent = Agent(
#     name="youtube_search_agent",
#     model="gemini-2.5-flash",
#     instruction="""
#         You return demonstration video links for exercises.

#         RULES:
#         - ONLY call the youtube_search_tool with the list of exercise names.
#         - NEVER guess or generate a URL manually.
#         - NEVER search by yourself.
#         - NEVER output text except the JSON result.

#         Expected output format:
#         {
#             "videos": {
#                 "Exercise Name": "url",
#                 ...
#             }
#         }
#     """,
#     tools=[youtube_tool],
#     description="Provides validated tutorial links for exercises."
# )

# # Timer for rest periods between exercises
# def break_timer(seconds: int, exercise: str):
#     """
#     Break timer that streams countdown updates to the ADK interface.
#     Replaces print() with yield events so the UI displays live updates.
#     """
#     for remaining in range(seconds, 0, -1):

#         # Only send updates at 10s intervals OR the last 5 seconds
#         if remaining % 10 == 0 or remaining <= 5:
#             yield {
#                 "event": "break_timer_tick",
#                 "exercise": exercise,
#                 "seconds_left": remaining,
#                 "message": f"â�³ {exercise}: {remaining} seconds remaining..."
#             }

#         time.sleep(1)

#     # Final message
#     yield {
#         "event": "break_timer_done",
#         "exercise": exercise,
#         "message": f"ğŸ”¥ {exercise} break finished!"
#     }

#     return {"status": "finished", "exercise": exercise}

# break_timer_tool = FunctionTool(break_timer)

# # Interactive step-by-step workout execution with timer
# def step_by_step_workout_tool(workout_plan: dict, video_urls: dict):
#     """
#     Executes a single day's workout step-by-step.
#     Handles warm-ups first, then main exercises.
#     Calls break timer after each exercise.
#     """
#     results = []
#     all_exercises = workout_plan.get("warmups", []) + workout_plan.get("exercises", [])
    
#     for exercise in all_exercises:
#         name = exercise["name"]
#         sets = exercise["sets"]
#         reps = exercise["reps"]
#         rest = exercise.get("rest_seconds", 60)
#         video = video_urls.get(name, "Video not found")
        
#         print(f"\nExercise: {name}")
#         print(f"Sets x Reps: {sets} x {reps}")
#         print(f"Video Demo: {video}")
        
#         user_input = input("Type 'done' when completed, or 'skip' to skip: ").strip().lower()
        
#         if user_input == "skip":
#             results.append({"exercise": name, "status": "skipped"})
#             continue
        
#         break_timer(rest, name)
#         results.append({"exercise": name, "status": "completed"})
    
#     return {"day": workout_plan.get("day"), "results": results}

# step_guide_tool = FunctionTool(step_by_step_workout_tool)

# MEMORY_FILE = "fitness_memory.json"

# # Load memory from JSON file
# def _load_memory():
#     if not os.path.exists(MEMORY_FILE):
#         return {"entries": []}
#     with open(MEMORY_FILE, "r") as f:
#         return json.load(f)

# # Save memory to JSON file
# def _save_memory(mem):
#     with open(MEMORY_FILE, "w") as f:
#         json.dump(mem, f, indent=4)

# # Write new memory entry with timestamp
# def memory_bank_write(category: str, data: dict) -> dict:
#     """
#     Store a memory event under 'category', with arbitrary data (dict).
#     Returns status and stored entry.
#     """
#     mem = _load_memory()
#     entry = {
#         "timestamp": datetime.utcnow().isoformat(),
#         "category": category,
#         "data": data
#     }
#     mem.setdefault("entries", []).append(entry)
#     _save_memory(mem)
#     return {"status": "success", "entry": entry}

# # Read memory entries with optional category filter
# def memory_bank_read(category: Optional[str] = None) -> dict:
#     """
#     Read memory entries. If category given, filters by that category.
#     """
#     mem = _load_memory()
#     entries = mem.get("entries", [])
#     if category:
#         entries = [e for e in entries if e.get("category") == category]
#     return {"entries": entries}

# # Compact memory by keeping only recent entries
# def memory_bank_compact(max_entries: int = 100) -> dict:
#     """
#     Optionally trim memory to keep only the most recent max_entries entries.
#     """
#     mem = _load_memory()
#     entries = mem.get("entries", [])
#     if len(entries) <= max_entries:
#         return {"status": "unchanged", "count": len(entries)}
#     new_entries = entries[-max_entries:]
#     mem["entries"] = new_entries
#     _save_memory(mem)
#     return {"status": "compacted", "remaining": len(new_entries)}

# # Wrap memory functions as ADK tools
# memory_bank_write_tool = FunctionTool(memory_bank_write)
# memory_bank_read_tool  = FunctionTool(memory_bank_read)
# memory_bank_compact_tool = FunctionTool(memory_bank_compact)

# # Agent responsible for tracking and analyzing long-term fitness progress
# progress_memory_agent = Agent(
#     name="ProgressMemoryAgent",
#     model="gemini-2.5-flash-lite",
#     instruction="""
#         You are the FitGenie Progress Memory Agent. You receive data each week and update the user's long-term fitness memory.  
        
#         Inputs (via orchestration context):
#         - completed_workouts: list describing which exercises were done/skipped
#         - body_weight: numeric or string (e.g. kg)
#         - performance_data: dict mapping exercise names â†’ performance notes (reps, difficulty, changes)
#         - pain_reports: optional list describing pain/injury notes
#         - current_plan: the workout plan JSON used this week
        
#         Tasks:
#         1. Store a memory entry summarizing the week:
#            - adherence (percentage of workout days completed)
#            - body weight
#            - performance_changes (if any)
#            - pain_reports (if any)
#            - timestamp
#         2. Analyze trends: compare with past weeks (if available) and note improvements or issues.
#         3. Suggest modifications to next week's plan: for example, if pain was reported, mark for lighter load or alternative exercises. If performance improved consistently, suggest small progressive overload.  
#         4. Return JSON with:
#            - stored_entry (what was written)
#            - trend_analysis (brief summary: improving / needs deload / pain alert)
#            - plan_modifications (if any, list of suggested changes)
        
#         You must use the memory_bank tools (write/read/compact) to persist and retrieve history.
#         """,
#     tools=[memory_bank_write_tool, memory_bank_read_tool, memory_bank_compact_tool],
# )

# # Agent responsible for weekly review and plan adjustment
# review_agent = Agent(
#     name="ReviewAgent",
#     model="gemini-2.5-flash-lite",
#     instruction="""
#         You are FitGenie Review Agent. Every 7 days you collect user feedback to adjust next week's training.
#         Inputs (via orchestration context):
#         - past_progress: JSON from ProgressMemoryAgent (last 7 days)
#         - current_plan: JSON for the upcoming week
#         - user_feedback: optional input (weight, energy, soreness)
#         Tasks:
#         1. Ask the user for:
#            - weight
#            - energy level (1-10)
#            - muscle soreness (body parts, 1-10)
#         2. Compare this feedback with past week:
#            - Identify if workouts were too easy or too hard
#            - Suggest a deload week if soreness is high or performance declined
#         3. Adjust next week's difficulty:
#            - Increase load for improved performance
#            - Reduce load or swap exercises for soreness/pain
#         4. Update nutrition plan if needed based on feedback and adjusted difficulty
#         5. Return JSON with:
#            - adjusted_plan
#            - deload_recommendation (bool)
#            - nutrition_update (if applicable)
#            - feedback_summary
#         6. Use memory_bank tools to log this review for long-term tracking.
#         """,
#     tools=[memory_bank_write_tool, memory_bank_read_tool, memory_bank_compact_tool]
# )

# # Define root orchestrator agent
# root_agent = Agent(
#     name="FitGenieRootAgent",
#     model="gemini-2.5-flash-lite",
#     instruction=
#     """
#         You are FitGenie â€” the Master Fitness Coach and Orchestrator.
#         You coordinate all agents, control the workflow, pass data, and guarantee the user's fitness system runs smoothly and consistently.
        
#         GLOBAL RULES:
#             - Always ask only ONE question at a time.
#             - Always output in clean, user-friendly Markdown.
#             - Never wait for the user to ask for videos.
#             - Never ask the user for data already stored.
#             - Every exercise must include a YouTube tutorial link.
#             - Never use or trust video links created by WeeklyPlannerAgent.
#             - Only use video URLs returned by youtube_search_agent after validation.
#             - If the workout plan contains any video URL that does not match the validated URL, overwrite it automatically.
#             - Final workout plan MUST only include verified working URLs.
#             - All break timers MUST stream countdown updates to the user interface.
#             - Whenever break_timer_tool yields a countdown event, you MUST immediately forward (yield) it to the user so the UI shows it in real time.
#             - Never hide or delay countdown events.
        
#         WORKFLOW:
        
#         STEP 1 â€” Profile Intake
#             - Ask for one field at a time:
#                 weight â†’ height â†’ age â†’ sex â†’ training status â†’ goal â†’ weekly frequency â†’ injuries
#             - After collecting all fields, show a profile summary and ask:
#                 "Is this correct?"
#             - If user says â€œnoâ€�, request ONLY the incorrect fields (one at a time).
        
#         STEP 2 â€” Workout Plan Generation
#             - Call WeeklyPlannerAgent with full profile.
#             - WeeklyPlannerAgent must return clean JSON containing:
#                 â€¢ weekly split
#                 â€¢ exercises per day
#                 â€¢ sets, reps, rest_seconds
#                 â€¢ video placeholder: "video": ""
#             - After receiving the plan:
#                 **Immediately display the full weekly workout plan to the user in Markdown.**
#                 - Do not say â€œI generated your planâ€� without showing it.
#                 - Never wait for user request.
        
#         STEP 3 â€” Automatic Video Fetch
#             - Extract all unique exercise names.
#             - Call youtube_search_agent immediately.
#             - Validate all YouTube URLs:
#                 â€¢ API search
#                 â€¢ availability check
#                 â€¢ fallback if unavailable
#             - Replace ALL placeholder video URLs in the workout plan with validated URLs.
#             - Never display broken or missing URLs.
#             - After updating with valid URLs, re-display the final complete plan.
#             - Store validated tutorial URLs for StepGuideAgent.
        
#         STEP 4 â€” Nutrition Plan (Optional)
#             - Ask a single question: â€œWould you like a nutrition plan?â€�
#             - If yes:
#                 - Call nutrition_agent with:
#                     â€¢ full profile
#                     â€¢ full workout plan JSON
#                 - If plan is missing, return an error to the orchestrator (never to the user).
        
#         STEP 5 â€” Step-by-Step Workout Execution
#             - When the user requests to train:
#                 - Call StepGuideAgent with:
#                     â€¢ chosen day
#                     â€¢ exercise list
#                     â€¢ video URLs
#             - StepGuideAgent:
#                 â€¢ Shows exercises one at a time with video.
#                 â€¢ Waits for user: â€œdoneâ€� or â€œskipâ€�.
#                 â€¢ After each exercise:
#                     - MUST call break_timer_tool with rest_seconds.
#                     - The break_timer_tool yields:
#                         â€¢ countdown events (e.g., {"seconds_left": 30, ...})
#                         â€¢ final "break finished" event
#                     - You (FitGenie Root) MUST immediately forward every countdown tick to the user.
#                     - Do not buffer, hide, or wait until the timer finishes.
#                 â€¢ Only move to the next exercise when the break timer is fully completed.
#             - Record:
#                 â€¢ each exercise completion status
#                 â€¢ break-timer completion
        
#         STEP 6 â€” Weekly Progress
#             - ProgressAgent stores:
#                 â€¢ completed exercises
#                 â€¢ sets performed
#                 â€¢ consistency
#                 â€¢ soreness
#                 â€¢ weight changes
#                 â€¢ pain/injury flags
#                 â€¢ performance improvements
#             - Orchestrator only forwards updates.
        
#         STEP 7 â€” Weekly Review
#             - Every 7 days:
#                 - Call ReviewAgent with:
#                     â€¢ last weekâ€™s progress
#                     â€¢ current plan JSON
#             - ReviewAgent asks (one at a time):
#                 â€¢ weight
#                 â€¢ energy
#                 â€¢ soreness
#             - It returns adjustments.
#             - Orchestrator forwards adjustments to:
#                 â€¢ WeeklyPlannerAgent (plan updates)
#                 â€¢ nutrition_agent (calorie/macro changes)
        
#         STEP 8 â€” Final Output Assembly
#             - Always display:
#                 â€¢ Profile summary
#                 â€¢ Full workout plan (with verified video links only)
#                 â€¢ Nutrition plan (if selected)
#                 â€¢ Weekly review changes (when applicable)
#                 â€¢ Progress and insights
#             - Never hide or delay the workout plan.
#         """
#         ,
#     tools=[
#         AgentTool(plan_agent),          
#         AgentTool(nutrition_agent),
#         AgentTool(youtube_agent),
#         AgentTool(progress_memory_agent),
#         AgentTool(review_agent),
#         step_guide_tool,                
#         break_timer_tool 
#     ],
#     description="Master orchestrator of FitGenie workflow."
# )


# Get the ADK web UI proxy URL for Kaggle environment
# url_prefix = get_adk_proxy_url()


# Start the ADK web interface with debug logging
# !adk web --log_level DEBUG --url_prefix {url_prefix}

