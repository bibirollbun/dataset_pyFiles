# Kaggle Notebook: Scent Select AI Agent - Agents Intensive - Capstone Project

# Participant: Lijun Wang (Aaron) 

# ---
# ## Table of Contents
# 1.  **Project Introduction**
# 2.  **Setup and Imports**
#     * Gemini API Key Configuration
#     * ADK Library Imports (Bare minimum)
# 3.  **Simulated Data (Non-LLM Tools)**
# 4.  **Custom Tools Implementation (Functions only, now WITH ADK FunctionTool integration)**
# 5.  **ADK Agent Definitions (Now with FunctionTool/AgentTool integration)**
# 6.  **ADK Session & Memory Management**
#     * Initializing Services
#     * Automatic Memory Saving Callback
# 7.  **Main Agent Orchestration (Simplified using SequentialAgent and FunctionTools)**
# ---

# ## 1. Project Introduction
# This project builds an AI agent to recommend daily perfumes. This version focuses on
# getting the ADK `LlmAgent` and `Runner` instances to initialize and leverage ADK's
# built-in workflow patterns (`SequentialAgent`) and `FunctionTool` for custom functions,
# aligning with the approach demonstrated in the Kaggle multi-agent course.

# ---

# ## 2. Setup and Imports

# ### Gemini API Key Configuration
# You must have your Google API key stored as a Kaggle secret named `GOOGLE_API_KEY`.
# To add it:
# 1.  Click "Add-ons" in the notebook editor.
# 2.  Select "Secrets".
# 3.  Add a new secret named `GOOGLE_API_KEY` and paste your API key.

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

# ### Install ADK and Google Generative AI Libraries
# Uncomment and run these if they are not already installed in your Kaggle environment.
# !pip install -q google-generativeai google-adk

# ### ADK Library Imports
# Now importing more specific components needed for multi-agent systems from the course example.
from google.adk.agents import Agent, SequentialAgent 
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner 
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import FunctionTool, preload_memory 

from google.genai import types

print("âœ… ADK components imported successfully.")

import datetime
import random
import json
import uuid
from typing import Optional, List# <--- Add this import!

# Retry configuration for robust API calls
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Define constants for the application
APP_NAME = "PerfumeAgentApp"
DEMO_USER_ID = "perfume_enthusiast_123"

# ---

# ## 3. Simulated Data (Non-LLM Tools)
# These are the in-memory "databases" for perfumes and simulated weather data,
# which our custom functions will interact with.

# ### Simulated Perfume Database
PERFUME_DATABASE = {
    "Chanel No. 5": {
        "scent_family": "Floral-Aldehyde",
        "notes": ["aldehydes", "ylang-ylang", "neroli", "bergamot", "lemon", "iris", "rose", "lily-of-the-valley", "jasmine", "sandalwood", "vetiver", "vanilla", "amber", "patchouli"],
        "moods": ["elegant", "classic", "confident"],
        "occasions": ["formal", "evening", "work"],
        "weather": ["cool", "cold"],
        "description": "A timeless, sophisticated floral-aldehyde fragrance."
    },
    "Light Blue by Dolce & Gabbana": {
        "scent_family": "Citrus-Fruity",
        "notes": ["Granny Smith apple", "Sicilian lemon", "bellflower", "jasmine", "white rose", "bamboo", "cedar", "musk", "amber"],
        "moods": ["energetic", "fresh", "joyful"],
        "occasions": ["casual", "daytime", "outdoors"],
        "weather": ["warm", "hot"],
        "description": "A sparkling, fresh, and vibrant fragrance inspired by the Mediterranean."
    },
    "Acqua di Gio by Giorgio Armani": {
        "scent_family": "Aquatic-Aromatic",
        "notes": ["lime", "lemon", "bergamot", "jasmine", "orange", "mandarin orange", "neroli", "calone", "sea notes", "peach", "freesia", "hyacinth", "cyclamen", "rose", "rosemary", "coriander", "nutmeg", "oakmoss", "cedar", "patchouli", "amber", "musk", "white musk"],
        "moods": ["calm", "fresh", "masculine"],
        "occasions": ["casual", "daytime", "sport"],
        "weather": ["warm", "hot"],
        "description": "A classic aquatic fragrance evoking the sea and fresh air."
    },
    "Black Opium by Yves Saint Laurent": {
        "scent_family": "Amber Vanilla",
        "notes": ["pear", "pink pepper", "orange blossom", "coffee", "jasmine", "almond", "licorice", "vanilla", "patchouli", "cedar", "cashmere wood"],
        "moods": ["seductive", "bold", "mysterious"],
        "occasions": ["evening", "party", "date"],
        "weather": ["cool", "cold"],
        "description": "An addictive gourmand floral with a shot of black coffee."
    },
    "Jo Malone Wood Sage & Sea Salt": {
        "scent_family": "Aromatic Aquatic",
        "notes": ["Ambrette seeds", "Sea Salt", "Sage", "Grapefruit", "Seaweed"],
        "moods": ["relaxed", "natural", "free-spirited"],
        "occasions": ["casual", "daytime", "outdoors"],
        "weather": ["mild", "warm"],
        "description": "A sophisticated, fresh, and woody fragrance, reminiscent of the British coast."
    },
    "Baccarat Rouge 540 by Maison Francis Kurkdjian": {
        "scent_family": "Amber Floral",
        "notes": ["saffron", "jasmine", "amberwood", "ambergris", "fir resin", "cedar"],
        "moods": ["luxurious", "confident", "unique"],
        "occasions": ["special event", "evening", "statement"],
        "weather": ["cool", "cold", "mild"],
        "description": "A radiant and sophisticated amber floral with a unique signature."
    }
}

# ### Simulated Weather Data
SIMULATED_WEATHER_DATA = {
    "New York": {
        "2023-10-27": {"temperature": "15Â°C", "conditions": "partly cloudy", "feel": "mild"},
        "2023-10-28": {"temperature": "10Â°C", "conditions": "rainy", "feel": "cool"},
        "2023-10-29": {"temperature": "5Â°C", "conditions": "snowy", "feel": "cold"},
        "2023-10-30": {"temperature": "22Â°C", "conditions": "sunny", "feel": "warm"},
        (datetime.date.today().strftime("%Y-%m-%d")): {"temperature": "18Â°C", "conditions": "clear sky", "feel": "mild"} # Dynamic today's weather
    },
    "Los Angeles": {
        "2023-10-27": {"temperature": "25Â°C", "conditions": "sunny", "feel": "warm"},
        "2023-10-28": {"temperature": "23Â°C", "conditions": "sunny", "feel": "warm"},
        "2023-10-29": {"temperature": "20Â°C", "conditions": "foggy", "feel": "mild"},
        "2023-10-30": {"temperature": "28Â°C", "conditions": "sunny", "feel": "hot"},
        (datetime.date.today().strftime("%Y-%m-%d")): {"temperature": "26Â°C", "conditions": "sunny", "feel": "warm"} # Dynamic today's weather
    }
}

# ---

# ## 4. Custom Tools Implementation (Functions only, now WITH ADK FunctionTool integration)
# Define the Python functions. These will now be wrapped in `FunctionTool` objects
# and provided directly to the `LlmAgent`s.

# ### `get_weather_func`
def get_weather_func(city: str, date: str = datetime.date.today().strftime("%Y-%m-%d")) -> dict:
    """
    Fetches simulated weather information for a given city and date.
    Args:
        city (str): The city for which to get weather.
        date (str): The date in 'YYYY-MM-DD' format. Defaults to today.
    Returns:
        dict: Weather conditions (temperature, conditions, feel) or an error message.
    """
    print(f"âš™ï¸� [Func: get_weather_func]: Looking up weather for {city} on {date}...")
    city_data = SIMULATED_WEATHER_DATA.get(city)
    if city_data:
        weather = city_data.get(date)
        if weather:
            print(f"âš™ï¸� [Func: get_weather_func]: Found weather: {weather}")
            return weather
        else:
            print(f"âš™ï¸� [Func: get_weather_func]: Weather data not available for {date}.")
            return {"error": f"Weather data not available for {date}"}
    else:
        print(f"âš™ï¸� [Func: get_weather_func]: City '{city}' not found in database.")
        return {"error": f"City '{city}' not found in database"}

# ### `lookup_perfumes_func`
def lookup_perfumes_func(
    scent_family: Optional[str] = None,
    moods: Optional[List[str]] = None,
    occasions: Optional[List[str]] = None,
    weather_feel: Optional[List[str]] = None
) -> List[str]:
    """
    Searches the perfume database based on provided criteria.
    Args:
        scent_family (str): E.g., 'Floral-Aldehyde', 'Citrus-Fruity'.
        moods (list): List of moods, e.g., ['energetic', 'fresh'].
        occasions (list): List of occasions, e.g., ['casual', 'daytime'].
        weather_feel (list): List of weather feels, e.g., ['warm', 'hot'].
    Returns:
        list: A list of perfume names that match the criteria.
    """
    criteria = {
        "scent_family": scent_family,
        "moods": moods,
        "occasions": occasions,
        "weather_feel": weather_feel
    }
    # Filter out None values from criteria
    filtered_criteria = {k: v for k, v in criteria.items() if v is not None}

    print(f"âš™ï¸� [Func: lookup_perfumes_func]: Searching perfumes with criteria: {filtered_criteria}")
    matching_perfumes = []
    for perfume_name, details in PERFUME_DATABASE.items():
        match = True
        if "scent_family" in filtered_criteria:
            if filtered_criteria["scent_family"].lower() not in details["scent_family"].lower():
                match = False
        if "moods" in filtered_criteria:
            # If list is empty, it still matches, or if a mood is found
            if not filtered_criteria["moods"] or not any(m.lower() in [dm.lower() for dm in details["moods"]] for m in filtered_criteria["moods"]):
                match = False
        if "occasions" in filtered_criteria:
            if not filtered_criteria["occasions"] or not any(o.lower() in [do.lower() for do in details["occasions"]] for o in filtered_criteria["occasions"]):
                match = False
        if "weather_feel" in filtered_criteria:
            if not filtered_criteria["weather_feel"] or not any(w.lower() in [dw.lower() for dw in details["weather"]] for w in filtered_criteria["weather_feel"]):
                match = False

        if match:
            matching_perfumes.append(perfume_name)
    print(f"âš™ï¸� [Func: lookup_perfumes_func]: Found {len(matching_perfumes)} matching perfumes.")
    return matching_perfumes

# ### `get_perfume_details_func`
def get_perfume_details_func(perfume_name: str) -> dict:
    """
    Retrieves full details for a specific perfume.
    Args:
        perfume_name (str): The name of the perfume.
    Returns:
        dict: Details of the perfume or an empty dict if not found.
    """
    print(f"âš™ï¸� [Func: get_perfume_details_func]: Retrieving details for {perfume_name}...")
    return PERFUME_DATABASE.get(perfume_name, {})

print("âœ… Custom tool functions defined, ready for FunctionTool integration.")


# ---

# ## 5. ADK Agent Definitions (Now with FunctionTool/AgentTool integration)
# We'll define `LlmAgent` instances and integrate the custom functions as `FunctionTool`s.
# We also modify the `PerfumeRecommenderAgent` to directly receive its tools.

# ### `UserContextAgent`
# This agent's primary job is to extract user preferences and city, then call `get_weather_func`.
user_context_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="UserContextAgent",
    instruction=(
        "You are an assistant for gathering user preferences and current context for perfume recommendations."
        "Your task is to identify the user's current mood, the occasion, their city, and any preferred scent families."
        "Once you have a city, you MUST call the `get_weather` tool to find today's weather for that city."
        "After gathering all information (mood, occasion, city, scent preference, and weather feel from the tool), "
        "present the full context in a concise JSON format: "
        "{'mood': '...', 'occasion': '...', 'city': '...', 'scent_preference': '...', 'weather_feel': '...'}. "
        "Only output the JSON. If a piece of information is missing, use 'unknown'."
    ),
    tools=[
        FunctionTool(get_weather_func) # Now integrating get_weather_func as a FunctionTool
    ],
    output_key="user_context", # The result of this agent will be stored in the session state with this key.
)
print("âœ… UserContextAgent created with `get_weather_func` tool.")


# ### `PerfumeSearchAgent` (New Agent for direct tool calls)
# This new agent will encapsulate the logic of looking up perfumes and getting details.
perfume_search_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="PerfumeSearchAgent",
    instruction=(
        "You are a specialized perfume database agent. Your task is to use the `lookup_perfumes` and `get_perfume_details` tools "
        "to find the best matching perfume based on the provided user context. "
        "User Context: {user_context}. " # This will be injected by SequentialAgent
        "First, carefully parse the `user_context` to identify scent_preference, mood, occasion, and weather_feel. "
        "Then, call the `lookup_perfumes` tool with these criteria. "
        "If matches are found, select the first matching perfume and call `get_perfume_details` for it. "
        "Finally, summarize the details of the recommended perfume in a concise JSON format, "
        "including 'name', 'scent_family', 'notes', 'moods', 'occasions', 'weather', and 'description'. "
        "If no perfumes match, output: {'recommendation': 'No specific match found. Consider broadening your preferences.'}"
    ),
    tools=[
        FunctionTool(lookup_perfumes_func),
        FunctionTool(get_perfume_details_func)
    ],
    output_key="perfume_details_json",
)
print("âœ… PerfumeSearchAgent created with `lookup_perfumes_func` and `get_perfume_details_func` tools.")


# ### `PerfumeRecommenderAgent` (Now acts as a final output formatter)
perfume_recommender_agent = Agent( 
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="PerfumeRecommenderAgent",
    instruction=(
        "You are an expert perfume sommelier. Your task is to provide a friendly, detailed, and engaging perfume recommendation "
        "based on the provided user context and the found perfume details. "
        "User Context: {user_context}. " # Injected by SequentialAgent
        "Found Perfume Details (JSON): {perfume_details_json}. " # Injected by SequentialAgent
        "If the `perfume_details_json` indicates no specific match, acknowledge that and recommend a versatile classic."
        "Otherwise, elaborate on the recommended perfume, explaining its scent profile, suitable occasions, and why it's a good fit for the user's context. "
        "Use the `preload_memory` tool if you need to recall past preferences or conversations with the user to enhance your recommendation, "
        "especially if 'scent_preference' in the provided context was 'unknown' and memory might hold hints."
        "Always output the recommendation in a user-friendly, conversational format, not just raw data."
    ),
    tools=[preload_memory], # preload_memory is a built-in ADK tool and works directly.
    output_key="final_recommendation_text" # This will be the final output of the entire pipeline
)
print("âœ… PerfumeRecommenderAgent created with `preload_memory`.")


# ---

# ## 6. ADK Session & Memory Management

# ### Initializing Services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

print("âœ… ADK Session and Memory Services initialized.")

# ### Automatic Memory Saving Callback
async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    print("ğŸ’¾ [Callback]: Auto-saving session to memory...")
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )
    print("ğŸ’¾ [Callback]: Session saved.")

print("âœ… Auto-save memory callback created.")

# Re-create PerfumeRecommenderAgent with the auto-save callback
perfume_recommender_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="PerfumeRecommenderAgent",
    instruction=(
        "You are an expert perfume sommelier. Your task is to provide a friendly, detailed, and engaging perfume recommendation "
        "based on the provided user context and the found perfume details. "
        "User Context: {user_context}. "
        "Found Perfume Details (JSON): {perfume_details_json}. "
        "If the `perfume_details_json` indicates no specific match, acknowledge that and recommend a versatile classic."
        "Otherwise, elaborate on the recommended perfume, explaining its scent profile, suitable occasions, and why it's a good fit for the user's context. "
        "Use the `preload_memory` tool if you need to recall past preferences or conversations with the user to enhance your recommendation, "
        "especially if 'scent_preference' in the provided context was 'unknown' and memory might hold hints."
        "Always output the recommendation in a user-friendly, conversational format, not just raw data."
    ),
    tools=[preload_memory],
    after_agent_callback=auto_save_to_memory, # Saves after each turn!
    output_key="final_recommendation_text"
)

print("âœ… PerfumeRecommenderAgent re-created with auto-save memory callback.")


# ---

# ## 7. Main Agent Orchestration (Simplified using SequentialAgent and FunctionTools)
# This section defines the main `SequentialAgent` that orchestrates the entire flow.
# Each sub-agent will run in order, passing its output (via `output_key`) to the
# next agent, which can then use these as `input_key` placeholders in its instruction.

# Define the overall SequentialAgent pipeline
perfume_pipeline_agent = SequentialAgent(
    name="PerfumeRecommendationPipeline",
    sub_agents=[
        user_context_agent,      # Phase 1: Gathers initial context and calls weather tool
        perfume_search_agent,    # Phase 2: Uses context to lookup perfumes and get details
        perfume_recommender_agent # Phase 3: Formats the final recommendation
    ],
    # The output of the last agent in the sequence is the final output of the SequentialAgent.
    # In this case, it will be the "final_recommendation_text" from perfume_recommender_agent.
)

print("âœ… Perfume Recommendation Pipeline (SequentialAgent) created.")

# Now, define the `run_session` function that will use this pipeline.
# `InMemoryRunner` and `run_debug` method.


async def run_perfume_session(runner_instance: InMemoryRunner, user_input_prompt: str, session_id: str):
    """
    Runs the full perfume recommendation pipeline using the provided InMemoryRunner.
    This function simplifies the interaction to match the Kaggle course's style.
    """
    print("\n" + "#"*60)
    print(f"### Starting Perfume Recommendation Flow for Session ID: {session_id} ###")
    print("#"*60)

    response_events = await runner_instance.run_debug(
        user_input_prompt,
        session_id=session_id,
        user_id=DEMO_USER_ID
    )

    final_response_text = "Error: Could not retrieve final recommendation."

    # OPTION 1: Look for the specific output_key/output_value pattern
    for event in reversed(response_events):
        if hasattr(event, 'output_key') and event.output_key == "final_recommendation_text":
            if hasattr(event, 'output_value') and isinstance(event.output_value, str) and event.output_value.strip():
                final_response_text = event.output_value.strip()
                print(f"DEBUG: Found recommendation via output_key: {final_response_text[:50]}...")
                break
        # DEBUGGING AID: Print other AgentOutputEvents to see if we're missing something
        elif hasattr(event, 'output_key') and hasattr(event, 'output_value'):
             print(f"DEBUG: Found other AgentOutputEvent with key: {event.output_key}, value: {str(event.output_value)[:50]}...")


    # OPTION 2: If OPTION 1 failed, try to find the last *LLM generated text* event.
    if final_response_text == "Error: Could not retrieve final recommendation.":
        print("DEBUG: Option 1 (output_key) failed. Trying Option 2 (last LLM text content).")
        for event in reversed(response_events):
            # Check for events that contain direct LLM text output for the user
            if hasattr(event, 'content') and hasattr(event.content, 'parts') and event.content.parts:
                if len(event.content.parts) > 0 and hasattr(event.content.parts[0], 'text') and isinstance(event.content.parts[0].text, str) and event.content.parts[0].text.strip():
                    text = event.content.parts[0].text.strip()
                    # Filter out short or tool-call responses. We want the full agent's narrative.
                    if len(text) > 100 and not text.startswith('{' or 'function_call'): # Heuristic for a full LLM response
                        final_response_text = text
                        print(f"DEBUG: Found recommendation via content.parts[0].text (fallback): {final_response_text[:50]}...")
                        break
            # Check for generic text_content (e.g., from ToolCodeExecutionEvent, but also LLM output sometimes)
            elif hasattr(event, 'data') and hasattr(event.data, 'text_content') and isinstance(event.data.text_content, str) and event.data.text_content.strip():
                text = event.data.text_content.strip()
                if len(text) > 100 and not text.startswith('{' or 'function_call'):
                    final_response_text = text
                    print(f"DEBUG: Found recommendation via data.text_content (fallback): {final_response_text[:50]}...")
                    break
    if final_response_text == "Error: Could not retrieve final recommendation.":
        print("DEBUG: Option 2 (last LLM text content) also failed.")


    print("\n--- Final Recommendation from Pipeline ---")
    print(final_response_text)

    print("\n" + "#"*60)
    print(f"### Flow Complete for Session ID: {session_id} ###")
    print("#"*60)
    return final_response_text

print("âœ… `run_perfume_session` helper function defined, using SequentialAgent pipeline.")

# ---

print("\n--- End of Notebook Code ---")


perfume_runner = InMemoryRunner(
    agent=perfume_pipeline_agent,
    app_name=APP_NAME,
)


# Demo 1
await run_perfume_session(
    perfume_runner, # Pass the initialized runner instance
    "My city is New York. I'm feeling energetic, going to a casual outing, and I prefer citrus scents.",
    "conversation-01", # Session ID
)


#Demo 2
await run_perfume_session(
     perfume_runner,
     "I'm in Paris, feeling romantic, for an evening date. I like rose scents.",
     "conversation-02",
 )


 # Demo 3 Example demonstrating memory (first turn)
 await run_perfume_session(
     perfume_runner,
     "What perfume should I wear? I like sweet scents.",
     "conversation-03",
 )


 # Demo 3 Example demonstrating memory (second turn, for the same session_id)
 await run_perfume_session(
     perfume_runner,
     "And for a formal business meeting in a cool climate?",
     "conversation-03",
 )


 # Demo 3 Example demonstrating memory (Third turn, for the same session_id)
 await run_perfume_session(
     perfume_runner,
     "New York",
     "conversation-03",
 )


# Demo 4


await run_perfume_session(
    perfume_runner,
    "I'm in New York, today is 2023-10-30, feeling relaxed, going to a casual, and I prefer Aromatic Aquatic scents.",
    "conversation-04",
)

