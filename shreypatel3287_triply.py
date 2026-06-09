# ------------------------------------------------------------
# LOAD GOOGLE API KEY FROM KAGGLE SECRETS
# ------------------------------------------------------------
# Kaggle does NOT allow hardcoding API keys inside notebooks.
# Instead, we securely store keys using "Kaggle Secrets".
#
# This block retrieves your GOOGLE_API_KEY safely and sets it
# as an environment variable so that Gemini + Google tools
# (like google_search) can authenticate properly.
# ------------------------------------------------------------

import os
from kaggle_secrets import UserSecretsClient

try:
    # Load the API key stored under the name "GOOGLE_API_KEY"
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")

    # Set it as an environment variable so the Agent Framework,
    # Gemini models, and Google Search tool can automatically use it.
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    print("âœ… Setup and authentication complete.")

except Exception as e:
    # If the user forgot to add the key in Kaggle Secrets,
    # show a clear, friendly error message.
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



# ------------------------------------------------------------
# IMPORTING ALL REQUIRED MODULES FROM THE AGENT DEVELOPMENT KIT
# ------------------------------------------------------------
# These imports give us access to:
# - LLM models (Gemini)
# - Agent abstractions (Agent, SequentialAgent, ParallelAgent)
# - Runners and Sessions for executing agent workflows
# - Tools such as Google Search
# - Code execution utilities for running Python within agents
#
# This block forms the foundation of the entire multi-agent system.
# ------------------------------------------------------------

# Core types for model configuration
from google.genai import types

# Agent classes:
# Agent = base class
# SequentialAgent = executes agents in order
# ParallelAgent = executes agents simultaneously when no dependency exists
from google.adk.agents import Agent, SequentialAgent, ParallelAgent

# Google LLM model wrapper (Gemini)
from google.adk.models.google_llm import Gemini

# Runner that executes agents in-memory (perfect for development & Kaggle)
from google.adk.runners import InMemoryRunner

# Session service to maintain agent state (conversation history, memory)
from google.adk.sessions import InMemorySessionService

# Tools such as Google Search that agents can call for real-world information
from google.adk.tools import AgentTool, ToolContext, google_search

# Optional: enables Python code execution within agents if needed
from google.adk.code_executors import BuiltInCodeExecutor


print("âœ… ADK modules imported.")



# ------------------------------------------------------------
# HTTP RETRY CONFIGURATION FOR GEMINI / TOOL CALLS
# ------------------------------------------------------------
# Sometimes API requests fail due to:
#  - rate limits (HTTP 429)
#  - server errors (500, 503, 504)
#
# To make the system more reliable, we configure automatic retries
# with exponential backoff. This ensures that temporary network or
# API instability does not break the entire agent workflow.
# ------------------------------------------------------------

retry_config = types.HttpRetryOptions(
    attempts=5,           # Try up to 5 times before failing
    exp_base=7,           # Exponential backoff multiplier (7^n delay growth)
    initial_delay=1,      # First retry delay starts at 1 second
    http_status_codes=[
        429,  # Too many requests â†’ often resolved by retrying
        500,  # Internal server error
        503,  # Service unavailable
        504   # Gateway timeout â†’ retry helps in most cases
    ],
)



# ------------------------------------------------------------
# REQUIREMENTS AGENT
# ------------------------------------------------------------
# Purpose:
#   - This is the FIRST agent in the entire pipeline.
#   - It takes the user's natural-language trip request and converts it
#     into a clean, structured JSON object.
#   - This structured output becomes the foundation for every other agent:
#       hotel_agent, weather_agent, places_agent, itinerary_agent, etc.
#
# Responsibilities:
#   âœ“ Extract location (e.g., Goa)
#   âœ“ Extract duration (days)
#   âœ“ Extract number of travellers
#   âœ“ Extract preferences (beaches, food, adventure, etc.)
#   âœ“ Extract or infer budget breakdown
#
# Budget Logic:
#   - If the user provides ONLY a total budget:
#       hotel = 45%,
#       food = 25%,
#       transport = 15%,
#       activities = 15%
#
#   - If user specifies custom values, they override defaults.
#
# Output:
#   Returns JSON in a STRICT schema (prevents hallucination and ensures
#   smooth context passing to the next agents).
#
# Why Important:
#   - EVERYTHING in the pipeline depends on this.
#   - Good structure here ensures the rest of the agents behave correctly.
# ------------------------------------------------------------
requirements_agent = Agent(
    name="requirements_agent",
    model=Gemini(model="gemini-2.5-flash"),
    instruction="""
You are a travel requirements extractor. Your job is to parse user messages 
and extract structured travel details.

Return a JSON object in this EXACT schema:

{
  "location": string,
  "days": integer,
  "travellers": integer,
  "preferences": [string],
  "budget": {
      "total": string,
      "per_traveller": string,
      "breakdown": {
          "hotel": string,
          "food": string,
          "transport": string,
          "activities": string
      }
  }
}

Rules:
- If the user provides only TOTAL budget, apply these default per-traveller ratios:
    - hotel = 45%
    - food = 25%
    - transport = 15%
    - activities = 15%

- If the user provides any specific costs, override defaults.
- If anything is missing â†’ return "missing".
""",
    output_key="requirements"
)



# ------------------------------------------------------------
# HOTEL AGENT
# ------------------------------------------------------------
# Purpose:
#   - This agent is responsible for finding hotel options for the user's trip.
#   - It uses the Google Search tool to fetch REAL hotel information.
#   - It filters hotels based on the user's hotel budget-per-night.
#
# Why this agent matters:
#   - Hotel selection is a major component of travel planning.
#   - Other agents (itinerary_agent, restaurants_agent) may reference hotel area.
#   - It demonstrates TOOL-AUGMENTED agent behavior (Google Search).
#
# Tool Usage:
#   - google_search: Allows the agent to pull live data from the web.
#
# Input:
#   The agent expects structured input containing:
#     { "location": "...", "days": <int>, "budget_per_night": <int> }
#   This input is automatically provided by earlier agents (requirements_agent).
#
# Output:
#   Must return STRICT JSON:
#     - A list of 4â€“6 hotel objects
#     - Each with: name, area, approx_price_per_night
#     - approx_price_per_night MUST be integer â†’ prevents LLM hallucinations.
#
# Notes:
#   - NO paragraphs, NO markdown, NO explanations allowed.
#   - This strictness ensures that downstream agents can parse and use this data.
# ------------------------------------------------------------
hotel_agent = Agent(
    name="hotel_agent",
    model=Gemini(model="gemini-2.5-flash"),

    tools=[google_search],

    instruction="""
You must ALWAYS return ONLY VALID JSON.
Never return paragraphs, markdown, tables, or descriptions.

Your job:
- Use the google_search tool to find 4â€“6 hotels.
- Filter hotels based on the user's hotel budget per night.
- Only return hotels with approx_price_per_night <= given budget.

Expected Input:
{
  "location": "...",
  "days": <number>,
  "budget_per_night": <number>   // derived from requirements_agent
}

OUTPUT FORMAT (strict):
{
  "hotels": [
    {
      "name": "...",
      "area": "...",
      "approx_price_per_night": <number>
    }
  ]
}

RULES:
- approx_price_per_night MUST be an integer.
- Must ALWAYS be valid JSON.
- No text, no explanations.
- No markdown.
""",
    output_key="hotels"
)



# ------------------------------------------------------------
# PLACES AGENT (ATTRACTIONS DISCOVERY)
# ------------------------------------------------------------
# Purpose:
#   - This agent searches for tourist attractions, activities, and sightseeing
#     spots for the user's destination.
#   - It uses the Google Search tool to pull real-world data.
#   - It filters out attractions that exceed the user's activities budget.
#   - It returns the correct number of attractions based on trip duration.
#
# Why this agent is important:
#   - Attractions are the core of itinerary planning.
#   - Downstream agents like itinerary_agent depend on high-quality,
#     budget-filtered, structured attraction data.
#   - It demonstrates tool usage + filtering logic + structured JSON outputs.
#
# Tool Usage:
#   - google_search: fetches real descriptions, costs, and suggestions.
#
# Input expected (auto-provided by requirements_agent):
#   {
#     "location": "...",
#     "days": <number>,
#     "activities_budget_per_person": <number>
#   }
#
# Output:
#   - STRICT JSON containing an "attractions" list.
#   - Each attraction includes:
#       name, description, best_time_to_visit, approx_cost
#   - approx_cost must ALWAYS be an integer
#   - No explanations, markdown, bullet points, or extra fluff
#
# Notes:
#   - Attraction count = days Ã— (1 to 2), giving the itinerary sufficient options.
#   - If cost cannot be found with certainty, the agent estimates a reasonable number.
#   - This ensures itinerary_agent does not break due to missing data.
# ------------------------------------------------------------
places_agent = Agent(
    name="places_agent",
    model=Gemini(model="gemini-2.5-flash"),

    tools=[google_search],

    instruction="""
You are an attractions discovery agent.

Input will be in this format:
{
  "location": "...",
  "days": <number>,
  "activities_budget_per_person": <number>
}

Your job:
1. Use google_search to find the most popular tourist attractions,
   activities, or sightseeing spots for the destination.
2. Only include attractions whose approx_cost PER PERSON
   is <= activities_budget_per_person.
3. Return the correct number of attractions based on days , which people can travel.

Your response MUST ALWAYS be valid JSON with this schema:

{
  "attractions": [
    {
      "name": "...",
      "description": "...",
      "best_time_to_visit": "...",
      "approx_cost": <number>
    }
  ]
}

RULES:
- approx_cost must be an INTEGER.
- No explanations.
- No markdown.
- No bullet points.
- If cost is not found online, estimate a reasonable cost.
- Filter out attractions that exceed the budget.
""",
    output_key="places"
)


# ------------------------------------------------------------
# WEATHER AGENT
# ------------------------------------------------------------
# Purpose:
#   - Fetch real-time weather information for the user's destination.
#   - Provides usable data for itinerary planning:
#       * Temperature
#       * Humidity
#       * Rainfall chance
#       * Weather warnings
#       * Clothing recommendations
#
# Why this is important:
#   - The itinerary_agent uses weather data to plan suitable activities
#       (e.g., avoid beaches during heavy rain, choose indoor attractions).
#   - restaurants_agent also considers weather (e.g., avoid rooftop dining if rainy).
#   - This agent demonstrates tool usage + structured reasoning.
#
# Tool Usage:
#   - google_search: Used to fetch live weather data for the location.
#
# Output Format:
#   Must ALWAYS return strict JSON:
#   {
#     "weather_summary": "...",
#     "temperature": "...",
#     "rainfall_chance": "...",
#     "humidity": "...",
#     "travel_warnings": "...",
#     "packing_suggestions": ["...", "..."]
#   }
#
# Notes:
#   - Output must be concise, actionable, and trip-focused.
#   - No markdown, no explanations, no long paragraphs.
# ------------------------------------------------------------
weather_agent = Agent(
    name="weather_agent",
    model=Gemini(model="gemini-2.5-flash"),

    tools=[google_search],   # IMPORTANT: tool passed directly

    instruction="""
You are a weather information agent.

Use the google_search tool to retrieve:
- temperature
- humidity
- rainfall prediction
- best clothing suggestions
- any weather warnings

Your response MUST be a JSON object like:
{
  "weather_summary": "...",
  "temperature": "...",
  "rainfall_chance": "...",
  "humidity": "...",
  "travel_warnings": "...",
  "packing_suggestions": ["...", "..."]
}

Always keep responses short, clean, and useful for planning a trip.
""",
    output_key="weather"
)


# ------------------------------------------------------------
# ITINERARY AGENT
# ------------------------------------------------------------
# Purpose:
#   - This agent is responsible for building the full **day-wise itinerary**
#     based on:
#       * weather data
#       * hotel location
#       * list of attractions
#       * number of days
#
#   - It transforms raw attraction data into a practical travel plan.
#
# Why this agent is important:
#   - This is the "brain" of the trip planning pipeline.
#   - The itinerary influences:
#       * transport_agent  (needs route sequence)
#       * restaurants_agent (restaurants must match planned locations)
#   - It demonstrates context-aware, multi-input LLM reasoning.
#
# How it works:
#   - Uses weather to decide indoor/outdoor scheduling.
#   - Uses hotel_area to start days near the hotel.
#   - Distributes attractions evenly across trip days.
#   - Creates a logical chronological flow with simple time slots.
#
# Input (automatically supplied by previous agents):
#   {
#     "location": "...",
#     "days": <int>,
#     "hotel_area": "...",
#     "weather": {...},
#     "attractions": [...]
#   }
#
# Output (STRICT JSON):
#   {
#     "itinerary": [
#       {
#         "day": <int>,
#         "plan": [
#           { "time": "...", "activity": "...", "location": "...", "notes": "..." }
#         ]
#       }
#     ]
#   }
#
# Notes:
#   - No markdown, no extra explanations, no narrative text.
#   - Time slots must be simple and consistent (e.g., "9:00 AM").
#   - The number of activities per day should be reasonable (2â€“3).
#   - If attractions are fewer than required, the agent distributes them logically.
# ------------------------------------------------------------
itinerary_agent = Agent(
    name="itinerary_agent",
    model=Gemini(model="gemini-2.5-flash"),

    instruction="""
You are a travel itinerary builder.

INPUT FORMAT:
{
  "location": "...",
  "days": <number>,
  "hotel_area": "...",
  "weather": {
      "temperature": "...",
      "rainfall_chance": "...",
      "humidity": "...",
      "travel_warnings": "..."
  },
  "attractions": [
    {
      "name": "...",
      "description": "...",
      "best_time_to_visit": "...",
      "approx_cost": <number>
    }
  ]
}

Your job:
1. Create a **day-wise itinerary**.
2. Use weather data:
   - If rainfall_chance is high, schedule **indoor attractions** first.
   - If weather is clear, schedule outdoor attractions earlier.
3. Use hotel_area:
   - Start each day with attractions close to the hotel.
4. Distribute attractions evenly across days.
5. Provide 2â€“3 activities per day if available.
6. Keep itinerary practical and in logical geographic flow.

OUTPUT FORMAT (STRICT JSON):
{
  "itinerary": [
    {
      "day": <number>,
      "plan": [
        {
          "time": "...",
          "activity": "...",
          "location": "...",
          "notes": "..."
        }
      ]
    }
  ]
}

RULES:
- Must ALWAYS return VALID JSON.
- NO markdown, NO explanations.
- If number of attractions < needed, spread reasonably.
- Use simple time slots like: "9:00 AM", "12:00 PM", "4:00 PM".
""",
    output_key="itinerary"
)


# ------------------------------------------------------------
# TRANSPORT AGENT
# ------------------------------------------------------------
# Purpose:
#   - This agent determines the best transport mode for each movement in the
#     itinerary (hotel â†’ attraction 1 â†’ attraction 2 â†’ ...).
#   - It ensures that total transport cost stays within the per-person budget.
#
# Why this agent is important:
#   - Transport directly affects budget feasibility.
#   - It ensures the trip is realistic, safe in bad weather, and cost-effective.
#   - Downstream agent (restaurants_agent) uses transport mode to choose
#     nearby / accessible restaurants.
#
# Inputs (automatically provided from previous agents):
#   {
#     "location": "...",
#     "transport_budget_per_person": <int>,
#     "hotel_area": "...",
#     "itinerary": [... day wise plan ...]
#   }
#
# Reasoning criteria:
#   - Walking for same-area attractions
#   - Auto-rickshaw for <5 km
#   - Taxi for >5 km
#   - Scooter rental if budget allows
#   - Bus for cheap & long routes
#   - Avoid scooters when rainfall chance is high
#
# Output (STRICT JSON):
#   {
#     "transport_plan": [
#       {
#         "day": <int>,
#         "routes": [
#           {
#             "from": "...",
#             "to": "...",
#             "recommended_mode": "...",
#             "estimated_cost": <int>,
#             "notes": "..."
#           }
#         ],
#         "total_day_cost": <int>
#       }
#     ],
#     "total_trip_cost": <int>
#   }
#
# Notes:
#   - ZERO markdown, ZERO explanations â€” strict JSON only.
#   - Costs MUST be integers â†’ prevents parsing issues for next agents.
#   - If itinerary lacks distance info:
#         same area â†’ walk
#         nearby (0â€“5 km) â†’ auto
#         long (>5 km) â†’ taxi/scooter
#   - Must NOT exceed transport_budget_per_person.
# ------------------------------------------------------------
transport_agent = Agent(
    name="transport_agent",
    model=Gemini(model="gemini-2.5-flash"),

    instruction="""
You are a transport planning agent.

INPUT FORMAT:
{
  "location": "...",
  "transport_budget_per_person": <number>,
  "hotel_area": "...",
  "itinerary": [
    {
      "day": <number>,
      "plan": [
        {
          "time": "...",
          "activity": "...",
          "location": "...",
          "notes": "..."
        }
      ]
    }
  ]
}

Your tasks:
1. For every DAY, and every SEQUENCE of locations:
      hotel_area â†’ attraction 1 â†’ attraction 2 â†’ ...
   determine the best transport mode.
2. Consider:
   - walk if distance is short (within same area).
   - local auto/tuk-tuk for <5 km.
   - taxi for >5 km.
   - scooter rental if it fits whole-day budget.
   - bus if route is common and cheaper.
3. Total transport cost per person MUST NOT exceed transport_budget_per_person.
4. If budget is low, prefer bus, shared taxis, walking, or scooter rental.
5. If rainfall_chance in itinerary notes is â€œHighâ€�, avoid scooters.

OUTPUT FORMAT (STRICT JSON):
{
  "transport_plan": [
    {
      "day": <number>,
      "routes": [
        {
          "from": "...",
          "to": "...",
          "recommended_mode": "...",
          "estimated_cost": <number>,
          "notes": "..."
        }
      ],
      "total_day_cost": <number>
    }
  ],
  "total_trip_cost": <number>
}

RULES:
- ALWAYS return valid JSON.
- estimated_cost MUST be an integer.
- total_day_cost and total_trip_cost must be integers.
- No markdown, no explanations.
- If itinerary gives no distance, assume:
     - same area = walking
     - nearby areas (0â€“5 km) = auto
     - different zones (>5 km) = taxi/scooter
""",
    output_key="transport"
)


# ------------------------------------------------------------
# RESTAURANTS AGENT
# ------------------------------------------------------------
# Purpose:
#   - Recommend restaurants for each day of the trip based on:
#       * itinerary locations
#       * hotel area
#       * transport routes
#       * weather conditions
#       * food budget per person
#
# Why this agent is important:
#   - It finalizes one of the most important parts of travel planning.
#   - It depends heavily on ALL previous agents (weather, attractions,
#     transport, itinerary).
#   - It demonstrates deep multi-agent dependency reasoning.
#   - Combines tool usage + filtering logic + contextual alignment.
#
# Tools:
#   - google_search: fetches real restaurants, ratings, cuisine types, pricing,
#     and location details.
#
# Inputs (auto-supplied by earlier agents in pipeline):
#   {
#     "location": "...",
#     "food_budget_per_person": <int>,
#     "hotel_area": "...",
#     "weather": {...},
#     "itinerary": [...],
#     "transport_plan": [...]
#   }
#
# Agent Logic:
#   1. For every day â†’ look at the attraction and activity locations.
#   2. Use google_search to find restaurants near these locations.
#   3. Filter restaurants by:
#        - food_budget_per_person
#        - weather suitability (no rooftop/open air when rainy)
#        - accessibility based on transport
#   4. Recommend 2â€“3 restaurants per day.
#
# Weather rules:
#   - Rainy weather â†’ indoor dining
#   - Clear weather â†’ allow rooftop, sea-facing, or open-air restaurants
#
# Output (STRICT JSON):
#   {
#     "restaurants_plan": [
#       {
#         "day": <int>,
#         "restaurants": [
#           {
#             "name": "...",
#             "area": "...",
#             "cuisine": "...",
#             "approx_price_per_person": <int>,
#             "distance_from_spot": "...",
#             "suitable_for_weather": true/false
#           }
#         ]
#       }
#     ]
#   }
#
# Notes:
#   - approx_price_per_person MUST be an integer.
#   - ZERO markdown, ZERO explanations, ZERO extra text.
#   - If exact pricing is not known â†’ estimate reasonable integer cost.
# ------------------------------------------------------------
restaurants_agent = Agent(
    name="restaurants_agent",
    model=Gemini(model="gemini-2.5-flash"),

    tools=[google_search],

    instruction="""
You are a restaurant recommendation agent.

INPUT FORMAT:
{
  "location": "...",
  "food_budget_per_person": <number>,
  "hotel_area": "...",
  "weather": {
      "temperature": "...",
      "rainfall_chance": "...",
      "humidity": "...",
      "travel_warnings": "..."
  },
  "itinerary": [
    {
      "day": <number>,
      "plan": [
        {
          "time": "...",
          "activity": "...",
          "location": "...",
          "notes": "..."
        }
      ]
    }
  ],
  "transport_plan": [
    {
      "day": <number>,
      "routes": [
        {
          "from": "...",
          "to": "...",
          "recommended_mode": "...",
          "estimated_cost": <number>,
          "notes": "..."
        }
      ],
      "total_day_cost": <number>
    }
  ]
}

Your tasks:
1. For each day, find restaurants NEAR:
      - each attraction location
      - hotel_area
      - transport routes
2. Use google_search to find real restaurants.
3. Filter restaurants whose approx_price_per_person > food_budget_per_person.
4. Consider weather:
      - If rainfall_chance is high â†’ avoid rooftop/open-air restaurants.
      - If weather is clear â†’ allow outdoor cafes.
5. For each day, recommend 2â€“3 restaurants.

OUTPUT FORMAT (STRICT JSON):
{
  "restaurants_plan": [
    {
      "day": <number>,
      "restaurants": [
        {
          "name": "...",
          "area": "...",
          "cuisine": "...",
          "approx_price_per_person": <number>,
          "distance_from_spot": "...",
          "suitable_for_weather": true
        }
      ]
    }
  ]
}

RULES:
- ALWAYS return valid JSON.
- approx_price_per_person MUST be an integer.
- No explanations, no markdown, no text outside JSON.
- If exact price is not found, estimate a reasonable integer.
""",
    output_key="food"
)


# ------------------------------------------------------------
# TRIP SUMMARY AGENT
# ------------------------------------------------------------
# Purpose:
#   - This is the FINAL agent in the entire multi-agent travel pipeline.
#   - It takes structured outputs from all previous agents:
#       * requirements_agent
#       * hotel_agent
#       * weather_agent
#       * places_agent
#       * itinerary_agent
#       * transport_agent
#       * restaurants_agent
#   - And transforms them into a clean, human-readable, day-wise trip plan.
#
# Why this agent is important:
#   - Users do NOT want raw JSON â€” they want a natural, friendly travel plan.
#   - This is the moment where the entire pipeline becomes understandable.
#   - It demonstrates aggregation, reasoning, and human-friendly generation.
#
# Input:
#   A large combined context dictionary containing:
#       - Budget breakdown
#       - Hotel details
#       - Weather summary
#       - Daily itinerary
#       - Restaurant plan
#       - Transport plan
#
# Output:
#   A warm, friendly, narrative-style trip plan.
#   STRICT RULE: No JSON. No code. No schemas. Natural language only.
#
# Output Style:
#   For each day:
#       - Attractions
#       - Transport advice
#       - Restaurant suggestions
#       - Weather notes
#       - Tips and reminders
#
#   At the end:
#       - Overall summary
#       - Packing suggestions
#       - Weather warnings
#       - Budget insights
#
# Notes:
#   - This agent turns structured data â†’ readable text.
#   - It is the MOST user-facing part of the system.
# ------------------------------------------------------------
trip_summary_agent = Agent(
    name="trip_summary_agent",
    model=Gemini(model="gemini-2.5-flash"),

    instruction="""
You are a travel summary generator.  
Your job is to take all structured outputs from previous agents and create a **clean, human-readable, day-wise trip plan**.

INPUT FORMAT:
{
  "location": "...",
  "days": <number>,
  "budget": {
      "total": "...",
      "per_traveller": "...",
      "breakdown": {
          "hotel": "...",
          "food": "...",
          "transport": "...",
          "activities": "..."
      }
  },
  "hotel": {
      "name": "...",
      "area": "...",
      "approx_price_per_night": <number>
  },
  "weather": {
      "weather_summary": "...",
      "temperature": "...",
      "rainfall_chance": "...",
      "humidity": "...",
      "travel_warnings": "...",
      "packing_suggestions": ["...", "..."]
  },
  "itinerary": [
    {
      "day": <number>,
      "plan": [
        {"time": "...", "activity": "...", "location": "...", "notes": "..."}
      ]
    }
  ],
  "restaurants_plan": [
    {
      "day": <number>,
      "restaurants": [
        {
          "name": "...",
          "area": "...",
          "cuisine": "...",
          "approx_price_per_person": <number>,
          "suitable_for_weather": true
        }
      ]
    }
  ],
  "transport_plan": [
    {
      "day": <number>,
      "routes": [
        {
          "from": "...",
          "to": "...",
          "recommended_mode": "...",
          "estimated_cost": <number>,
          "notes": "..."
        }
      ],
      "total_day_cost": <number>
    }
  ]
}

YOUR TASK:
- Combine ALL the above info into a clear, friendly, day-by-day narrative.
- The user should feel like they are reading a finalized trip plan.
- For each day, include:
    â€¢ Key attractions  
    â€¢ Transport between attractions  
    â€¢ Recommended restaurants  
    â€¢ Weather considerations  
    â€¢ Practical tips  

FORMAT (Human-readable, NO JSON):
----------------------------------------------------
ğŸŒ� Day 1:
- Morning: Visit ...
- Afternoon Transport: Taxi from ... to ...
- Lunch Suggestion: ...
- Evening: ...
- Tips: ...
----------------------------------------------------

At the end, give a small summary:
- Total estimated transport cost
- Per day food/hotel/activity highlights
- Weather warnings (if any)
- Packing suggestions

RULES:
- ABSOLUTELY NO JSON.
- No markdown formatting beyond simple emojis or clean bullet points.
- Make it friendly, warm, and easy to read.
- Ensure the summary matches the actual data provided.
""",
    output_key="summary"
)


# ------------------------------------------------------------
# PARALLEL AGENT BLOCK
# ------------------------------------------------------------
# Purpose:
#   - This block runs multiple agents *simultaneously*:
#         1. hotel_agent
#         2. weather_agent
#         3. places_agent
#
#   - These agents do not depend on each otherâ€™s outputs.
#   - Running them in parallel significantly reduces total
#     execution time because all three make Google Search tool
#     calls, which are slow if executed sequentially.
#
# Why Parallel Execution Matters:
#   - Demonstrates your understanding of one of the key course concepts.
#   - Improves system performance in real-world use.
#   - Matches the ideal design pattern for independent subtasks.
#
# Data Flow:
#   requirements_agent â†’ (parallel) â†’ itinerary_agent
#
#   After extracting user requirements, we can immediately:
#       * Fetch hotel options
#       * Fetch weather details
#       * Fetch tourist attractions
#
# Output:
#   The ParallelAgent collects outputs from all sub-agents and
#   merges them into a single context dictionary, making it
#   available to the next agent in sequence (itinerary_agent).
# ------------------------------------------------------------
parallel_hotel_weather_places = ParallelAgent(
    name="Parallel_Hotel_Weather_Places",
    sub_agents=[
        hotel_agent,
        weather_agent,
        places_agent
    ],
)


# ------------------------------------------------------------
# ROOT SEQUENTIAL PIPELINE (THE HEART OF THE SYSTEM)
# ------------------------------------------------------------
# Purpose:
#   - This SequentialAgent defines the *entire orchestration* of the
#     multi-agent travel planning workflow.
#   - Each agent runs in a strict top-to-bottom order,
#     ensuring that dependent agents always receive the correct data.
#
# Why this pipeline is crucial:
#   - Demonstrates "Sequential Agents" â€” a major course requirement.
#   - Makes the whole system behave like a real AI product.
#   - Ensures clean data flow from extraction â†’ research â†’ planning â†’ summary.
#
# Pipeline Flow:
#
#   1. requirements_agent
#        - Extracts clean structured travel inputs.
#
#   2. parallel_hotel_weather_places   (ParallelAgent)
#        - Fetches hotels, weather, attractions simultaneously.
#        - Greatly improves efficiency.
#
#   3. itinerary_agent
#        - Builds a day-wise trip plan using outputs from step #2.
#
#   4. transport_agent
#        - Determines best transport for each route in the itinerary.
#
#   5. restaurants_agent
#        - Suggests restaurants that match:
#            * itinerary locations
#            * food budget
#            * weather
#            * transport constraints
#
#   6. trip_summary_agent
#        - Converts all collected JSON into a readable final travel plan.
#
#
# Why SequentialAgent is perfect here:
#   - Every step depends on data from the previous one.
#   - Enforces deterministic, clean, predictable logic.
#   - Removes ambiguity and hallucination issues.
#
# Output:
#   - The final output is a warm, narrative-style trip summary
#     produced by the last agent (trip_summary_agent).
# ------------------------------------------------------------

root_agent = SequentialAgent(
    name="TravelPlanningSystem",

    # Ordered list of sub-agents defining the complete workflow
    sub_agents=[
        requirements_agent,                  # 1. Extract structured user inputs

        parallel_hotel_weather_places,       # 2. Parallel: hotels + weather + attractions

        itinerary_agent,                     # 3. Build day-wise itinerary

        transport_agent,                     # 4. Generate transport plan

        restaurants_agent,                   # 5. Recommend restaurants based on all above

        trip_summary_agent                   # 6. Produce final human-friendly trip plan
    ],
)



# Create the runner for your travel system
runner = InMemoryRunner(agent=root_agent, app_name="Triply")

# Run in debug mode to see each agent step clearly
response = await runner.run_debug(
    "Plan a Dwarka trip for 3 days for 4 people under 25,000 INR with temple visits and vegetarian food."
)


