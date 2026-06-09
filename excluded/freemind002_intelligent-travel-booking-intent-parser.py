import os
# from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field


# Flight Query Model
class FlightQuery(BaseModel):
    category: Literal["Flight"] = Field("Flight", description="Query type: Flight")
    airline: Optional[str] = Field(
        None,
        description="Preferred airline, e.g., 'EVA Air', 'JAL'. If not specified, use null.",
    )
    departure_city: str = Field(..., description="Departure city or airport code.")
    arrival_city: str = Field(..., description="Arrival city or airport code.")
    departure_date: str = Field(..., description="Departure date, e.g., '2025-12-25'.")
    return_date: Optional[str] = Field(
        None, description="Return date, if round trip. If not specified, use null."
    )
    adults: Optional[int] = Field(
        None,
        ge=1,
        description="Number of adults. If not specified in the query, the value must be null.",
    )


# Group Tour Query Model
class TourQuery(BaseModel):
    category: Literal["GroupTour"] = Field(
        "GroupTour", description="Query type: Group Tour"
    )
    city_or_region: str = Field(..., description="Target city or region.")
    start_date: Optional[str] = Field(
        None, description="Preferred start date. If not specified, use null."
    )
    duration_days: Optional[int] = Field(
        None, description="Preferred duration in days. If not specified, use null."
    )
    keywords: Optional[List[str]] = Field(
        None,
        description="Keywords for the tour, e.g., 'family-friendly', 'skiing'. If not specified, use null.",
    )


# Hotel Query Model
class HotelQuery(BaseModel):
    category: Literal["Hotel"] = Field("Hotel", description="Query type: Hotel")
    city: str = Field(..., description="Target city or area for the hotel.")
    check_in_date: str = Field(..., description="Check-in date.")
    check_out_date: str = Field(..., description="Check-out date.")
    guests: Optional[int] = Field(
        None,
        ge=1,
        description="Number of guests. If not specified in the query, the value must be null.",
    )
    keywords: Optional[List[str]] = Field(
        None,
        description="Keywords for hotel preference, e.g., 'near Shinjuku station', 'pet-friendly'. If not specified, use null.",
    )


# Itinerary Models
class DailyDesc(BaseModel):
    day: int = Field(..., ge=1, description="Day number, starting from 1")
    travel_point: str = Field(
        ..., description="Description of activities and places to visit"
    )
    breakfast: Optional[str] = Field(None, description="Breakfast recommendation")
    lunch: Optional[str] = Field(None, description="Lunch recommendation")
    dinner: Optional[str] = Field(None, description="Dinner recommendation")
    hotel: Optional[str] = Field(None, description="Hotel/accommodation for the night")


class TravelInfo(BaseModel):
    prod_name: str = Field(..., description="Trip name/title")
    daily_desc: List[DailyDesc] = Field(
        None, description="Daily itinerary descriptions"
    )


# Classification Result (for reference)
class ClassificationResult(BaseModel):
    category: str = Field(..., description="The category of the user's request.")
    parameters: Union[FlightQuery, TourQuery, HotelQuery, TravelInfo] = Field(
        ..., description="The structured data matching the determined category."
    )


# Flight Query Agent
flight_agent = Agent(
    name="flight_agent",
    model=Gemini(retry_options=retry_config),
    instruction="""Extract flight booking parameters from user query and output ONLY a JSON object:
    {
      "category": "Flight",
      "airline": "airline name or null",
      "departure_city": "departure city",
      "arrival_city": "arrival city",
      "departure_date": "YYYY-MM-DD",
      "return_date": "YYYY-MM-DD or null",
      "adults": number or null
    }
    Output ONLY the JSON object, no other text.""",
    tools=[],
    output_key="flight_result",
)

# Group Tour Query Agent
tour_agent = Agent(
    name="tour_agent",
    model=Gemini(retry_options=retry_config),
    instruction="""Extract tour booking parameters from user query and output ONLY a JSON object:
    {
      "category": "GroupTour",
      "city_or_region": "destination",
      "start_date": "YYYY-MM-DD or null",
      "duration_days": number or null,
      "keywords": ["keyword1", "keyword2"] or null
    }
    Output ONLY the JSON object, no other text.""",
    tools=[],
    output_key="tour_result",
)

# Hotel Query Agent
hotel_agent = Agent(
    name="hotel_agent",
    model=Gemini(retry_options=retry_config),
    instruction="""Extract hotel booking parameters from user query and output ONLY a JSON object:
    {
      "category": "Hotel",
      "city": "target city",
      "check_in_date": "YYYY-MM-DD",
      "check_out_date": "YYYY-MM-DD",
      "guests": number or null,
      "keywords": ["keyword1", "keyword2"] or null
    }
    Output ONLY the JSON object, no other text.""",
    tools=[],
    output_key="hotel_result",
)

# Research Agent (Stage 1 for Itineraries)
research_agent = Agent(
    name="research_agent",
    model=Gemini(retry_options=retry_config),
    instruction="""You are a travel research assistant. Use Google Search to find comprehensive information about:
    - Popular attractions and must-visit places
    - Restaurant and cuisine recommendations
    - Hotel and accommodation options
    - Transportation tips
    - Weather and seasonal considerations
    - Cultural tips and local customs
    
    Compile findings into a detailed, well-organized text summary. Do NOT output JSON.""",
    tools=[google_search],
    output_key="research_result",
)

# Itinerary Agent (Stage 2 for Itineraries)
itinerary_agent = Agent(
    name="itinerary_agent",
    model=Gemini(retry_options=retry_config),
    instruction="""Transform travel research into a day-by-day itinerary in JSON format:
    {
      "prod_name": "trip name",
      "daily_desc": [
        {
          "day": 1,
          "travel_point": "activities description",
          "breakfast": "recommendation or null",
          "lunch": "recommendation or null",
          "dinner": "recommendation or null",
          "hotel": "accommodation or null"
        }
      ]
    }
    Output ONLY the JSON object, no markdown.""",
    tools=[],
    output_key="itinerary_result",
)

# Root Agent (Coordinator)
root_agent = Agent(
    name="root_agent",
    model=Gemini(retry_options=retry_config),
    description="Travel Concierge that routes requests to specialized agents",
    instruction="""Analyze user request and immediately transfer to appropriate agent:
    
    - FLIGHT queries â†’ transfer to flight_agent
    - GROUP TOUR queries â†’ transfer to tour_agent
    - HOTEL queries â†’ transfer to hotel_agent
    
    Transfer immediately. Do not try to answer yourself.
    
    Note: Itinerary requests are handled separately by research and planning agents.""",
    sub_agents=[flight_agent, tour_agent, hotel_agent],
)

# Exit Intent Detection Agent
exit_intent_agent = Agent(
    name="exit_intent_agent",
    model=Gemini(retry_options=retry_config),
    instruction="""Detect if user wants to end the conversation. Output ONLY JSON:
    {
      "wants_to_exit": true/false,
      "confidence": "high"/"medium"/"low",
      "reason": "brief explanation"
    }
    
    Examples of exit intent:
    - "bye", "goodbye", "see you", "exit", "quit", "stop"
    - "I'm done", "That's all", "No more questions"
    - "Thanks, I'm good", "All set, thanks"
    - Any polite way of indicating they want to leave
    
    Output ONLY the JSON object, no other text.""",
    tools=[],
    output_key="exit_intent_result",
)

print("âœ… Agents configured successfully")


import json
import re
from pydantic import ValidationError
from pprint import pprint


def is_itinerary_request(query: str) -> bool:
    """Check if query is requesting an itinerary"""
    keywords = ["itinerary", "plan", "trip", "schedule", "travel plan"]
    return any(keyword in query.lower() for keyword in keywords)


def extract_content(state):
    """Extract text content from agent response state"""
    if hasattr(state.content, "parts") and state.content.parts:
        return state.content.parts[0].text
    elif hasattr(state.content, "text"):
        return state.content.text
    elif isinstance(state.content, str):
        return state.content
    return str(state.content)


def clean_json_text(text: str) -> str:
    """Remove markdown formatting from JSON text"""
    text = text.lstrip("itinerary_agent > ")
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


async def run_itinerary_flow_with_research(user_query: str):
    """Two-stage flow for itinerary: Research â†’ Structure"""
    print("ğŸ”� Processing itinerary...")

    # Stage 1: Research with Google Search
    research_runner = InMemoryRunner(agent=research_agent)
    research_responses = await research_runner.run_debug(user_query)
    research_text = extract_content(research_responses[-1])

    # Stage 2: Structure itinerary from research
    itinerary_runner = InMemoryRunner(agent=itinerary_agent)
    itinerary_prompt = f"""Based on the following travel research, create a detailed day-by-day itinerary:

=== TRAVEL RESEARCH ===
{research_text}

=== ORIGINAL REQUEST ===
{user_query}

Please create a structured JSON itinerary with prod_name and daily_desc fields."""

    itinerary_responses = await itinerary_runner.run_debug(itinerary_prompt)
    raw_json_text = extract_content(itinerary_responses[-1])
    raw_json_text = clean_json_text(raw_json_text)

    try:
        parsed_json = json.loads(raw_json_text)
        model_instance = TravelInfo(**parsed_json)
        result_dict = model_instance.model_dump(mode="json", exclude_none=False)
        print("âœ… Itinerary created successfully")
        return result_dict
    except Exception as e:
        print(f"â�Œ Error: {e}")
        return parsed_json if "parsed_json" in locals() else raw_json_text


async def run_root_agent_flow(user_query: str):
    """Main entry point: routes queries to appropriate handlers"""

    if is_itinerary_request(user_query):
        return await run_itinerary_flow_with_research(user_query)

    print("ğŸ”� Processing query...")

    root_runner = InMemoryRunner(agent=root_agent)
    response_list = await root_runner.run_debug(user_query)
    final_state = response_list[-1]

    try:
        raw_final_json_text = extract_content(final_state)
        author = final_state.author if hasattr(final_state, "author") else None

        # Handle research agent response
        if author == "research_agent" and not raw_final_json_text.strip().startswith(
            "{"
        ):
            itinerary_runner = InMemoryRunner(agent=itinerary_agent)
            itinerary_prompt = f"""Based on the following travel research, create a detailed itinerary:

{raw_final_json_text}

Original user request: {user_query}

Please output a structured JSON itinerary."""

            itinerary_responses = await itinerary_runner.run_debug(itinerary_prompt)
            raw_final_json_text = extract_content(itinerary_responses[-1])

        # Clean and parse JSON
        raw_final_json_text = clean_json_text(raw_final_json_text)
        parsed_json = json.loads(raw_final_json_text)
        category = parsed_json.get("category", "")

        # Infer category from JSON structure if not present
        if not category:
            if "prod_name" in parsed_json and "daily_desc" in parsed_json:
                category = "ItinerarySuggestion"
            elif "departure_city" in parsed_json and "arrival_city" in parsed_json:
                category = "Flight"
            elif "city_or_region" in parsed_json:
                category = "GroupTour"
            elif "check_in_date" in parsed_json:
                category = "Hotel"

        # Validate with Pydantic model
        model_map = {
            "Flight": FlightQuery,
            "GroupTour": TourQuery,
            "Hotel": HotelQuery,
            "ItinerarySuggestion": TravelInfo,
        }

        if category in model_map:
            model_instance = model_map[category](**parsed_json)
            result_dict = model_instance.model_dump(mode="json", exclude_none=False)
            print(f"âœ… Query classified as: {category}")
            return result_dict
        else:
            print(f"âš ï¸� Unknown category: {category}")
            return json.dumps(parsed_json, indent=2, ensure_ascii=False)

    except json.JSONDecodeError as e:
        print(f"â�Œ JSON parsing error: {e}")
        return raw_final_json_text
    except ValidationError as e:
        print(f"â�Œ Validation error: {e}")
        return json.dumps(parsed_json, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"â�Œ Error: {e}")
        return str(final_state)


async def is_exit_intent(query: str) -> bool:
    """Check if user wants to exit using Gemini AI"""
    try:
        exit_runner = InMemoryRunner(agent=exit_intent_agent)
        response_list = await exit_runner.run_debug(query)
        final_state = response_list[-1]

        # Extract response
        if hasattr(final_state.content, "parts") and final_state.content.parts:
            raw_text = final_state.content.parts[0].text
        elif hasattr(final_state.content, "text"):
            raw_text = final_state.content.text
        else:
            raw_text = str(final_state.content)

        # Clean and parse JSON
        raw_text = re.sub(r"^```json\s*", "", raw_text.strip())
        raw_text = re.sub(r"\s*```$", "", raw_text.strip())
        result = json.loads(raw_text)

        wants_to_exit = result.get("wants_to_exit", False)
        confidence = result.get("confidence", "low")

        # Only exit if high or medium confidence
        if wants_to_exit and confidence in ["high", "medium"]:
            return True

        return False

    except Exception as e:
        # Fallback to keyword matching
        exit_keywords = ["bye", "goodbye", "exit", "quit", "stop", "end"]
        return any(keyword in query.lower() for keyword in exit_keywords)


async def process_query(user_query: str):
    """
    Process a single user query with intent detection.
    Returns structured JSON or appropriate message.
    """
    print(f"\n{'=' * 80}")
    print(f"Query: {user_query}")
    print(f"{'=' * 80}")

    # Check for exit intent first
    if await is_exit_intent(user_query):
        print("ğŸ‘‹ Thank you for using Travel Assistant!\n")
        return {"intent": "exit", "message": "Thank you for using our service"}

    # Process the query
    try:
        result = await run_root_agent_flow(user_query)

        # Check if result is valid
        if isinstance(result, dict):
            category = result.get("category", "")
            # Check for both category field and itinerary structure
            has_itinerary_structure = "prod_name" in result and "daily_desc" in result

            if (
                category in ["Flight", "Hotel", "GroupTour", "ItinerarySuggestion"]
                or has_itinerary_structure
            ):
                # Successfully classified - return clean result
                print(f"\nğŸ“Š Result:")
                return result
            else:
                # Unclassified query
                print("âš ï¸� No corresponding category for your query")
                print("ğŸ’¡ Supported: Flight, Hotel, Tour, Itinerary\n")
                return {
                    "error": "No corresponding category",
                    "message": "Query type not supported",
                }
        else:
            # Failed to parse
            print("âš ï¸� Unable to classify query\n")
            return {"error": "Unclassified", "message": "Unable to classify query"}

    except Exception as e:
        print(f"â�Œ Error: {e}\n")
        return {"error": str(e), "message": "Processing error occurred"}


# Example 1: Flight Query
user_query = "I want to book a flight from Taipei to Tokyo on December 25, 2025"
result = await process_query(user_query)
print("==== Example 1: Flight Query Result =====")
final_result = {}
final_result["user_query"] = user_query
final_result["result"] = result
pprint(final_result)
json_data = json.dumps(final_result, indent=4)
with open("/kaggle/working/example_01.json", "w") as json_file:
    json_file.write(json_data)


# Example 2: Exit Intent
user_query = "I want to book a flight from Taipei to Tokyo on December 25, 2025"
result = await process_query(user_query)
print("==== Example 2: Exit Intent Result =====")
final_result = {}
final_result["user_query"] = user_query
final_result["result"] = result
pprint(final_result)
json_data = json.dumps(final_result, indent=4)
with open("/kaggle/working/example_02.json", "w") as json_file:
    json_file.write(json_data)


# Example 3: Unclassified Query
user_query = "What's the weather like today?"
result = await process_query(user_query)
print("==== Example 3: Unclassified Query Result =====")
final_result = {}
final_result["user_query"] = user_query
final_result["result"] = result
pprint(final_result)
json_data = json.dumps(final_result, indent=4)
with open("/kaggle/working/example_03.json", "w") as json_file:
    json_file.write(json_data)


# Example 4: Itinerary Planning
user_query = "Plan a 7-day trip to Japan starting March 10, 2026"
result = await process_query(user_query)
print("==== Example 4: Itinerary Planning Result =====")
final_result = {}
final_result["user_query"] = user_query
final_result["result"] = result
pprint(final_result)
json_data = json.dumps(final_result, indent=4)
with open("/kaggle/working/example_04.json", "w") as json_file:
    json_file.write(json_data)


# Example 5: Tour Package Query
user_query = "Looking for a 5-day tour package in Hokkaido"
result = await process_query(user_query)
print("==== Example 5: Tour Package Query Result =====")
final_result = {}
final_result["user_query"] = user_query
final_result["result"] = result
pprint(final_result)
json_data = json.dumps(final_result, indent=4)
with open("/kaggle/working/example_05.json", "w") as json_file:
    json_file.write(json_data)


# Example 6: Hotel Query
user_query = "Find a hotel in Kyoto for 3 nights, check-in January 5, check-out January 8"
result = await process_query(user_query)
print("==== Example 6: Hotel Query Result =====")
final_result = {}
final_result["user_query"] = user_query
final_result["result"] = result
pprint(final_result)
json_data = json.dumps(final_result, indent=4)
with open("/kaggle/working/example_06.json", "w") as json_file:
    json_file.write(json_data)


print("âœ… All tests completed")

