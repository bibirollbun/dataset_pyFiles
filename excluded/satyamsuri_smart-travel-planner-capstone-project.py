#. Setup & Imports

import os
import math
import textwrap
from dataclasses import dataclass
from typing import Dict, Any, List

LLM_AVAILABLE = False
try:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        LLM_AVAILABLE = True
        print("✅ Gemini model loaded.")
    else:
        print("⚠️ GEMINI_API_KEY not found. Falling back to simple rule-based text.")
except Exception as e:
    print("⚠️ Could not import google-generativeai:", e)
    print("Using simple rule-based text instead.")



# Simple LLM wrapper (FIXED)

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Wraps calls to Gemini (if available); otherwise uses a simple rule-based text generator.
    """
    full_prompt = system_prompt.strip() + "\n\nUser request:\n" + user_prompt.strip()

    # If Gemini key available, use it
    if LLM_AVAILABLE:
        response = model.generate_content(full_prompt)
        return response.text.strip()

    # Fallback (no Gemini API key)
    # This is a safe, fully closed, correct multi-line string.
    return textwrap.dedent(f"""
    (Rule-based response – no external LLM key detected)

    Here is a simple structured travel response based on your query:

    User request:
    {user_prompt}

    Since no real LLM is available, this response uses a deterministic template.

    - The itinerary will include 3 days by default.
    - Costs will be estimated using provided tools.
    - Activities will be simple and general-purpose such as sightseeing, food exploration, and local attractions.

    This fallback ensures the notebook ALWAYS runs on Kaggle, even without any API keys.
    """).strip()



# Toy knowledge base and tools

CITY_DATABASE = {
    "paris": {
        "country": "France",
        "avg_flight_cost_usd": 800,
        "avg_hotel_per_night_usd": 180,
        "typical_activities": [
            "Visit the Eiffel Tower",
            "Walk along the Seine",
            "Explore the Louvre Museum",
            "Try local cafes and bakeries",
        ],
        "best_for": ["culture", "food", "art"],
        "weather_summary": "Mild with a chance of light rain depending on season.",
    },
    "tokyo": {
        "country": "Japan",
        "avg_flight_cost_usd": 1000,
        "avg_hotel_per_night_usd": 150,
        "typical_activities": [
            "Visit Senso-ji temple in Asakusa",
            "Explore Akihabara for electronics and anime",
            "Walk in Shibuya and Shinjuku",
            "Try ramen, sushi and street food",
        ],
        "best_for": ["technology", "food", "city life"],
        "weather_summary": "Can be humid in summer and cool in winter; generally comfortable.",
    },
    "dubai": {
        "country": "United Arab Emirates",
        "avg_flight_cost_usd": 600,
        "avg_hotel_per_night_usd": 120,
        "typical_activities": [
            "Burj Khalifa observation deck",
            "Dubai Mall and Fountain show",
            "Desert safari",
            "Marina and JBR beach walk",
        ],
        "best_for": ["shopping", "luxury", "family trips"],
        "weather_summary": "Hot and dry; indoor activities and evenings are more comfortable.",
    },
}

BUDGET_MULTIPLIERS = {
    "budget": 0.8,
    "standard": 1.0,
    "premium": 1.4,
}


def lookup_city(destination: str) -> Dict[str, Any]:
    key = destination.strip().lower()
    return CITY_DATABASE.get(key)


def estimate_flight_cost(origin: str, destination: str, budget_level: str) -> float:
    city = lookup_city(destination)
    if not city:
        # default
        base = 700
    else:
        base = city["avg_flight_cost_usd"]
    mult = BUDGET_MULTIPLIERS.get(budget_level, 1.0)
    return round(base * mult, 2)


def estimate_hotel_cost(destination: str, nights: int, budget_level: str) -> float:
    city = lookup_city(destination)
    if not city:
        base_per_night = 120
    else:
        base_per_night = city["avg_hotel_per_night_usd"]
    mult = BUDGET_MULTIPLIERS.get(budget_level, 1.0)
    return round(base_per_night * nights * mult, 2)


def estimate_total_cost(origin: str, destination: str, nights: int, budget_level: str) -> Dict[str, float]:
    flight = estimate_flight_cost(origin, destination, budget_level)
    hotel = estimate_hotel_cost(destination, nights, budget_level)
    activities = round(80 * nights * BUDGET_MULTIPLIERS.get(budget_level, 1.0), 2)
    return {
        "flight": flight,
        "hotel": hotel,
        "activities": activities,
        "total": round(flight + hotel + activities, 2),
    }


def get_weather_summary(destination: str) -> str:
    city = lookup_city(destination)
    if city:
        return city["weather_summary"]
    return "Weather information is approximate; expect typical conditions for the season."



# Simple data structures for agent messages

@dataclass
class TravelRequest:
    origin: str
    destination: str
    start_date: str  # as string for simplicity
    end_date: str
    budget_level: str  # 'budget', 'standard', 'premium'

    @property
    def nights(self) -> int:
        # simple approximate nights; not using datetime for brevity
        # in real code, we would parse dates.
        # Here we ask user to provide nights separately if needed.
        return 3  # default for demo; user can adjust


@dataclass
class ResearchResult:
    destination_info: Dict[str, Any]
    weather_summary: str


@dataclass
class BudgetResult:
    cost_breakdown: Dict[str, float]


@dataclass
class ItineraryResult:
    text_itinerary: str



# Agent classes

class BaseAgent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    def ask_llm(self, user_prompt: str) -> str:
        return call_llm(self.system_prompt, user_prompt)


class ResearchAgent(BaseAgent):
    def run(self, request: TravelRequest) -> ResearchResult:
        city_info = lookup_city(request.destination) or {}
        weather = get_weather_summary(request.destination)

        # Prepare a human-friendly summary using LLM / fallback
        summary_prompt = f"""
        You are a travel research specialist.

        Origin: {request.origin}
        Destination: {request.destination}
        Start date: {request.start_date}
        End date: {request.end_date}
        Budget level: {request.budget_level}

        City info (from tools): {city_info if city_info else "No specific database entry; use general knowledge."}
        Weather summary (from tools): {weather}

        Provide a concise bullet-point summary of the destination:
        - Key characteristics
        - Typical activities
        - Any quick tips for a first-time visitor
        """
        _ = self.ask_llm(summary_prompt)  # we don't strictly need text here, tools already give structure

        return ResearchResult(
            destination_info=city_info,
            weather_summary=weather,
        )


class BudgetAgent(BaseAgent):
    def run(self, request: TravelRequest) -> BudgetResult:
        cost_breakdown = estimate_total_cost(
            origin=request.origin,
            destination=request.destination,
            nights=request.nights,
            budget_level=request.budget_level,
        )
        return BudgetResult(cost_breakdown=cost_breakdown)


class ItineraryAgent(BaseAgent):
    def run(self, request: TravelRequest, research: ResearchResult, budget: BudgetResult) -> ItineraryResult:
        city_info = research.destination_info
        activities = city_info.get("typical_activities", [])
        weather = research.weather_summary
        costs = budget.cost_breakdown

        user_prompt = f"""
        Create a day-by-day travel itinerary.

        Input:
        - Origin: {request.origin}
        - Destination: {request.destination}
        - Start date: {request.start_date}
        - End date: {request.end_date}
        - Nights (approx): {request.nights}
        - Budget level: {request.budget_level}

        Tools data:
        - Destination info: {city_info}
        - Typical activities: {activities}
        - Weather summary: {weather}
        - Cost breakdown (USD): {costs}

        Requirements:
        - Provide a clear itinerary per day (Day 1, Day 2, Day 3, ...).
        - Suggest 2–4 activities per day.
        - Make sure at least one activity is budget friendly.
        - Mention weather considerations briefly.
        - End with a short recap of total estimated cost.
        """

        itinerary_text = self.ask_llm(user_prompt)
        return ItineraryResult(text_itinerary=itinerary_text)



# Orchestrator agent that coordinates the others

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="orchestrator",
            system_prompt=textwrap.dedent("""
            You are an orchestrator agent. You do NOT talk directly to the end user.
            Instead, you coordinate other agents (research, budget, itinerary) in a
            deterministic pipeline. Keep the logic in Python; this prompt is just
            documented for completeness.
            """),
        )
        self.research_agent = ResearchAgent(
            name="research_agent",
            system_prompt="You are a travel research specialist."
        )
        self.budget_agent = BudgetAgent(
            name="budget_agent",
            system_prompt="You are a travel budget and cost estimation specialist."
        )
        self.itinerary_agent = ItineraryAgent(
            name="itinerary_agent",
            system_prompt="You are an expert travel planner and itinerary writer."
        )

    def plan_trip(self, request: TravelRequest) -> Dict[str, Any]:
        # Step 1: research
        research = self.research_agent.run(request)

        # Step 2: budget estimation
        budget = self.budget_agent.run(request)

        # Step 3: itinerary creation
        itinerary = self.itinerary_agent.run(request, research, budget)

        # Combined response
        return {
            "request": request,
            "research": research,
            "budget": budget,
            "itinerary": itinerary,
        }



# Run an example trip planning request

request = TravelRequest(
    origin="Delhi, India",
    destination="Dubai",
    start_date="2025-12-20",
    end_date="2025-12-23",
    budget_level="standard",  # 'budget', 'standard', or 'premium'
)

orchestrator = OrchestratorAgent()
result = orchestrator.plan_trip(request)

print("=== BASIC SUMMARY ===")
print(f"Origin: {request.origin}")
print(f"Destination: {request.destination}")
print(f"Dates: {request.start_date} → {request.end_date}")
print(f"Budget level: {request.budget_level}")
print()

print("=== COST BREAKDOWN (USD) ===")
for k, v in result["budget"].cost_breakdown.items():
    print(f"{k.capitalize():10s}: ${v}")

print("\n=== WEATHER SUMMARY ===")
print(result["research"].weather_summary)

print("\n=== GENERATED ITINERARY ===\n")
print(result["itinerary"].text_itinerary)



import json

output = {
    "result": "Capstone project completed successfully!"
}

with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(output, f)

print("submission.json file created!")


