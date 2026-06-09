# tools/mock_flight_tool.py
import random
from google_adk.tools import AgentTool

@AgentTool
def search_flights(origin: str, destination: str, departure_date: str, return_date: str) -> dict:
    """
    Searches for available flights between an origin and destination on specific dates.
    
    Args:
        origin: The starting city or airport code.
        destination: The destination city or airport code.
        departure_date: The departure date in YYYY-MM-DD format.
        return_date: The return date in YYYY-MM-DD format.
    
    Returns:
        A dictionary containing a list of flight options with airline, price, and times.
    """
    print(f"Tool: Searching flights from {origin} to {destination}...")
    airlines = ["SkyLink Airways", "CloudHopper", "Velocity Jet", "AeroVista"]
    flight_options = []
    for _ in range(random.randint(2, 4)):
        flight_options.append({
            "airline": random.choice(airlines),
            "departure_time": f"{random.randint(6, 20):02d}:00",
            "arrival_time": f"{random.randint(7, 22):02d}:00",
            "price_usd": random.randint(350, 950)
        })
    return {"flights": flight_options}


# tools/mock_hotel_tool.py
import random
from google_adk.tools import AgentTool

@AgentTool
def search_hotels(city: str, checkin_date: str, checkout_date: str, budget_per_night: int) -> dict:
    """
    Searches for available hotels in a city within a given budget.
    
    Args:
        city: The city where the user wants to stay.
        checkin_date: The check-in date in YYYY-MM-DD format.
        checkout_date: The check-out date in YYYY-MM-DD format.
        budget_per_night: The maximum price per night in USD.

    Returns:
        A dictionary containing a list of hotel options with name, rating, and price.
    """
    print(f"Tool: Searching hotels in {city} with budget ${budget_per_night}/night...")
    hotel_names = ["The Grand Vista", "Metropolis Inn", "Harborview Hotel", "The Sapphire Lodge"]
    hotel_options = []
    for _ in range(random.randint(2, 3)):
        price = random.randint(int(budget_per_night * 0.7), budget_per_night)
        hotel_options.append({
            "name": random.choice(hotel_names),
            "rating": round(random.uniform(3.8, 5.0), 1),
            "price_per_night_usd": price
        })
    return {"hotels": hotel_options}


# agents/flight_agent.py
from google_adk.agents import LlmAgent
from tools.mock_flight_tool import search_flights

flight_research_agent = LlmAgent(
    model="gemini-1.5-flash",
    name="flight_researcher",
    description="Specialist agent for researching and finding flight options.",
    instruction="""You are an expert flight researcher. Your sole job is to use the `search_flights` tool to find flight options based on the user's request. 
    Do not answer any other questions. Once you have the flight results, return them directly.
    """,
    tools=[search_flights]
)


# agents/hotel_agent.py
from google_adk.agents import LlmAgent
from tools.mock_hotel_tool import search_hotels

hotel_research_agent = LlmAgent(
    model="gemini-1.5-flash",
    name="hotel_researcher",
    description="Specialist agent for researching and finding hotel options.",
    instruction="""You are an expert hotel researcher. Your only job is to use the `search_hotels` tool to find accommodations based on the user's destination and budget.
    Do not engage in any other conversation. Return the hotel results once found.
    """,
    tools=[search_hotels]
)


# agents/activity_agent.py
from google_adk.agents import LlmAgent
from google_adk.tools.google_search_tool import google_search

activity_planner_agent = LlmAgent(
    model="gemini-1.5-flash",
    name="activity_planner",
    description="Specialist agent for finding popular activities and attractions in a city.",
    instruction="""You are a local tour guide. Your job is to use the Google Search tool to find 'top 3 things to do in [city]'.
    Summarize your findings into a short, exciting list. Do not use any other tools or answer other questions.
    """,
    tools=[google_search]
)


# agents/travel_planner_agent.py
from google_adk.agents import LlmAgent
from .flight_agent import flight_research_agent
from .hotel_agent import hotel_research_agent
from .activity_agent import activity_planner_agent

travel_planner_agent = LlmAgent(
    model="gemini-1.5-pro", # Use a more powerful model for orchestration
    name="travel_planner",
    description="The main travel concierge agent that plans a full trip.",
    instruction="""You are a world-class travel concierge. Your goal is to create a complete travel itinerary for the user.

    Here is your plan:
    1.  First, greet the user and confirm you have all the necessary information: origin, destination, dates, and total budget. If anything is missing, ask for it.
    2.  Once you have all the information, delegate the task of finding flights to the `flight_researcher` agent.
    3.  Next, delegate the task of finding hotels to the `hotel_researcher` agent. Calculate a nightly budget based on the total budget and trip duration.
    4.  Then, delegate the task of finding activities to the `activity_planner` agent.
    5.  Finally, synthesize all the information from the specialist agents into a single, well-formatted, and helpful travel itinerary. Present this final plan to the user.
    
    Do not use the tools of the sub-agents directly. Delegate to them.
    """,
    sub_agents=[flight_research_agent, hotel_research_agent, activity_planner_agent]
)


# main.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
from google_adk.runner import run_agent
from google_adk.sessions import InMemorySessionService
from agents.travel_planner_agent import travel_planner_agent

def main():
    """
    The main function to run the Smart Travel Concierge agent.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

    genai.configure(api_key=api_key)

    print("Welcome to the Smart Travel Concierge!")
    print("I can help you plan your next trip.")
    print("Type 'exit' to end the conversation.")
    print("-" * 20)

    # Using InMemorySessionService to manage conversation state
    session_service = InMemorySessionService()

    run_agent(
        agent=travel_planner_agent,
        session_service=session_service,
        # Starting with an example prompt
        # initial_query="Plan a trip for me from San Francisco to Paris from 2025-12-20 to 2025-12-27 with a budget of $5000."
    )

if __name__ == "__main__":
    main()


[
  {
    "query": "I want to plan a trip to Tokyo from New York City. We'll be going from 2026-03-10 to 2026-03-17. Our total budget is $4000.",
    "ideal_output_keywords": ["flight", "hotel", "Tokyo", "activity", "itinerary", "Shibuya", "temple"]
  }
]


# evaluation/evaluate.py
import os
import json
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from google_adk.integrations.langchain import AdkLangchainRunnable
from agents.travel_planner_agent import travel_planner_agent

async def run_agent_for_evaluation(query: str) -> str:
    """Runs the agent with a single query and returns the final response."""
    runnable = AdkLangchainRunnable(agent=travel_planner_agent)
    final_result = ""
    async for chunk in runnable.astream_log(query):
        for op in chunk.ops:
            if op["path"] == "/logs/ChatEntry:final_output/content":
                final_result = op["value"]
    return final_result

async def llm_as_judge(query: str, agent_output: str, ideal_keywords: list) -> dict:
    """Uses Gemini to judge the agent's output based on a rubric."""
    judge_model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    You are an expert evaluator for an AI travel agent.
    Your task is to assess the quality of the agent's response based on the user's query and a set of ideal keywords.

    USER QUERY: "{query}"

    AGENT'S FINAL ITINERARY:
    "{agent_output}"

    EVALUATION CRITERIA:
    1.  Completeness: Did the agent provide options for flights, hotels, AND activities? (Score 1-5)
    2.  Relevance: Does the itinerary match the user's request (destination, dates)? (Score 1-5)
    3.  Inclusion of Keywords: Does the output contain relevant information, as suggested by these keywords: {ideal_keywords}? (Score 1-5)

    Please provide a score for each criterion and a final "pass" or "fail" judgment. A "pass" requires an average score of 4 or higher.
    Return your evaluation as a JSON object with keys: "completeness_score", "relevance_score", "keyword_score", "reasoning", "final_judgment".
    """
    
    response = await judge_model.generate_content_async(prompt)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse judge's response", "raw_response": response.text}

async def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
    genai.configure(api_key=api_key)

    with open("evaluation/golden_dataset.json", "r") as f:
        dataset = json.load(f)

    print("Running Agent Evaluation...")
    for i, item in enumerate(dataset):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Query: {item['query']}")
        
        agent_output = await run_agent_for_evaluation(item['query'])
        print(f"Agent Output: {agent_output[:200]}...") # Print snippet

        judge_result = await llm_as_judge(item['query'], agent_output, item['ideal_output_keywords'])
        
        print("\n--- Judge's Verdict ---")
        print(json.dumps(judge_result, indent=2))
        print(f"Final Judgment: {judge_result.get('final_judgment', 'ERROR')}")

if __name__ == "__main__":
    asyncio.run(main())


# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for the API key (to be set at runtime)
ENV GOOGLE_API_KEY=""

# Run main.py when the container launches
CMD ["python", "main.py"]


# Build the docker image
gcloud builds submit --tag gcr.io/your-gcp-project/smart-travel-agent

# Deploy to Cloud Run
gcloud run deploy smart-travel-agent \
  --image gcr.io/your-gcp-project/smart-travel-agent \
  --platform managed \
  --region us-central1 \
  --set-secrets=GOOGLE_API_KEY=your-secret-name:latest

