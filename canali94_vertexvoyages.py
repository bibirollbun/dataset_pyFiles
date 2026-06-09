import os
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("âœ… Setup and authentication complete.")


# Core ADK imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.tools import (
    AgentTool,
    FunctionTool,
    ToolContext,
    google_search,
    load_memory,
    preload_memory,
)
from google.adk.code_executors import BuiltInCodeExecutor

# Memory and state management
from google.adk.memory import InMemoryMemoryService
from google.adk.memory import VertexAiMemoryBankService

# MCP integration
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# App and resumability for long-running operations
from google.adk.apps.app import App, ResumabilityConfig, EventsCompactionConfig

# Observability
from google.adk.plugins import LoggingPlugin

# API types and retry configuration
from google.genai import types

# Standard library
import uuid
import json
from datetime import datetime

print("âœ… All dependencies imported successfully")


# Configure retry logic for API calls
retry_config = types.HttpRetryOptions(
    attempts=5,              # Maximum retry attempts
    exp_base=7,              # Exponential backoff multiplier
    initial_delay=1,         # Initial delay in seconds
    http_status_codes=[429, 500, 503, 504]  # Retry on these errors
)

print("âœ… Retry configuration set")
print("   â€¢ Max attempts: 5")
print("   â€¢ Retry on: 429 (rate limit), 500, 503, 504 errors")


def calculate_trip_budget(
    destination: str,
    num_days: int,
    num_travelers: int,
    accommodation_level: str,
    tool_context: ToolContext
) -> dict:
    """Calculates estimated trip budget based on destination and preferences.
    
    Args:
        destination: City or country name (e.g., "Paris, France")
        num_days: Number of days for the trip
        num_travelers: Number of people traveling
        accommodation_level: "budget", "mid-range", or "luxury"
    
    Returns:
        Dictionary with budget breakdown and total cost
    """
    
    # Mock pricing database (in production, this would call a real API)
    base_costs = {
        "paris": {"budget": 80, "mid-range": 150, "luxury": 350},
        "tokyo": {"budget": 70, "mid-range": 140, "luxury": 400},
        "bali": {"budget": 40, "mid-range": 90, "luxury": 250},
        "new york": {"budget": 100, "mid-range": 200, "luxury": 500},
        "istanbul": {"budget": 50, "mid-range": 100, "luxury": 220},
    }
    
    # Normalize destination
    dest_key = destination.lower().split(",")[0].strip()
    
    # Find matching destination
    daily_cost = None
    for key in base_costs:
        if key in dest_key or dest_key in key:
            daily_cost = base_costs[key].get(accommodation_level.lower(), 150)
            break
    
    if daily_cost is None:
        # Default for unknown destinations
        daily_cost = {"budget": 60, "mid-range": 120, "luxury": 300}[accommodation_level.lower()]
    
    # Calculate breakdown
    accommodation = daily_cost * 0.4 * num_days * num_travelers
    food = daily_cost * 0.3 * num_days * num_travelers
    activities = daily_cost * 0.2 * num_days * num_travelers
    transport = daily_cost * 0.1 * num_days * num_travelers
    
    total = accommodation + food + activities + transport
    
    # Store in session state
    tool_context.state["last_budget"] = total
    tool_context.state["budget_breakdown"] = {
        "accommodation": accommodation,
        "food": food,
        "activities": activities,
        "transport": transport,
        "total": total
    }
    
    return {
        "status": "success",
        "destination": destination,
        "num_days": num_days,
        "num_travelers": num_travelers,
        "accommodation_level": accommodation_level,
        "breakdown": {
            "accommodation": f"${accommodation:.2f}",
            "food": f"${food:.2f}",
            "activities": f"${activities:.2f}",
            "local_transport": f"${transport:.2f}"
        },
        "total_estimated_cost": f"${total:.2f}"
    }

print("âœ… Budget calculator tool created")
print("   ğŸ’° Supports: budget, mid-range, luxury levels")
print("   ğŸŒ� Pre-configured: Paris, Tokyo, Bali, New York, Istanbul")


def validate_destination(
    destination: str,
    travel_dates: str,
    tool_context: ToolContext
) -> dict:
    """Validates if a destination is safe and suitable for travel during specified dates.
    
    Args:
        destination: City or country name (e.g., "Bali, Indonesia")
        travel_dates: Date range in format "YYYY-MM-DD to YYYY-MM-DD"
    
    Returns:
        Dictionary with validation status, safety rating, and recommendations
    """
    
    # Mock validation database (in production, use real travel advisory APIs)
    destination_info = {
        "paris": {
            "safe": True,
            "safety_rating": 4.2,
            "best_months": ["Apr", "May", "Sep", "Oct"],
            "warnings": []
        },
        "tokyo": {
            "safe": True,
            "safety_rating": 4.8,
            "best_months": ["Mar", "Apr", "Oct", "Nov"],
            "warnings": ["Typhoon season: Aug-Sep"]
        },
        "bali": {
            "safe": True,
            "safety_rating": 4.5,
            "best_months": ["Apr", "May", "Jun", "Sep"],
            "warnings": ["Rainy season: Nov-Mar"]
        },
        "new york": {
            "safe": True,
            "safety_rating": 4.0,
            "best_months": ["Apr", "May", "Sep", "Oct"],
            "warnings": ["Very cold winters"]
        },
        "istanbul": {
            "safe": True,
            "safety_rating": 4.3,
            "best_months": ["Apr", "May", "Sep", "Oct"],
            "warnings": []
        }
    }
    
    # Normalize destination
    dest_key = destination.lower().split(",")[0].strip()
    
    # Find matching destination
    info = None
    matched_dest = "Unknown"
    for key in destination_info:
        if key in dest_key or dest_key in key:
            info = destination_info[key]
            matched_dest = key.title()
            break
    
    if info is None:
        # Default for unknown destinations
        info = {
            "safe": True,
            "safety_rating": 3.5,
            "best_months": [],
            "warnings": ["Limited information available - verify travel advisories"]
        }
    
    # Store in session state
    tool_context.state["validated_destination"] = destination
    tool_context.state["destination_safe"] = info["safe"]
    tool_context.state["safety_rating"] = info["safety_rating"]
    
    return {
        "status": "success",
        "destination": destination,
        "matched_location": matched_dest,
        "is_safe": info["safe"],
        "safety_rating": f"{info['safety_rating']}/5.0",
        "best_months_to_visit": info["best_months"],
        "travel_warnings": info["warnings"],
        "recommendation": "Approved for travel" if info["safe"] else "Check travel advisories",
        "travel_dates": travel_dates
    }

print("âœ… Destination validator tool created")
print("   ğŸ›¡ï¸� Validates safety and seasonal suitability")
print("   ğŸ“… Recommends best travel months")


def request_booking_approval(
    total_cost: float,
    destination: str,
    num_travelers: int,
    tool_context: ToolContext
) -> dict:
    """Requests human approval for expensive bookings (>$1000).
    
    This is a long-running operation that pauses the agent workflow
    and waits for human confirmation before proceeding.
    
    Args:
        total_cost: Total trip cost in USD
        destination: Destination name
        num_travelers: Number of travelers
        tool_context: Context for pause/resume functionality
    
    Returns:
        Dictionary with approval status
    """
    
    APPROVAL_THRESHOLD = 1000.0
    
    # SCENARIO 1: Cost under threshold - auto-approve
    if total_cost <= APPROVAL_THRESHOLD:
        tool_context.state["booking_approved"] = True
        tool_context.state["approval_reason"] = "auto_approved"
        return {
            "status": "approved",
            "reason": "auto_approved",
            "message": f"Booking auto-approved (${total_cost:.2f} â‰¤ ${APPROVAL_THRESHOLD})",
            "total_cost": total_cost
        }
    
    # SCENARIO 2: First call - request approval and PAUSE
    if not tool_context.tool_confirmation:
        approval_details = {
            "destination": destination,
            "num_travelers": num_travelers,
            "total_cost": total_cost,
            "threshold": APPROVAL_THRESHOLD
        }
        
        tool_context.request_confirmation(
            hint=f"âš ï¸� High-cost booking detected!\n"
                 f"Destination: {destination}\n"
                 f"Travelers: {num_travelers}\n"
                 f"Total Cost: ${total_cost:.2f}\n"
                 f"Threshold: ${APPROVAL_THRESHOLD}\n\n"
                 f"Do you approve this booking?",
            payload=approval_details
        )
        
        return {
            "status": "pending",
            "message": f"Booking requires approval (${total_cost:.2f} > ${APPROVAL_THRESHOLD})",
            "awaiting_confirmation": True
        }
    
    # SCENARIO 3: Resumed after human response
    if tool_context.tool_confirmation.confirmed:
        tool_context.state["booking_approved"] = True
        tool_context.state["approval_reason"] = "human_approved"
        return {
            "status": "approved",
            "reason": "human_approved",
            "message": f"Booking approved by user for ${total_cost:.2f}",
            "total_cost": total_cost
        }
    else:
        tool_context.state["booking_approved"] = False
        tool_context.state["approval_reason"] = "rejected"
        return {
            "status": "rejected",
            "message": f"Booking rejected by user for ${total_cost:.2f}",
            "total_cost": total_cost
        }

print("âœ… Booking approval tool created (Long-Running)")
print("   ğŸ’³ Auto-approves bookings â‰¤ $1,000")
print("   â�¸ï¸�  Pauses for human approval > $1,000")


def check_for_approval(events):
    """Check if events contain an approval request.
    
    Args:
        events: List of event objects from agent execution
    
    Returns:
        dict with approval details or None
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None


def create_approval_response(approval_info, approved: bool):
    """Create approval response message.
    
    Args:
        approval_info: Dictionary with approval_id and invocation_id
        approved: Boolean indicating approval decision
    
    Returns:
        Content object with function response
    """
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", 
        parts=[types.Part(function_response=confirmation_response)]
    )


def print_agent_response(events):
    """Print agent's text responses from events.
    
    Args:
        events: List of event objects from agent execution
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"ğŸ¤– Agent: {part.text}")


print("âœ… Helper functions defined")
print("   ğŸ“‹ check_for_approval - Detects pause requests")
print("   âœ‰ï¸� create_approval_response - Builds resume messages")
print("   ğŸ’¬ print_agent_response - Displays agent output")


# # MCP Weather Tool Integration - Using Open-Meteo via NPM (No API Key)
# try:
#     mcp_weather_tool = McpToolset(
#         connection_params=StdioConnectionParams(
#             server_params=StdioServerParameters(
#                 command="uvx",
#                 args=[
#                     "--from",
#                     "git+https://github.com/microagents/mcp-servers.git#subdirectory=mcp-weather-free",
#                     "mcp-weather-free",
#                 ],
#             ),
#             timeout=30,
#         )
#     )
#     print("âœ… MCP Weather Tool configured (Open-Meteo)")
#     print("   ğŸŒ¤ï¸� Multiple weather tools available")
#     print("   ğŸ†“ No API key required")
#     print("   â�±ï¸� Timeout: 30 seconds")
#     MCP_AVAILABLE = True
# except Exception as e:
#     print("âš ï¸� MCP Weather Tool not available (optional)")
#     print(f"   Reason: {e}")
#     mcp_weather_tool = None
#     MCP_AVAILABLE = False

print("âš ï¸� Skipping MCP Weather Tool (connection issues)")
print("âœ… Using Google Search for reliable weather data")
mcp_weather_tool = None
MCP_AVAILABLE = False


# Destination Research Agent - Uses Google Search to find information
destination_researcher = Agent(
    name="DestinationResearcher",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a destination research specialist.
    
    Your task:
    1. Research the given destination using Google Search
    2. Find information about: top attractions, local culture, must-see landmarks, hidden gems
    3. Focus on recent and popular recommendations
    4. Keep your findings concise (150-200 words)
    5. Include 3-5 specific attraction names with brief descriptions
    
    Format your response as:
    **Top Attractions:**
    - [Attraction 1]: Brief description
    - [Attraction 2]: Brief description
    ...
    """,
    tools=[google_search],
    output_key="destination_research",
)

print("âœ… Destination Researcher Agent created")
print("   ğŸ”� Tool: Google Search")
print("   ğŸ“� Focus: Attractions, culture, landmarks")


# Activity Finder Agent - Discovers activities and experiences
activity_finder = Agent(
    name="ActivityFinder",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are an activity and experience specialist.
    
    Your task:
    1. Use Google Search to find activities, tours, and experiences at the destination
    2. Focus on: outdoor activities, cultural experiences, food tours, adventure sports
    3. Include family-friendly and adult options
    4. Provide realistic time estimates for each activity
    5. Keep findings concise (150-200 words)
    
    Format your response as:
    **Recommended Activities:**
    - [Activity 1]: Description (Duration: X hours)
    - [Activity 2]: Description (Duration: X hours)
    ...
    """,
    tools=[google_search],
    output_key="activity_research",
)

print("âœ… Activity Finder Agent created")
print("   ğŸ�¯ Tool: Google Search")
print("   ğŸ�¨ Focus: Activities, tours, experiences")


# REPLACE the Weather Checker Agent cell with this RELIABLE version:
weather_checker = Agent(
    name="WeatherChecker",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
        generation_config=types.GenerateContentConfig(
            temperature=0.7,
        )
    ),
    instruction="""You are a weather research specialist for travel planning.

YOUR MISSION:
Extract the destination and travel dates/months from the user's request, then use Google Search to find accurate weather information.

SEARCH STRATEGY:
1. Search: "[destination] weather [specific month/season] average"
2. Search: "[destination] climate [month]" 
3. Look for: temperature ranges, rainfall, seasonal patterns, typical conditions

WHAT TO FIND:
- Average temperatures (highs and lows)
- Precipitation likelihood (rainy/dry season)
- Weather patterns for that specific time period
- Any extreme weather warnings

OUTPUT FORMAT:
**Weather for [Destination] in [Month/Season]:**
- ğŸŒ¡ï¸� Temperature: [typical range in Â°C/Â°F]
- â˜�ï¸� Conditions: [sunny/rainy/mixed + seasonal patterns]
- ğŸ§³ What to Pack: 
  - [Specific clothing items]
  - [Weather accessories needed]
  - [Activity-specific gear if applicable]

Keep your response concise (120-150 words), accurate, and immediately actionable for travelers.

EXAMPLE:
"**Weather for Paris in September:**
- ğŸŒ¡ï¸� Temperature: 15-22Â°C (59-72Â°F), mild and pleasant
- â˜�ï¸� Conditions: Generally sunny with occasional rain, comfortable autumn weather
- ğŸ§³ What to Pack:
  - Light layers (long sleeves, light jacket)
  - Umbrella or rain jacket for occasional showers
  - Comfortable walking shoes
  - Sunglasses for sunny days"
""",
    tools=[google_search],  # ONLY Google Search - reliable and fast
    output_key="weather_research",
)

print("âœ… Weather Checker Agent created (RELIABLE VERSION)")
print("   ğŸŒ¤ï¸� Tool: Google Search (100% reliable)")
print("   ğŸš€ Fast, no connection issues")
print("   ğŸ�¯ Optimized instructions for weather research")


# # Run this test to verify the fix works:
# async def test_weather_agent_fixed():
#     """Test the fixed weather agent with Google Search only."""
#     print("\n" + "="*70)
#     print("ğŸ§ª TESTING FIXED WEATHER AGENT (Google Search)")
#     print("="*70 + "\n")
    
#     test_runner = InMemoryRunner(agent=weather_checker)
    
#     test_cases = [
#         ("Paris, France in September", "What's the weather like in Paris, France during September?"),
#         ("Bali, Indonesia in February", "Check weather conditions for Bali, Indonesia in February"),
#         ("Tokyo, Japan in November", "Weather forecast for Tokyo, Japan in November")
#     ]
    
#     for location, query in test_cases:
#         print(f"ğŸ“� Testing: {location}")
#         print("-" * 60)
        
#         try:
#             response = await test_runner.run_debug(query)
#             print(f"âœ… SUCCESS - Weather agent responded for {location}\n")
#         except Exception as e:
#             print(f"â�Œ FAILED for {location}: {e}\n")
    
#     print("="*70)
#     print("âœ… WEATHER AGENT TEST COMPLETE")
#     print("="*70 + "\n")

# # Run the test:
# await test_weather_agent_fixed()


# Parallel Research Team - All agents run simultaneously
parallel_research_team = ParallelAgent(
    name="ResearchTeam",
    sub_agents=[
        destination_researcher,
        activity_finder,
        weather_checker
    ],
)

print("âœ… Parallel Research Team created")
print("   ğŸ”„ Runs 3 agents concurrently:")
print("      â€¢ Destination Researcher")
print("      â€¢ Activity Finder")
print("      â€¢ Weather Checker")
print("   âš¡ Speeds up research phase significantly")


# Itinerary Builder - Creates day-by-day travel plan
itinerary_builder = Agent(
    name="ItineraryBuilder",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are an expert travel itinerary planner.
    
    Using the research data:
    - Destination Info: {destination_research}
    - Activities: {activity_research}
    - Weather: {weather_research}
    
    Create a day-by-day itinerary that:
    1. Balances popular attractions with unique experiences
    2. Groups nearby locations together to minimize travel time
    3. Considers weather conditions and best times for outdoor activities
    4. Includes realistic time allocations (travel time, activity duration, meals)
    5. Provides morning, afternoon, and evening plans for each day
    6. Leaves some flexibility for spontaneous exploration
    
    Format as:
    **Day 1:**
    - Morning (9:00-12:00): [Activity + location]
    - Afternoon (13:00-17:00): [Activity + location]
    - Evening (18:00-21:00): [Activity + location]
    
    Keep it concise and practical. Total length: 200-300 words.
    """,
    output_key="itinerary_draft",
)

print("âœ… Itinerary Builder Agent created")
print("   ğŸ“… Creates day-by-day schedules")
print("   ğŸ—ºï¸� Optimizes for location proximity")


# Budget Calculator Agent - Uses custom tool
budget_calculator_agent = Agent(
    name="BudgetCalculator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a travel budget specialist.
    
    Your task:
    1. Use the calculate_trip_budget tool with the user's preferences
    2. Present the budget breakdown clearly
    3. Provide money-saving tips specific to the destination
    4. Suggest budget adjustments if costs seem too high
    
    Always call the calculate_trip_budget tool first, then explain the results.
    
    Format your response as:
    **Budget Breakdown:**
    [Show the breakdown from the tool]
    
    **Money-Saving Tips:**
    - [Tip 1]
    - [Tip 2]
    - [Tip 3]
    """,
    tools=[FunctionTool(func=calculate_trip_budget)],
    output_key="budget_analysis",
)

print("âœ… Budget Calculator Agent created")
print("   ğŸ’° Tool: calculate_trip_budget (custom)")
print("   ğŸ’¡ Provides cost breakdown + savings tips")


# Optimizer Agent - Reviews and improves the itinerary
optimizer_agent = Agent(
    name="OptimizerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a travel plan optimization specialist.
    
    Review the itinerary and budget:
    - Itinerary: {itinerary_draft}
    - Budget: {budget_analysis}
    
    Your task:
    1. Identify potential issues (too rushed, too expensive, poor timing)
    2. Suggest optimizations (better routes, cheaper alternatives, time savings)
    3. Ensure the plan is realistic and enjoyable
    4. Keep the final output concise (150-200 words)
    
    Format as:
    **Optimized Plan:**
    [Key improvements made]
    
    **Final Recommendations:**
    - [Recommendation 1]
    - [Recommendation 2]
    - [Recommendation 3]
    """,
    output_key="optimized_plan",
)

print("âœ… Optimizer Agent created")
print("   âš™ï¸� Reviews itinerary + budget")
print("   ğŸ�¯ Suggests improvements")


# Sequential Planning Pipeline - Runs agents in order
sequential_planning_pipeline = SequentialAgent(
    name="PlanningPipeline",
    sub_agents=[
        itinerary_builder,
        budget_calculator_agent,
        optimizer_agent
    ],
)

print("âœ… Sequential Planning Pipeline created")
print("   ğŸ“Š Pipeline order:")
print("      1. Itinerary Builder")
print("      2. Budget Calculator")
print("      3. Optimizer")
print("   â†’ Each agent uses outputs from previous agents")


# Booking Agent - Handles approvals for expensive trips
booking_agent = Agent(
    name="BookingAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a travel booking specialist.
    
    Your task:
    1. Extract the total cost from the budget analysis: {budget_analysis}
    2. Extract destination and traveler count from the user's request
    3. Use the request_booking_approval tool to check if approval is needed
    4. If status is "pending", inform the user that approval is required
    5. If status is "approved" or "rejected", provide a clear summary
    
    Always use the request_booking_approval tool first, then respond based on the result.
    
    Response format:
    **Booking Status:** [Approved/Pending/Rejected]
    **Total Cost:** $X,XXX.XX
    **Next Steps:** [What happens next]
    """,
    tools=[FunctionTool(func=request_booking_approval)],
    output_key="booking_status",
)

print("âœ… Booking Agent created")
print("   ğŸ’³ Tool: request_booking_approval (long-running)")
print("   â�¸ï¸� Auto-approves â‰¤ $1,000")
print("   â�¯ï¸� Pauses for human approval > $1,000")


# Validation Agent - Uses custom validator tool
validation_agent = Agent(
    name="ValidationAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a travel safety and feasibility validator.
    
    Your task:
    1. Extract the destination and travel dates from the user's request
    2. Use the validate_destination tool with these parameters
    3. Report the safety rating and any travel warnings
    4. Provide recommendations based on the validation results
    
    IMPORTANT: 
    - If dates are mentioned in the request (like "2025-06-15 to 2025-06-20"), use them directly
    - If destination is mentioned, use it directly
    - Do NOT ask for more information - extract from the request
    
    Always call validate_destination first, then summarize the results clearly.
    
    Format:
    **Destination Validation:**
    - Safety Rating: X/5.0
    - Best Months: [list]
    - Warnings: [list or "None"]
    - Recommendation: [Your advice]
    """,
    tools=[FunctionTool(func=validate_destination)],
    output_key="validation_result",
)

print("âœ… Validation Agent created (UPDATED)")
print("   ğŸ›¡ï¸� Tool: validate_destination (custom)")
print("   âœ… Extracts destination + dates from request")


# Root Coordinator Agent - Orchestrates the entire workflow SEQUENTIALLY
root_coordinator = Agent(
    name="VertexVoyagesCoordinator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are the Vertex Voyages travel planning coordinator.

    CRITICAL: You MUST complete ALL 4 steps in sequence. Do NOT stop until all 4 are done.
    
    **WORKFLOW (MANDATORY - EXECUTE ALL 4 STEPS):**
    
    ğŸ“‹ Step 1: VALIDATION (REQUIRED)
    - Call ValidationAgent with: "Destination: [name], Dates: [dates], Travelers: [num]"
    - Wait for response
    - After receiving validation, IMMEDIATELY proceed to Step 2 - DO NOT STOP
    
    ğŸ“‹ Step 2: RESEARCH (REQUIRED)
    - Call ResearchTeam with: "Research [destination] for [num] travelers from [dates]"
    - Wait for response
    - After receiving research, IMMEDIATELY proceed to Step 3 - DO NOT STOP
    
    ğŸ“‹ Step 3: PLANNING (REQUIRED)
    - Call PlanningPipeline with: "Create itinerary and budget for [destination], [num_days] days, [num_travelers] travelers, [accommodation_level] accommodation"
    - Wait for response
    - After receiving plan, IMMEDIATELY proceed to Step 4 - DO NOT STOP
    
    ğŸ“‹ Step 4: BOOKING (REQUIRED)
    - Call BookingAgent with: "Process booking for [destination], [num_travelers] travelers, total budget from planning"
    - Wait for response
    - After receiving booking status, NOW you can stop and provide final output
    
    **MANDATORY RULES:**
    1. Extract ALL details from user query (destination, dates, days, travelers, accommodation level)
    2. Pass COMPLETE information to each agent
    3. Call agents ONE AT A TIME in the exact order above
    4. After EACH agent response, explain what you received and what you'll do NEXT
    5. You MUST call all 4 agents - validation â†’ research â†’ planning â†’ booking
    6. Only stop AFTER Step 4 is complete
    
    **After ALL 4 steps complete, provide this final output:**
    
    **ğŸŒ� Vertex Voyages Travel Plan**
    
    **Destination:** [Name]
    **Dates:** [Travel dates]
    **Travelers:** [Number]
    
    **âœ… Validation:** [Summary from ValidationAgent]
    
    **ğŸ”� Research Highlights:**
    [Key findings from ResearchTeam]
    
    **ğŸ“… Itinerary & Budget:**
    [Key details from PlanningPipeline]
    
    **ğŸ’³ Booking Status:**
    [Status from BookingAgent - Approved or Pending Approval]
    
    Remember: You must complete ALL 4 steps before stopping!
    """,
    tools=[
        AgentTool(agent=validation_agent),
        AgentTool(agent=parallel_research_team),
        AgentTool(agent=sequential_planning_pipeline),
        AgentTool(agent=booking_agent)
    ],
)

print("âœ… Root Coordinator Agent created (FORCED SEQUENTIAL)")
print("   ğŸ�¯ Workflow: MUST complete all 4 steps")
print("   ğŸ“‹ Step 1: Validation â†’ Step 2: Research â†’ Step 3: Planning â†’ Step 4: Booking")
print("   âš ï¸�  Will not stop until all 4 steps complete")


# Wrap root coordinator in a resumable app for long-running operations
vertex_voyages_app = App(
    name="VertexVoyages",
    root_agent=root_coordinator,
    resumability_config=ResumabilityConfig(
        is_resumable=True  # Enables pause/resume for booking approvals
    ),
    plugins=[LoggingPlugin()]  # Plugins go in the App, not the Runner
)

print("âœ… Vertex Voyages App created (Resumable)")
print("   â�¸ï¸� Supports pause/resume for approval workflows")
print("   ğŸ”„ Can handle long-running booking operations")
print("   ğŸ“� Logging enabled via LoggingPlugin")


# Initialize session service for state management
session_service = InMemorySessionService()

print("âœ… Session Service initialized")
print("   ğŸ’¾ Type: InMemorySessionService")
print("   ğŸ“Š Stores: conversation history, state, context")


# Initialize memory service for long-term storage
try:
    memory_service = InMemoryMemoryService()
    print("âœ… Memory Service initialized")
    print("   ğŸ§  Type: InMemoryMemoryService")
    print("   ğŸ“� Stores: user preferences, past trips, patterns")
    MEMORY_AVAILABLE = True
except Exception as e:
    print("âš ï¸� Memory Service not available (optional)")
    print(f"   Reason: {e}")
    memory_service = None
    MEMORY_AVAILABLE = False


# Configure runner with session service (no plugins here!)
runner = Runner(
    app=vertex_voyages_app,
    session_service=session_service,
)

print("âœ… Runner configured")
print("   ğŸ“Š Session Service: InMemorySessionService")
print("   ğŸ“� Logging: Enabled in App (LoggingPlugin)")
print("   ğŸ”� Captures: Agent calls, tool usage, state changes")


async def plan_trip(
    user_query: str,
    destination: str,
    travel_dates: str,
    num_days: int,
    num_travelers: int,
    accommodation_level: str = "mid-range",
    auto_approve: bool = True
) -> dict:
    """Main workflow function for Vertex Voyages travel planning.
    
    Args:
        user_query: Natural language travel request
        destination: Destination name (e.g., "Paris, France")
        travel_dates: Date range "YYYY-MM-DD to YYYY-MM-DD"
        num_days: Number of days for the trip
        num_travelers: Number of travelers
        accommodation_level: "budget", "mid-range", or "luxury"
        auto_approve: Auto-approve bookings for testing (True) or require manual approval (False)
    
    Returns:
        Dictionary with complete travel plan and status
    """
    
    print(f"\n{'='*70}")
    print(f"ğŸŒ� VERTEX VOYAGES - Travel Planning System")
    print(f"{'='*70}")
    print(f"ğŸ“� Destination: {destination}")
    print(f"ğŸ“… Dates: {travel_dates}")
    print(f"ğŸ‘¥ Travelers: {num_travelers}")
    print(f"ğŸ�¨ Level: {accommodation_level}")
    print(f"{'='*70}\n")
    
    # Generate unique session ID
    session_id = f"trip_{uuid.uuid4().hex[:8]}"
    
    # Create session
    await session_service.create_session(
        app_name="VertexVoyages",
        user_id="traveler_001",
        session_id=session_id
    )
    
    # Store trip parameters in session state
    initial_state = {
        "destination": destination,
        "travel_dates": travel_dates,
        "num_days": num_days,
        "num_travelers": num_travelers,
        "accommodation_level": accommodation_level,
        "timestamp": datetime.now().isoformat()
    }
    
    # Prepare user message
    enhanced_query = f"""{user_query}
    
Trip Details:
- Destination: {destination}
- Dates: {travel_dates}
- Duration: {num_days} days
- Travelers: {num_travelers}
- Accommodation: {accommodation_level}
"""
    
    query_content = types.Content(
        role="user",
        parts=[types.Part(text=enhanced_query)]
    )
    
    events = []
    
    # -----------------------------------------------------------------------------------------------
    # STEP 1: Send initial request to coordinator
    print("ğŸš€ Starting travel planning workflow...\n")
    
    async for event in runner.run_async(
        user_id="traveler_001",
        session_id=session_id,
        new_message=query_content
    ):
        events.append(event)
    
    # -----------------------------------------------------------------------------------------------
    # STEP 2: Check for approval request (long-running operation)
    approval_info = check_for_approval(events)
    
    if approval_info:
        print(f"\n{'='*70}")
        print("â�¸ï¸�  BOOKING APPROVAL REQUIRED")
        print(f"{'='*70}")
        print(f"ğŸ’° Trip cost exceeds $1,000 threshold")
        print(f"ğŸ¤” Simulated Human Decision: {'APPROVE âœ…' if auto_approve else 'REJECT â�Œ'}\n")
        
        # -----------------------------------------------------------------------------------------------
        # STEP 3: Resume with approval decision
        async for event in runner.run_async(
            user_id="traveler_001",
            session_id=session_id,
            new_message=create_approval_response(approval_info, auto_approve),
            invocation_id=approval_info["invocation_id"]
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"ğŸ¤– Agent: {part.text}")
    else:
        # No approval needed - print response
        print_agent_response(events)
    
    print(f"\n{'='*70}")
    print("âœ… TRAVEL PLANNING COMPLETE")
    print(f"{'='*70}\n")
    
    return {
        "session_id": session_id,
        "status": "complete",
        "destination": destination,
        "dates": travel_dates
    }

print("âœ… Main workflow function defined")
print("   ğŸ�¯ Function: plan_trip()")
print("   ğŸ”„ Handles: Full planning + approval workflow")


# Test Case 1: Budget-friendly trip to Bali
print("ğŸ§ª TEST CASE 1: Low-Cost Trip to Bali")
print("Expected: Auto-approval (cost < $1,000)\n")

result_1 = await plan_trip(
    user_query="I want to plan a relaxing beach vacation to Bali with my partner.",
    destination="Bali, Indonesia",
    travel_dates="2026-02-10 to 2026-02-15",
    num_days=5,
    num_travelers=2,
    accommodation_level="budget",
    auto_approve=True  # Not needed for low-cost, but included for consistency
)

print(f"\nâœ… Test 1 Complete - Session ID: {result_1['session_id']}")


# # Test Case 2: Luxury trip to Paris requiring approval
# print("\n" + "="*70)
# print("ğŸ§ª TEST CASE 2: High-Cost Trip to Paris")
# print("Expected: Pauses for approval (cost > $1,000)\n")

# result_2 = await plan_trip(
#     user_query="Plan a romantic luxury getaway to Paris for our anniversary.",
#     destination="Paris, France",
#     travel_dates="2025-09-01 to 2025-09-08",
#     num_days=7,
#     num_travelers=2,
#     accommodation_level="luxury",
#     auto_approve=True  # Change to False to simulate rejection
# )

# print(f"\nâœ… Test 2 Complete - Session ID: {result_2['session_id']}")


# # Test Case 3: High-cost trip that gets rejected
# print("\n" + "="*70)
# print("ğŸ§ª TEST CASE 3: High-Cost Trip to Tokyo (REJECTED)")
# print("Expected: Pauses for approval, user rejects\n")

# result_3 = await plan_trip(
#     user_query="Plan a luxury adventure trip to Tokyo with gourmet dining experiences.",
#     destination="Tokyo, Japan",
#     travel_dates="2025-11-10 to 2025-11-18",
#     num_days=8,
#     num_travelers=2,
#     accommodation_level="luxury",
#     auto_approve=False  # Simulate user rejection
# )

# print(f"\nâœ… Test 3 Complete - Session ID: {result_3['session_id']}")




