import os
from typing import List, Dict

# --- 1. Setup and Tool Definitions ---

# NOTE: In your capstone, you will replace this with the actual Gemini client and ADK imports.
# For example: from google_adk import Agent, Tool, ...
# For this example, we'll simulate the Tool Manager's execution.

class ToolManager:
    """
    Manages all external functions (Tools) the agent can call.
    In the capstone, these would be registered with the ADK Tool class.
    """
    def __init__(self):
        # A dictionary to hold the available tools
        self.tools = {
            "search_events": self.search_local_events,
            "find_flights": self.find_flight_options,
            "book_hotel": self.book_hotel,
        }

    def search_local_events(self, destination: str, dates: List[str]) -> str:
        """Searches for events/activities in the destination during the travel dates."""
        print(f"-> TOOL CALL: Searching for events in {destination} on {dates}...")
        # Placeholder API call simulation
        if "Paris" in destination:
            return "Found a list of Michelin-starred restaurants and a Monet exhibit."
        return f"Found interesting activities and weather for {destination}."

    def find_flight_options(self, origin: str, destination: str, date: str) -> str:
        """Searches for flight options."""
        print(f"-> TOOL CALL: Finding flights from {origin} to {destination} on {date}...")
        # Placeholder API call simulation
        return "Top 3 flights: 1) $500, direct, 8hr. 2) $450, 1 stop, 10hr. 3) $380, 2 stops, 12hr."

    def book_hotel(self, destination: str, check_in: str, check_out: str, preferences: str) -> str:
        """Searches for and books accommodation."""
        print(f"-> TOOL CALL: Searching hotels in {destination} with preferences: {preferences}...")
        # Placeholder API call simulation
        return "Recommended Hotel: The Grand Paris Hotel ($250/night, 4.5 stars) - near the Louvre."

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Executes the requested tool function."""
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        return f"Error: Tool '{tool_name}' not found."


# --- 2. The Main Multi-Agent Class ---

class TravelAgent:
    """
    The main orchestrator agent responsible for decomposition, delegation, 
    tool calling, and final synthesis (Multi-Agent System).
    """
    def __init__(self, model_name: str = 'gemini-2.5-pro'): # Use a powerful model for planning
        # In a real notebook, you would initialize the Gemini client here.
        # self.client = client.create() 
        self.tool_manager = ToolManager()
        self.model_name = model_name
        
        self.PLANNER_PROMPT = """
        You are the Travel Planner Agent. Your goal is to generate a comprehensive, 
        day-by-day travel itinerary based on the user's request.
        
        1. **Decomposition:** Break the request into finding Flights, Hotels, and Local Activities.
        2. **Tool Use:** Use the available tools (find_flights, book_hotel, search_events) to gather all necessary facts. 
           Wait for the tool results before proceeding.
        3. **Synthesis:** Once all facts are gathered, compile them into a beautiful, formatted 
           Markdown itinerary. Be enthusiastic and helpful.
        """

    def generate_response(self, user_query: str) -> str:
        """
        The core loop simulating the LLM's thought process and tool execution.
        """
        print(f"\n--- ğŸ—ºï¸� TRAVEL AGENT START ---")
        print(f"User Query: {user_query}")
        
        # In the capstone, the LLM handles the multi-step planning and function calling loop.
        # We will simulate that sequence here for clarity.
        
        collected_facts = {}
        
        # --- Simulated LLM Planning Step 1: Flights ---
        print("\n[STEP 1/3: Planning Flights]")
        flight_result = self.tool_manager.execute_tool(
            "find_flights", 
            origin="London", 
            destination="Paris", 
            date="2025-12-04"
        )
        collected_facts['flights'] = flight_result
        
        # --- Simulated LLM Planning Step 2: Hotel ---
        print("\n[STEP 2/3: Planning Hotel]")
        hotel_result = self.tool_manager.execute_tool(
            "book_hotel", 
            destination="Paris", 
            check_in="2025-12-04", 
            check_out="2025-12-08",
            preferences="romantic, near Eiffel Tower"
        )
        collected_facts['hotel'] = hotel_result

        # --- Simulated LLM Planning Step 3: Local Events ---
        print("\n[STEP 3/3: Researching Local Events]")
        event_result = self.tool_manager.execute_tool(
            "search_events", 
            destination="Paris", 
            dates=["2025-12-05", "2025-12-06", "2025-12-07"]
        )
        collected_facts['events'] = event_result
        
        # --- Simulated LLM Synthesis (Final Itinerary Writer) ---
        # The final step is to feed the PLANNER_PROMPT and collected_facts into the LLM 
        # for final formatting (this simulates the Writer Agent in the multi-agent design).

        print("\n[STEP 4/4: Final Synthesis]")
        
        # Concatenate facts into the context for the final output generation (the synthesis step)
        context = f"""
        USER REQUEST: {user_query}
        ---
        FACTS GATHERED:
        Flights: {collected_facts['flights']}
        Hotel: {collected_facts['hotel']}
        Events: {collected_facts['events']}
        ---
        Use the facts above to generate the final, detailed itinerary.
        """

        # In your notebook, this is where you would call: 
        # response = self.client.models.generate_content(model=self.model_name, contents=self.PLANNER_PROMPT + context)
        
        # SIMULATED FINAL OUTPUT:
        itinerary_output = f"""
        ## âœ¨ Your Bespoke 4-Day Paris Itinerary âœ¨

        Based on your request for a **romantic trip focusing on art and food**!

        ---

        ### âœˆï¸� Flights & Accommodation
        
        **Flights:** {collected_facts['flights']}
        
        **Accommodation:** {collected_facts['hotel']} - *Perfectly located for your romantic stay!*

        ---
        
        ### ğŸ—“ï¸� Day-by-Day Plan

        **Day 1 (Dec 4): Arrival & Check-In**
        * Arrive at the airport (refer to flight details).
        * Check into The Grand Paris Hotel.
        * Evening: Romantic dinner at a suggested **Michelin-starred restaurant**.

        **Day 2 (Dec 5): Art Immersion**
        * Morning: Visit the **Monet exhibit** found by your agent.
        * Afternoon: Explore the Louvre Museum.
        
        **Day 3 (Dec 6): Parisian Charm**
        * Enjoy the other interesting local activities found by the agent.
        * Evening: A leisurely walk near the Eiffel Tower.

        **Day 4 (Dec 7): Departure**
        * Enjoy a final Parisian breakfast before heading home.

        ---
        """
        
        return itinerary_output


# --- 3. Execution ---

if __name__ == "__main__":
    # The agent is instantiated
    planner = TravelAgent()
    
    # The user request is processed
    USER_REQUEST = "Plan a romantic 4-day trip to Paris from London for a couple, checking in on Dec 4th, with a focus on art and good food."
    final_itinerary = planner.generate_response(USER_REQUEST)
    
    print("\n\n" + "="*70)
    print("âœ… FINAL AGENT-GENERATED ITINERARY")
    print("="*70)
    print(final_itinerary)

