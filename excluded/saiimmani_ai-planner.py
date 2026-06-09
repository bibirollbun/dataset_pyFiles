# --- Cell 1: Imports and Tool Definitions ---
# Run this cell first to define your tools and import necessary libraries.

import json
import time
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from kaggle_secrets import UserSecretsClient # <-- ADDED IMPORT

print("Cell 1 executed: Imports complete.")

# --- Cell 2: Configure Gemini API ---
# Run this cell to configure the Gemini API.
# IMPORTANT: Set your GOOGLE_API_KEY in your environment (e.g., Kaggle Secrets)

# --- CORRECTED SECTION START ---
# Actively fetch the secret from Kaggle Secrets and set it as an environment variable.
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("Cell 2: Successfully fetched GOOGLE_API_KEY from Kaggle Secrets.")
except Exception as e:
    print(f"Cell 2: Error accessing Kaggle secret: {e}")
    print("Cell 2: Please make sure 'GOOGLE_API_KEY' is added to this notebook's secrets (Add-ons > Secrets).")
# --- CORRECTED SECTION END ---

# Load the API key from environment variables (which we just set)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("--- WARNING ---")
    print("GOOGLE_API_KEY environment variable not set.")
    print("The agent will NOT work. Please set the key in your environment (e.g., Kaggle Secrets) and restart the kernel.")
    print("---------------")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Cell 2 executed: Gemini API configured successfully.")
    except Exception as e:
        print(f"Cell 2 Error: Could not configure Gemini API. Error: {e}")

# --- Cell 3: Tool Definitions ---
# These are the specific functions (tools) our specialist agents can use.

def tool_search_restaurants(cuisine):
    """Mock Tool: Searches for a restaurant based on cuisine."""
    print(f"[Tool Called]: tool_search_restaurants(cuisine='{cuisine}')")
    time.sleep(0.5) # Simulate API call
    if not isinstance(cuisine, str):
        cuisine = "generic"
        
    if cuisine.lower() == 'italian':
        return {"name": "Luigi's Pasta Palace", "address": "123 Main St", "rating": 4.5}
    elif cuisine.lower() == 'mexican':
        return {"name": "Taco Fiesta", "address": "456 Oak Ave", "rating": 4.2}
    else:
        return {"name": "The Generic Diner", "address": "789 Pine Ln", "rating": 3.0}

def tool_search_movies(genre):
    """Mock Tool: Searches for a movie based on genre."""
    print(f"[Tool Called]: tool_search_movies(genre='{genre}')")
    time.sleep(0.5) # Simulate API call
    if not isinstance(genre, str):
        genre = "generic"

    if genre.lower() == 'comedy':
        return {"title": "Laugh Out Loud", "time": "8:00 PM", "theater": "CinemaPlex 1"}
    elif genre.lower() == 'action':
        return {"title": "Danger Zone 4", "time": "9:00 PM", "theater": "MegaMovies 2"}
    else:
        return {"title": "A Generic Film", "time": "7:30 PM", "theater": "IndieHouse"}

print("Cell 3 executed: Tools defined.")


# --- Cell 4: Real LLM Brain (Replaces Mock) ---
# This cell defines the actual call to the Gemini API using JSON mode.

# Define the JSON schemas for the LLM's structured output
SCHEMA_TOOL_CALL = {
    "type": "OBJECT",
    "properties": {
        "thought": { "type": "STRING", "description": "Your internal reasoning for this step."},
        "action": { "type": "STRING", "enum": ["use_tool"], "description": "The action to take."},
        "tool_name": { "type": "STRING", "description": "The exact name of the tool to use."},
        "tool_input": { "type": "STRING", "description": "The query or input for the tool. For example, 'italian' or 'comedy'."}
    },
    "required": ["thought", "action", "tool_name", "tool_input"]
}

SCHEMA_FINAL_RESPONSE = {
    "type": "OBJECT",
    "properties": {
        "thought": { "type": "STRING", "description": "Your internal reasoning for formulating the final response."},
        "action": { "type": "STRING", "enum": ["respond"], "description": "The action to take."},
        "action_input": { "type": "STRING", "description": "The final, user-facing response as a single string."}
    },
    "required": ["thought", "action", "action_input"]
}

def get_llm_response(system_prompt, user_prompt, json_schema):
    """
    Calls the Gemini API with a system prompt, user prompt, and a JSON schema.
    """
    if not GOOGLE_API_KEY:
        print("[LLM Error]: GOOGLE_API_KEY is not set. Returning empty response.")
        return "{}"

    print(f"\n[LLM Call Starting]")
    print(f"[LLM System Prompt]: {system_prompt}")
    print(f"[LLM User Prompt]: {user_prompt}")
    
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            system_instruction=system_prompt,
        )
        
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
            temperature=0.3
        )

        response = model.generate_content(
            user_prompt,
            generation_config=generation_config
        )
        
        response_text = response.text
        print(f"[LLM Response Received]: {response_text}")
        return response_text
    
    except Exception as e:
        print(f"[LLM Error]: Failed to generate content. {e}")
        # Return a valid JSON string matching the expected structure, but with an error message
        if json_schema == SCHEMA_FINAL_RESPONSE:
            return json.dumps({
                "thought": "An error occurred.",
                "action": "respond",
                "action_input": "I'm sorry, I encountered an error while processing your request."
            })
        else:
            # This case is harder to handle, as it expects a tool call.
            # We'll return an empty JSON and let the agent handle it.
            return "{}"

print("Cell 4 executed: Real LLM Brain defined.")


# --- Cell 5: Specialist Agents (Updated) ---
# Run this cell to define the specialist agents that now use the real LLM.

class RestaurantAgent:
    def __init__(self):
        self.tools = {"search_restaurants": tool_search_restaurants}
        self.agent_type = "RESTAURANT"
        self.system_prompt = (
            "You are a helpful Restaurant Agent. Your job is to find a restaurant for the user. "
            "You have one tool: 'search_restaurants(cuisine)'. "
            "Analyze the user's prompt, extract the *cuisine* they want, and then call your tool. "
            "If no cuisine is specified, default to 'generic'."
        )

    def run(self, prompt):
        print(f"\n--- {self.agent_type} Activated ---")
        llm_response_str = get_llm_response(self.system_prompt, prompt, SCHEMA_TOOL_CALL)
        
        try:
            llm_response = json.loads(llm_response_str)
        except json.JSONDecodeError:
            print("[Agent Error]: Failed to decode LLM JSON response.")
            return "Sorry, I had trouble understanding the restaurant request."

        if llm_response.get("action") == "use_tool":
            tool_name = llm_response.get("tool_name")
            tool_input = llm_response.get("tool_input")
            if tool_name == "search_restaurants" and tool_name in self.tools:
                return self.tools[tool_name](tool_input)
        
        print("[Agent Error]: LLM did not return a valid tool call.")
        return "Sorry, I couldn't find a restaurant."

class MovieAgent:
    def __init__(self):
        self.tools = {"search_movies": tool_search_movies}
        self.agent_type = "MOVIE"
        self.system_prompt = (
            "You are a helpful Movie Agent. Your job is to find a movie for the user. "
            "You have one tool: 'search_movies(genre)'. "
            "Analyze the user's prompt, extract the *genre* they want, and then call your tool. "
            "If no genre is specified, default to 'generic'."
        )

    def run(self, prompt):
        print(f"\n--- {self.agent_type} Activated ---")
        llm_response_str = get_llm_response(self.system_prompt, prompt, SCHEMA_TOOL_CALL)
        
        try:
            llm_response = json.loads(llm_response_str)
        except json.JSONDecodeError:
            print("[Agent Error]: Failed to decode LLM JSON response.")
            return "Sorry, I had trouble understanding the movie request."
            
        if llm_response.get("action") == "use_tool":
            tool_name = llm_response.get("tool_name")
            tool_input = llm_response.get("tool_input")
            if tool_name == "search_movies" and tool_name in self.tools:
                return self.tools[tool_name](tool_input)
                
        print("[Agent Error]: LLM did not return a valid tool call.")
        return "Sorry, I couldn't find a movie."

print("Cell 5 executed: Specialist Agents defined.")


# --- Cell 6: Main "Planner" Agent (Updated) ---
# Run this cell to define the main agent.

class PlannerAgent:
    def __init__(self):
        # Feature 1: Multi-agent system
        self.restaurant_agent = RestaurantAgent()
        self.movie_agent = MovieAgent()
        
        # Feature 3: Sessions & State Management
        self.plan = {} 
        
        self.summary_system_prompt = (
            "You are a helpful planner. You have been given a JSON object containing a user's original request, "
            "a chosen restaurant, and a chosen movie. Your job is to summarize this information into a "
            "friendly, single-paragraph response for the user. "
            "Do not just repeat the JSON, make it sound natural and helpful."
        )

    def run(self, user_input):
        """
        This is a "Sequential" agent flow.
        Step 1: Get restaurant.
        Step 2: Get movie.
        Step 3: Summarize.
        """
        print(f"\n--- New User Request ---")
        print(f"[User]: {user_input}")
        
        self.plan = {"user_request": user_input}
        
        # Step 1: Call RestaurantAgent
        # We pass the full user_input, and the agent's system prompt
        # helps it focus on only the restaurant part.
        restaurant_result = self.restaurant_agent.run(user_input)
        self.plan["restaurant"] = restaurant_result
        print(f"[PlannerAgent]: Stored restaurant in plan: {restaurant_result}")
        
        # Step 2: Call MovieAgent
        movie_result = self.movie_agent.run(user_input)
        self.plan["movie"] = movie_result
        print(f"[PlannerAgent]: Stored movie in plan: {movie_result}")

        # Step 3: Summarize the complete plan
        summary_prompt = (
            "Here is the completed plan. Please summarize it for the user: \n"
            f"{json.dumps(self.plan, indent=2)}"
        )
        
        llm_response_str = get_llm_response(self.summary_system_prompt, summary_prompt, SCHEMA_FINAL_RESPONSE)
        
        try:
            llm_response = json.loads(llm_response_str)
        except json.JSONDecodeError:
            print("[PlannerAgent Error]: Failed to decode final summary JSON.")
            return "I've gathered the details, but I had trouble summarizing them."

        final_response = llm_response.get("action_input", "I'm sorry, I couldn't complete the plan.")
        
        print(f"\n[PlannerAgent Final Response]: {final_response}")
        return final_response

print("Cell 6 executed: PlannerAgent defined.")


# --- Cell 7: Run the System ---
# This is the final cell you run to test the agent.
# Make sure you have run all the cells above first.

if not GOOGLE_API_KEY:
    print("Please set your GOOGLE_API_KEY (Cell 2) and restart the kernel to run the agent.")
else:
    print("\n--- Running PlannerAgent System ---")
    concierge = PlannerAgent()
    final_plan = concierge.run("I want to go out tonight! Find me a fun Italian dinner and a comedy movie.")
    print("--- System Run Complete ---")

