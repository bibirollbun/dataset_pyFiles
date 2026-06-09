# --- CELL 1: Install Dependencies ---
!pip install google-genai

# --- CELL 2: Setup & Imports ---
import os
from google import genai
from google.genai import types

# [IMPORTANT] REPLACE WITH YOUR API KEY
os.environ["GOOGLE_API_KEY"] = "AIzaSyBedfES4h8BdYN9hxQvgX_ZXiNzaS3QmwI"

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
print("Setup complete.")

# --- CELL 3: Robust Tool Definition (The Fix) ---

# 1. The actual Python function we want to run
def get_destination_weather(city: str):
    """Retrieves weather for a city."""
    mock_weather = {
        "Tokyo": "Sunny, 22C",
        "Paris": "Rainy, 14C",
        "London": "Foggy, 12C"
    }
    return mock_weather.get(city, "Sunny, 25C")

# 2. Manual Tool Definition (Bypasses the 'from_function' error)
# We explicitly tell the AI what the function looks like.
weather_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_destination_weather",
            description="Get the weather for a specific city.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "city": types.Schema(
                        type="STRING",
                        description="The city name"
                    )
                },
                required=["city"]
            )
        )
    ]
)

print("Tool defined manually (Robust mode).")
# --- CELL 4: Agent Class (FIXED MODEL NAME) ---
class SimpleAgent:
    def __init__(self, name, instruction, tools=None):
        self.name = name
        self.instruction = instruction
        self.tools = tools
        # FIX: Use the specific version '001' to avoid 404 errors
        self.model = "gemini-1.5-flash-001" 

    def run(self, prompt):
        # Combine system instruction with user prompt
        full_content = f"SYSTEM: {self.instruction}\nUSER: {prompt}"
        
        # specific config for tools
        config = None
        if self.tools:
            config = types.GenerateContentConfig(tools=self.tools)

        response = client.models.generate_content(
            model=self.model,
            contents=[full_content],
            config=config
        )
        return response

# --- CELL 5: Run the Capstone ---
def run_project():
    print("--- Starting Smart Travel Concierge ---")
    
    # 1. Setup Agents
    researcher = SimpleAgent(
        name="Researcher",
        instruction="You are a travel researcher. Extract the city from the user request and call 'get_destination_weather' to check the weather.",
        tools=[weather_tool]
    )
    
    planner = SimpleAgent(
        name="Planner",
        instruction="You are a trip planner. Write a 3-day itinerary based on the City and Weather provided."
    )

    # 2. User Input
    user_input = "I want to visit Paris."
    print(f"User Request: {user_input}")

    # 3. Agent 1: Research (with Tool)
    print("\n[Researcher Agent working...]")
    res_response = researcher.run(user_input)

    # Manual Tool Execution Handler
    # This part "catches" the AI's request to run code
    research_output = res_response.text or "" # Default to text
    
    # Check if the AI wants to call a function
    if res_response.function_calls:
        for fc in res_response.function_calls:
            if fc.name == "get_destination_weather":
                # Get arguments
                city = fc.args["city"]
                print(f" > Tool Triggered: Checking weather for {city}...")
                
                # Run actual function
                weather_data = get_destination_weather(city)
                print(f" > Tool Result: {weather_data}")
                
                # Add to output
                research_output = f"City: {city}, Weather: {weather_data}"

    print(f"Researcher Found: {research_output}")

    # 4. Agent 2: Planner
    print("\n[Planner Agent working...]")
    plan_response = planner.run(f"Create itinerary for: {research_output}")
    
    print("\n--- FINAL ITINERARY ---")
    print(plan_response.text)

# Run it
run_project()



