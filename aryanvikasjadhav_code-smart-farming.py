# 1. Install necessary libraries 

# 2. Import modules for authentication
import os
from kaggle_secrets import UserSecretsClient

# 3. Securely load the API Key from Kaggle Secrets
try:
    # This line retrieves the key you saved with the label 'GOOGLE_API_KEY'
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup complete. Gemini API Key loaded securely.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# 4. Import the main agent modules
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Initialize the Gemini Client
# It automatically uses the GEMINI_API_KEY environment variable
try:
    client = genai.Client()
    print("âœ… Gemini Client initialized.")
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    


# --- ğŸ›°ï¸� Tool Definitions (Simulating Data Sources) ---

# Tool 1: Sensor Data (e.g., from an IoT device)
class SensorData(BaseModel):
    """Current sensor readings for a specific field/crop."""
    crop_type: str = Field(description="The specific crop being monitored (e.g., Wheat, Corn).")
    soil_moisture_percent: int = Field(description="Current soil moisture level (0-100%).")
    air_temperature_celsius: float = Field(description="Current ambient air temperature in Celsius.")
    ph_level: float = Field(description="Current soil pH level.")

def get_current_sensor_data(crop_name: str) -> dict:
    """
    Retrieves the latest real-time sensor data for a specified crop type.
    Example: get_current_sensor_data(crop_name="Wheat")
    """
    # [attachment_0](attachment)
    print(f"\n[TOOL CALLED: get_current_sensor_data] for {crop_name}...")
    # Simulate different sensor data based on the crop
    if "wheat" in crop_name.lower():
        # Dry, acidic soil, hot (Wheat optimal pH 6.0-7.0)
        data = SensorData(crop_type="Wheat", soil_moisture_percent=18, air_temperature_celsius=31.5, ph_level=5.8)
    elif "corn" in crop_name.lower():
        # Optimal moisture, neutral pH (Corn optimal pH 6.0-7.5)
        data = SensorData(crop_type="Corn", soil_moisture_percent=55, air_temperature_celsius=24.0, ph_level=6.5)
    else:
        # Default scenario
        data = SensorData(crop_type=crop_name, soil_moisture_percent=30, air_temperature_celsius=28.0, ph_level=6.0)
        
    return {"status": "SUCCESS", "data": data.model_dump()}


# Tool 2: Pest & Disease Reference (e.g., from a database)
def lookup_pest_and_disease_info(crop_name: str) -> dict:
    """
    Searches a reference database for common pests and diseases affecting the specified crop.
    Example: lookup_pest_and_disease_info(crop_name="Wheat")
    """
    print(f"\n[TOOL CALLED: lookup_pest_and_disease_info] for {crop_name}...")
    if "wheat" in crop_name.lower():
        info = "Common Wheat Pests: Aphids, Hessian Fly. Diseases: Rust (Puccinia spp.), Powdery Mildew. Recommended action: Monitor for orange/yellow spots (Rust)."
    elif "corn" in crop_name.lower():
        info = "Common Corn Pests: Corn Earworm, Fall Armyworm. Diseases: Blight, Gray Leaf Spot. Recommended action: Check leaf collars for 'window-pane' lesions (Gray Leaf Spot)."
    else:
        info = f"No specific high-priority pest or disease information found for {crop_name} at this moment."
        
    return {"status": "SUCCESS", "info": info}

print("âœ… Agent tools defined.")



def run_farming_guide_agent(farmer_query: str, field_name: str):
    """
    The central agent that uses tools to answer the farmer's query.
    """
    print(f"--- ğŸ§‘â€�ğŸŒ¾ Agent Starting for Field: {field_name} ---\n")
    print(f"Farmer's Query: {farmer_query}\n")

    # 1. Define the agent's core instructions and expertise
    system_instruction = f"""
    You are a professional Smart Farming Guide Assistant. Your current focus is **{field_name}**.
    
    Your goal is to provide comprehensive, actionable, and safe advice to the farmer.
    You must always use the available tools to retrieve factual data before formulating a response.
    
    Analysis Protocol:
    1. If the query relates to **soil, water, or climate**, use `get_current_sensor_data`.
    2. If the query relates to **pests or diseases**, use `lookup_pest_and_disease_info`.
    3. Synthesize all retrieved tool data into a clear, professional farming guide.
    4. Provide specific recommendations (e.g., "Irrigate 1 inch now" or "Apply fungicide X").
    """

    # 2. Configure the model with tools
    model_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[get_current_sensor_data, lookup_pest_and_disease_info] # List all available tools
    )

    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=model_config,
    )

    # 3. Initial message to trigger tool use
    response = chat.send_message(farmer_query)
    
    # 4. Handle Tool Calls Loop (Important for multi-step reasoning)
    while response.function_calls:
        print("\n--- Model requested Tool Calls ---")
        function_calls = response.function_calls
        tool_responses = []

        for f_call in function_calls:
            print(f"  -> Calling Tool: {f_call.name} with args: {dict(f_call.args)}")
            
            # --- Execute the Python Function based on the model's request ---
            tool_args = dict(f_call.args)
            if f_call.name == "get_current_sensor_data":
                result = get_current_sensor_data(**tool_args)
            elif f_call.name == "lookup_pest_and_disease_info":
                result = lookup_pest_and_disease_info(**tool_args)
            else:
                result = {"error": f"Tool '{f_call.name}' not found"}
            # --- End Tool Execution ---
            
            tool_responses.append(
                types.Part.from_function_response(
                    name=f_call.name,
                    response={"result": result}
                )
            )

        # Send tool results back to the model for the next step of reasoning
        print("--- Sending Tool Results back to Model for Final Analysis ---")
        response = chat.send_message(tool_responses)

    # 5. Final Output
    print("\n" + "="*50)
    print("      FINAL FARMING GUIDE RECOMMENDATION")
    print("="*50)
    print(response.text)
    print("="*50 + "\n")
    


# --- Example Runs ---

print("="*20 + " EXAMPLE 1 (WHEAT - Dry/Pest Focus) " + "="*20)
# A query requiring two different tools (Sensor + Pest)
run_farming_guide_agent(
    farmer_query="I have a wheat field. What are the current soil conditions, and should I be worried about any diseases right now?",
    field_name="Wheat Field 3A"
)

print("\n" + "#"*70 + "\n")

print("="*20 + " EXAMPLE 2 (CORN - Irrigation Focus) " + "="*20)
# A query focused primarily on a single tool (Sensor)
run_farming_guide_agent(
    farmer_query="My corn field looks dry. Should I irrigate and what's the pH?",
    field_name="Corn Field B-2"
)


