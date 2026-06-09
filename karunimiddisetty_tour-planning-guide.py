# --- 0. Install the required library ---
# We must do this first in the Kaggle environment
!pip install -q google-generativeai

print("Installed google-generativeai library.")

# --- 1. Import libraries and access the API Key ---
import os
import google.generativeai as genai
from google.generativeai import types

# This is the special way to access Kaggle Secrets
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    # This line retrieves the secret you just added
    my_secret_key = user_secrets.get_secret("GOOGLE_API_KEY")
    
    # Configure the genai library with your key
    genai.configure(api_key=my_secret_key)
    print("Google API Key configured successfully.")
    
    # --- 2. Helper Function to Call the Gemini LLM ---
    def get_llm_response(prompt):
        """
        Calls the Gemini API with a given prompt and returns
        the text content of the response.
        """
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            system_instruction = "You are an expert tour guide."
            
            response = model.generate_content(
                prompt,
                generation_config=types.GenerationConfig(),
                system_instruction=system_instruction
            )
            return response.text
        except Exception as e:
            # Handle potential API errors, e.g., quota
            if "quota" in str(e).lower():
                return "Error: API quota exceeded. Please check your Google AI Studio account."
            print(f"Error calling Gemini API: {e}")
            return "Error: Could not generate plan."

    # --- 3. Define the Main "Loop" ---

    user_request = "a 2-day relaxed tour of Tokyo, focusing on food and modern culture"
    itinerary = [] # An empty list to store our plans

    print(f"\nðŸ¤– Generating your 2-day tour for: {user_request}\n")

    # This 'for' loop is our "Loop Agent"
    for day in range(1, 3): # This will run for day=1 and day=2
        print(f"--- Planning Day {day}... ---")

        # 1. Plan the morning ("Parallel" agent call 1)
        morning_prompt = (
            f"Plan a fantastic **morning** for Day {day} "
            f"of a trip to {user_request}. "
            f"Keep it concise (2-3 sentences)."
        )
        morning_plan = get_llm_response(morning_prompt)
        
        # 2. Plan the afternoon ("Parallel" agent call 2)
        afternoon_prompt = (
            f"Plan a fantastic **afternoon** for Day {day} "
            f"of a trip to {user_request}. "
            f"Keep it concise (2-3 sentences)."
        )
        afternoon_plan = get_llm_response(afternoon_prompt)

        # Combine the results for the day
        day_plan = (
            f"**Day {day}:**\n"
            f"* **Morning:** {morning_plan}\n"
            f"* **Afternoon:** {afternoon_plan}"
        )
        
        itinerary.append(day_plan)


    # --- 4. Show the Final Result ---
    print("\nâœ… Here is your complete 2-day itinerary!\n")
    final_plan = "\n\n".join(itinerary)
    print(final_plan)

# This handles the case where the secret key wasn't found
except Exception as e:
    print(f"Error: Could not retrieve the secret key.")
    print("Please make sure you have added a secret named 'GOOGLE_API_KEY' under Add-ons > Secrets.")
    print(f"Details: {e}")

