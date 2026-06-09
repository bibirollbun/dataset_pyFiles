# --- 1. INSTALL AND IMPORT LIBRARIES ---

# Install the Google GenAI SDK silently
!pip install google-genai -q

import os
from kaggle_secrets import UserSecretsClient
from google import genai
from google.genai.errors import APIError

# --- 2. SECURE API KEY LOADING ---


# The name 'GOOGLE_API_KEY' must match the name of the secret you saved in Kaggle.
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ['GEMINI_API_KEY'] = GOOGLE_API_KEY # Set for the SDK
    print("âœ… Google API Key loaded successfully from Kaggle Secrets.")
except Exception as e:
    print(f"â�Œ Error loading API Key: {e}")
    print("Please ensure your API Key is saved in Kaggle Secrets with the name 'GOOGLE_API_KEY'.")

# Initialize the Gemini client
try:
    client = genai.Client()
    print("âœ… Gemini client initialized successfully.")
except Exception as e:
    print(f"â�Œ Error initializing client: {e}")


# --- 3. DEFINE THE TASK BREAKER AGENT FUNCTION ---

def task_breaker_agent(complex_task: str, output_filename: str = None):
    """
    Uses the Gemini model to break a complex task into smaller, actionable steps
    and optionally saves the output to a file.
   
    Args:
        complex_task (str): The large task to be broken down.
        output_filename (str, optional): The name of the file to save the output.
       
    Returns:
        str: The numbered list of sub-tasks, or an error message.
    """
   
    # System Instruction: Defines the agent's specialized role
    system_instruction = (
        "You are a professional Task Breaker Agent. Your sole purpose is to take a "
        "complex task provided by the user and break it down into 5 to 10 smaller, "
        "specific, and actionable sub-tasks. Present the output ONLY as a numbered list. "
        "Do not include any other commentary, introductions, or conclusions."
    )
   
    prompt = f"Break down the following complex task:\n\n'{complex_task}'"
   
    print("\n--- Sending request to the Task Breaker Agent... ---")
   
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config={'system_instruction': system_instruction}
        )
       
        sub_tasks = response.text
       
        # Action: Write output to a file if a filename is provided
        if output_filename:
            with open(output_filename, 'w') as f:
                f.write(f"Original Task: {complex_task}\n\n")
                f.write("--- Actionable Sub-Tasks ---\n")
                f.write(sub_tasks)
            print(f"âœ… Output successfully saved to **{output_filename}** in the Kaggle working directory.")
       
        return sub_tasks
   
    except APIError as e:
        return f"An API Error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- 4. EXECUTE AND TEST THE AGENT ---

# Define the complex task and the desired output file name
MY_COMPLEX_TASK = "Develop a strategy for migrating a legacy SQL database to a cloud-native NoSQL solution."
OUTPUT_FILENAME = "migration_plan.txt"

# Run the agent
sub_tasks_result = task_breaker_agent(
    complex_task=MY_COMPLEX_TASK,
    output_filename=OUTPUT_FILENAME
)

# Print the results to the console
print("\n## ğŸ“‹ Task Breakdown Results\n")
print(f"**Original Task:** {MY_COMPLEX_TASK}\n")
print("**Actionable Sub-Tasks:**")
print("--------------------------")
print(sub_tasks_result)


# --- 5. (OPTIONAL) VERIFY FILE CREATION ---
print(f"\nVerifying file content (first 300 chars of {OUTPUT_FILENAME}):")
print("----------------------------------------------------------------")
try:
    with open(OUTPUT_FILENAME, 'r') as f:
        print(f.read()[:300] + "...")
except FileNotFoundError:
    print(f"File {OUTPUT_FILENAME} was not found.")




