# Install the Google GenAI SDK
!pip install google-genai -q


import os
from kaggle_secrets import UserSecretsClient

# Load the API key securely
try:
    user_secrets = UserSecretsClient()
    # The name here ('GOOGLE_API_KEY') must match the secret name in Kaggle
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ['GEMINI_API_KEY'] = GOOGLE_API_KEY # Set the environment variable for the SDK
    print("Google API Key loaded successfully from Kaggle Secrets.")
except Exception as e:
    print(f"Error loading API Key: {e}")
    print("Please ensure you have added your Google API Key to Kaggle Secrets with the name 'GOOGLE_API_KEY'.")

# Import the rest of the libraries
from google import genai
from google.genai.errors import APIError


# Initialize the GenAI client
try:
    client = genai.Client()
    print("Gemini client initialized successfully.")
except Exception as e:
    print(f"Error initializing client: {e}")
    


def task_breaker_agent(complex_task: str, output_filename: str = None):
    """
    Uses a generative model to break a complex task into smaller, actionable steps
    and optionally saves the output to a file.
    
    Args:
        complex_task (str): The large task to be broken down.
        output_filename (str, optional): The name of the file to save the output to. 
                                         Defaults to None (no file output).
        
    Returns:
        str: The numbered list of sub-tasks, or an error message.
    """
    
    system_instruction = (
        "You are a professional Task Breaker Agent. Your sole purpose is to take a "
        "complex task provided by the user and break it down into 5 to 10 smaller, "
        "specific, and actionable sub-tasks. Present the output as a numbered list. "
        "Do not include any other commentary, introductions, or conclusions."
    )
    
    prompt = f"Break down the following complex task:\n\n'{complex_task}'"
    
    print("--- Sending request to the agent... ---")
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config={'system_instruction': system_instruction}
        )
        
        sub_tasks = response.text
        
        # --- NEW CODE: File Output ---
        if output_filename:
            with open(output_filename, 'w') as f:
                f.write(f"Task: {complex_task}\n\n")
                f.write("--- Actionable Sub-Tasks ---\n")
                f.write(sub_tasks)
            print(f"âœ… Output successfully saved to **{output_filename}**")
        # --- END NEW CODE ---
        
        return sub_tasks
    
    except APIError as e:
        return f"An API Error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# Example complex task
my_task = "Plan a comprehensive marketing campaign for a new mobile game launch."

# Define the output file name
output_file = "game_launch_plan.txt"

# Run the agent and save the output
sub_tasks = task_breaker_agent(
    complex_task=my_task, 
    output_filename=output_file
)

# Print the results to the console
print("\n## ðŸ“‹ Task Breakdown Results\n")
print(f"**Original Task:** {my_task}\n")
print("**Actionable Sub-Tasks:**")
print("--------------------------")
print(sub_tasks)




