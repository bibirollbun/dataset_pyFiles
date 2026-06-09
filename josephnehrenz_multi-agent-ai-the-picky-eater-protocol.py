# Install libraries
!pip install google-adk > /dev/null 2>&1

import asyncio
import warnings
import random
import os
from kaggle_secrets import UserSecretsClient

# Suppress the specific framework warning related to tool calls.
warnings.filterwarnings("ignore", "there are non-text parts in the response")

# Google API authentication
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# Import ADK Components
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")

# Configure Retry Options
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


# --- Define the Custom Tool ---
# This function will be called by the Analyst Agent.
def calculate_logistics(recipe_details: str) -> str:
    """
    Analyzes the recipe details to calculate logistics (simulated prep time and shopping list).
    """
    import random
    
    # Base time is faster for 'quick'/'speedy' recipes
    base_time = 20 if 'quick' in recipe_details.lower() or 'speedy' in recipe_details.lower() else 45
    prep_time = f"{random.randint(base_time - 10, base_time + 15)} minutes"
    
    # Logic to identify the main shopping item
    if "chicken" in recipe_details.lower():
        main_item = "Chicken Breast (1.5 lbs)"
    elif "pasta" in recipe_details.lower():
        main_item = "Dry Pasta (1 box)"
    else:
        main_item = "Protein Source (Check label)"

    # Shopping list is formatted for the Analyst Agent to present cleanly
    shopping_list_snippet = f"* {main_item}\n* Spices\n* Milk/Dairy\n* Bread/Buns"

    return f"""
    PREP_TIME_RESULT: {prep_time}
    SHOPPING_LIST_RESULT: {shopping_list_snippet}
    """

# We must wrap the function for use by the ADK framework
logistics_tool = FunctionTool(func=calculate_logistics)

print("âœ… Custom Logistics Tool defined.")


# --- Define the Specialized Agents ---
# The Chef Agent (Idea Generator)
chef_agent = Agent(
    name="Chef_Agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Generates a simple, single recipe idea based on a main ingredient and provides ingredients.",
    instruction="""
    You are the 'Picky Eater Chef.' Your sole task is to generate a recipe idea and a corresponding Key Ingredients list based on the user's prompt.    
    Format your response with the **Recipe Name:**, a **Description:**, and then the **Key Ingredients:**.    
    CRITICAL FORMATTING: List the **Key Ingredients** as a single, comma-separated sentence (e.g., 'Pasta, butter, garlic, Parmesan cheese, milk, salt, and pepper.'), NOT as a bulleted or numbered list.
    """,
    tools=[google_search], # Tool: Built-in Google Search
)

# The Censor Agent (Context Engineering/Filter) - THE CORE VALUE
censor_agent = Agent(
    name="Censor_Agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Filters recipes based on strict family sensory and ingredient rules.",
    instruction=
    """
    You are the 'Sensory Safety Officer.' Review the proposed recipe against the Family Blacklist.
    
    **Family Blacklist (STRICTLY FORBIDDEN):**
    1. **Texture/Preparation:** Anything described as 'Mushy,' 'Slimy,' 'Lumpy,' or 'Viscous.'
    2. **Ingredient:** Onions (any form), Bell Peppers, Mushrooms, or any dish that requires a sauce with chunks.
    
    If safe, output only the single word '\nAPPROVED'.
    If you reject it, output only the text '\nREJECTED: [The Blacklisted Item that caused the failure]'.
    """,
    tools=[],  
)

# The Analyst Agent (Logistics & Tool User)
analyst_agent = Agent(
    name="Analyst_Agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Processes the final approved recipe into actionable logistics using a custom tool.",
    instruction=
    """
    You are the final reviewer. Your task is to use the 'calculate_logistics' tool on the recipe. 
    Based on the tool's output, generate the final report using Markdown for clarity.
    
    Format:
    **Estimated Prep Time:** [TIME FROM TOOL]
    **Shopping List:**
    [LIST ITEMS FROM TOOL]
    
    DO NOT include any other commentary or introductory text.
    """,
    tools=[logistics_tool],
)
print("âœ… Specialized Agents defined.")


# --- Define the Sequential Workflow (ORCHESTRATOR CODE) ---
def extract_content_text(response_object):
    """
    Safely extracts the final text from the ADK's Message/Event structure.
    """
    if hasattr(response_object, 'text') and response_object.text:
        return response_object.text
    try:
        if hasattr(response_object, 'content') and response_object.content and response_object.content.parts:
            if response_object.content.parts[0].text:
                return response_object.content.parts[0].text
    except Exception:
        pass 
    if hasattr(response_object, 'output') and isinstance(response_object.output, str):
        return response_object.output
    
    return "ADK_EXTRACT_ERROR"

async def rerun_chef_agent(prompt: str):
    """Reruns the Chef Agent with a new prompt and returns the text output."""
    # Assumes chef_agent is defined globally
    chef_runner = InMemoryRunner(agent=chef_agent)
    chef_response_list = await chef_runner.run_debug(prompt)
    return extract_content_text(chef_response_list[-1])

async def run_picky_eater_protocol_simplified(user_prompt: str):
    """
    Runs the sequential chain: Chef -> Censor -> Analyst.
    """
    
    # --- 1. Chef Agent (Generates) ---
    print(f"\nğŸ‘¨â€�ğŸ�³ Step 1: Chef Agent generating idea for: '{user_prompt}'...")
    chef_runner = InMemoryRunner(agent=chef_agent)
    
    chef_response_list = await chef_runner.run_debug(user_prompt)
    recipe_proposal = extract_content_text(chef_response_list[-1])
    
    if "ADK_EXTRACT_ERROR" in recipe_proposal:
         print("â�Œ Critical Error: Cannot reliably extract text from Chef Agent output.")
         return
    
    # --- ISOLATE KEY DATA FOR CLEANER SUBSEQUENT INPUTS ---
    # 1. Find Recipe Name
    name_start = recipe_proposal.find("**Recipe Name:**")
    name_end = recipe_proposal.find("**Description:**") if recipe_proposal.find("**Description:**") != -1 else len(recipe_proposal)
    recipe_name_block = recipe_proposal[name_start:name_end].strip()

    # 2. Find Key Ingredients
    ingredients_start = recipe_proposal.find("**Key Ingredients:**")
    ingredients_block = recipe_proposal[ingredients_start:].strip()

    # Create the minimal, compressed payload for Censor and Analyst
    minimal_payload = f"{recipe_name_block}\n{ingredients_block}"
    
    # --- 2. Censor Agent (Filters/Context Engineering) ---
    print("\nğŸš¨ Step 2: Censor Agent applying the 'Family Blacklist' protocol...")
    censor_runner = InMemoryRunner(agent=censor_agent)
    
    # Send the minimal payload to reduce the size of the repeated "User >" print
    censor_response_list = await censor_runner.run_debug(minimal_payload)
    censor_output = extract_content_text(censor_response_list[-1]).strip()

    # Check for the REJECTED signal (Failure Case)
    if censor_output.startswith("REJECTED"):
        blacklisted_item = censor_output.split(":")[1].strip()
        print(f"\nâ�Œ PROTOCOL FAILED. Censor Agent Report: {censor_output}")
        
        # --- CONTEXT ENGINEERING: Refine the prompt for the Chef Agent ---
        # Capture the rejected item (blacklisted_item) and inject it into a new, concise prompt.
        # This context compaction ensures the Chef Agent is highly constrained and efficient in the retry.
        original_recipe_name = minimal_payload.split('\n')[0].replace('**Recipe Name:**', '').strip()
        
        recovery_prompt = (
            f"The recipe '{original_recipe_name}' was rejected. CRITICAL CONTEXT: The item '{blacklisted_item}' is forbidden. "
            f"Propose a similar recipe idea that is safe and does NOT contain {blacklisted_item}. Focus on ingredients that are known to be safe."
        )
        
        print("\nğŸ”„ Step 4: Chef Agent searching for alternative recipe (using Engineered Context)...")
        replacement_recipe = await rerun_chef_agent(recovery_prompt)
        
        print("\nâœ… New Proposal Approved: (Alternative)")
        print(replacement_recipe)
        
        # --- NEW STEP 5: Analyst Agent calculates logistics for alternative ---
        print("\nğŸ’° Step 5: Analyst Agent calculating logistics for alternative recipe...")
        
        analyst_runner = InMemoryRunner(agent=analyst_agent)
        
        # Send the replacement_recipe text to the analyst for tool execution
        analyst_response_list = await analyst_runner.run_debug(replacement_recipe)
        final_report = extract_content_text(analyst_response_list[-1])

        # Print the final report (Recovery Path)
        print("="*70)
        print("--- FINAL APPROVED DINNER REPORT (Alternative) ---")
        print(final_report)
        print("="*70)
        
        return # Exit the protocol after successful recovery
        
    # --- 3. Analyst Agent (Success Case) ---
    print("âœ… Recipe Approved by Censor Agent. Proceeding to logistics.")
    print("\nğŸ’° Step 3: Analyst Agent calculating logistics using the custom tool...")
    analyst_runner = InMemoryRunner(agent=analyst_agent)
    
    # Send the FULL recipe proposal to the Analyst for reliable tool execution
    # CHANGE THIS LINE:
    analyst_response_list = await analyst_runner.run_debug(recipe_proposal) 
    
    # The Analyst's output will be the clean, formatted final report
    final_report = extract_content_text(analyst_response_list[-1])

    # Print the final report (Success Path)
    print("\n" + "="*70)
    print("--- FINAL APPROVED DINNER REPORT ---")
    print(final_report)
    print("="*70)


# --- Run the Agents ---
print("--- SCENARIO A: Success Case (Asking for a safe food) ---")
# Use 'await' to run the async function directly in the notebook environment
await run_picky_eater_protocol_simplified("Give me a meal idea for chicken and cheese.")


print("--- SCENARIO B: Failure Case (Asking for a blacklisted ingredient) ---")
# Use 'await' to run the async function directly in the notebook environment
await run_picky_eater_protocol_simplified("I need a complex recipe for Beef Chili with chopped onions.")


import sys
import platform
import json
import pkg_resources
from colorama import Fore, Style

def print_section_header(title):
    """Standardized section header with color."""
    print(f"\n{Fore.GREEN}{'='*20} {title} {'='*20}{Style.RESET_ALL}")

def get_package_version(package_name):
    """Safely retrieves the installed version of a package."""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return f"{package_name} Not Installed"
    except Exception as e:
        return f"Error: {e}"

def print_aligned_dict(data_dict, title):
    """Prints a dictionary with key-value pairs aligned for neatness."""
    print(f"\n{title}:")
    
    # Calculate max key length for alignment
    max_key_len = max(len(str(k)) for k in data_dict.keys())
    
    for key, value in data_dict.items():
        print(f"  - {key:<{max_key_len}} : {value}")


# --- Environment Summary Generation ---
print_section_header("ADK MULTI-AGENT ENVIRONMENT SUMMARY")

env_summary = {
    "python_version": sys.version.split('\n')[0].strip(),
    "operating_system": platform.platform(),
}

dependency_versions = {
    "google_adk": get_package_version('google-adk'),
    "numpy": get_package_version('numpy'),
    "pandas": get_package_version('pandas'),
}

# Print the core environment details
print_aligned_dict(env_summary, "# CORE ENVIRONMENT")

# Print the package dependencies
print_aligned_dict(dependency_versions, "# KEY DEPENDENCIES")

print(f"\n{'-'*50}")

