!pip install google-adk


#Set up Google_API_Key

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


#Import ADK (Agent Development Kit) components
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


#Configure Retry Options
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Recipe Finder Agent: Its job is to use the google_search tool and present findings.
recipe_finder_agent = Agent(
    name="RecipeFinderAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to look for recipes.
    Base you search on the list of provided ingredients. 
    Exlude any ingredients listed as allergens.
    Present the most relevant recipes that match the allowed ingredients and avoid allergens
    for the next agents to use. Also provide short descriptions and links, 
    if available, for each suggested recipe.""",
    tools=[google_search],
    output_key="recipe_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… Recipe Finder Agent created.")


# Recipe Selector Agent: Its job is to select 5 best recipes.
recipe_selector_agent = Agent(
    name="RecipeSelectorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Read the provided list of recipes: {recipe_findings}. 
    Select the three best recipes based on ratings, reviews, cooking time, simplicity, 
    ingredient match, and allergen restrictions; 
    if ratings or reviews are unavailable, choose based on cooking time, 
    clarity and simplicity of instructions, ingredient match, allergen safety, 
    and overall practicality.""",
    output_key="final_recipies",
)

print("âœ… Recipe Selector Agent created.")





# RecipeDetailAgent to fetch full cooking instructions
recipe_detail_agent = Agent(
    name="RecipeDetailAgent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""
    You are a Recipe Detail Agent. Your job is to provide full cooking instructions for a selected recipe.
    1. Receive the recipe title from: {final_recipies}.
    2. Return the full cooking instructions and ingredients, and the detailed recipe.
    """,
       output_key="detailed_final_recipies",
)
print("âœ… Recipe Detail Agent created.")


# Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.

root_agent = SequentialAgent(
    name="ResearchCoordinator",
    sub_agents = [recipe_finder_agent,recipe_selector_agent,recipe_detail_agent],
)
    
print("âœ… root_agent created.")


runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug(
    "I need recipe of soup with beef, carrot, celery, potato, tometo. Allegens are sesame seed, almond, rice"
)


response = await runner.run_debug(
    "I need a carrot cake recipe. Allergens: sesame seed, almond, rice"
)


# Suppose user selects a recipe
chosen_title = "Hearty Vegetable Beef Soup"

# Fetch full instructions using RecipeDetailAgent
detail_response = await runner.run_debug(chosen_title)
full_instructions = detail_response[-1].content.parts[0].text

# Save to memory
save_recipe(chosen_title, full_instructions)

print(f"Recipe '{chosen_title}' saved successfully!")


