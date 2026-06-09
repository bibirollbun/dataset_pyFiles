# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# =================================================================
# SETUP: INSTALLING LIBRARIES & CONFIGURING THE ENVIRONMENT
# =================================================================
# This cell prepares the entire environment to ensure a smooth, uninterrupted workflow.

# --- 1. Installing all required packages ---
print("â�³ Installing necessary libraries...")
%pip install -q langchain langchain-core langchain-google-genai google-generativeai "pydantic<2" langchain_community beautifulsoup4
print("âœ… Libraries installed successfully.")

# --- 2. Importing all necessary modules ---
print("\nâ�³ Importing modules...")
import os
import PIL.Image
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
print("âœ… Modules imported successfully.")

# --- 3. Configuring API Keys and Initializing the LLM ---
print("\nâ�³ Configuring API keys and initializing the LLM...")
try:
    user_secrets = UserSecretsClient()
    # Gemini API Key
    gemini_api_key = user_secrets.get_secret("GEMINI_API_KEY")
    os.environ['GOOGLE_API_KEY'] = gemini_api_key
    genai.configure(api_key=gemini_api_key)

    # Google Search API Keys
    os.environ["GCS_DEVELOPER_KEY"] = user_secrets.get_secret("GCS_DEVELOPER_KEY")
    os.environ["GCS_CX"] = user_secrets.get_secret("GCS_CX")
    
    # LLM Initialization
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    
    print("âœ… API keys configured and LLM initialized successfully.")
except Exception as e:
    print(f"ğŸš¨ Error: Please ensure all 3 secrets ('GEMINI_API_KEY', 'GCS_DEVELOPER_KEY', 'GCS_CX') are correctly set.")

print("\n--- ENVIRONMENT SETUP COMPLETE ---")



# =================================================================
# TOOL 1: THE INVENTORY AGENT
# =================================================================
@tool
def inventory_agent(image_path: str) -> str:
    """
    Analyzes an image of a fridge/pantry from a file path to identify ingredients and estimate their quantities.
    This is the first tool to use when a user provides a photo.
    """
    print("\n    [Tool Call: inventory_agent]")
    try:
        img = PIL.Image.open(image_path)
        vision_model = genai.GenerativeModel('gemini-pro-vision')
        prompt = """
        You are an expert image analysis tool. Your job is to identify food items in this image and estimate their quantity.
        Return a single, comma-separated string. Example: 'eggs (6), milk (approx. 500ml), carrots (3)'
        """
        response = vision_model.generate_content([prompt, img])
        return response.text.strip()
    except FileNotFoundError:
        return f"Error: The image file was not found at {image_path}."
    except Exception as e:
        return f"An unexpected error occurred during image analysis: {str(e)}"

print("âœ… Tool 1 (inventory_agent) is defined.")



# =================================================================
# UNIT TEST: INVENTORY AGENT
# =================================================================
# Before assembling the full agent, we perform a quick unit test on our vision tool.

# --- IMPORTANT: Upload an image and update the path below ---
test_image_path = "../input/fridge-photo/my-fridge.jpg" # <--- !!! CHANGE THIS PATH !!!

print(f"ğŸ§ª Testing the inventory_agent with image: {test_image_path}")

if not os.path.exists(test_image_path):
    print(f"ğŸš¨ Test Skipped: Image file not found at '{test_image_path}'. Please upload an image and set the correct path.")
else:
    # We call the tool function directly
    ingredients = inventory_agent.invoke(test_image_path)
    print("\n--- Test Result ---")
    print(f"Identified Ingredients: {ingredients}")

print("\nâœ… Unit test for inventory_agent complete.")



# =================================================================
# TOOL 2 & 3: THE WEB TOOLS
# =================================================================

# --- Tool 2: The Recipe Search Tool ---
# This is a standard tool from the LangChain community.
try:
    search_tool = GoogleSearchAPIWrapper()
    print("âœ… Tool 2 (search_tool) is defined.")
except Exception as e:
    print(f"ğŸš¨ Error defining search_tool. This is likely an issue with the Google Search API keys.")

# --- Tool 3: The Web Scraper Tool ---
# This tool allows the agent to read the content of a webpage.
@tool
def web_scraper_tool(url: str) -> str:
    """
    Fetches the main text content of a given webpage URL. 
    Use this to read the details of a recipe after finding its URL with the search tool.
    """
    print(f"\n    [Tool Call: web_scraper_tool, URL: {url}]")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        content = " ".join([doc.page_content for doc in documents])
        # Truncate content to avoid exceeding the agent's context limit
        return content[:4000]
    except Exception as e:
        return f"An error occurred while scraping the URL: {str(e)}"

print("âœ… Tool 3 (web_scraper_tool) is defined.")



# =================================================================
# UNIT TEST: WEB TOOLS
# =================================================================
# We test the two web tools in sequence to ensure they work together.

print("ğŸ§ª Testing the web tools...")

# --- Test the Searcher ---
print("\n--- Testing search_tool ---")
test_query = "vegetarian pasta recipes with tomatoes"
search_results = search_tool.run(test_query)
print(f"Search results are of type: {type(search_results)}")
print("Search test complete.")

# --- Test the Reader ---
# We will try to scrape the first result from a known good recipe site.
# (Note: Web scraping can be fragile. This test might fail if the website structure changes.)
print("\n--- Testing web_scraper_tool ---")
# Let's use a reliable URL for this test
test_url = "https://www.allrecipes.com/recipe/22285/fresh-tomato-pasta/"
print(f"Attempting to scrape: {test_url}")
scraped_content = web_scraper_tool.invoke(test_url)

if "error" in scraped_content.lower():
    print("Scraping test encountered an error (this is sometimes normal).")
else:
    print("Scraping test successful! First 200 characters of content:")
    print(scraped_content[:200] + "...")

print("\nâœ… Unit test for web tools complete.")



# =================================================================
# TOOL 4: THE SHOPPING LIST AGENT
# =================================================================
@tool
def shopping_list_agent(required_ingredients: str, available_ingredients: str) -> str:
    """
    Compares a comma-separated string of required ingredients with a comma-separated string of available ingredients
    and returns a final, comma-separated string of only the missing items.
    """
    print("\n    [Tool Call: shopping_list_agent]")
    try:
        # Process strings into clean sets for accurate comparison.
        # We handle potential variations in spacing and capitalization.
        required_set = {item.strip().lower() for item in required_ingredients.split(',')}
        available_set = {item.strip().lower() for item in available_ingredients.split(',')}
        
        missing_items = required_set - available_set
        
        if not missing_items:
            return "Great news! You have all the ingredients you need."
        else:
            # Return a clean, sorted, comma-separated string.
            return "Here is your shopping list: " + ", ".join(sorted(list(missing_items)))
    except Exception as e:
        return f"An error occurred while creating the shopping list: {str(e)}"

print("âœ… Tool 4 (shopping_list_agent) is defined.")



# =================================================================
# UNIT TEST: SHOPPING LIST AGENT
# =================================================================
# We test our logic-based tool with a sample scenario.

print("ğŸ§ª Testing the shopping_list_agent...")

# --- Test Scenario ---
available = "eggs, milk, flour, sugar"
required = "eggs, milk, flour, sugar, butter, chocolate chips"

print(f"\nAvailable: {available}")
print(f"Required:  {required}")

# --- Call the tool function directly ---
shopping_list = shopping_list_agent.invoke({
    "available_ingredients": available,
    "required_ingredients": required
})

print("\n--- Test Result ---")
print(shopping_list)

# --- Verification ---
assert "butter" in shopping_list
assert "chocolate chips" in shopping_list
assert "eggs" not in shopping_list

print("\nâœ… Unit test for shopping_list_agent complete.")



# =================================================================
# ORCHESTRATOR AGENT ASSEMBLY
# =================================================================
print("â�³ Assembling the main orchestrator agent...")

# --- 1. Define the complete list of tools for the agent ---
tools = [
    inventory_agent,
    search_tool,
    web_scraper_tool,
    shopping_list_agent
]
print("âœ… Toolset defined.")

# --- 2. Define the Master Prompt (The Agent's "Brain") ---
# This prompt is critical. It tells the agent who it is, what its goal is,
# and how to use its tools in a logical sequence.
prompt_template = """
You are "Personal E-Chef", a friendly and highly intelligent kitchen assistant. Your goal is to help users create meals from the ingredients they have.

You MUST operate in a step-by-step manner. Here is your mandatory workflow:

1.  **Inventory:** When the user provides an image, your absolute first step is to use the `inventory_agent` tool to identify the ingredients and their quantities.

2.  **Clarification:** After getting the inventory, talk to the user. Confirm the ingredients and ask about their dietary preferences or allergies for today (e.g., "I see you have eggs, milk, and cheese. Any allergies I should be aware of?").

3.  **Recipe Search:** Once you have the preferences, use the `search_tool` to find 2-3 relevant recipe URLs. Formulate a good search query using the ingredients and preferences.

4.  **Recipe Analysis:** For EACH recipe URL you found, you MUST use the `web_scraper_tool` to read the content of the page. Then, analyze the scraped text to extract the full list of required ingredients for that recipe.

5.  **Recommendation:** Present the recipe ideas to the user in a clear, friendly format. For each recipe, state that you have analyzed it and are ready to provide the shopping list.

6.  **Shopping List:** Once the user chooses a recipe, use the `shopping_list_agent` tool. You will provide it with two arguments:
    - `required_ingredients`: The list you extracted from the webpage in step 4.
    - `available_ingredients`: The list you got from the `inventory_agent` in step 1.
    
7.  **Final Answer:** Present the final shopping list to the user.

Begin!

User's Request: {input}
Agent's Thought Process:
{agent_scratchpad}
"""
print("âœ… Master Prompt defined.")

# --- 3. Create the Agent and the AgentExecutor ---
# This binds the LLM, the tools, and the prompt together into a runnable agent.
try:
    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    print("âœ… Orchestrator Agent 'Personal E-Chef' is assembled and ready!")
except Exception as e:
    print(f"ğŸš¨ ERROR: Could not assemble the agent. Details: {e}")




# =================================================================
# FINAL END-TO-END TEST
# =================================================================
print("â–¶ï¸� Running the final end-to-end test...")

# --- 1. Define the path to your uploaded image ---
# !!! IMPORTANT !!!
# YOU MUST CHANGE THIS PATH to match the name of your dataset and image file.
# Example: "../input/my-fridge-pictures/fridge_01.jpg"
image_path = "../input/fridge-photo/my-fridge.jpg" # <--- !!! CHANGE THIS PATH !!!

# --- 2. Formulate the initial user query ---
# This is the first message the user sends to the agent.
query = f"Hi! I want to cook something for dinner. Here is a picture of my fridge, can you help me out? The image is at: {image_path}"

# --- 3. Invoke the agent ---
# We run the agent and let it perform its step-by-step reasoning.
try:
    # First, we check if the image file actually exists to prevent a common error.
    if not os.path.exists(image_path):
        print(f"ğŸš¨ FATAL ERROR: The image file was not found at the specified path: '{image_path}'")
        print("    Please upload your image to a Kaggle dataset and update the 'image_path' variable in this cell.")
    else:
        print("\n--- AGENT IS NOW THINKING... ---")
        response = agent_executor.invoke({"input": query})
        
        # --- 4. Print the final, user-facing answer ---
        print("\n\n==========================================")
        print("âœ… FINAL AGENT RESPONSE TO THE USER")
        print("==========================================")
        print(response['output'])

except Exception as e:
    print(f"ğŸš¨ A critical error occurred during the agent's run: {e}")

print("\n--- TEST FINISHED ---")


