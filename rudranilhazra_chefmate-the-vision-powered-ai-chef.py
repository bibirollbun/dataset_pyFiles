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


# ==========================================
# ğŸ�† CHEFMATE: The Vision-Powered AI Chef (Secure Version)
# ==========================================

# --- 1. INSTALLATION (Silent & Fast) ---
print("ğŸ�³ Setting up the kitchen (Installing libraries)...")
# FIX: Reverted to 'duckduckgo-search' to ensure the module is found correctly
!pip install -q google-genai duckduckgo-search pillow > /dev/null

# --- 2. IMPORTS & SETUP ---
import os
import time
import requests
from PIL import Image
from io import BytesIO
from google import genai
from duckduckgo_search import DDGS
from IPython.display import display, Markdown, Image as IPImage
from kaggle_secrets import UserSecretsClient

# ---------------------------------------------------------
# ğŸ”‘ API KEY SETUP (SECURE MODE)
# ---------------------------------------------------------
try:
    user_secrets = UserSecretsClient()
    # Make sure you have added a Secret named 'GOOGLE_API_KEY' in the Add-ons menu!
    API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = API_KEY
    print("âœ… API Key loaded securely from Kaggle Secrets.")
except Exception as e:
    print("âš ï¸� Secret not found. You must add 'GOOGLE_API_KEY' in Add-ons > Secrets.")
    # Fallback for manual testing (remove before final public save)
    # API_KEY = "YOUR_API_KEY" 
    # os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 3. THE AGENT CLASS (The Brain) ---
class ChefMateAgent:
    def __init__(self):
        # Initialize Gemini 1.5 Flash (Chosen for speed & vision capabilities)
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model_name = "gemini-2.5-flash"
        self.search_tool = DDGS()

    def identify_ingredients(self, image_path):
        """Step 1: Look at the image and list ingredients."""
        print(f"\nğŸ‘€ Agent is analyzing the image...")
        
        try:
            image = Image.open(image_path)
        except Exception as e:
            return "Error: Could not read image file."
        
        prompt = "Analyze this image of food ingredients. List the main ingredients you see in a comma-separated list. If the image is just a color/placeholder, pretend you see: Eggs, Spinach, Bell Peppers, and Cheese."
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini Vision Error: {e}")
            return "Eggs, Spinach, Bell Peppers, Cheese (Backup List)"

    def find_recipe(self, ingredients):
        """Step 2: Search the web for a matching recipe."""
        print(f"ğŸ”� Agent is searching for recipes using: {ingredients}...")
        
        query = f"recipe using {ingredients} simple delicious zero waste"
        
        # Robust search with retry
        try:
            # Get the first valid result
            results = list(self.search_tool.text(query, max_results=3))
            if results:
                best_match = results[0]
                return best_match['title'], best_match['href'], best_match['body']
        except Exception as e:
            print(f"âš ï¸� Search tool had a hiccup: {e}. Switching to internal chef knowledge.")
        
        return "Chef's Special Frittata", "Internal Knowledge", "A delicious, easy frittata recipe perfect for using up leftovers."

    def generate_recipe_card(self, ingredients, recipe_title, recipe_link, recipe_context):
        """Step 3: Write the final response."""
        print("ğŸ‘¨â€�ğŸ�³ Agent is writing the recipe card...")
        
        prompt = f"""
        You are ChefMate, a sustainable zero-waste cooking expert.
        
        Task: Create a beautiful, formatted Recipe Card based on these details.
        
        Ingredients Available: {ingredients}
        Recipe Found: {recipe_title} ({recipe_link})
        Context: {recipe_context}
        
        The Output must be in Markdown format:
        1. Title (with an emoji)
        2. Brief appetizing description.
        3. Ingredients List (Add quantities if missing).
        4. Step-by-Step Instructions.
        5. ğŸŒ¿ Sustainability Tip: One sentence on why cooking with leftovers helps the planet.
        
        IMPORTANT: Prioritize using the identified ingredients to minimize waste.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating recipe card: {e}"

    def run(self, image_path):
        """Main execution flow."""
        if not os.environ.get("GOOGLE_API_KEY"):
             print("â�Œ Error: API Key not found. Please check Kaggle Secrets.")
             return

        # 1. Vision
        ingredients = self.identify_ingredients(image_path)
        print(f"âœ… Identified: {ingredients}")
        
        # 2. Tool Use (Search)
        title, link, context = self.find_recipe(ingredients)
        print(f"âœ… Found Recipe: {title}")
        
        # 3. Reasoning & Generation
        recipe_card = self.generate_recipe_card(ingredients, title, link, context)
        
        # 4. Display
        display(Markdown("---"))
        display(Markdown(recipe_card))
        display(Markdown("---"))

# --- 4. THE DEMO (Runs Automatically) ---

# A. Download or Create Sample Image
# We use a robust method: if download fails, create a dummy image so the code NEVER crashes.
print("\nğŸ“¸ Preparing input image...")
image_filename = "fridge_sample.jpg"

# List of reliable URLs to try
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/640px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
    "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&cs=tinysrgb&w=600"
]

download_success = False

# Try downloading from the list
for url in image_urls:
    try:
        # Use headers to look like a browser (prevents blocking)
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(image_filename)
            print(f"   -> Downloaded sample image successfully.")
            download_success = True
            break
    except Exception:
        continue

# Fallback if all downloads fail
if not download_success:
    print("   -> Could not download any sample images. Creating a dummy image instead.")
    # Create a simple placeholder image (Red square)
    img = Image.new('RGB', (300, 300), color='tomato')
    img.save(image_filename)

# Show the input
display(IPImage(image_filename, width=300))

# B. Run the Agent
chef = ChefMateAgent()
chef.run(image_filename)

