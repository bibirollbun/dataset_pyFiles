# Core imports
import os
import json
import base64
import time
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# Data & visualization
import pandas as pd
from IPython.display import display, HTML, clear_output
from ipywidgets import Button, Output, VBox, HBox, Label

# API clients
import google.generativeai as genai
from google.cloud import firestore
from google.oauth2 import service_account

# Setup logging
# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging - Suppress output
logging.basicConfig(
    level=logging.CRITICAL, # Suppress INFO, WARNING, ERROR
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("âœ… Dependencies loaded!")


# Configuration - KAGGLE SECRETS
from kaggle_secrets import UserSecretsClient
import json
import os

class Config:
    try:
        user_secrets = UserSecretsClient()
        
        # Gemini API
        GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
        
        # Model Selection
        GEMINI_MODEL = "gemini-2.0-flash" 
        
        # Firestore
        GOOGLE_CLOUD_PROJECT = user_secrets.get_secret("GOOGLE_CLOUD_PROJECT")
        FIRESTORE_DATABASE = user_secrets.get_secret("FIRESTORE_DATABASE")
        
        # Service Account (Directly from secrets)
        service_account_json = user_secrets.get_secret("GCP_SERVICE_ACCOUNT")
        service_account_info = json.loads(service_account_json)
            
    except Exception as e:
        print(f"âš ï¸� Error loading secrets: {e}")
        GEMINI_API_KEY = None
        GOOGLE_CLOUD_PROJECT = None
        FIRESTORE_DATABASE = None
        service_account_info = None

    # User configuration
    USER_ID = "kaggle_demo_user"
    
config = Config()

# Initialize Gemini
try:
    if config.GEMINI_API_KEY:
        genai.configure(api_key=config.GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)
        print(f"âœ… Gemini API configured with model: {config.GEMINI_MODEL}")
    else:
        print("âš ï¸� GEMINI_API_KEY not found in secrets")
        gemini_model = None
except Exception as e:
    print(f"âš ï¸� Gemini configuration failed: {e}")
    gemini_model = None

# Initialize Firestore
db = None
if config.service_account_info:
    try:
        credentials = service_account.Credentials.from_service_account_info(config.service_account_info)
        print(f"ğŸ”Œ Connecting to Firestore Project: {config.GOOGLE_CLOUD_PROJECT}, Database: {config.FIRESTORE_DATABASE}...")
        db_client = firestore.Client(project=config.GOOGLE_CLOUD_PROJECT, credentials=credentials, database=config.FIRESTORE_DATABASE)
        try:
            list(db_client.collection('users').limit(1).stream())
            db = db_client
            print("âœ… Firestore connected")
        except Exception as db_err:
            print(f"âš ï¸� Firestore available but connection failed (using memory): {db_err}")
            db = None
    except Exception as e:
        print(f"âš ï¸� Firestore initialization failed (using memory): {e}")
        db = None
else:
    print("âš ï¸� Firestore credentials not found - using in-memory storage")

print("\nğŸ“‹ Configuration complete!")



# --- Data Models ---
# We use Pydantic/Dataclasses to ensure structured communication between agents.
# This prevents 'hallucinations' in data format and ensures type safety.

@dataclass
class UserProfile:
    """User profile data model storing preferences and state."""
    user_id: str
    name: Optional[str] = None
    dietary_preferences: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    budget_weekly: float = 0.0
    household_size: int = 1
    onboarding_completed: bool = False
    current_step: str = "start"  # Tracks conversation state
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class PantryItem:
    """Represents a physical item in the user's pantry."""
    item_name: str
    quantity: str
    category: str
    expiry_date: Optional[str] = None
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Meal:
    """A single meal definition."""
    name: str
    description: str
    ingredients: List[str]
    calories: Optional[int] = None
    prep_time: Optional[int] = None

@dataclass
class DayMealPlan:
    """Daily meal plan container."""
    day: str
    breakfast: Meal
    lunch: Meal
    dinner: Meal
    snacks: List[Meal] = field(default_factory=list)

@dataclass
class GroceryItem:
    """Item to be purchased."""
    item_name: str
    quantity: str
    category: str
    estimated_price: Optional[float] = None

# In-memory storage (fallback if Firestore not available)
memory_storage = {
    'users': {},
    'pantry': {},
    'meal_plans': {},
    'grocery_lists': {}
}

print("âœ… Data models defined")


class LLMHelper:
    """Helper for Gemini LLM interactions"""
    
    @staticmethod
    def extract_json(text: str) -> Any:
        """Extract JSON from text using regex."""
        try:
            # Find JSON block
            match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', text)
            if match:
                return json.loads(match.group(0))
            # Fallback to cleanup
            cleaned = text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {text}")
            return {}

    @staticmethod
    async def generate_text(prompt: str) -> str:
        """Generate text using Gemini"""
        if not gemini_model:
            return "LLM not configured"
        try:
            response = await gemini_model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error: {e}"
    
    @staticmethod
    def generate_text_sync(prompt: str) -> str:
        """Synchronous text generation"""
        if not gemini_model:
            return "LLM not configured"
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error: {e}"
    
    @staticmethod
    def analyze_image(image_data: str, prompt: str = "Identify food items in this image") -> str:
        """Analyze image using Gemini Vision"""
        if not gemini_model:
            return "Vision model not configured"
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Prepare prompt for structured JSON output
            full_prompt = """
            Analyze this image and identify pantry items.
            Return ONLY a JSON array where each object has:
            - "item_name": Name of the item
            - "quantity": Estimated quantity (e.g. "1L", "2 count")
            - "category": Category (e.g. "Dairy", "Vegetable")
            - "expiry_date": Estimated YYYY-MM-DD based on typical shelf life from today
            
            Do not include markdown formatting or backticks. Just the raw JSON string.
            """
            
            # Generate content with image
            response = gemini_model.generate_content([
                {"mime_type": "image/jpeg", "data": image_bytes},
                full_prompt
            ])
            return response.text
        except Exception as e:
            return f"Vision analysis error: {e}"

# Database helpers
class DatabaseHelper:
    """Helper for database operations"""
    
    @staticmethod
    def save_user_profile(profile: UserProfile):
        """Save user profile"""
        if db:
            try:
                db.collection('users').document(profile.user_id).set(asdict(profile))
                print(f"âœ… Saved profile for {profile.user_id} to Firestore")
            except Exception as e:
                print(f"âš ï¸� Firestore save failed: {e}")
                memory_storage['users'][profile.user_id] = asdict(profile)
        else:
            memory_storage['users'][profile.user_id] = asdict(profile)
            print(f"â„¹ï¸� Saved profile for {profile.user_id} to memory (Firestore unavailable)")
    
    @staticmethod
    def get_user_profile(user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        if db:
            try:
                doc = db.collection('users').document(user_id).get()
                if doc.exists:
                    return UserProfile(**doc.to_dict())
            except Exception as e:
                print(f"âš ï¸� Firestore get failed: {e}")
        
        # Fallback to memory
        if user_id in memory_storage['users']:
            return UserProfile(**memory_storage['users'][user_id])
        return None
    
    @staticmethod
    def save_pantry_items(user_id: str, items: List[PantryItem]):
        """Save pantry items"""
        if db:
            try:
                batch = db.batch()
                pantry_ref = db.collection('users').document(user_id).collection('pantry')
                for item in items:
                    # Use item name as ID, sanitize it
                    doc_id = item.item_name.replace("/", "-")
                    batch.set(pantry_ref.document(doc_id), asdict(item))
                batch.commit()
                print(f"âœ… Saved {len(items)} items to Firestore")
            except Exception as e:
                print(f"âš ï¸� Firestore save failed: {e}")
                memory_storage['pantry'][user_id] = [asdict(i) for i in items]
        else:
            memory_storage['pantry'][user_id] = [asdict(i) for i in items]
            print(f"â„¹ï¸� Saved {len(items)} items to memory (Firestore unavailable)")
    
    @staticmethod
    def get_pantry_items(user_id: str) -> List[PantryItem]:
        """Get pantry items"""
        items = []
        if db:
            try:
                docs = db.collection('users').document(user_id).collection('pantry').stream()
                items = [PantryItem(**doc.to_dict()) for doc in docs]
            except Exception as e:
                print(f"âš ï¸� Firestore get pantry failed: {e}")
        
        if not items and user_id in memory_storage['pantry']:
            items = [PantryItem(**i) for i in memory_storage['pantry'][user_id]]
        
        return items

print("âœ… Helper functions defined")



from typing import List, Dict, Optional, Tuple, Any
import logging
import re
import json
from datetime import datetime, timedelta

class OnboardingAgent:
    """
    Handles user onboarding and profile creation.
    """
    @staticmethod
    def process_message(user_id: str, message: str) -> str:
        profile = DatabaseHelper.get_user_profile(user_id)
        
        if not profile:
            profile = UserProfile(user_id=user_id)
            DatabaseHelper.save_user_profile(profile)
        
        if profile.onboarding_completed:
            # Check for cooking intent
            lower_msg = message.lower()
            if any(k in lower_msg for k in ["cook", "recipe", "make", "how to", "prepare"]):
                return CookingAssistantAgent.answer_query(message)
            return "You're already onboarded! Try 'scan pantry', 'plan meals', or ask me how to cook something."
        
        step = profile.current_step
        
        if step == "start":
            profile.current_step = "name"
            DatabaseHelper.save_user_profile(profile)
            return "Hi! I'm your Smart Nutrition Planner. Let's get you set up. What should I call you?"
        
        elif step == "name":
            profile.name = message
            profile.current_step = "diet"
            DatabaseHelper.save_user_profile(profile)
            return f"Nice to meet you, {profile.name}! Do you have any dietary preferences? (e.g., Vegetarian, Vegan, Keto, None)"
        
        elif step == "diet":
            profile.dietary_preferences = [message]
            profile.current_step = "allergies"
            DatabaseHelper.save_user_profile(profile)
            return "Got it. Do you have any food allergies I should know about?"
        
        elif step == "allergies":
            profile.allergies = [message] if message.lower() != "no" else []
            profile.current_step = "budget"
            DatabaseHelper.save_user_profile(profile)
            return "Noted. What is your weekly grocery budget (in INR)?"
        
        elif step == "budget":
            nums = re.findall(r'\d+', message)
            if nums:
                profile.budget_weekly = float(nums[0])
                profile.current_step = "household"
                DatabaseHelper.save_user_profile(profile)
                return "Okay. Finally, how many people are in your household?"
            else:
                return "I didn't catch a number. What's your weekly budget?"
        
        elif step == "household":
            nums = re.findall(r'\d+', message)
            if nums:
                profile.household_size = int(nums[0])
                profile.onboarding_completed = True
                profile.current_step = "complete"
                DatabaseHelper.save_user_profile(profile)
                return "All set! Your profile is ready. You can now start planning meals or scanning your pantry."
            else:
                return "How many people? (Just a number please)"
        
        return "Onboarding in progress..."

class PantryScannerAgent:
    """
    Scans images to identify pantry items using Gemini Vision.
    """
    @staticmethod
    def scan_image(user_id: str, image_data: str) -> str:
        result_json = LLMHelper.analyze_image(image_data)
        
        try:
            items_data = LLMHelper.extract_json(result_json)
            
            real_items = []
            if isinstance(items_data, list):
                for item in items_data:
                    real_items.append(PantryItem(
                        item_name=item.get("item_name", "Unknown"),
                        quantity=item.get("quantity", "1"),
                        category=item.get("category", "General"),
                        expiry_date=item.get("expiry_date", (datetime.now() + timedelta(days=7)).date().isoformat())
                    ))
            
            DatabaseHelper.save_pantry_items(user_id, real_items)
            
            return f"âœ… Scanned and added {len(real_items)} items to your pantry!\n" + ", ".join([i.item_name for i in real_items])
        except Exception as e:
            return f"â�Œ Failed to parse vision response: {e}. Raw response: {result_json}"

class MealPlannerAgent:
    """
    Generates personalized meal plans.
    """
    @staticmethod
    def generate_meal_plan(user_id: str) -> Dict:
        profile = DatabaseHelper.get_user_profile(user_id)
        pantry = DatabaseHelper.get_pantry_items(user_id)
        
        if not profile or not profile.onboarding_completed:
            return {"error": "Please complete onboarding first"}
        
        prompt = f"""
        Create a 7-day meal plan for:
        - Dietary preferences: {', '.join(profile.dietary_preferences)}
        - Allergies: {', '.join(profile.allergies) or 'None'}
        - Household size: {profile.household_size}
        - Budget: â‚¹{profile.budget_weekly}/week
        
        Available pantry items: {', '.join([f"{i.item_name} ({i.quantity})" for i in pantry])}
        
        Return a JSON object with this EXACT structure:
        {{
            "days": [
                {{
                    "day": "Monday",
                    "breakfast": {{ "name": "Meal Name", "description": "Short description", "ingredients": ["ing1", "ing2"] }},
                    "lunch": {{ "name": "Meal Name", "description": "Short description", "ingredients": ["ing1", "ing2"] }},
                    "dinner": {{ "name": "Meal Name", "description": "Short description", "ingredients": ["ing1", "ing2"] }},
                    "snacks": [ {{ "name": "Snack Name", "description": "Short description", "ingredients": ["ing1"] }} ]
                }}
            ]
        }}
        IMPORTANT: Return ONLY valid JSON. No markdown formatting.
        """
        
        try:
            response_text = LLMHelper.generate_text_sync(prompt)
            meal_plan_data = LLMHelper.extract_json(response_text)
            
            meal_plan_data["user_id"] = user_id
            meal_plan_data["week_start_date"] = datetime.now().date().isoformat()
            
            memory_storage['meal_plans'][user_id] = meal_plan_data
            return meal_plan_data
            
        except Exception as e:
            logger.error(f"Meal plan generation failed: {e}")
            return {"error": f"Failed to generate plan: {e}"}

class GroceryOptimizerAgent:
    """
    Creates optimized grocery lists.
    """
    @staticmethod
    def generate_grocery_list(user_id: str) -> Dict:
        meal_plan = memory_storage['meal_plans'].get(user_id)
        pantry = DatabaseHelper.get_pantry_items(user_id)
        
        if not meal_plan:
            return {"error": "No meal plan found. Generate one first."}
        
        all_ingredients = []
        if 'days' in meal_plan:
            for day in meal_plan['days']:
                for meal_type in ['breakfast', 'lunch', 'dinner']:
                    if meal_type in day:
                        all_ingredients.extend(day[meal_type].get('ingredients', []))
        
        pantry_items = [item.item_name for item in pantry]
        
        prompt = f"""
        Based on the following meal plan ingredients and current pantry items, create an optimized grocery list.
        
        Meal Plan Ingredients: {', '.join(all_ingredients)}
        Current Pantry: {', '.join(pantry_items) if pantry_items else 'Empty'}
        
        Generate a shopping list for items NOT in the pantry. Consolidate similar items, estimate quantities.
        Return ONLY a JSON array where each object has:
        - "item_name": Name of the grocery item
        - "quantity": Estimated quantity (e.g. "500g", "1kg", "2L")
        - "category": Category (e.g. "Meat", "Vegetable", "Grains")
        
        Do not include markdown formatting or backticks. Just the raw JSON array.
        """
        
        try:
            response_text = LLMHelper.generate_text_sync(prompt)
            items_data = LLMHelper.extract_json(response_text)
            
            grocery_list = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "items": items_data
            }
            
            memory_storage['grocery_lists'][user_id] = grocery_list
            return grocery_list
        except Exception as e:
            logger.error(f"Grocery list generation failed: {e}")
            return {"error": f"Failed to generate grocery list: {e}"}

class PriceComparatorAgent:
    """
    Compares prices across providers using a 3-tier fallback strategy.
    
    Role: Savvy Shopper.
    Strategy:
    1. Google Search (Real-time)
    2. DuckDuckGo Search (Fallback)
    3. LLM Estimation (Last Resort)
    """
    _price_cache = {} # Cache to avoid redundant API/LLM calls

    @staticmethod
    def extract_price_from_text(text: str) -> Optional[float]:
        """
        Regex to find prices in Indian Rupee formats (â‚¹100, Rs. 100, INR 100, 100/-).
        Returns the first valid float found, or None.
        """
        # Patterns: â‚¹ 100, Rs. 100, INR 100, 100 rs
        pattern = r'(?:â‚¹|Rs\.?|INR)\s*(\d+(?:,\d+)*(?:\.\d{2})?)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            try:
                price_str = match.group(1).replace(',', '')
                return float(price_str)
            except ValueError:
                return None
        return None

    @staticmethod
    def search_price(item_name: str, provider: str) -> Dict:
        """Search for item price with fallbacks."""
        cache_key = f"{item_name}_{provider}"
        if cache_key in PriceComparatorAgent._price_cache:
            return PriceComparatorAgent._price_cache[cache_key]

        query = f"{item_name} price {provider} india"
        
        # 1. Try Google
        price, source = PriceComparatorAgent._search_google(query)
        if price: 
            result = {"price": price, "status": "available", "source": source}
            PriceComparatorAgent._price_cache[cache_key] = result
            return result
        
        # 2. Try DuckDuckGo
        price, source = PriceComparatorAgent._search_ddg(query)
        if price: 
            result = {"price": price, "status": "available", "source": source}
            PriceComparatorAgent._price_cache[cache_key] = result
            return result
        
        # 3. Fallback to LLM Estimate
        try:
            result = PriceComparatorAgent._estimate_price_llm(item_name, provider)
            PriceComparatorAgent._price_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"LLM fallback failed for {item_name}: {e}")
            return {"price": 0, "status": "not_found", "error": str(e)}

    @staticmethod
    def _search_google(query: str) -> Tuple[Optional[float], Optional[str]]:
        try:
            from googlesearch import search
            # logger.info(f"ğŸ”� Searching (Google): {query}")
            
            results = list(search(query, num_results=2, advanced=True))
            
            for result in results:
                content = f"{result.title} {result.description}"
                price = PriceComparatorAgent.extract_price_from_text(content)
                if price:
                    # logger.info(f"âœ… Google found price: â‚¹{price}")
                    return price, "Google"
            
            # logger.warning(f"â�Œ Google found no price for: {query}")
        except Exception as e:
            logger.warning(f"âš ï¸� Google search failed: {e}")
        return None, None

    @staticmethod
    def _search_ddg(query: str) -> Tuple[Optional[float], Optional[str]]:
        try:
            # Handle package renaming/versions - prioritize ddgs
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS
                except ImportError:
                    logger.warning("duckduckgo_search/ddgs not found, skipping DDG search")
                    return None, None
            
            # logger.info(f"ğŸ¦† Searching (DDG): {query}")
            
            with DDGS() as ddgs:
                # Use backend='html' to avoid unwanted API calls (like Yandex/Mojeek) and improve stability
                results = list(ddgs.text(query, max_results=2, backend='html'))
            
            for res in results:
                content = f"{res.get('title', '')} {res.get('body', '')}"
                price = PriceComparatorAgent.extract_price_from_text(content)
                if price:
                    # logger.info(f"âœ… DDG found price: â‚¹{price}")
                    return price, "DuckDuckGo"
            
            # logger.warning(f"â�Œ DDG found no price for: {query}")
        except Exception as e:
            logger.warning(f"âš ï¸� DDG search failed: {e}")
        return None, None

    @staticmethod
    def _estimate_price_llm(item_name: str, provider: str) -> Dict:
        logger.warning(f"ğŸ¤– Using LLM estimation for {item_name} (Quota risk)")
        prompt = f"""
        Estimate a realistic price in INR for "{item_name}" from {provider} in India.
        Consider typical grocery prices.
        
        Return ONLY a JSON object with:
        - "price": numeric value in INR
        - "status": "estimated"
        """
        time.sleep(1) # Throttle to avoid 429
        response = LLMHelper.generate_text_sync(prompt)
        return LLMHelper.extract_json(response)
    
    @staticmethod
    def compare_prices(grocery_list: Dict) -> Dict:
        """Compare prices across stores using real search"""
        items = grocery_list.get('items', [])
        providers = ["BigBasket", "Blinkit", "Zepto", "Amazon Fresh"]
        comparison = {}
        
        print("ğŸ”� Searching for prices across providers (via Google/DuckDuckGo/LLM Estimate)...")
        
        for provider in providers:
            provider_total = 0
            provider_items = []
            
            for item in items:
                item_name = item.get('item_name', item) if isinstance(item, dict) else item
                
                # Search for price
                price_data = PriceComparatorAgent.search_price(item_name, provider)
                
                if price_data.get('status') in ['available', 'estimated'] and price_data.get('price'):
                    price = price_data['price']
                    provider_total += price
                    provider_items.append({"item": item_name, "price": price, "status": price_data.get('status')})
                else:
                    provider_items.append({"item": item_name, "price": 0, "status": "not_found"})
                
                time.sleep(0.5) # Throttle
            
            comparison[provider] = {
                "total_cost": round(provider_total, 2),
                "items": provider_items
            }
        
        # Find best provider
        valid_providers = {k: v for k, v in comparison.items() if all(i["status"] in ["available", "estimated"] for i in v["items"])}
        
        if valid_providers:
            min_provider = min(valid_providers.items(), key=lambda x: x[1]['total_cost'])
            recommendation = f"Best option: Buy from {min_provider[0]} for â‚¹{min_provider[1]['total_cost']}"
        else:
            recommendation = "Could not find all items on a single provider. Showing best estimates."
        
        return {
            "price_comparison": comparison,
            "recommendation": recommendation
        }

class NotifierAgent:
    """Sends notifications to users"""
    
    @staticmethod
    def send_notification(user_id: str, message: str, channel: str = "whatsapp"):
        timestamp = datetime.now().strftime("%H:%M")
        print(f"ğŸ“± [{timestamp}] Notification to {user_id} via {channel}: {message}")
        return True

class CookingAssistantAgent:
    """Guides users through cooking recipes"""
    
    @staticmethod
    def answer_query(query: str) -> str:
        """Responds to natural language cooking queries"""
        prompt = f"""
        You are a helpful cooking assistant. The user asks: "{query}"
        Provide a friendly, step-by-step guide for the requested dish.
        Keep the instructions clear and concise.
        """
        return LLMHelper.generate_text_sync(prompt)
    
    
    @staticmethod
    def get_recipe_steps(meal_name: str) -> List[str]:
        """Get cooking steps for a meal using LLM"""
        prompt = f"""
        Provide simple, step-by-step cooking instructions for: {meal_name}
        Return ONLY a JSON array of strings, where each string is one step.
        Example: ["Step 1: Chop vegetables", "Step 2: Heat oil"]
        """
        
        try:
            response_text = LLMHelper.generate_text_sync(prompt)
            steps = LLMHelper.extract_json(response_text)
            if isinstance(steps, list):
                return steps
            return ["Could not parse recipe steps."]
        except Exception as e:
            return [f"Error generating recipe: {e}"]

print("âœ… All agents implemented (Updated with Real LLM & Dynamic Grocery/Price)")



# Comprehensive Dashboard with Tabs
import ipywidgets as widgets
from IPython.display import clear_output, display
import asyncio

class Dashboard:
    def __init__(self):
        self.user_id = config.USER_ID
        self.out = widgets.Output()
        self.setup_ui()
        
    def setup_ui(self):
        # --- Tab 1: Dashboard (Onboarding) ---
        self.chat_history = widgets.Output(layout={'border': '1px solid #ccc', 'height': '300px', 'overflow_y': 'scroll'})
        self.chat_input = widgets.Text(placeholder="Type a message and press Enter...", layout={'width': '80%'})
        self.send_btn = widgets.Button(description="Send", button_style='primary', layout={'width': 'auto'})
        self.send_btn.on_click(self.on_chat_send)
        self.chat_input.continuous_update = False
        self.chat_input.observe(self.on_chat_send, names='value')
        
        tab1 = widgets.VBox([
            widgets.HTML("<h3>ğŸ‘‹ Welcome to Smart Nutrition Planner</h3>"),
            self.chat_history,
            widgets.HBox([self.chat_input, self.send_btn])
        ])
        
        # --- Tab 2: Pantry ---
        self.upload_btn = widgets.FileUpload(accept='image/*', multiple=False, description='ğŸ“¸ Upload Pantry Image', button_style='primary', layout={'width': 'max-content'})
        self.upload_btn.observe(self.on_upload_change, names='value')
        self.pantry_out = widgets.Output()
        
        tab2 = widgets.VBox([
            widgets.HTML("<h3>ğŸ“¦ Pantry Management</h3>"),
            self.upload_btn,
            self.pantry_out
        ])
        
        # --- Tab 3: Meal Plan ---
        self.plan_btn = widgets.Button(description="ğŸ�½ï¸� Generate Meal Plan", button_style='primary', layout={'width': 'max-content'})
        self.plan_btn.on_click(self.on_plan)
        self.plan_out = widgets.Output()
        
        tab3 = widgets.VBox([
            widgets.HTML("<h3>ğŸ“… Weekly Meal Plan</h3>"),
            self.plan_btn,
            self.plan_out
        ])
        
        # --- Tab 4: Grocery List ---
        self.grocery_btn = widgets.Button(description="ğŸ›’ Generate Grocery List", button_style='primary', layout={'width': 'max-content'})
        self.grocery_btn.on_click(self.on_grocery)
        self.grocery_out = widgets.Output()
        
        tab4 = widgets.VBox([
            widgets.HTML("<h3>ğŸ“� Grocery List & Price Comparison</h3>"),
            self.grocery_btn,
            self.grocery_out
        ])
        
        # --- Tab 5: Cooking ---
        self.cook_meal_select = widgets.Dropdown(options=['Breakfast', 'Lunch', 'Dinner'], description='Meal:')
        self.cook_btn = widgets.Button(description="ğŸ‘¨â€�ğŸ�³ Start Cooking", button_style='primary', layout={'width': 'max-content'})
        self.cook_btn.on_click(self.on_cook)
        self.cook_out = widgets.Output()
        
        tab5 = widgets.VBox([
            widgets.HTML("<h3>ğŸ�³ Cooking Assistant</h3>"),
            widgets.HBox([self.cook_meal_select, self.cook_btn]),
            self.cook_out
        ])
        
        # --- Tab 6: Settings ---
        self.notify_toggle = widgets.Checkbox(value=True, description='Enable WhatsApp Notifications')
        self.save_settings_btn = widgets.Button(description="Save Settings", button_style='primary', layout={'width': 'max-content'})
        self.save_settings_btn.on_click(lambda b: self.log("Settings saved!"))
        
        tab6 = widgets.VBox([
            widgets.HTML("<h3>âš™ï¸� Settings</h3>"),
            self.notify_toggle,
            self.save_settings_btn
        ])
        
        # --- Main Tabs ---
        self.tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4, tab5, tab6])
        self.tabs.set_title(0, 'Dashboard')
        self.tabs.set_title(1, 'Pantry')
        self.tabs.set_title(2, 'Meal Plan')
        self.tabs.set_title(3, 'Grocery')
        self.tabs.set_title(4, 'Cooking')
        self.tabs.set_title(5, 'Settings')
        
        display(self.tabs)
        self.log("System ready! Start by saying 'Hi' in the Dashboard tab.")

    def log(self, message, output_widget=None):
        target = output_widget if output_widget else self.chat_history
        with target:
            print(message)

    def on_chat_send(self, b):
        if isinstance(b, dict) and b.get('name') == 'value': pass
        msg = self.chat_input.value
        if not msg: return
        self.chat_input.value = ""
        self.log(f"ğŸ‘¤ You: {msg}")
        response = OnboardingAgent.process_message(self.user_id, msg)
        self.log(f"ğŸ¤– Agent: {response}")
        if "All set" in response:
            NotifierAgent.send_notification(self.user_id, "Welcome to Smart Nutrition Planner! Profile created.")

    def on_upload_change(self, change):
        if not change['new']: return
        uploaded_file = change['new'][0] if isinstance(change['new'], (list, tuple)) else change['new']
        content = uploaded_file.get('content') or uploaded_file.get('data')
        if not content: return
        image_data = base64.b64encode(content).decode('utf-8')
        with self.pantry_out:
            clear_output()
            print("ğŸ“¸ Scanning uploaded image...")
            result = PantryScannerAgent.scan_image(self.user_id, image_data)
            print(result)
            items = DatabaseHelper.get_pantry_items(self.user_id)
            if items:
                df = pd.DataFrame([asdict(i) for i in items])
                display(df[['item_name', 'quantity', 'expiry_date']])
        self.upload_btn.value = ()

    def on_plan(self, b):
        with self.plan_out:
            clear_output()
            print("ğŸ�½ï¸� Generating personalized meal plan... (this may take a moment)")
            plan = MealPlannerAgent.generate_meal_plan(self.user_id)
            if "error" in plan:
                print(f"â�Œ {plan['error']}")
                return
            
            clear_output()
            print(f"âœ¨ Weekly Meal Plan for {self.user_id}")
            for day in plan.get('days', []):
                print(f"\nğŸ“… {day['day']}")
                for meal_type in ['breakfast', 'lunch', 'dinner']:
                    if meal_type in day:
                        meal = day[meal_type]
                        print(f"  {meal_type.capitalize()}: {meal.get('name', 'Unknown')}")
                        print(f"    - {meal.get('description', '')}")
                        print(f"    - Ingredients: {', '.join(meal.get('ingredients', []))}")

    def on_grocery(self, b):
        with self.grocery_out:
            clear_output()
            print("ğŸ›’ Generating grocery list...")
            grocery = GroceryOptimizerAgent.generate_grocery_list(self.user_id)
            if "error" in grocery:
                print(f"â�Œ {grocery['error']}")
                return
            
            print("ğŸ›’ Optimizing list...")
            df = pd.DataFrame(grocery['items'])
            df.index = df.index + 1
            display(df)
            print("\nğŸ’° Price Comparison:")
            comp = PriceComparatorAgent.compare_prices(grocery)
            for prov, data in comp['price_comparison'].items():
                print(f"   {prov}: â‚¹{data['total_cost']}")
            print(f"\nâœ¨ {comp['recommendation']}")

    def on_cook(self, b):
        meal_type = self.cook_meal_select.value
        with self.cook_out:
            clear_output()
            print(f"ğŸ‘¨â€�ğŸ�³ Finding recipe for {meal_type}...")
            
            # Try to find specific meal from plan
            meal_name = f"{meal_type}" # Default
            try:
                plan = memory_storage['meal_plans'].get(self.user_id)
                if plan:
                    today = datetime.now().strftime("%A")
                    # Find today's plan
                    day_plan = next((d for d in plan.get('days', []) if d['day'] == today), None)
                    if day_plan and meal_type.lower() in day_plan:
                        meal_name = day_plan[meal_type.lower()].get('name', meal_type)
            except Exception as e:
                print(f"(Using generic recipe as plan lookup failed: {e})")
            
            print(f"ğŸ�³ Cooking: {meal_name}")
            steps = CookingAssistantAgent.get_recipe_steps(meal_name)
            for i, step in enumerate(steps, 1):
                print(f"{i}. {step}")

dashboard = Dashboard()

