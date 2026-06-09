from google.genai import types

from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor
from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import logging

import os

import warnings
logging.getLogger('google_genai').setLevel(logging.ERROR)



from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Setup and authentication complete")
except Exception as e:
    print(
        f" Authentication error"
    )


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


from IPython.display import IFrame, display
import re

async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
    user_id: str = "user",
):
    print(f"\n ### Session: {session_name}")

    app_name = runner_instance.app_name

    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_name
        )

    if user_queries:
        if isinstance(user_queries, str):
            user_queries = [user_queries]

        for query in user_queries:
            print(f"\n{user_id} > {query}")

            query = types.Content(role="user", parts=[types.Part(text=query)])

            async for event in runner_instance.run_async(
                user_id=user_id, session_id=session.id, new_message=query
            ):

                if event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text and text != "None":
                        print(f"{MODEL_NAME} > ", text)

                
    else:
        print("No queries!")



import pandas as pd

CAKE_DATA = [
    {
        "name": "Chocolate Lava Cake",
        "product_id": "CLC001",
        "price": 10.00,
        "base_discount": 0.05,
        "discount_threshold": 3,
        "quantity": 120,
        "tags": ["chocolate", "rich", "high_sugar"],
        "health_level": "normal",
        "recommended_for": []
    },
    {
        "name": "Vanilla Bean Cake",
        "product_id": "VBC002",
        "price": 5.00,
        "base_discount": 0.00,
        "discount_threshold": 0,
        "quantity": 150,
        "tags": ["vanilla", "light", "moderate_sugar"],
        "health_level": "normal",
        "recommended_for": []
    },
    {
        "name": "Red Velvet Cake",
        "product_id": "RVC003",
        "price": 15.00,
        "base_discount": 0.10,
        "discount_threshold": 2,
        "quantity": 80,
        "tags": ["cream_cheese", "sweet", "high_sugar"],
        "health_level": "normal",
        "recommended_for": []
    },
    {
        "name": "Cheesecake Delight Cake",
        "product_id": "CCD004",
        "price": 20.00,
        "base_discount": 0.15,
        "discount_threshold": 1,
        "quantity": 75,
        "tags": ["creamy", "high_fat", "high_protein"],
        "health_level": "high_protein",
        "recommended_for": ["muscle_gain", "high_protein_diet"]
    },
    {
        "name": "Lemon Zest Cake",
        "product_id": "LZC005",
        "price": 12.50,
        "base_discount": 0.05,
        "discount_threshold": 5,
        "quantity": 90,
        "tags": ["lemon", "fresh", "moderate_sugar"],
        "health_level": "normal",
        "recommended_for": []
    },

    # NEW HEALTHY OPTIONS BELOW 
    {
        "name": "Sugar-Free Almond Cake",
        "product_id": "SFA006",
        "price": 18.00,
        "base_discount": 0.07,
        "discount_threshold": 2,
        "quantity": 100,
        "tags": ["sugar_free", "almond", "low_carb"],
        "health_level": "low_sugar",
        "recommended_for": ["diabetes", "weight_loss", "low_sugar_diet"]
    },
    {
        "name": "Oats & Honey Healthy Cake",
        "product_id": "OHC007",
        "price": 16.00,
        "base_discount": 0.05,
        "discount_threshold": 3,
        "quantity": 110,
        "tags": ["oat", "fiber_rich", "low_fat"],
        "health_level": "high_fiber",
        "recommended_for": ["heart_health", "cholesterol_control", "weight_loss"]
    },
    {
        "name": "Protein Power Cake",
        "product_id": "PPC008",
        "price": 22.00,
        "base_discount": 0.10,
        "discount_threshold": 1,
        "quantity": 60,
        "tags": ["protein", "low_sugar"],
        "health_level": "high_protein",
        "recommended_for": ["gym_diet", "muscle_gain", "low_sugar_diet"]
    },
    {
        "name": "Gluten-Free Berry Cake",
        "product_id": "GFBC009",
        "price": 19.00,
        "base_discount": 0.05,
        "discount_threshold": 2,
        "quantity": 70,
        "tags": ["gluten_free", "berries", "light_sweet"],
        "health_level": "gluten_free",
        "recommended_for": ["gluten_intolerance", "celiac", "light_diet"]
    }
]

# Convert to DataFrame
product_df = pd.DataFrame(CAKE_DATA)
print(product_df)



def get_all_cakes() -> str:
    """
    Retrieves the name, price per piece, and discount rule for all cakes available.
    
    Returns:
        A formatted string listing all cakes and their details.
    """
    if not CAKE_DATA:
        return "There are currently no cakes available in the catalog."

    cake_details = []
    
    for cake in CAKE_DATA:
        price = f"${cake['price']:.2f} per piece"
        
        # Determine the discount rule description
        if cake['discount_threshold'] > 0:
            discount_rule = f"{int(cake['base_discount'] * 100)}% off for {cake['discount_threshold']} pieces or more"
        else:
            discount_rule = "No volume discount available"
            
        cake_details.append(f"- **{cake['name']}**: {price} | Discount: {discount_rule}")
        
    # Join all the formatted strings into a single, comprehensive response
    return "Our current cake selection includes:\n" + "\n".join(cake_details)


def get_cake_details(cake_name: str) -> str:
    """
    Looks up the base price and discount rule for a specific cake by its name.
    
    Args:
        cake_name: The name of the cake (e.g., 'Chocolate Lava Cake').
        
    Returns:
        A JSON string with the cake's price and discount rules, or an error message.
    """
    try:
        # Simple case-insensitive search
        result = product_df[product_df['name'].str.contains(cake_name, case=False, na=False)]
        
        if result.empty:
            return f'{{"error": "Cake named {cake_name} not found."}}'
            
        # Get the first match
        cake = result.iloc[0]
        
        details = {
            "name": cake['name'],
            "price": f"${cake['price']:.2f} per piece",
            "discount_rule": f"{int(cake['base_discount'] * 100)}% discount for orders of {cake['discount_threshold']} pieces or more."
        }
        return str(details)
        
    except Exception as e:
        return f'{{"error": "An internal error occurred: {str(e)}"}}'



!pip install reportlab



import os
import random

def get_cartoon_image(source_folder="/kaggle/input/cartoon-cake-images/cartoon_cake_images"):

    # List all image files in source folder
    images = [f for f in os.listdir(source_folder) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    if not images:
        raise FileNotFoundError(f"No images found in folder: {source_folder}")

    # Pick a random image
    chosen_image = random.choice(images)
    return os.path.join(source_folder, chosen_image)





import json
import uuid
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
# Directory to store order JSONs
ORDER_DIRECTORY = "orders"
os.makedirs(ORDER_DIRECTORY, exist_ok=True)

def open_pdf(filename):
    print(f"\n Open PDF file: {filename}")
    # Display PDF inline
    try:
        display(IFrame(filename, width=600, height=400))
    except:
        print("Unable to open PDF automatically.")


def generate_pdf_receipt(cake_name: str, pieces: float, price: float, discounted_price: float, joke: str):
    filename = f"receipt_{uuid.uuid4().hex}.pdf"

    # Generate unique order ID
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

    # --- Save order as JSON ---
    order_data = {
        "order_id": order_id,
        "cake_name": cake_name,
        "pieces": pieces,
        "price_per_piece": price,
        "discounted_price": discounted_price
    }
    
    json_filename = os.path.join(ORDER_DIRECTORY, f"{order_id}.json")
    with open(json_filename, "w") as f:
        json.dump(order_data, f, indent=4)
    
    # --- Generate PDF ---
    width = 320
    height = 520
    c = canvas.Canvas(filename, pagesize=(width, height))
    c.setFont("Helvetica", 12)

    c.setFillColor(colors.black)
    center_x = width / 2

    # Get cartoon cake image
    cartoon_path = get_cartoon_image()
    try:
        img_w, img_h = 150, 90
        img_x = (width - img_w) / 2
        c.drawImage(ImageReader(cartoon_path), img_x, height - 150, width=img_w, height=img_h)
    except:
        print("Failed to insert cartoon image.")

    # Header
    c.drawCentredString(center_x, height - 170, "=============================")
    c.drawCentredString(center_x, height - 185, "Sweet Delights - Cake Receipt")
    c.drawCentredString(center_x, height - 200, "=============================")

    # Add Order ID
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, height - 220, f"Order ID: {order_id}")
    c.setFont("Helvetica", 12)

    # Receipt details
    c.drawString(60, height - 240, f"Cake Name: {cake_name}")
    c.drawString(60, height - 260, f"Pieces: {pieces}")
    c.drawString(60, height - 280, f"Original Price: ${price:.2f}")
    c.drawString(60, height - 300, f"Discounted Price: ${discounted_price:.2f}")

    # Footer text
    c.drawCentredString(center_x, height - 330, "Have a delicious day!")
    c.drawCentredString(center_x, height - 350, "=============================")

    # Joke section
    c.drawCentredString(center_x, height - 370, "Joke:")

    words = joke.split()
    line_length = 7
    lines = [" ".join(words[i:i+line_length]) for i in range(0, len(words), line_length)]

    y = height - 390
    for line in lines:
        c.drawCentredString(center_x, y, line)
        y -= 15

    c.save()
    open_pdf(filename)
    return {"receipt_file": filename}



MODEL_NAME = "gemini-2.5-flash-lite"


discount_calculation_agent = LlmAgent(
    name="discount_calculation_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""You are a specialized calculator that ONLY responds with Python code. You are forbidden from providing any text, explanations, or conversational responses.
 
     Your task is to take a request for a calculation and translate it into a single block of Python code that calculates the answer.
     
     **RULES:**
    1.  Your output MUST be ONLY a Python code block.
    2.  Do NOT write any text before or after the code block.
    3.  The Python code MUST calculate the result.
    4.  The Python code MUST print the final result to stdout.
    5.  You are PROHIBITED from performing the calculation yourself. Your only job is to generate the code that will perform the calculation.
   
    Failure to follow these rules will result in an error.
       """,
    code_executor=BuiltInCodeExecutor(),
)


joke_agent = LlmAgent(
    name="joke_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
        You are a comedian agent. 
        Your ONLY job is to generate a short, funny cake-related joke.
        Always reply with just the joke text and nothing else.
    """,
)



health_agent = LlmAgent(
    name="health_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
        You are the Health & Nutrition Advisor for Sweet Delights Cake Shop.
        
        Your responsibilities:
        
        1. **Explain Cake Nutrition**
           - Describe protein, sugar, calories, fat, fiber, and ingredients.
           - Keep explanations simple and friendly for customers.
        
        2. **Handle Customer Health Conditions Safely**
           - If a customer mentions diabetes, high blood pressure, heart disease, kidney disease,
             severe allergies, obesity, or any serious medical condition:
               • DO NOT recommend any cake immediately.
               • First politely warn them that cakes may not be suitable for serious conditions.
               • If the customer still insists, then provide the safest possible cake option.
           - For mild conditions (dieting, weight loss, low sugar preference, lactose intolerance, etc.):
               • Provide general-friendly suggestions only.
           - NEVER give medical advice.
           - ALWAYS remind them to consult a doctor for strict dietary needs.
        
        3. **Recommend Healthier Cake Options**
           - When it is safe to recommend:
               • Suggest sugar-free, low-calorie, fruit-based, oat-based, whole-wheat,
                 keto-friendly, or high-protein cakes depending on the condition.
           - Match recommendations to the customer’s needs.
             Examples:
               • Diabetes → sugar-free cakes  
               • High BP → low-sodium fruit cakes  
               • Weight loss → low-calorie yogurt cakes  
               • High protein → protein-based cakes  
        
        4. **Polite & Clear Communication**
           - Maintain a friendly tone.
           - Keep responses short, helpful, and easy to understand.
        
        **Safety Rule (Very Important):**
           - If a customer’s health condition is SERIOUS:
               → Do NOT recommend a cake first.  
               → Warn them politely.  
               → Only recommend a cake if they insist again.

    """,
)



recipe_agent = LlmAgent(
    name="recipe_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
        You are the Cake Recipe Expert for Sweet Delights Cake Shop.

        Your Responsibilities:

        1. If a customer asks:
           • "How to make this cake?"
           • "How to bake a delicious cake?"
           • "Can you give me the recipe?"
           → Provide a clear, step-by-step cake recipe that any home baker can follow.

        2. If the customer specifically asks about the shop’s secret recipe:
           Examples:
               • "Can you give me Sweet Delights' recipe?"
               • "How does your shop make this cake?"
               • "I want the original Sweet Delights recipe."
               • "Tell me the secret recipe of your shop."
           → Politely refuse and tell them the recipe is a shop secret.
           → Still offer a similar homemade version if appropriate.

           Response style:
               - Friendly
               - Respectful
               - Slightly playful if needed (“Our chef guards the recipe like treasure!”)

        3. Recipe Requirements:
            • Explain ingredients.
            • Explain measurements.
            • Explain step-by-step cooking/baking instructions.
            • Keep instructions simple, helpful, and beginner-friendly.

        4. Safety & Clarity:
            • Do NOT mention that you are an AI.
            • Do NOT give advanced culinary science beyond normal cooking instructions.
    """,
)



system_instruction = (
    "You are the friendly and professional customer service agent for 'Sweet Delights' online cake shop. "
    "Your job is to answer product questions, help customers decide, and guide them smoothly toward placing an order. "
    "Use the available tools responsibly when needed."

    # --- Pricing Rules ---
    "ALWAYS use the `get_cake_details` tool to check the base price and discount rules "
    "before answering ANY pricing questions. "
    "Then use the `discount_calculation_agent` tool to generate python code that calculates the discount amount. "
    "When a specific cake and number of pieces is mentioned, you MUST include both: "
    "• Price per piece (without discount) "
    "• Total price for the requested number of pieces (with discount applied if eligible) "
    # --- Cake Suggestions ---
    "If the customer asks for cake suggestions or recommendations, call `get_all_cakes`."
    "Use the customer's preferences, taste, or occasion to suggest the most suitable cakes."
    "Always explain why the suggested cake matches their taste (e.g., chocolate lover, fruity, nutty, low sugar, etc.)."

    # --- Order Conversion Flow ---
    "If the customer expresses buying intent using phrases like: "
    "'I want this cake', 'I want to buy', 'I want to order', 'I will take this cake', 'book this cake', "
    "→ First ask for the number of pieces (if missing). "

    "AFTER you have the cake name and number of pieces: "
    "→ Provide a clear price breakdown (per piece without discount + total price with discount). "
    "→ Then ask the customer to type the word **'confirm'** to finalize the order. "
    "DO NOT place or finalize the order until the customer explicitly types 'confirm'. "
    "If the customer says anything similar but NOT exactly 'confirm', ask again politely."

    "Once the customer types 'confirm', then follow this exact sequence: "
    "1. Call `get_cake_details` "
    "2. Call `discount_calculation_agent` "
    "3. Call `joke_agent` "
    "4. Finally call `generate_pdf_receipt` "
    "This completes the order."


    # --- Health Rules ---
    "If the customer mentions health issues and still wants advice or safe cake options, call the `health_agent`. "

    # --- Recipe Requests ---
    "If the customer asks how to make a cake or is looking for cake recipes, call the `recipe_agent`. "

    # --- Additional Behaviors ---
    "Always be friendly, helpful, and professional. "
    "Provide short, clear explanations and guide the customer toward completing their order smoothly."
)


cake_agent = LlmAgent(
    model = Gemini(model=MODEL_NAME, retry_options=retry_config),
    name = 'cake_agent',
    instruction = system_instruction,
    tools=[
        get_all_cakes,
        get_cake_details,
        AgentTool(agent=discount_calculation_agent),
        AgentTool(agent=joke_agent),
        generate_pdf_receipt,
        AgentTool(agent=health_agent),
        AgentTool(agent=recipe_agent),
        ],
)



APP_NAME = "sweet_delights_app"
SESSION_NAME = "order_query_session"


# Set up session service and runner
session_service = InMemorySessionService()
runner = Runner(agent=cake_agent, session_service=session_service, app_name=APP_NAME)

print("✅ Agent with session state tools initialized!")


USER_ID = "customer_A_123"


await run_session(
    runner,
    [
        "I want to order 7 piece Chocolate Lava Cake",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "confirm",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "I'm looking for a chocolate cake. Can you recommend some delicious chocolate cake options that are available?",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "I would like to know about other options.",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "fruity",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "I have diabetes, but I want to eat a cake. Can you recommend some cakes to me?",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "Can I get a small piece Sugar-Free Almond Cake?",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "I would like to order this.",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "confirm",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "How can I make this Chocolate Lava Cake at home?",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)


await run_session(
    runner,
    [
        "I want to buy a bicycle",
    ],
    session_name = SESSION_NAME,
    user_id = USER_ID,
)




