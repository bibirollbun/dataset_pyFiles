from IPython.display import Image, display
display(Image("/kaggle/input/archietecture/archietecture.png"))



!pip install google-adk



import os
import json
import base64
from typing import Any, Dict, List

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk import Runner
from google.adk.tools import FunctionTool, google_search
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from kaggle_secrets import UserSecretsClient
import asyncio



# Configure your Gemini API Key
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_Key")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



from typing import List, Dict
from pydantic import BaseModel, Field


class ProductInfo(BaseModel):
    """
    Schema for storing extracted product information from an image.

    Attributes:
        product (str): Name of the detected product (e.g., "Boo Headset").
        category (str): High-level product category (e.g., "Headphones").
        features (List[str]): List of key visual or functional features of the product.
        audience (List[str]): Target audience segments likely to buy the product.
        mood (str): Aesthetic or stylistic mood of the product (e.g., "Modern", "Luxury").
        keywords (List[str]): Semantic keywords extracted from the image or product description.
        short_description (str): Concise product description (1-2 sentences).
        long_description (str): Detailed product description suitable for e-commerce or marketing pages.
    """
    product: str = Field(..., description="Name of the detected product")
    category: str = Field(..., description="High-level product category")
    features: List[str] = Field(..., description="List of key visual or functional features")
    audience: List[str] = Field(..., description="Target audience segments for the product")
    mood: str = Field(..., description="Aesthetic or stylistic mood of the product")
    keywords: List[str] = Field(..., description="Semantic keywords from the image or product")
    short_description: str = Field(..., description="Concise short description for marketing")
    long_description: str = Field(..., description="Detailed long description for e-commerce")



# ------------------------------
# Marketing Output Schema
# ------------------------------
class MarketingOutput(BaseModel):
    """
    Schema for storing generated marketing content for a product.

    Attributes:
        taglines (Dict[str, List[str]]): Taglines categorized by tone/style (e.g., Professional, Energetic, Playful, Tech, Luxury).
        winner_tagline (str): The selected best tagline from all generated options.
        short_description (str): Refined short product description optimized for marketing.
        long_description (str): Refined long product description optimized for e-commerce or promotional use.
    """
    taglines: Dict[str, List[str]] = Field(..., description="Dictionary of taglines categorized by style")
    winner_tagline: str = Field(..., description="The selected best tagline")
    short_description: str = Field(..., description="Refined short description for marketing")
    long_description: str = Field(..., description="Refined long description for e-commerce or ads")



# ------------------------------
# SEO Output Schema
# ------------------------------
class SEOOutput(BaseModel):
    """
    Schema for storing SEO and advertising outputs for a product.

    Attributes:
        seo_keywords (List[str]): Recommended SEO keywords for the product.
        ad_copies (List[str]): List of ad copy variations suitable for digital campaigns.
        upsell_products (List[str]): Suggested complementary or upsell products to increase revenue.
    """
    seo_keywords: List[str] = Field(..., description="List of recommended SEO keywords")
    ad_copies: List[str] = Field(..., description="Suggested advertisement copy texts")
    upsell_products: List[str] = Field(..., description="Related or complementary products for upselling")


def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image file to a Base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



# ImageAgent: Detects product info from image
image_agent = LlmAgent(
    name="ImageAgent",
    model="gemini-2.0-flash",
    instruction="""
You are an image analyzer specialized in consumer electronics and e-commerce products.
Input: JSON containing 'image_b64' (base64 string).
Task:
- Identify the real-world product in the image.
- Provide category, features, audience, mood, keywords.
- Generate short_description (1-2 sentences) and long_description (marketing-ready).
Output JSON only, matching the ProductInfo schema.
Do not hallucinate unrelated categories; focus on headphones, smartwatches, phones, speakers, etc.
""",
    tools=[ProductInfo]
)



# MarketingAgent: Generates taglines and refines descriptions
marketing_agent = LlmAgent(
    name="MarketingAgent",
    model="gemini-2.0-flash",
    instruction= """
You are a marketing AI.
Input: JSON containing product information including short_description and long_description, plus brand_voice.
Task:
- Generate taglines (Professional, Energetic, Playful, Techy, Luxury)
- Pick winner_tagline
- Refine short_description and long_description for marketing
Output only JSON matching MarketingOutput schema.
""",
    tools=[MarketingOutput]
)


# SEOAgent: Generates keywords, ad copies, and upsells
seo_agent = LlmAgent(
    name="SEOAgent",
    model="gemini-2.0-flash",
    instruction="""
You are an e-commerce SEO assistant.
Input: JSON containing product information including short_description and long_description.
Task:
- Generate SEO keywords
- Generate ad_copies
- Suggest upsell_products
Output only JSON matching SEOOutput schema.
""",
    tools=[SEOOutput]
)


pipeline_agent = SequentialAgent(
    name="MarketingPipelineAgent",
    sub_agents=[image_agent, marketing_agent, seo_agent],
    description="Analyzes an image, generates marketing content, SEO and upsells sequentially."
)

root_agent = pipeline_agent



# Session setup
session_service = InMemorySessionService()
USER_ID = "user123"
SESSION_ID = "marketing_session"

await session_service.create_session(user_id=USER_ID, app_name="marketing_pipeline", session_id=SESSION_ID)

runner = Runner(
    agent=root_agent,
    session_service=session_service,
    app_name="marketing_pipeline"
)



async def run_pipeline(image_path: str, brand_voice: str):
    image_b64 = encode_image_to_base64(image_path)
    user_input = {"image_b64": image_b64, "brand_voice": brand_voice}
    content = types.Content(role="user", parts=[types.Part(text=json.dumps(user_input))])
    
    final_result = None
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            final_result = event.content.parts[0].text
            break
    return final_result



image_path = "/kaggle/input/sample-image/sample2.png"  # Replace with your product image path
brand_voice = "Premium & Minimal"

# Run pipeline and get final output
final_output = await run_pipeline(image_path, brand_voice)



print(final_output)


import re

# Automatically remove ```json fences if present
clean_json = re.sub(r'^```json|```$', '', final_output, flags=re.MULTILINE).strip()

# Parse with Pydantic v2
product_info = ProductInfo.model_validate_json(clean_json)

# âœ… Correct way to print nicely in Pydantic v2
print("âœ… Parsed ProductInfo:\n", product_info.model_dump_json(indent=2))



import sqlite3
import json
# -----------------------------
# 1ï¸�âƒ£ Connect to SQLite database
# -----------------------------
conn = sqlite3.connect("product_data.db")
cursor = conn.cursor()

# -----------------------------
# 2ï¸�âƒ£ Create table if it doesn't exist
# -----------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product TEXT,
    category TEXT,
    features TEXT,
    audience TEXT,
    mood TEXT,
    keywords TEXT,
    short_description TEXT,
    long_description TEXT
)
""")



# -----------------------------
# 3ï¸�âƒ£ Insert product_info into the database
# -----------------------------
cursor.execute("""
INSERT INTO products (product, category, features, audience, mood, keywords, short_description, long_description)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
    product_info.product,
    product_info.category,
    json.dumps(product_info.features),  # convert list to JSON string
    json.dumps(product_info.audience),  # convert list to JSON string
    product_info.mood,
    json.dumps(product_info.keywords),  # convert list to JSON string
    product_info.short_description,
    product_info.long_description
))

# Commit changes
conn.commit()
print("âœ… Product info saved to SQLite database!")



# -----------------------------
# 4ï¸�âƒ£ Retrieve and display data
# -----------------------------
cursor.execute("SELECT * FROM products")
rows = cursor.fetchall()

print(rows)
for row in rows:
    print("ID:", row[0])
    print("Product:", row[1])
    print("Category:", row[2])
    print("Features:", json.loads(row[3]))  # convert back to list
    print("Audience:", json.loads(row[4]))
    print("Mood:", row[5])
    print("Keywords:", json.loads(row[6]))
    print("Short Description:", row[7])
    print("Long Description:", row[8])
    print("-" * 50)




# -----------------------------
# Install Gradio if needed
# -----------------------------
# !pip install gradio

import gradio as gr
import asyncio

# -----------------------------
# Function to run pipeline and save to DB
# -----------------------------
def process_and_save(image, brand_voice="Premium & Minimal"):
    # 1ï¸�âƒ£ Save uploaded image temporarily
    image_path = "/tmp/uploaded_image.png"
    image.save(image_path)

    # 2ï¸�âƒ£ Run ADK pipeline asynchronously
    final_output = asyncio.run(run_pipeline(image_path, brand_voice))

    # 3ï¸�âƒ£ Clean JSON fences
    clean_json = re.sub(r'^```json|```$', '', final_output, flags=re.MULTILINE).strip()

    # 4ï¸�âƒ£ Parse with Pydantic v2
    product_info = ProductInfo.model_validate_json(clean_json)

    # 5ï¸�âƒ£ Save to SQLite
    conn = sqlite3.connect("product_data.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product TEXT,
        category TEXT,
        features TEXT,
        audience TEXT,
        mood TEXT,
        keywords TEXT,
        short_description TEXT,
        long_description TEXT
    )
    """)

    cursor.execute("""
    INSERT INTO products (
        product, category, features, audience, mood, keywords, short_description, long_description
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        product_info.product,
        product_info.category,
        json.dumps(product_info.features),
        json.dumps(product_info.audience),
        product_info.mood,
        json.dumps(product_info.keywords),
        product_info.short_description,
        product_info.long_description
    ))

    conn.commit()
    conn.close()

    # 6ï¸�âƒ£ Return summary to display
    return f"âœ… Product '{product_info.product}' saved to database!", product_info.model_dump_json(indent=2)

# -----------------------------
# Gradio Interface
# -----------------------------
iface = gr.Interface(
    fn=process_and_save,
    inputs=[
        gr.Image(type="pil", label="Upload Product Image"),
        gr.Textbox(label="Brand Voice (optional)", placeholder="Premium, Minimal, Energetic...", lines=1)
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Code(label="Product Info JSON", language="json")
    ],
    
    title="Vision-to-Marketing: Product Image to Marketing Package",
    description="Upload a product image and automatically generate marketing content, SEO keywords, and save to SQLite DB."
)

# Launch the UI
iface.launch()



#Sample output images
from IPython.display import Image, display
display(Image("/kaggle/input/example-output/example_output.png"))

