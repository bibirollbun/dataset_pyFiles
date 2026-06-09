!pip install intelli -q


import os
import asyncio
import base64
import pandas as pd
from kaggle_secrets import UserSecretsClient
from intelli.flow import Agent, Task, Flow, TextTaskInput, ImageTaskInput
from IPython.display import Markdown, display, HTML, Image

# Initialize Google Gemini 3 API Key
user_secrets = UserSecretsClient()
try:
    # Add 'GEMINI_API_KEY' to your Kaggle Secrets
    os.environ['GEMINI_API_KEY'] = user_secrets.get_secret("GEMINI_API_KEY")
    GOOGLE_API_KEY = os.environ['GEMINI_API_KEY']
    print("Gemini 3 Pro initialized ✅")
except:
    print("API Key Error: Add GEMINI_API_KEY to your Kaggle Secrets.")


# Model parameters for the flagship Gemini 3 Pro
gemini_3_text_config = {
    'key': GOOGLE_API_KEY,
    'model': 'gemini-3-pro-preview'
}

imagen_3_config = {
    'key': GOOGLE_API_KEY,
    'model': 'gemini-3-pro-image-preview'
}


# Agent 1: The Creative Director (Brain)
strategist = Agent(
    agent_type='text',
    provider='gemini',
    mission='Analyze product specs to define a high-end "Vibe", "Title", and "Tagline".',
    model_params=gemini_3_text_config
)

# Agent 2: The Frontend Coder 
frontend_dev = Agent(
    agent_type='text',
    provider='gemini',
    mission='''You are a UI Engineer. 
    Output ONLY a single, valid HTML document (including <style> tags).
    Use {product_image} as the exact src for the product <img> tag.
    IMPORTANT: Do not wrap your response in markdown code blocks (```html). 
    Start directly with <!DOCTYPE html>.''',
    model_params=gemini_3_text_config
)

# Agent 3: The Studio Photographer (Visual Generation)
artist = Agent(
    agent_type='image',
    provider='gemini',
    mission='photorealistic luxury product photography, 4k, studio lighting.',
    model_params=imagen_3_config
)


data = {
    'raw_name': ['cheap plastic chair white', 'gaming laptop 16gb ram rtx fast'],
    'raw_specs': ['polypropelene, 5kg weight, stackable, stain resistant', '144hz screen, rgb keyboard, 1tb ssd, no os, dual fans'],
    'target_audience': ['Budget Cafes & Startups', 'Competitive E-Sports Gamers']
}

df = pd.DataFrame(data)
display(df)


def build_gemini_3_flow(row):
    """Constructs the dependency graph for a single product row."""
    
    input_context = f"Product: {row['raw_name']} | Audience: {row['target_audience']}"
    
    # Task 1: Strategy (The Root)
    t1_strat = Task(TextTaskInput(input_context), strategist)

    # Task 2: Code (Branch A)
    t2_code = Task(TextTaskInput("Generate a complete HTML product card."), frontend_dev)

    # Task 3: Image (Branch B)
    t3_image = Task(TextTaskInput("Generate a high-end product visual."), artist)

    # Define the Flow Graph
    return Flow(
        tasks={"strategy_task": t1_strat, "code_task": t2_code, "image_task": t3_image},
        map_paths={"strategy_task": ["code_task", "image_task"]},
        log=False
    )


product_index = 0 
active_row = df.iloc[product_index]
print(f"Active Product: {active_row['raw_name']}")


# build the flow
swarm_flow = build_gemini_3_flow(active_row)

# visual the agents
swarm_flow.generate_graph_img(name="swarm_logic", save_path=".")
display(Image(filename='swarm_logic.png'))


# Execute the swarm for the single item
results = await swarm_flow.start()


# 1. Extract and Clean the HTML Code
raw_html = results['code_task']['output']
clean_html = raw_html.replace("```html", "").replace("```", "").strip()


# 3. Extract the Image
img_data = results['image_task']['output']


print("--- Raw Image Output (Artist Agent) ---\n")
image_uri = f"data:image/jpeg;base64,{img_data}"
display(HTML(f'<img src="{image_uri}" style="width:300px; border-radius:12px; margin-bottom:20px;">'))


def render_ai_output(clean_html, image_src):
    """
    Directly injects the Imagen 3 asset into the Gemini 3 HTML.
    """
    # Simply replace the placeholder with the Base64 data URI
    final_output = clean_html.replace("{product_image}", image_src)
    
    # Render the full AI-designed component
    display(HTML(final_output))


render_ai_output(clean_html, image_uri)




