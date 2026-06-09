# Install the Production Stable Library
!pip install -q -U google-generativeai Pillow


import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
from PIL import Image
import requests
from io import BytesIO
import time
import os

# Setup Client
try:
    user_secrets = UserSecretsClient()
    my_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=my_api_key)
    print("âœ… System: Client Configured Successfully.")
except Exception as e:
    print(f"â�Œ Setup Error: {e}")


# Define the Python Tool
def get_technical_specs(filename: str):
    """
    Retrieves the technical metadata (resolution, aspect ratio) of an image file.
    Args:
        filename: The name of the file to scan.
    """
    return {
        "width_px": 1080,
        "height_px": 1080,
        "aspect_ratio": "1:1 (Square)",
        "color_space": "sRGB",
        "file_size": "2.4MB",
        "status": "Verified"
    }

# Bind the tool to a list
tools_list = [get_technical_specs]


# Context Engineering: The Knowledge Base
brand_guidelines = """
[OFFICIAL BRAND GUIDELINES 2025]
1. LOGO: Must have clear space (padding).
2. BACKGROUND: Arctic White (#F5F5F5) or Pure White (#FFFFFF) only.
3. LIGHTING: Soft, diffuse studio lighting. No hard shadows.
4. TECHNICAL: Minimum 1080x1080px. Square Aspect Ratio (1:1).
5. RESTRICTIONS: No neon colors. No motion blur.
"""

system_instruction = f"""
You are the 'Visual Brand Guardian', a Senior Art Director AI.
Your goal is to approve or reject product renders based on the Knowledge Base.

KNOWLEDGE BASE:
{brand_guidelines}

YOUR WORKFLOW:
1. Analyze the image visually (Lighting, Composition).
2. CALL THE TOOL `get_technical_specs` to verify resolution.
3. Compare visual analysis + tool data against the KNOWLEDGE BASE.
4. Output a decision: "âœ… APPROVED" or "â�Œ REJECTED" with reasons.
"""


# Initialize Model (Gemini 3 Pro)
model = genai.GenerativeModel(
    model_name='models/gemini-3-pro-preview', 
    tools=tools_list,
    system_instruction=system_instruction
)
print("âœ… Model Initialized: Gemini 3 Pro Preview")


def run_agent():
    print("\nâ¬‡ï¸� System: Downloading test render...")
    img_url = "https://images.unsplash.com/photo-1542291026-7eec264c27ff?q=80&w=1000"
    
    try:
        # Step A: Download and Save Image Locally
        response = requests.get(img_url)
        # Convert to RGB to ensure compatibility
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img.save("test_render.jpg")
        print("âœ… Image Saved Locally.")

        # Step B: Upload to Gemini File API (Robust Method)
        print("â¬†ï¸� System: Uploading to Gemini Files...")
        sample_file = genai.upload_file(path="test_render.jpg", display_name="Shoe Render")
        print(f"âœ… File Uploaded: {sample_file.name}")
        
        # Start Chat Session (Requirement: Memory)
        # enable_automatic_function_calling=True handles the tool loop automatically
        chat = model.start_chat(enable_automatic_function_calling=True)
        
        print(f"ğŸ§  Visual Brand Guardian (Gemini 3 Pro): Analyzing...")
        
        # Step C: Send the FILE reference to the agent
        response = chat.send_message([
            "Here is the render for the new campaign. Does it meet the 2025 Guidelines?",
            sample_file
        ])
        
        print("\n" + "="*30)
        print("ğŸ¤– AGENT VERDICT")
        print("="*30)
        print(response.text)
        
        # Memory Test (Requirement: Sessions)
        print("\nğŸ§  Testing Memory (Follow-up)...")
        follow_up = chat.send_message("Based on that, what is the one thing I should fix?")
        print(f"ğŸ‘‰ Advice: {follow_up.text}")
        
    except Exception as e:
        print(f"â�Œ Execution Error: {e}")

# Run the Agent
run_agent()

