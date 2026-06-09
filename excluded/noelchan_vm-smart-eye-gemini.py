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


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



import json
import requests
import subprocess
import time
import uuid

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# ---   Installation and Configuration   ---
!pip install -U -q "google-generativeai"  
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown, Image
import PIL.Image
import os

# 1. Load API Key  
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 2. Set VM Smart Eye Role ( 
sys_instruction = """You are "VM Smart Eye," a Senior Visual Merchandising Manager.
Your Goal: Analyze store photos for compliance with brand guidelines.
Your Tone: Professional, constructive, and detail-oriented."""

# --- ğŸ› ï¸� Key Addition: Define Custom Tool   ---
def save_report_to_disk(report_content: str):
    """
    Saves the generated VM compliance report to a local markdown file.
    Args:
        report_content: The full text of the report to be saved.
    Returns:
        A confirmation message.
    """
    filename = "vm_smart_eye_report.md"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        return f"âœ… File saved successfully: {filename}"
    except Exception as e:
        return f"â�Œ Error saving file: {str(e)}"

# --- 3. Initialize Model (Bind Tools)   ---
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash", 
    system_instruction=sys_instruction,
    tools=[save_report_to_disk]
)

print("âœ… System setup complete! Model locked: Gemini 2.0 Flash")
print("âœ… Tool mounted: save_report_to_disk (File Saving function)")


def analyze_vm_display(image_path, guideline_text):
    # 1. Load the image (Unchanged)
    img = PIL.Image.open(image_path)
    print("ğŸ“¸ VM Smart Eye is performing visual analysis...")
    display(img.resize((300, int(300*img.height/img.width))))

    # 2. Professional Prompt (Unchanged Structure)
    prompt = f"""
    You are a Senior Visual Merchandising Manager (VM Smart Eye) with 15 years of experience.
    Your task is to review the store display photo for compliance with the current season's Guidelines.
    
    ---
    ğŸ“‹ Current Guidelines:
    {guideline_text}
    ---
    
    Please perform the following steps:
    1. **Visual Analysis and Report Generation**: 
       Carefully observe the image, compare it against the guidelines, and generate a professional compliance report IN ENGLISH.
       
       The report format must be Markdown:
       ## ğŸ‘�ï¸� VM Smart Eye Smart Audit Report
       **ğŸ“Š Compliance Score:** [0-10] / 10
       **âœ… Highlights:** ...
       **âš ï¸� Non-Compliance & Improvement Suggestions:** ...
       **ğŸ’¡ Expert Insights:** ...

    2. **Tool Execution**:
       After writing the report, you MUST immediately call the `save_report_to_disk` tool to save the complete report content you just generated.
    """

    messages = [prompt, img]
    final_report_content = "" # Initialize a variable to store the report text
    
    try:
        # 4. Critical: Start the Tool Calling Loop
        while True:
            response = model.generate_content(
                messages,
                tool_config={'function_calling_config': 'AUTO'} 
            )

            # --- Observability ---
            print("\nğŸ“Š [System Log] Observability Metrics (Token Usage):")
            print(response.usage_metadata)
            print("--------------------------------------------------\n")
            
            # â­�ï¸� Core Fix: Safely extracting Function Call information â­�ï¸�
            tool_calls = []
            if response.candidates:
                content = response.candidates[0].content
                if content.parts:
                    tool_calls = [
                        p.function_call for p in content.parts if p.function_call
                    ]
            
            # Check for Function Call
            if tool_calls:
                print(f"ğŸ”§ Tool Triggered: {len(tool_calls)} function call(s) detected.")
                
                for fc in tool_calls:
                    # ğŸ’¥ Ultimate Bypass: We know the report content is here, so we extract it first!
                    if fc.name == "save_report_to_disk" and 'report_content' in fc.args:
                        final_report_content = fc.args['report_content']
                        print("âœ¨ Report content successfully extracted from Tool Call arguments.")
                        
                        # Execute the actual saving action
                        function_result = save_report_to_disk(**dict(fc.args))
                        print(f"   -> Tool Execution Result: {function_result}")
                        
                        # Due to environment issues, we no longer try to pass the result back to the model (to prevent errors)
                        # Instead, we break the loop and output the content we already retrieved.
                        
                        # The failing code for replying to the model would be:
                        # messages.append(content)
                        # messages.append(genai.types.Part.from_function_response(...))
                        
                        break # Break the loop, as we have the necessary report and save status.
                
                if final_report_content:
                    break # Exit the outer while loop if report content was extracted
            
            # Check for final text response (This path is now less critical)
            elif response.text:
                final_report_content = response.text
                break
            
            else:
                print("âš ï¸� Warning: Agent did not return text or tool call. Breaking loop.")
                break

        return final_report_content # Return the extracted report content
        
    except Exception as e:
        print(f"\nâ�Œ FATAL ERROR CATCHED: {str(e)}")
        # If failed, return the report content captured during the Tool Call
        if final_report_content:
            return f"âš ï¸� Report was generated successfully via the save tool, but final display failed due to environment error. Report content:\n\n{final_report_content}"
        
        return f"â�Œ Analysis Failed (Final Report): {str(e)}"

print("âœ… Cell 2 update complete! Ultimate bypass mode enabled to secure the report and score.")


# --- Simulate a real-world scenario: Extracted text from corporate PDF guidelines ---

# 1. Set the image path  
image_path = "/kaggle/input/photo-3/IMG_4370.jpeg"   
# 2. Input the 'correct' guidelines (2025 Spring Green Collection)
# This demonstrates the value of an Enterprise Agent: it can change its criteria based on seasonal documents.
current_guidelines = """
ã€�2025 Spring Collection - Visual Guidelinesã€‘

1. **Color Palette:**
   - Key colors: "Sage Green" and "Pistachio".
   - Must display a "Monochromatic" layered look.

2. **Window Display:**
   - Must include the "2025 Spring Collection" Decal, placed centered.
   - Glass must be clean and free of fingerprints.

3. **Mannequin Styling:**
   - Mannequins must wear the key green collection items.
   - Use the "Relaxed Logic" pose to showcase natural drape.

4. **Housekeeping:**
   - Pantone Floor Decal must be clearly visible and undamaged.
   - Rails must be level, and hanger spacing should be 2 fingers wide.
"""

# --- Execute Analysis ---
# Check if path is correct
import os
if not os.path.exists(image_path):
    print(f"âš ï¸� Image not found, please check path: {image_path}")
else:
    # Call the AI Agent
    result = analyze_vm_display(image_path, current_guidelines)
    
    # Display the result
    display(Markdown(result))

