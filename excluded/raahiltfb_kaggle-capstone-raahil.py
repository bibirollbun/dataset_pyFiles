# ==============================================================================
# 0. Setup and Environment Configuration
# ==============================================================================

# Core Kaggle imports
import numpy as np
import pandas as pd
import os
import json
from typing import List

# Install necessary packages for structured output and API interaction
!pip install -q google-genai pydantic requests

# Agent Libraries (from the newly installed packages)
from google import genai
from pydantic import BaseModel, Field
import requests

# --- API Keys Configuration (Your keys inserted) ---
os.environ['GEMINI_API_KEY'] = 'AIzaSyDil2EV22Cq0i2wbIP8NktekpIWGfHe9SY' 
PEXELS_API_KEY = 'OMVPphU7K8XcWbE63y9MJAPdIysTgD0iooosQXJZhfcSHdDN7QTJmWdO' 

try:
    client = genai.Client()
    print("âœ… Gemini Client Initialized.")
except Exception as e:
    print(f"â�Œ Failed to initialize Gemini Client: {e}")
    
# ==============================================================================
# 1. Agent 1: Script Writer Agent (Context Engineering & Memory)
# ==============================================================================

# 1.1. Sessions & Memory Schema (New Element!)
class UserPreferences(BaseModel):
    """Represents a simple memory/state of the user's preferred style."""
    tone: str = Field(description="The desired tone for the video (e.g., Motivational, Formal, Sarcastic).")
    audience: str = Field(description="The target audience (e.g., Young Investors, Retirees, Students).")

# 1.2. Structured Output Schema (Pydantic Models)
class Scene(BaseModel):
    """A structured representation of a single video scene."""
    scene_id: int = Field(description="Sequential ID for the scene, starting at 1.")
    duration: int = Field(description="The approximate duration of the scene in seconds (must be an integer).")
    voiceover: str = Field(description="The complete, polished voiceover script for this scene.")
    visual_keywords: str = Field(description="A comma-separated string of highly specific keywords for searching stock footage.")
    video_url: str = Field(default="", description="Placeholder for the video URL, to be filled by the Visual Selector Agent.")

class VideoScript(BaseModel):
    """The master script containing all scenes for video assembly."""
    title: str = Field(description="The catchy title of the video short.")
    total_duration_estimate: int = Field(description="The calculated total duration of the script in seconds.")
    scenes: List[Scene]

# 1.3. Script Generation Function (Now accepts memory)
def generate_video_script(prompt: str, user_memory: UserPreferences) -> dict:
    """
    Uses the Gemini model to generate a structured video script, guided by user memory.
    """
    if not os.environ.get('GEMINI_API_KEY'):
        print("â�Œ Gemini API Key not set. Skipping script generation.")
        return None
        
    system_instruction = (
        "You are the **Script Writer Agent**, an expert financial content creator specializing in 30-second shorts. "
        "Current User Preferences/Memory: "
        f"Tone='{user_memory.tone}', Audience='{user_memory.audience}'. "
        "Your task is to generate a compelling script tailored to these preferences. "
        "Strictly adhere to the provided JSON schema for output."
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": VideoScript,
            }
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"An error occurred during script generation: {e}")
        return None

# ==============================================================================
# 2. Agent 2: Visual Selector Agent (Custom Tool Implementation)
# ==============================================================================

# 2.1. The Custom Tool: Pexels Video Search
def search_stock_video(keywords: str) -> str:
    """
    Custom Tool to search the Pexels API for a free stock video URL based on keywords.
    """
    if not PEXELS_API_KEY or PEXELS_API_KEY == 'YOUR_PEXELS_API_KEY_HERE':
        return "ERROR: Pexels API Key is missing. Cannot fetch video URL."
        
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": keywords,
        "orientation": "portrait", # Optimized for Reels/Shorts
        "size": "medium",
        "per_page": 1
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if data['videos']:
            # Find the highest resolution MP4 file link available
            best_video = max(
                data['videos'][0]['video_files'], 
                key=lambda x: int(x['width']) if 'width' in x else 0
            )
            return best_video['link']
        
        return f"Fallback: No specific video found for '{keywords}'. Use a generic stock asset."

    except requests.exceptions.RequestException as e:
        return f"API ERROR: Could not connect to Pexels or API key is invalid. Details: {e}"


# 2.2. Visual Selector Agent Logic
def select_visual_assets(script_data: dict) -> dict:
    """
    Iterates through the script and uses the Custom Tool to assign a video URL to each scene.
    """
    if not script_data:
        return {"error": "No script data provided."}

    print("\n--- Running Visual Selector Agent (Applying Custom Tool) ---")
    
    updated_scenes = []
    # Convert the dict back to the Pydantic model for type safety
    script_model = VideoScript(**script_data) 
    
    for scene in script_model.scenes:
        print(f" -> Searching for scene {scene.scene_id}: '{scene.visual_keywords}'")
        video_link = search_stock_video(scene.visual_keywords)
        
        scene.video_url = video_link
        updated_scenes.append(scene.model_dump())
        
        print(f"   -> Result: {'SUCCESS' if 'http' in video_link else 'FALLBACK/ERROR'}")
        
    script_data['scenes'] = updated_scenes
    return script_data

# ==============================================================================
# 3. Agent 3: Final Assembler Agent (Placeholder)
# ==============================================================================

def assemble_final_assets(final_script: dict):
    """
    Generates the final assembly instructions and voiceover placeholder.
    """
    print("\n--- Running Final Assembler Agent (Placeholder) ---")
    
    if not final_script or final_script.get('error'):
        print("â�Œ Cannot assemble. Script data is missing or corrupted.")
        return

    assembly_guide = f"## ğŸ�¬ Video Assembly Instructions: {final_script.get('title', 'Untitled Video')}\n\n"
    assembly_guide += f"**Total Estimated Duration:** {final_script.get('total_duration_estimate')} seconds\n\n"
    assembly_guide += "### Scene Breakdown:\n"
    
    full_voiceover = ""
    for scene in final_script.get('scenes', []):
        full_voiceover += scene['voiceover'] + " "
        assembly_guide += f"--- Scene {scene['scene_id']} ---\n"
        assembly_guide += f"Duration: {scene['duration']}s\n"
        assembly_guide += f"**Video URL (Pexels):** {scene['video_url']}\n"
        assembly_guide += f"**Voiceover:** \"{scene['voiceover']}\"\n\n"

    print("--- FULL VOICE-OVER TEXT ---")
    print(full_voiceover.strip())
    
    filename = final_script.get('title', 'final_video').replace(' ', '_').lower().replace(':', '') + "_assembly_guide_with_memory.txt"
    with open(filename, 'w') as f:
        f.write(assembly_guide)
        
    print(f"\nâœ… Assembly Agent Complete. Final instructions saved to: {filename}")


# ==============================================================================
# 4. EXECUTION FLOW (Multi-Agent System Orchestration with Memory)
# ==============================================================================

print("\n\n######################################################")
print("# STARTING AGENTS INTENSIVE CAPSTONE PROJECT EXECUTION #")
print("######################################################\n")

# A. User Input and Memory Initialization (New Memory Object)
video_concept = "Starbuck's Secret: A 30-second video explaining the power of compound interest if you invested your daily coffee money into the S&P500 instead."

user_state = UserPreferences(
    tone="Motivational and Urgent",
    audience="Young Professionals (22-35)"
)

print(f"ğŸ§  Current User Session Memory: {user_state.model_dump_json(indent=2)}")
print(f"ğŸ�¬ User Concept: {video_concept}")

# 1. Run Script Writer Agent (Passing Memory/State)
script_output = generate_video_script(video_concept, user_state)

if script_output:
    # 2. Run Visual Selector Agent (Applies the Custom Tool)
    final_script_with_urls = select_visual_assets(script_output)
    
    # 3. Run Final Assembler Agent (Generates final assets/instructions)
    assemble_final_assets(final_script_with_urls)

    print("\n\n--- Final Structured Output (JSON with URLs) ---")
    print(json.dumps(final_script_with_urls, indent=2))
    
print("\n######################################################")
print("# AGENT WORKFLOW COMPLETE #")
print("######################################################")

