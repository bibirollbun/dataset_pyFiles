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


!pip install google-genai pydantic


import os
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No GEMINI_API_KEY found in Kaggle Secrets.")
    os.environ["GEMINI_API_KEY"] = api_key
    print("GEMINI_API_KEY loaded into environment.")
except Exception as e:
    print("Failed to load GEMINI_API_KEY from Kaggle Secrets:", e)


import os
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
mock_mode = False


from pydantic import BaseModel, Field

class PanelScript(BaseModel):
    panel_number: int = Field(description="The sequence number of the panel on the page.")
    character_focus: str = Field(description="The primary character or object in focus.")
    scene_description: str = Field(description="Detailed description of the action, setting, and emotion.")
    dialogue: str = Field(description="The text for the speech bubble or narration caption.")

class VisualPlan(BaseModel):
    final_image_prompt: str = Field(description="The complete, detailed, and consistent prompt for the image generation model.")
    camera_angle: str = Field(description="Specific camera technique (e.g., 'Extreme Close-up', 'Wide Shot', 'Low Angle').")
    lighting_mood: str = Field(description="Desired lighting and atmosphere (e.g., 'High contrast film noir lighting', 'soft morning light').")


CHARACTER_BLUEPRINT = {
    "name": "Captain Anya",
    "physical_description": "A stoic female space captain, mid-30s, with a severe scar running over her right eye and a silver cybernetic arm. She wears a navy-blue utility jumpsuit.",
    "art_style_guide": "Digital painting, high-resolution, retro sci-fi aesthetic, emphasis on dramatic storytelling."
}

_mock_panelscript = PanelScript(
    panel_number=1,
    character_focus="Captain Anya",
    scene_description="Captain Anya turns sharply toward the viewport as the asteroid base detonates outside, shards of glass and light scattering; she clenches her jaw, resolve hardening.",
    dialogue="Not today. Hold together — I'm on my way."
)

_mock_visualplan = VisualPlan(
    final_image_prompt=(
        "CHARACTER: Captain Anya (A stoic female space captain, mid-30s, with a severe scar over her right eye and a silver cybernetic arm.) "
        "STYLE GUIDE: Digital painting, high-resolution, retro sci-fi aesthetic, emphasis on dramatic storytelling. "
        "Render Captain Anya in a low-angle medium shot on the command deck, dramatic backlight from exploding debris outside the viewport, "
        "dust and spark particles, cinematic composition, ultra-detailed, filmic lighting, 4k."
    ),
    camera_angle="Low-angle medium shot",
    lighting_mood="Dramatic backlight with rim light, high contrast"
)



from google import genai

def initialize_client():
    """Initialize Gemini client or return None in mock mode."""
    if mock_mode:
        logger.info("Mock mode enabled — skipping Gemini client initialization.")
        return None
    try:
        client = genai.Client()
        logger.info("Gemini client initialized.")
        return client
    except Exception as e:
        raise RuntimeError(
            "Error initializing Gemini client. Check that GEMINI_API_KEY is set in Kaggle Secrets and Internet is enabled."
        ) from e



from google.genai import types

def writer_agent_task(client, synopsis: str) -> PanelScript:
    logger.info("Writer Agent: Generating panel script...")
    if mock_mode:
        logger.info("Returning mock PanelScript.")
        return _mock_panelscript

    prompt = f"""
You are the 'Writer Agent'. Your task is to take the following high-level story synopsis
and create the script for a single, dramatic comic panel.

Synopsis: "{synopsis}"

Focus on creating:
1. A single moment of intense action or emotion.
2. Dialogue that is concise and impactful.
3. A clear description of the scene's focus.

Output the result in the required JSON format.
"""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PanelScript,
        ),
    )
    try:
        script_data = PanelScript.model_validate_json(response.text)
    except Exception as e:
        logger.exception("Failed to parse Writer Agent response as PanelScript.")
        raise RuntimeError("Writer Agent returned unexpected output; try switching to mock_mode=True to inspect.") from e
    logger.info("Writer Agent task complete.")
    return script_data

def director_agent_task(client, script: PanelScript, blueprint: Dict[str,str]) -> VisualPlan:
    logger.info("Director Agent: Creating visual plan...")
    if mock_mode:
        logger.info("Returning mock VisualPlan.")
        return _mock_visualplan

    consistency_instructions = f"""
CHARACTER: {blueprint['name']} ({blueprint['physical_description']})
STYLE GUIDE: {blueprint['art_style_guide']}
"""
    prompt = f"""
You are the 'Director Agent'. Your goal is to translate the panel script into a powerful,
coherent, and visually consistent image prompt that adheres to the character and art style guides.

{consistency_instructions}

Action/Emotion: {script.scene_description}
Dialogue: "{script.dialogue}"

Based on the script and guides, fill in the JSON fields. The 'final_image_prompt' MUST
start with the CHARACTER and STYLE GUIDE details to ensure maximum consistency from the image model.
"""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VisualPlan,
        ),
    )
    try:
        visual_plan_data = VisualPlan.model_validate_json(response.text)
    except Exception as e:
        logger.exception("Failed to parse Director Agent response as VisualPlan.")
        raise RuntimeError("Director Agent returned unexpected output; try switching to mock_mode=True to inspect.") from e
    logger.info("Director Agent task complete.")
    return visual_plan_data



def orchestrate_comic_panel(return_outputs=True):
    client = initialize_client()
    story_synopsis = (
        "Captain Anya is in the command deck. She receives a frantic distress call and "
        "turns sharply toward the viewport just as the nearby asteroid base explodes into a cloud of debris. "
        "Her expression shifts from calm to determined resolve."
    )
    logger.info("Orchestration started.")
    script_output = writer_agent_task(client, story_synopsis)
    print("\n--- Panel Script Output (JSON) ---")
    print(script_output.model_dump_json(indent=2))
    visual_plan_output = director_agent_task(client, script_output, CHARACTER_BLUEPRINT)
    print("\n--- Visual Plan Output (JSON) ---")
    print(visual_plan_output.model_dump_json(indent=2))
    print("\n--- Illustrator Agent Input (ready) ---")
    print("Final Prompt:", visual_plan_output.final_image_prompt)
    print("Camera:", visual_plan_output.camera_angle)
    print("Lighting:", visual_plan_output.lighting_mood)
    logger.info("Orchestration complete.")
    if return_outputs:
        return {
            "client": client,
            "script_output": script_output,
            "visual_plan_output": visual_plan_output
        }

outputs = orchestrate_comic_panel()
client = outputs.get("client")
script_output = outputs.get("script_output")
visual_plan_output = outputs.get("visual_plan_output")



try:
    response = client.images.generate(
        model="imagen-3.0-generate-001",
        prompt=visual_plan_output.final_image_prompt,
    )

    img_data = response.images[0].data

    with open("panel.png", "wb") as f:
        f.write(img_data)

    print("Saved panel.png")

except Exception as e:
    print("Image generation failed:", e)


