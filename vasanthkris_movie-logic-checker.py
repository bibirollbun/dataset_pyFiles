import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


pip install google-genai


from kaggle_secrets import UserSecretsClient
from google import genai
from google.genai import types
import textwrap

# API Key
user_secrets = UserSecretsClient()
my_api_key = user_secrets.get_secret("GOOGLE_API_KEY")

# Google Gemini
client = genai.Client(api_key=my_api_key)

print("âœ… Client setup complete!")


from dataclasses import dataclass

@dataclass
class ScriptSession:
    previous_scene: str
    current_scene: str
    next_scene: str
    realism_notes: str = ""
    continuity_notes: str = ""

    def get_context_for_continuity(self):
        """Packages the scenes so the Continuity Agent can read them"""
        return f"""
        === PREVIOUS SCENE (Context) ===
        {self.previous_scene}
        
        === CURRENT SCENE (Analyze This) ===
        {self.current_scene}
        
        === NEXT SCENE (Context) ===
        {self.next_scene}
        """


# Google Search Tool
google_search_tool = types.Tool(
    google_search=types.GoogleSearch()
)

# The Realism Agent (Checks Physics/Facts)
def run_realism_agent(scene_text):
    print("   ... Realism Agent is researching facts ...")
    
    prompt = """
    You are a Fact-Checking Script Supervisor. 
    1. Read the scene below.
    2. Use Google Search to verify any physical claims, historical facts, weather, or distances.
    3. If you find a logical error (e.g., "It takes 5 hours to fly NY to London, not 1 hour"), list it.
    4. If the scene is logical, reply only with "PASS".
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=scene_text,
        config=types.GenerateContentConfig(
            tools=[google_search_tool],
            system_instruction=prompt,
            temperature=0.0
        )
    )
    return response.text

# The Continuity Agent (Checks Flow)
def run_continuity_agent(full_context_text):
    print("   ... Continuity Agent is checking timelines ...")
    
    prompt = """
    You are a Continuity Editor. Analyze the flow between the three scenes provided.
    Check for:
    - Object Permanence (Did they drop a gun but still have it?)
    - Time Flow (Is it night in scene 1 but noon in scene 2?)
    - Location (Did they teleport?)
    
    Output a bulleted list of errors. If none, reply "PASS".
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=full_context_text,
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.1
        )
    )
    return response.text

# The Re-Writer Agent (The Fixer)
def run_writer_agent(current_scene, logic_notes, continuity_notes):
    print("   ... Writer Agent is rewriting the scene ...")
    
    prompt = """
    You are a Senior Screenwriter. Rewrite the 'Current Scene' to fix the reported errors.
    - Keep the original dialogue style and emotion.
    - ONLY fix the logic/continuity issues.
    - Do not change the plot outcome.
    """
    
    user_content = f"""
    ORIGINAL SCENE:
    {current_scene}
    
    LOGIC ERRORS TO FIX:
    {logic_notes}
    
    CONTINUITY ERRORS TO FIX:
    {continuity_notes}
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.7 
        )
    )
    return response.text


def process_script(prev, curr, next_scn):

    session = ScriptSession(prev, curr, next_scn)
    
    print("ğŸ�¬ AGENT WORKFLOW STARTED")
    print("-" * 30)

    # Run Realism Check 
    session.realism_notes = run_realism_agent(session.current_scene)
    print(f"ğŸ“� Realism Feedback:\n{session.realism_notes}\n")

    # Run Continuity Check
    session.continuity_notes = run_continuity_agent(session.get_context_for_continuity())
    print(f"ğŸ”— Continuity Feedback:\n{session.continuity_notes}\n")

    if "PASS" in session.realism_notes and "PASS" in session.continuity_notes:
        print("âœ… No errors found. Scene is perfect!")
        return session.current_scene

    # Run Rewrite
    print("-" * 30)
    new_scene = run_writer_agent(
        session.current_scene, 
        session.realism_notes, 
        session.continuity_notes
    )
    
    return new_scene


# TAMIL MOVIE DATA (Inspired by 'Beast')

# CONTEXT: The Hero (Veera) is trapped inside a mall taken over by terrorists.

scene_1 = """
INT. MALL LOBBY - DAY
Veera stands surrounded by five masked terrorists. 
The leader shouts, "Drop your weapon!"
Veera sighs, pulls his pistol from his waistband, and drops it to the floor.
One terrorist kicks the pistol away, sliding it under a locked shutter.
Veera raises his empty hands. He is completely unarmed.
"""

scene_2 = """
EXT. MALL ENTRANCE - MOMENTS LATER
Veera escapes and jumps into a RED MINI COOPER. 
He floors the gas pedal. The speedometer hits 100 km/h.
Veera drives the car toward a small concrete ramp near the entrance.
The car LAUNCHES into the air. It soars vertically, flying upward past the first and second floors.
The car crashes through the glass window of the THIRD FLOOR atrium.
"""

scene_3 = """
INT. MALL THIRD FLOOR - CONTINUOUS
The Mini Cooper lands with a heavy thud, smoke pouring from the engine.
The driver's door opens. Veera steps out.
He pumps a SHOTGUN and aims it at the terrorists below.
"""

# EXECUTE
final_script = process_script(scene_1, scene_2, scene_3)

print("="*40)
print("FINAL POLISHED SCRIPT:")
print("="*40)
print(final_script)

