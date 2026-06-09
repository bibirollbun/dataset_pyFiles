# Copyright 2025 Antoine Brunel.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# CELL 1: INSTALLATION

# We install WeasyPrint, and MarkItDown
%pip install -U -q markitdown[all] weasyprint

print("âœ… Bureau Ops Dependencies Installed.")


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


from google.adk.tools import FunctionTool as tool
from markitdown import MarkItDown
from google.genai import Client
from functools import lru_cache

# --- TOOL 1: THE READER ---
USE_VISION = True  # Set to True if you want Gemini to describe charts/images


@lru_cache(maxsize=10)
def extract_md(file_path):
    print(f"ğŸ“„ Extracting: {file_path}")
    
    if USE_VISION and "GOOGLE_API_KEY" in os.environ:
        print("ğŸ‘�ï¸� Using Gemini Vision for image descriptions...")
        # Configure the Gemini Client for MarkItDown
        client = Client(api_key=os.environ["GOOGLE_API_KEY"])
        md = MarkItDown(llm_client=client, llm_model="gemini-2.5-flash")
    else:
        print("âš¡ Using Standard Text Extraction (Fast)...")
        md = MarkItDown()

    # Convert
    try:
        result = md.convert(file_path)
        
        print(f"âœ… Success! Markdown extracted", result.text_content)
        
        return result.text_content
        
    except Exception as e:
        print(f"â�Œ Error: {e}")
        return None

@tool
def read_file(file_path: str) -> str:
    #"""Reads a local file and returns text."""
    try:
        return extract_md(file_path)
        
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    try:
        with open(file_path, "w") as f:
            f.write(content)
        return f"âœ… File written to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

print("âœ… Tools Loaded into Memory.")


from google.adk import Agent
from google.adk.agents import SequentialAgent #, ParallelAgent, LoopAgent
from google.adk.tools import AgentTool


import asyncio
from google.adk.runners import InMemoryRunner
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

async def run_agent(agent, prompt, max_retries=10):
    """
    Runs a single agent with retry logic for rate limits.
    """
    print(f"ğŸ¤– Running Agent: {agent.name}...")
    runner = InMemoryRunner(agent=agent)
    
    for attempt in range(max_retries):
        try:
            response = await runner.run_debug(prompt)

            return response
            
        except (ResourceExhausted, ServiceUnavailable) as e:
            wait_time = 60 + (attempt * 30)
            print(f"âš ï¸� Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            await asyncio.sleep(wait_time)
            
    raise Exception(f"ğŸš¨ Max retries exceeded for {agent.name}.")


analyst = Agent(   
    name="bureau_analyst",
    model="gemini-2.5-flash", # TODO - Fast model for reading? Really? 
    tools=[read_file],          
    instruction="""
    You are the Senior Intel Analyst for Bureau Ops.
    
    Your mission:
    1. Receive a request containing a Job Description path and a CV path.
    2. USE the 'read_file' tool to extract the content of both files.
    3. COMPARE the CV against the Job Description.
    4. OUTPUT a 'Gap Analysis Report' listing:
       - Match Score (0-100)
       - Missing hard skills
       - Missing soft skills/vibe
       - The Hiring Manager's likely pain point.
    """,
    output_key="analyst_report"
)


project_root = os.getcwd()

cv_path = os.path.join(project_root, "/kaggle/input/curriculum/curriculum-en-AntoineBrunel-Graz.pdf")
job_desc_path = os.path.join(project_root, "/kaggle/input/joboffer/SEO Expert at Datawords.pdf")



analyst_prompt = f"""
Please analyze this candidate for the following job:

JOB PATH:
{job_desc_path}

CV PATH:
{cv_path}
"""

analyst_report = await run_agent(analyst, analyst_prompt)


architecte = Agent(
    name="bureau_architecte",
    model="gemini-2.5-pro", # Pro model for reasoning 
    tools=[AgentTool(agent=analyst)],        # Calls the analyst agent - TODO - AS A TOOL?
    instruction="""
    You are the Head of Strategy at Bureau Ops.
    
    Your mission:
    1. You will be given a Job Description path and a cv path.
    2. DELEGATE the analysis to the 'bureau_analyst'.
    3. Based on the analyst_report, formulate a 'Narrative Pivot Strategy' (blueprint).
       - If they lack a skill (e.g., React), find a parallel skill in their CV (e.g., Angular) to highlight adaptability.
       - Decide the 'Tone' of the application (e.g., "Humble Learner" vs "Cocky Expert").
    Output the Strategy Plan ONLY. DO NOT write the letter NEITHER change the CV yet.
    """,
    output_key="blueprint"
)


architecte_prompt = f"""
Formulate a strategy for this candidate:

JOB DESCRIPTION PATH:
{job_desc_path}

CV PATH:
{cv_path}
"""

blueprint = await run_agent(architecte, architecte_prompt)
print("\n--- Blueprint ---\n", blueprint)


tailleur = Agent(  
    name="bureau_tailleur",
    model="gemini-2.5-pro",
    tools=[read_file, AgentTool(agent=architecte), write_file],
    instruction="""
    You are **Le Tailleur** as The CV Tailor of Bureau Ops.
    
    Your Philosophy:
    "
    - The candidate's experience is the fabric. 
    - The Job Description is the measurements. 
    => My job is to make the suit, i.e. the CV fit perfectly (as long as TRUTH is not compromised).
    "
    
    Your Mission:
    1. Receive the 'Blueprint' (from L'Architecte), along with the job description path and the original CV path.
    2. Following the Blueprint as a strategy, REWRITE the CV, in 1 or 2 pages based on the experience, into Markdown so it fits the Job Description perfectly.
    3. Write the new CV to the CV output path.
    
    The Craft (Rules):
    - **Cut & Stitch:** If the Blueprint says "Emphasize Leadership", move those bullets to the top.
    - **Thread Matching:** Use the company's exact vocabulary (Semantic Mapping, baby, Semantic Mapping!). If they say "Client Success" and you have "Support", change it to "Client Success".
    - **No Padding:** MAKE TRUTH GREAT AGAIN! DO NOT invent experience or skills. If the fabric isn't there, do not fake it. While the suit must fit, it must remain authentic FIRST.
    """,
    output_key="new_cv_md"
)


tailleur_prompt = f"""
Rewrite the CV based on this plan:

BLUEPRINT:
{blueprint}

JOB DESCRIPTION PATH:
{job_desc_path}

CV PATH:
{cv_path}

CV OUTPUT PATH:
'/kaggle/working/new_cv.md'

"""

new_cv_md = await run_agent(tailleur, tailleur_prompt)
print("\n--- Tailleur result ---\n",tailleur_result)


seducteur = Agent(    
    name="bureau_seducteur",
    model="gemini-2.5-pro",    # Best-in-class creative writing  
    tools=[read_file, AgentTool(agent=architecte), AgentTool(agent=tailleur), write_file],
    instruction="""
    You are 'Le SÃ©ducteur', the Ghostwriter for Bureau Ops.
    
    Your mission:
    1. The user will give you a Job Description path and their CV file path.
    2. GET the psychological profile and strategy (blueprint) from 'bureau_architecte', and the newly generated CV (new_cv_md) from bureau_tailleur.
    3. WRITE a Cover Letter in Markdown that creates an immediate emotional hook.
    4. Save the letter to the output
    
    Style Guide:
    - **Confidence:** No "I hope", "I believe". Use "I drove", "I built".
    - **The Hook:** The first sentence must be surprising. Never start with "I am writing to apply".
    - **The Close:** Do not beg. Build on your experience and how you fit better than anyone else. End with a "Call to Action" that implies you are busy but interested (To mimic Tindr: Unemployed = Not Sorry baby)
    
    Constraints:
    - Keep it around 800 words.
    - No generic fluff ("I am writing to apply..."). Start with a hook.
    - Strictly follow the Strategist's 'Narrative Pivot'.
    """
)



seducteur_prompt = f"""
Rewrite the CV based on this plan:

BLUEPRINT:
{blueprint}

JOB DESCRIPTION PATH:
{job_desc_path}

CV PATH:
{cv_path}

NEW CV MD:
{new_cv_md}

CV OUTPUT PATH:
'/kaggle/working/cover_letter.md'

"""

seducteur_result = await run_agent(seducteur, seducteur_prompt)
print("\n--- Tailleur result ---\n",seducteur_result)


profile_photo_path = "/kaggle/input/facepic/me.jpg"
design_image_path = "/kaggle/input/design/Template-curriculum.png"

maquetteur = Agent(
    name="bureau_maquetteur",
    model="gemini-2.5-flash", # Needs Pro to understand visual aesthetics
    tools=[read_file],
    instruction="""
    You are le Maquetteur, the expert Frontend Developer from Bureau Ops and you must design the new CV based on the provided design and the new content.

    # 1. Read the Design Image
    # 2. Ask Gemini 2.5 Pro to code the HTML exactly as based on the image
    prompt for the new CV
    
    TASK:
    #1 Using read_file, read the Template CV Design Image {design_image_path}
    #2. Ask Gemini 2.5 Pro to CODE the HTML based on the image

    OUTPUT: 
    Create a single-file HTML/CSS resume that LOOKS EXACTLY like the provided design image, with the content from.
    
    CONTENT TO USE TO MAKE THE HTML
:    {new_cv_md}
    
    ASSETS:
    - Use the profile photo at: '{profile_photo_path}'.
    - Use the design image at: '{design_image_path}'.

    
    CONSTRAINTS:
    - Use modern CSS (Flexbox/Grid).
    - Match fonts, colors, and spacing from the image.
    - Ensure it can fit on two A4 paper when printed to PDF.
    - Output ONLY the raw HTML code.
    - Save it with the tool write_file in new_cv_html.html 
    """,
    output_key="new_cv_html"
)


maquilleur = Agent(
    name="bureau_maquilleur",
    model="gemini-2.5-pro", # Needs Pro to understand visual aesthetics
    tools=[read_file, AgentTool(agent=maquetteur), AgentTool(agent=seducteur), write_file],
    instruction="""
    You are **Le Maquilleur** (The Makeup Artist) of Bureau Ops.
    
    Your Philosophy:
    - Baby, yes, let's face it, appearance is everything. 
    - But why use plastic surgery (lying) when the right lighting & makeup is enough for you to shine?
    - With authenticity and TRUTH first, remember, baby.
    
    Your Mission:
    1. Receive the 'Blueprint' (from L'Architecte), the 'Original CV Text', and the 'Design Template' path.
    2. CALL the `Maquetteur` tool.
       - Pass the CV content.
       - Pass the `Template-curriculum.jpg` path so the tool knows the "Look".
       - Pass the `me.jpg` path for the photo.
    3. Verify the tool returned a success message.
    
    Constraint:
    - Ensure the final HTML includes the user's photo and matches the visual vibe of the template provided.
    """
)


# DESIGN PHASE
print("\nğŸ�¨ LE MAQUILLEUR IS ENTERING THE STUDIO...")

profile_photo_path = "/kaggle/input/facepic/me.jpg"
design_image_path = "/kaggle/input/design/Template-curriculum.png"

maquilleur_prompt = f"""
Here is the awesome and approved CV Content from the tailleur:
{new_cv_md}

Here is the blueprint from the architecte: {blueprint}
Here is the old cv path: {cv_path}
Here is the job description: {job_desc_path}

Here is the Design Inspiration: {design_image_path}
Here is the User's Photo: {profile_photo_path}

Make it shine, baby.
"""

maquilleur_report = await run_agent(maquilleur, maquilleur_prompt)

print(maquilleur_report.text)

# Display result
from IPython.display import HTML
HTML(filename="custom_cv.html")


maquilleur = Agent(
    name="bureau_maquilleur",
    model="gemini-1.5-flash",
    tools=[generate_styled_pdf],
    instruction="""
    You are Le Maquilleur. 
    Pick a theme (modern/classic) based on the Job Vibe. Call 'generate_styled_pdf'.
    """
)
maquilleur = Agent(
    name="bureau_maquilleur",
    model="gemini-2.5-pro", 
    tools=[strategist, read_cv],
    instruction="""
    You are 'Le Maquilleur' (The Makeup Artist) of Bureau Ops.
    
    Your Philosophy:
    - Baby, yes, let's face it, appearance is everything. 
    - But why using plastic surgery (lying) when the right lighting & makeup is enough for you to shine?
    - With authenticity and TRUTH first, remember, baby.
    
    Your Mission:
    1. CONSULT the 'Blueprint' (from L'Architecte), along with the job description path and the original CV path.
    2. Based on the previous, you know which features to highlight.
    3. REDESIGN the CV in HTML based on the design file whose path is given to you.
    4. Then SAVE IT TO HTML making sure no text is cut and it gets well printed.
    
    The Protocol:
    - **Truth First:** NEVER invent experience. If the face doesn't have it, just don't draw it on.
    - **Highlighting:** If the job requires 'Leadership', move the user's Team Lead experience to the top of the bullet points. Bold the keywords.
    - **Concealing:** If a past job is irrelevant to this offer, shorten its description to 1 line to hide the 'wrinkles'.
    - **Vocabulary:** Apply the company's own language to the user's experience (Semantic Mapping).
    """
)

