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
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
import asyncio
import json
import re
import google.genai as genai


import google.generativeai as genai
from PIL import Image
from io import BytesIO
import base64

print("âœ… ADK components imported successfully.")


# Define helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML, Markdown
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


drafting_agent = LlmAgent(
    name="StoryCreator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    tools=[], 
    instruction="""
    Your task is to draft a 2-page story based on an Initial Story Idea provided by the user. 
    
    You MUST ALWAYS follow these steps:
    1. FIRST, check if any feedback from the 'revision_agent' is provided in the input. If so, carefully analyze the feedback and incorporate it by revising the previously drafted plot and story elements, or by adjusting your brainstorming process. Prioritize this feedback.
    2. Then, brainstorm a captivating plot for a 2-page story, including a beginning, rising action, climax, falling action, and resolution. Ensure this brainstorm incorporates any received feedback.
    3. Next, write the complete 2-page story based on the brainstormed plot, ensuring proper pacing and character development.
    4. Return the complete 2-page story.
    """,
)
print("âœ… Drafting Agent defined.")


revision_agent = LlmAgent(
    name="ContentReviewer",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Your task is to review drafted stories and artwork descriptions. You will receive input containing a Story, and optionally, Artwork Descriptions.

    You MUST ALWAYS follow these steps:

    1. **Analyze the Input:** Determine if you have received only a 'Story', or both a 'Story' AND 'Artwork Descriptions'.

    2. **CASE A: Story Only** (No artwork descriptions provided)
       - Perform an initial story-only review. Evaluate if the story has a clear plot, engaging characters, and proper pacing.
       - If improvements are needed, start your response with 'FEEDBACK_DRAFTING:' followed by constructive critique.
       - If the story is high quality, return ONLY the string: 'STORY_APPROVED'

    3. **CASE B: Story AND Artwork Descriptions**
       - Perform a comprehensive review. Evaluate the story's narrative quality AND check if the artwork descriptions accurately capture the significant events and are detailed enough for generation.
       - If story improvements are needed, return 'FEEDBACK_DRAFTING: [details]'.
       - If artwork improvements are needed, return 'FEEDBACK_ARTWORK: [details]'.
       - If BOTH are good, return ONLY the string: 'COMPREHENSIVELY_APPROVED'
    """,
    tools=[],
)
print("âœ… Revision Agent defined.")


artwork_agent = LlmAgent(
    name="ArtworkGenerator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Your task is to create artwork descriptions for a given 2-page story. You will receive the complete story as input.

    You MUST ALWAYS follow these steps:
    1. FIRST, check if any feedback from the 'revision_agent' is provided. If so, carefully analyze the feedback and incorporate it by revising your process for identifying significant events or generating image prompts. Prioritize this feedback.
    2. Then, read the 2-page story carefully.
    3. For each page, identify the single most significant event that would make for a compelling image.
    4. For each identified event, generate a detailed prompt for a text-to-image model. The prompt should be descriptive enough to produce a high-quality, relevant image.
    5. Return a list of dictionaries, where each dictionary contains the page number, a brief description of the event, and the detailed image generation prompt for that event.

    IMPORTANT OUTPUT FORMAT:
    You MUST enclose the entire list of dictionaries as a JSON array within a markdown code block using the ```json ... ``` format.
    
    Example:
    ```json
    [ 
      { "page": 1, "event": "description...", "image_prompt": "detailed prompt..." }, 
      { "page": 2, "event": "description...", "image_prompt": "detailed prompt..." } 
    ]
    ```
    """,
    tools=[],
)
print("âœ… Artwork Agent defined.")


# --- CONFIGURATION: SELECT YOUR MODEL ---

# OPTION 1: Standard (Nano Banana) - Fast, efficient
IMAGE_MODEL_NAME = "gemini-2.5-flash-image"

# OPTION 2: Pro (New) - Higher fidelity, better complex prompt adherence
#IMAGE_MODEL_NAME = "gemini-3-pro-image-preview" 

print(f"ğŸ�¨ using Image Model: {IMAGE_MODEL_NAME}")

# Initialize the model
image_model = genai.GenerativeModel(IMAGE_MODEL_NAME)

async def generate_illustration(prompt: str, page_num: int):
    print(f"   ğŸ�¨ Painting Page {page_num} using {IMAGE_MODEL_NAME}...")
    
    try:
        # For Gemini models, we ask for the image in the prompt
        # and standard generate_content handles it.
        response = image_model.generate_content(
            contents=prompt,
            # We explicitly ask for image/jpeg generation if the API supports media_resolution configs,
            # but usually the model name implies the capability.
        )
        
        # Parse the response
        # Gemini usually returns the image as an inline data part
        if response.parts:
            for part in response.parts:
                if part.inline_data:
                    # Decode the image data
                    img_data = part.inline_data.data
                    # If it's raw bytes, we can use it directly
                    # If it comes as a base64 string, we might need to decode, 
                    # but the SDK usually handles 'part.inline_data.data' as bytes.
                    img = Image.open(BytesIO(img_data))
                    return img
                    
        print(f"   [!] No image data found in response for Page {page_num}.")
        return None

    except Exception as e:
        print(f"   [!] Failed to generate image for Page {page_num}: {e}")
        return None

print("âœ… Image Generator Ready.")


# --- HELPER 1: Extract Text from Agent Result ---
def get_agent_text(result):
    """
    Safely extracts text from the ADK run_debug result.
    run_debug returns a list of events/steps. We want the last model response.
    """
    if isinstance(result, list):
        # Get the last item (usually the final response)
        last_item = result[-1]
        
        # If it's an object with a 'text' attribute (ADK Event object)
        if hasattr(last_item, 'text'):
            return last_item.text
        # If it's a dictionary (JSON representation)
        elif isinstance(last_item, dict) and 'text' in last_item:
            return last_item['text']
        # If it's just a string
        elif isinstance(last_item, str):
            return last_item
        # Fallback: Convert object to string
        return str(last_item)
    return str(result)

# --- HELPER 2: Parse JSON from Artwork Agent ---
def parse_artwork_response(response_text):
    try:
        # Look for JSON inside markdown blocks first
        json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        # Fallback: try parsing the raw text
        return json.loads(response_text)
    except Exception as e:
        print(f"   [!] JSON Parsing failed. Raw text: {response_text[:100]}...")
        return []

# --- MAIN PIPELINE FUNCTION ---
async def run_creative_pipeline(user_idea: str, max_iterations: int = 3):
    print(f"ğŸš€ Starting Creative Pipeline for: '{user_idea}'")

    current_story = ""
    current_artwork = []
    drafting_feedback = None
    artwork_feedback = None
    iteration = 1
    
    # --- PHASE 1: TEXT & PROMPT REFINEMENT ---
    while iteration <= max_iterations:
        print(f"\n--- ğŸ”„ ITERATION {iteration}/{max_iterations} ---")

        # 1. DRAFTING
        if iteration == 1 or drafting_feedback:
            drafting_input = f"Initial Story Idea: {user_idea}"
            if drafting_feedback:
                print(f"   ğŸ“� Re-drafting story...")
                drafting_input += f"\n\nFeedback from Revision Agent: {drafting_feedback}"
            else:
                print("   ğŸ“� Drafting initial story...")

            drafting_runner = InMemoryRunner(agent=drafting_agent)
            result_list = await drafting_runner.run_debug(drafting_input)
            current_story = get_agent_text(result_list)
        else:
            print("   â�© Skipping re-drafting.")

        # 2. ARTWORK PROMPTS
        print("   Thinking about artwork...")
        artwork_input = f"Story: {current_story}"
        if artwork_feedback:
            artwork_input += f"\n\nFeedback from Revision Agent: {artwork_feedback}"

        artwork_runner = InMemoryRunner(agent=artwork_agent)
        result_list = await artwork_runner.run_debug(artwork_input)
        raw_artwork_response = get_agent_text(result_list)
        current_artwork = parse_artwork_response(raw_artwork_response)

        # 3. REVISION
        print("   ğŸ”� Reviewing content...")
        artwork_str = json.dumps(current_artwork, indent=2)
        revision_input = f"Story: {current_story}\n\nArtwork Descriptions: {artwork_str}"
        
        revision_runner = InMemoryRunner(agent=revision_agent)
        result_list = await revision_runner.run_debug(revision_input)
        review_result = get_agent_text(result_list).strip()
        print(f"   ğŸ¤– Verdict: {review_result}")

        # 4. DECISION
        if "COMPREHENSIVELY_APPROVED" in review_result:
            print("\nâœ… PHASE 1 COMPLETE: Story and Prompts Approved.")
            break
        elif "FEEDBACK_DRAFTING:" in review_result:
            drafting_feedback = review_result.replace("FEEDBACK_DRAFTING:", "").strip()
            artwork_feedback = None 
            print("   âš ï¸�  Drafting issues detected.")
        elif "FEEDBACK_ARTWORK:" in review_result:
            artwork_feedback = review_result.replace("FEEDBACK_ARTWORK:", "").strip()
            drafting_feedback = None 
            print("   âš ï¸�  Artwork issues detected.")
        else:
            print(f"   â�“ Unexpected response. Stopping.")
            break
        iteration += 1

    # --- PHASE 2: IMAGE GENERATION ---
    print("\n--- ğŸ�¨ PHASE 2: GENERATING IMAGES ---")
    generated_images = []
    
    if current_artwork:
        for item in current_artwork:
            page_num = item.get('page')
            prompt = item.get('image_prompt')
            event_desc = item.get('event')
            
            # Call the image generator helper (ensure you ran the previous cell!)
            img = await generate_illustration(prompt, page_num)
            
            if img:
                generated_images.append({
                    "page": page_num,
                    "event": event_desc,
                    "image_obj": img
                })
    else:
        print("   [!] No artwork prompts found to generate.")

    return current_story, generated_images

print("âœ… Orchestrator logic loaded.")


# Cell: Run & Display
USER_PROMPT = "A cyberpunk detective solves a crime involving a missing robotic cat."

# Run the pipeline
final_story, final_images = await run_creative_pipeline(USER_PROMPT)

# Display Output
display(Markdown(f"# ğŸ“œ Generated Story"))
display(Markdown(final_story))
display(Markdown("---"))

display(Markdown(f"# ğŸ–¼ï¸� Illustrations"))

for item in final_images:
    display(Markdown(f"### Page {item['page']}: {item['event']}"))
    # Display the PIL Image object directly
    display(item['image_obj'])
    display(Markdown("---"))

