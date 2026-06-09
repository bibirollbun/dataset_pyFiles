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


!pip -q install google-adk 


from typing import Any, Dict
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.runners import Runner, InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.tools.tool_context import ToolContext
from google.adk.models.google_llm import Gemini
from google.genai import types
import os
import json
import requests
import subprocess
import time
import uuid
import base64
from PIL import Image
from IPython.display import Markdown, display
from google.colab import userdata
import google.generativeai as genai

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
NVIDIA_API_KEY = userdata.get("NVIDIA_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY




# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")




retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)





APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"




# Step 1: Create the LLM Agent

def ContentCreator():
    """
    Generates a refined one-liner and hashtags using Gemini 2.5 Flash Lite.

    Workflow:
        1. Retrieves API key from environment/Colab userdata.
        2. Takes user input for a topic.
        3. Sends structured prompt to Gemini.
        4. Returns results in a dictionary structure with error catching.

    Returns:
        dict: {
            "success": bool,
            "refined_text": str or None,
            "hashtags": list or None,
            "error": str or None
        }
    """
    try:
        # --- 1. Load API Key ---
        # API key is already configured globally, no need to get it again here
        # But keeping this for local execution compatibility if ContentCreator is called directly.
        GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
        #if not GOOGLE_API_KEY:
            #return {
               # "success": False,
                #"refined_text": None,
                #"hashtags": None,
                #"error": "Google API key not found. Set GOOGLE_API_KEY in Colab userdata."
            #}

        genai.configure(api_key=GOOGLE_API_KEY)

        MODEL_NAME = "gemini-2.5-flash-lite"


        # --- 2. Ask user input ---
        user_topic = input("Enter your topic or rough text: ").strip()
        #user_topic = user_topic.strip()

        if not user_topic:
            return {
                "success": False,
                "refined_text": None,
                "hashtags": None,
                "error": "No topic provided."
            }

        # --- 3. Prepare prompt ---
        system_prompt = (
            "You are an assistant that creates social-media-ready content. "
            "Given a user's rough topic, produce:\n"
            "A refined one-sentence version of the topic.\n"
            "A short list of relevant hashtags.\n"
            "Return them in a clean and structured format:\n"
            "Refined: <text>\n"
            "Hashtags: #tag1 #tag2 #tag3"
        )

        full_prompt = f"{system_prompt}\n\nUser topic: {user_topic}"

        # --- 4. Call Gemini ---
        try:
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            #model =  MODEL_NAME
            response = model.generate_content(full_prompt)
        except Exception as api_error:
            return {
                "success": False,
                "refined_text": None,
                "hashtags": None,
                "error": f"API Error: {api_error}"
            }

        # --- 5. Extract model output ---
        raw_output = response.text.strip() if getattr(response, "text", None) else ""

        if not raw_output:
            return {
                "success": False,
                "refined_text": None,
                "hashtags": None,
                "error": "Received empty response from model."
            }

        # --- 6. Parse structured output ---
        refined_text = None
        hashtags = []

        for line in raw_output.split("\n"):
            line = line.strip()
            if line.lower().startswith("refined"):
                refined_text = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("hashtags"):
                hashtag_text = line.split(":", 1)[-1].strip()
                hashtags = hashtag_text.split()

        return {
            "success": True,
            "refined_text": refined_text,
            "hashtags": hashtags,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "refined_text": None,
            "hashtags": None,
            "error": f"Unexpected Error: {e}"
        }






def DigitalArtist(gen_prompt: str) -> dict:
    """
    Generates an image using NVIDIA Stable Diffusion XL API.

    Args:
        gen_prompt (str): Text prompt for image generation.

    Returns:
        dict: {
            "success": bool,
            "image_path": str or None,
            "error": str or None
        }
    """
    try:
        # --- 1. Input Validation ---
        if not isinstance(gen_prompt, str) or not gen_prompt.strip():
            return {
                "success": False,
                "image_path": None,
                "error": "Prompt must be a non-empty string."
            }

        #gen_prompt == {topic}


        gen_prompt = gen_prompt.strip()

        # --- 2. Load NVIDIA API Key ---
        nvidia_api_key = userdata.get("NVIDIA_API_KEY")
        if not nvidia_api_key:
            return {
                "success": False,
                "image_path": None,
                "error": "NVIDIA API key not found. Set 'NVIDIA_API_KEY' in Colab userdata."
            }

        invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl"

        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Accept": "application/json",
        }

        payload = {
            "text_prompts": [{"text": gen_prompt, "weight": 1}],
            "seed": 0,
            "sampler": "K_EULER_ANCESTRAL",
            "steps": 25,
            "height": 1024,
            "width": 1024,
            "cfg_scale": 5,
            "samples": 1,
            "clip_guidance_preset": "NONE",
            "style_preset": "none"
        }

        # --- 3. Call NVIDIA API ---
        try:
            response = requests.post(invoke_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as req_err:
            return {
                "success": False,
                "image_path": None,
                "error": f"Network/API error: {req_err}"
            }

        # --- 4. Parse JSON Response ---
        try:
            response_body = response.json()
            base64_img = response_body["artifacts"][0]["base64"]
        except Exception as parse_err:
            return {
                "success": False,
                "image_path": None,
                "error": f"Failed to parse image output: {parse_err}"
            }

        # --- 5. Decode Image ---
        try:
            img_data = base64.b64decode(base64_img)
        except Exception as decode_err:
            return {
                "success": False,
                "image_path": None,
                "error": f"Base64 decode error: {decode_err}"
            }

        # --- 6. Save Image File ---
        unique_name = f"output_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(".", unique_name)

        try:
            with open(save_path, "wb") as f:
                f.write(img_data)
            # Test that image can open
            Image.open(save_path).verify()
        except Exception as save_err:
            return {
                "success": False,
                "image_path": None,
                "error": f"Image save/open error: {save_err}"
            }

        return {
            "success": True,
            "image_path": save_path,
            "error": None
        }

    except Exception as e:
        # Catch-all for unexpected issues
        return {
            "success": False,
            "image_path": None,
            "error": f"Unexpected error: {e}"
        }





#import base64
#from IPython.display import Markdown

def get_media(image_path: str, show: bool = True) -> dict:
    """
    Generates Markdown output embedding a base64-encoded image while preserving
    literal `{}` placeholders (e.g., {{title}}). Optionally displays the Markdown.

    Args:
        image_path (str): Path to the image file.
        show (bool): Whether to display the Markdown in notebook/Colab.

    Returns:
        dict: {
            "success": bool,
            "markdown": str or None,
            "error": str or None
        }
    """
    image_path = image_path
    try:
        # --- 1. Validate image path ---
        if not isinstance(image_path, str) or not image_path.strip():
            return {
                "success": False,
                "markdown": None,
                "error": "Image path must be a non-empty string."
            }

        # --- 2. Encode image to base64 ---
        try:
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            return {
                "success": False,
                "markdown": None,
                "error": f"File not found: {image_path}"
            }
        except Exception as read_err:
            return {
                "success": False,
                "markdown": None,
                "error": f"Error reading file: {read_err}"
            }

        # --- 3. Build Markdown with preserved placeholders ---
        try:
            md = """
<img src="data:image/jpg;base64,{data}" width="600" height="480" class="center-img">

####

{{"refined_text"}}


""".format(data=data)
        except Exception as format_err:
            return {
                "success": False,
                "markdown": None,
                "error": f"Markdown formatting error: {format_err}"
            }

        # --- 4. Display or just return as dict ---
        if show:
            try:
                display(Markdown(md))
            except Exception as display_err:
                return {
                    "success": False,
                    "markdown": md,
                    "error": f"Display error: {display_err}"
                }

        return {
            "success": True,
            "markdown": md,
            "error": None
        }

    except Exception as e:
        # Catch-all for unexpected errors
        return {
            "success": False,
            "markdown": None,
            "error": f"Unexpected error: {e}"
        }


display(Markdown("markdown"))
# ==========================================================
# AGENTS (WelcomeAgent -> MediaExec -> Finisher -> root_agent)
# ==========================================================
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],) # Retry on these HTTP errors)


WelcomeAgent = Agent(
    name="WelcomeAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A media creation initiator to create text and image on user's query.",
    instruction= """You are to help user for creating social media content. At the start, you MUST
    call 'ContentCreator()' tool. 'ContentCreator()'asks for user-topic. With the input
    provided by user,  'ContentCreator()' returns a dictionary with keys "success","refined_text",
    "hashtags", and "error".Check for clarity and consistencies, and append the :{"hashtags"} at the end
    of the :{"refined_text"}. Your output key will be named "refined_text". """,
    tools=[ContentCreator],
    output_key="refined_text",
    )


MediaExec = Agent(
    name="MediaExec",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction=""" You will call 'DigitalArtist()' tool and pass to it's arg :{'refined_text'} for use as 'gen_prompt' for image generation. 'DigitalArtist()'returns
    a dictionary with keys "success", "image_path", "error". Your output will be "image_path".""",
    tools=[DigitalArtist],
    output_key="image_path",
    )



FinisherAgent = Agent(
    name="FinisherAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
        ),
    instruction="""
    (1)You will call 'get_media()' tool and pass to it in it's arg :{'image-path'}.
    (2)Next, you will take :{"refined_text"} and assign a suitable short Title to it.
    This is now "refined_text".
    (3)Your output will be "markdown", which will contain :{"refined_text"} in {{"refined_text"}},
    below the image.""",
    tools=[get_media],
    output_key="markdown",
    )


root_agent = SequentialAgent(
    name="MediaCoordinator",
    sub_agents=[WelcomeAgent, MediaExec, FinisherAgent],
)


# Step 2: Set up Session Management
# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()


# Step 3: Create the Runner
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

print("âœ… Stateful agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")



await run_session(
    runner,
    [
        "Hi, help me creating social-campaign material. "


    ],
    "stateful-agentic-session",
)






APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"


#InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()

# Step 3: Create the Runner
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

print("âœ… Stateful agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")

