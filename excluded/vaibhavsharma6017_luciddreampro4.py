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


# 1. Uninstall the conflicting libraries first to clear the slate
!pip uninstall -y diffusers transformers accelerate peft

# 2. Reinstall everything fresh, including 'peft' to stop the error
!pip install -U diffusers transformers accelerate scipy safetensors peft google-generativeai langchain-google-genai langchain


import torch
from diffusers import StableDiffusionPipeline
print("✓ Library Imported Successfully!")


# --- SETUP IMAGE GENERATION PIPELINE ---
print("Loading Stable Diffusion model... this may take a few minutes.")

model_id = "runwayml/stable-diffusion-v1-5" # A good, reliable base model

# Load the pipeline onto the GPU (cuda) using float16 to save memory
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# Optional: reduces memory usage slightly
pipe.enable_attention_slicing() 

print("✓ Model loaded successfully on GPU!")


from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
import matplotlib.pyplot as plt

# Define the input schema for the tool so the agent knows what to send it
class ImageGenInput(BaseModel):
    enhanced_prompt: str = Field(description="A highly detailed, artistically descriptive prompt for generating a surreal image.")

@tool(args_schema=ImageGenInput)
def generate_dream_image(enhanced_prompt: str):
    """Useful for visualizing a segment of a dream. Input must be a detailed visual prompt."""
    print(f"\n--- [TOOL] Generating image for: '{enhanced_prompt}' ---")
    
    # --- THE REAL GENERATION STEP ---
    # The agent calls the loaded pipeline. 
    # num_inference_steps=30 is a balance between speed and quality.
    image = pipe(enhanced_prompt, num_inference_steps=30).images[0]
    
    # Save the image to the Kaggle working directory
    # We use a timestamp so images don't overwrite each other
    import time
    timestamp = int(time.time())
    filename = f"/kaggle/working/dream_{timestamp}.png"
    image.save(filename)
    
    print(f"✓ Image generated and saved to: {filename}")
    
    # Important: The tool must return a string to the agent, not an image object.
    # The agent can't "see" the image, it just needs to know it worked.
    return f"Successfully visualized the dream segment based on the prompt: {enhanced_prompt}. The image is saved at {filename}."

# Add it to your toolkit list
tools = [generate_dream_image]


# Use your GOOGLE_API_KEY setup here as we learned in the course
import os
from google.colab import userdata
# os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY') # If using Colab
# If in Kaggle, use the Add-ons -> Secrets menu to add our key and load it:
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("YOUR_SECRET_NAME")


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# 1. Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8) # High temp for creativity

# 2. Define Persona
SYSTEM_PROMPT = """
You are the 'Lucid Dream Architect.' Your purpose is to take fragments of human dreams and visualize them as surreal art.
Your aesthetic is 'Into the Spider-Verse' meets Salvador Dalí—glitch art, deep colors, shifting perspectives.
YOUR PROCESS:
1. Receive user input (a dream fragment).
2. THOUGHT: Plan how to enhance this into your specific artistic style.
3. ACTION: Use the generate_dream_image tool. The input to the tool MUST be a highly descriptive, artistic prompt. Do not just repeat the user input. Add details about lighting, style, and mood.
4. OBSERVATION: Wait for the tool to confirm the image is generated.
5. FINAL RESPONSE: Narrate the scene briefly and ask the user "What happens next?" to continue the dream loop.
"""

# 3. Setup standard ReAct agent (using a default prompt template for speed)
prompt = hub.pull("hwchase17/react")
# We prepend our system prompt to the standard instructions
prompt.template = SYSTEM_PROMPT + "\n\n" + prompt.template

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


from kaggle_secrets import UserSecretsClient
import os

# Reload the key from the Secrets menu
try:
    user_secrets = UserSecretsClient()
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
    print("✅ Key successfully re-loaded!")
except:
    print("❌ Error: Check your Add-ons -> Secrets menu.")


# Use the newer Flash model which is faster and currently active
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.7,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Re-attach the agent (Run this part again too)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Run the test
print("--- Attempting with Gemini 1.5 Flash ---")
response = agent_executor.invoke({"input": "I dreamt I opened a door floating in the sky."})
print(response["output"])


# Define a new, scenic dream input
scenery_dream = "I dreamt I was walking through an ancient forest where the trees were made of glowing stained glass and the ground was a mirror reflecting a purple sky."

print(f"--- Inputting new dream: '{scenery_dream}' ---")

# Run the agent loop again
response_scenery = agent_executor.invoke({"input": scenery_dream})

print("\n--- Final Agent Response ---")
print(response_scenery["output"])

