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


# --- VisionAI Companion: Accessibility Project ---

print("# The VisionAI Companion Project: An Accessible AI Powered by Gemini")
print("")
print("Competition: Google DeepMind - Vibe Code with Gemini Pro in AI Studio")
print("Project: This is an intelligent, multimodal assistant built specifically for people with visual impairments.")
print("")
print("Quick Links:")
print("Live Demo (Go try it!): [Try VisionAI Companion](https://ai.studio/apps/drive/105oIMifoYdab0uUkio0pawFeMR50DLvM?fullscreenApplet=true )")
print("Video Demo (Walkthrough): [Watch on YouTube](https://www.youtube.com/watch?v=iZS_SLx-mS4 )")
print("--------------------------------------------------------------------------------")


print("## Overview: Building a True Companion")
print("")
print("The goal was to harness Gemini's huge multimodal power to provide robust, comprehensive assistance to the 285 million people worldwide who are visually impaired.")
print("")
print("### The 5 Core Capabilities I Focused On:")
print("1. Scene Understanding: It handles the 'where am I?' and spatial awareness.")
print("2. Text Reader: It can read everything, from clean print to messy handwriting (OCR).")
print("3. Object Identifier: Essential for safety and daily tasks—identifying products and medication.")
print("4. Navigation Assistant: Gives real-time, clear guidance for moving around.")
print("5. Social Scene Analyzer: Helps interpret social cues and context (e.g., body language).")
print("")
print("### The Tech Stack Innovation:")
print("- I relied on native multimodality (vision + reasoning + language).")
print("- It uses a large context window (1M tokens) for smooth, continuous conversations.")
print("- The spatial reasoning is advanced for accurate navigation.")


# Set Up: Time to install the packages we need.
!pip install -q --upgrade google-generativeai pillow requests


# Configuration: Setting up the Gemini API
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import os
from PIL import Image
import requests
from io import BytesIO


# Getting my API key safely from Kaggle secrets—never hardcode keys!
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY")

# Configure the client
genai.configure(api_key=api_key)

# Using Gemini 2.5 Flash for its speed and great multimodal capabilities.
MODEL_NAME = "gemini-2.5-flash"

print(f"Gemini API configured successfully with model: {MODEL_NAME}")


print("## Module 1: Scene Understanding")
print("This module is the spatial brain. It takes an image of a room or environment and describes it in detail for safe movement.")

def analyze_scene_preamble(image_input):
    # This helper handles the initial setup and image loading logic
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Check if we got a URL and need to fetch the image first
    if isinstance(image_input, str):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content))
    else:
        image = image_input
    return model, image

def analyze_scene(image_input):
    """
    Analyzes a scene for spatial understanding and navigation.
    
    Args:
        image_input: PIL Image or image URL
        
    Returns:
        Detailed scene description with spatial information
    """
    model, image = analyze_scene_preamble(image_input)
    # The actual function call is in the next block to split the code logically.
    
    # Returning the result of the main function logic
    return analyze_scene_complete(model, image)


# Continuing the definition of the analyze_scene function with the main prompt logic
def analyze_scene_complete(model, image):
    prompt = """You are an AI assistant helping a person with visual impairment navigate their environment.
    
Analyze this image and provide:
1. Overall scene description (what type of space is this?)
2. Objects present and their locations (use clock positions and distances)
3. Potential obstacles or hazards
4. Safe paths for navigation
5. Spatial relationships between objects

Be specific with directions (left, right, ahead) and estimated distances.
Format your response to be clear and actionable for someone who cannot see."""
    
    # Generate response
    response = model.generate_content([prompt, image])
    return response.text

print("Module 1: Scene Understanding - Ready to process images using vision and spatial reasoning.")


print("## Module 2: Text Reader")
print("This reads text—both clean printed text and challenging handwriting—from images like notes or documents.")

def read_text(image_input):
    """
    Extract and read text from images, including handwritten text.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = """You are helping a person with visual impairment read text.
    
Analyze this image and:
1. Extract ALL text visible (printed or handwritten)
2. Identify the type of document (letter, note, form, sign, etc.)
3. Provide context about the text (who wrote it, what it's about)
4. Highlight important information (dates, names, numbers)
5. If handwritten, do your best to decipher unclear writing

Read the text naturally as if reading aloud to someone."""
    
    # Load image if URL provided
    if isinstance(image_input, str):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content))
    else:
        image = image_input
        
    # Generate response
    response = model.generate_content([prompt, image])
    return response.text

print("Module 2: Text Reader - Ready to use Gemini's handwriting recognition and OCR.")


print("## Module 3: Object Identifier")
print("Critical for daily safety! This identifies products, food, and medication and reads out key safety information.")

def identify_object(image_input):
    """
    Identify objects, products, and read labels with safety information.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = """You are helping a person with visual impairment identify objects and read product labels.
    
Analyze this image and provide:
1. What is this object/product?
2. Brand name and product name
3. Key information from the label:
    - For medication: dosage, instructions, warnings, expiration date
    - For food: ingredients, allergens, nutrition facts, expiration date
    - For other products: usage instructions, warnings
4. Safety warnings or important notices
5. Any other relevant details

Prioritize safety-critical information (allergens, warnings, expiration dates)."""
    
    # Load image if URL provided
    if isinstance(image_input, str):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content))
    else:
        image = image_input
        
    # Generate response
    response = model.generate_content([prompt, image])
    return response.text

print("Module 3: Object Identifier - Ready to use vision and reasoning for product safety.")


print("## Module 4: Navigation Assistant")
print("This module provides clear, real-time guidance, which is so important when moving through unfamiliar spaces.")

def navigate_space(image_input, destination=None):
    """
    Provide navigation guidance based on visual input.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"""You are a navigation assistant for a person with visual impairment.
    
Analyze this image and provide navigation guidance:
1. Describe what's directly ahead
2. Identify doorways, hallways, stairs, or exits
3. Provide clear directional instructions (forward, left, right)
4. Estimate distances ("about 10 feet ahead")
5. Warn about obstacles or hazards
6. Suggest the safest path forward

{f'The person wants to reach: {destination}' if destination else ''}

Give step-by-step instructions that are clear and actionable."""
    
    # Load image if URL provided
    if isinstance(image_input, str):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content))
    else:
        image = image_input
        
    # Generate response
    response = model.generate_content([prompt, image])
    return response.text

print("Module 4: Navigation Assistant - Ready to use spatial reasoning for real-time guidance.")


print("## Module 5: Social Scene Analyzer")
print("Helps with social interaction by providing context—essentially 'reading the room' for the user.")

def analyze_social_scene(image_input):
    """
    Analyze social scenes to help understand people and interactions.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = """You are helping a person with visual impairment understand a social situation.
    
Analyze this image and describe:
1. How many people are present?
2. Their approximate ages and genders (if discernible)
3. What are they doing? (sitting, standing, talking, etc.)
4. Their spatial arrangement (who is near whom?)
5. Facial expressions and emotions (happy, serious, etc.)
6. The overall mood or atmosphere
7. The setting/context (meeting, party, family gathering, etc.)

Help the person understand the social dynamics and context."""
    
    # Load image if URL provided
    if isinstance(image_input, str):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content))
    else:
        image = image_input
        
    # Generate response
    response = model.generate_content([prompt, image])
    return response.text

print("Module 5: Social Scene Analyzer - Ready, using vision and context for social intelligence.")


# All modules are now defined.
print("--------------------------------------------------")
print("All five core accessibility modules are fully defined and ready to be used.")


print("## Impact & Innovation: Why This Project Works")
print("--------------------------------------------------")
print("### The Big Wins:")
print("1. Comprehensive Solution: It's 5 essential capabilities integrated, not 5 separate apps. This is a huge user experience improvement.")
print("2. Advanced AI: I relied on Gemini's cutting-edge multimodal features for better accuracy.")
print("3. Conversational: The large context window (up to 1M tokens) is key for keeping track of natural, continuous dialogue.")
print("4. Truly Accessible: The design is voice-first and works across any smartphone platform.")
print("5. Real-World Scale: The tool addresses the needs of 285 million people globally—that's the real impact.")


print("### Technical Highlights:")
print("- Seamless Multimodal Synthesis: Combining vision, spatial reasoning, and language generation.")
print("- Custom Spatial Intelligence: The core of the navigation and scene modules.")
print("- Robust Handwriting Recognition: Handles personal correspondence, not just standard text.")
print("- Context Awareness: The large context window enables natural conversations.")
print("- Safety Focus: Critical information (medication, hazards) is always prioritized.")
print("")
print("### Practical Uses for Users:")
print("- Gaining independence for daily living and personal tasks.")
print("- Safe medication and food management.")
print("- Confident navigation in public and unfamiliar spaces.")
print("- Reading personal letters and notes.")
print("- Full participation in social events.")


print("--------------------------------------------------")
print("## Final Thoughts")
print("This project shows the immense potential of modern foundation models to deliver truly life-changing accessibility tools.")
print("")
print("Check Out The Project:")
print("Live Demo: [VisionAI Companion](https://ai.studio/apps/drive/105oIMifoYdab0uUkio0pawFeMR50DLvM?fullscreenApplet=true )")
print("Video Demo: [YouTube](https://www.youtube.com/watch?v=iZS_SLx-mS4 )")
print("")
print("VisionAI Companion: Empowering 285 million people to perceive the world on their own terms.")

