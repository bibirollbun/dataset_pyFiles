!pip install -q openai-agents


import ee
from ee import ImageCollection, Image, Geometry

from openai import OpenAI
from agents import Agent, Runner, ModelSettings, function_tool, WebSearchTool, trace
from pydantic import BaseModel, Field
from typing import List

import base64
import io
import os
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image
import pprint


def load_secret(name):
    """Loads secret from Colab/Kaggle."""

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass 
    else:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception: 
            pass

    return 'Secret not found'


os.environ['OPENAI_API_KEY'] = load_secret('openai_key_2025') # openai key
iam_service_account = load_secret('iam_service_account') # the address of your project's IAM service account
ee_credentials_json = load_secret('ee_credentials') # the file path for the JSON file containing the relevant credentials
ee_creds = ee.ServiceAccountCredentials(iam_service_account, ee_credentials_json) # fetch your service account credentials
ee.Initialize(ee_creds) # initialize earth engine using your service account credentials


def ee_image_to_pil(img, vis_params, region):
    """
    Helper function to fetch and load image as PIL Image
    """
    url = img.getThumbURL({
        'region': region,
        'dimensions': 800,
        'format': 'jpg',
        **vis_params
    })
    with urllib.request.urlopen(url) as response:
        img_data = response.read()
    return Image.open(io.BytesIO(img_data)), img_data
        

@function_tool
def visualize_on_ee(coordinates: list[float], poi_name: str):
    """
    Visualize the POI using the `coordinates` in the Earth Engine (ee) library in Python for the given `poi_name`.
    """
    # Credits to https://www.kaggle.com/code/paultimothymooney/how-to-ask-gpt-4o-about-google-earth-engine-data
    poi = ee.Geometry.Point(coordinates)
    region = poi.buffer(1500).bounds()
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(poi) \
        .filterDate('2024-01-01', '2024-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE')

    sentinel = collection.first()
    if sentinel is None:
        raise ValueError("No suitable Sentinel-2 image found.")
    
    # True Color (B4, B3, B2), from https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA?hl=it#bands
    true_color_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 1500,
        'gamma': 1.3
    }

    # Infrared
    infrared_color_params = {
        'bands': ['B5'],
        'min': 0,
        'max': 1500,
        'gamma': 1.3
    }

    # Get images
    img_true, img_to_save = ee_image_to_pil(sentinel, true_color_params, region)
    img_infra, infra_to_save = ee_image_to_pil(sentinel, infrared_color_params, region)

    # Side-by-side plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(img_true)
    axs[0].set_title(f"{poi_name} - Coords: {coordinates}\n(RGB)")
    axs[0].axis('off')

    axs[1].imshow(img_infra, cmap='gray')
    axs[1].set_title("Infrared")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Save the images for later use by following agents
    poi_images.append({
        "poi": poi_name,
        "coordinates": coordinates,
        "true_color_image": img_to_save,
        "infra_image": infra_to_save,
    })


instr = """
You are an expert in archaeological site prediction and geospatial analysis.
Given the Amazon basin, your task is to identify and prioritize points (point of interest, POIs) most likely to contain undiscovered archaeological sites. 

Consider:
- Proximity of current or old source of water (rivers, lakes etc)
- Environmental preservation (areas less affected by modern development)
- Accessibility (areas not yet surveyed or with limited access)
- Historical Indigenous presence (if data available)

Return 2-3 POIs in different sites
"""

class POIOutput(BaseModel):
    coordinates: List[List[float]] = Field(..., description="POI Longitude and latitude")
    poi_name: List[str] = Field(..., description="Name of the POI")
    
POIAgent = Agent(
    name="POI Agent",
    model='gpt-4.1',
    model_settings=ModelSettings(temperature=0.1, max_tokens=1000, tool_choice='required'),
    instructions=instr,
    output_type=POIOutput,
    tools=[visualize_on_ee])


instr_search = """
You are an expert in archaeological site prediction and geospatial analysis.
Given the provided POIs, conduct a web search to gather further insights and validate each proposed archaeological site.
Your task is to find if the given POI is most likely to contain undiscovered archaeological sites. 
"""

SearchAgent = Agent(
    name="Search Agent",
    model='gpt-4.1',
    model_settings=ModelSettings(temperature=0.5, max_tokens=3000, tool_choice='required'),
    instructions=instr_search,
    tools=[WebSearchTool()])


instr_vision = """
You are an expert in archaeological site prediction and geospatial analysis.
Given the provided images of POIs and related informations, describe what you see and if the given POI is most likely to contain undiscovered archaeological sites.
Return a score for each POI, together with your reasoning.
"""

class VisionOutput(BaseModel):
    image_description: str = Field(..., description="Description of the provided images, clearly specifying the POI name and main features")
    undiscovered_likelihood: str = Field(..., description="For each POI, the likelihood score from 1 to 10 that the site is likely to contain undiscovered sites")
    reasoning: str = Field(..., description="Reasoning of the likelihood score given, divided by POI")
    
VisionAgent = Agent(
    name="Search Agent",
    model='gpt-4.1',
    model_settings=ModelSettings(temperature=0.5, max_tokens=2500),
    output_type=VisionOutput,
    instructions=instr_vision)


def prepare_vision_prompt(prompt, dict_images):
    """
    Prepare a prompt payload for a vision-capable language model.

    This function builds a message payload containing:
        - User instructions or questions.
        - A summary of the points of interest (POIs) from the provided image dictionary.
        - The corresponding true color and infrared images, each encoded as base64 data URLs.
    """

    # Insert the prompt in the correct way and add the POIs names
    content = []
    content.append(
        {'role': 'user',
         'content': "{}.\nThese are the images (rgb and infrared) gathered, taken from: {}".format(prompt, '; '.join(
             [dict_images[i]['poi'] for i in range(len(dict_images))]
         ))}
    )

    # Loop over each image and append it to a list
    imgs = []
    for i in range(len(dict_images)):
        im = base64.b64encode(dict_images[i]['true_color_image']).decode("utf-8")
        im_inf = base64.b64encode(dict_images[i]['infra_image']).decode("utf-8")
        imgs.append({"type": "input_image", 'detail': 'auto', "image_url": f"data:image/jpeg;base64,{im}"})
        imgs.append({"type": "input_image", 'detail': 'auto', "image_url": f"data:image/jpeg;base64,{im_inf}"})

    # Pass the list to the payload
    content.append({'role': 'user', 
                    'content': imgs})
    
    return content


# Download a random image from the web
img_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"

with urllib.request.urlopen(img_url) as response:
    fake_binary = response.read()
    
fake_data = [
    {
        "poi": 'Test A',
        "coordinates": [1, 2],
        "true_color_image": fake_binary,
        "infra_image": fake_binary,
    },
    {
        "poi": 'Test B',
        "coordinates": [3, 4],
        "true_color_image": fake_binary,
        "infra_image": fake_binary,
    }
     ]

fake_prompt = 'This is a test!'

str(prepare_vision_prompt(fake_prompt, fake_data))[:400]


# Ensure the entire workflow is a single trace
with trace("Flow"):
    poi_images = []
    print("> GATHERING POIs FROM AGENT\n")
    # 1. Generate POIs
    poi_result = await Runner.run(
        POIAgent,
        'Tell me what you got',
    )
    print("> AGENT OUTPUT:\n\n{}".format(poi_result.final_output.poi_name))
    print("\n\n> CALLING WEB SEARCH AGENT")
    
    # 2. Web search on results from previous step
    websearch_result = await Runner.run(
        SearchAgent,
        "Conduct a detailed search on the following POIs {}".format(poi_result.final_output.poi_name)
    )
    print("> AGENT OUTPUT:\n\n{}".format(websearch_result.final_output))

    print("\n\n> CALLING VISION AGENT")
    # 3. Vision on images gathered so far
    vision_result = await Runner.run(
        VisionAgent,
        prepare_vision_prompt(websearch_result.final_output, poi_images) # Passing the web search result and the dict of images from step 1
    )

    print("> AGENT OUTPUT:\n\n")
    print("Image analysis:\n{}".format(vision_result.final_output.image_description))
    print("\nLikelihood score:\n{}".format(vision_result.final_output.undiscovered_likelihood))
    print("\nReasoning:\n{}".format(vision_result.final_output.reasoning))

