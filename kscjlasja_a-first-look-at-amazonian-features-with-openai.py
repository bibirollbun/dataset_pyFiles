import folium

# Coordinates for our region of interest in Acre, Brazil
lat, lon = -10.2857, -67.8403

# Create a map centered on the location
m = folium.Map(
    location=[-10.2857, -67.8403],
    zoom_start=6,
    tiles="CartoDB positron"
)

# Add a marker or cross
folium.Marker([lat, lon], popup="Acre, Brazil (Study Area)", icon=folium.Icon(color="red")).add_to(m)

# Display the map
m


# Load the satellite image

from PIL import Image
import matplotlib.pyplot as plt

# Path to image
# The image was retrieved using the Earth Engine API and uploaded as part of the Kaggle dataset
image_path = "/kaggle/input/sentinel2-images/20220630T144729_20220630T144731_T19LFJ.png"

# Load and display the image
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
img.save("submission.png")



import openai

# Securely retrieve the OpenAI API key stored as a Kaggle secret
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OPENAI_API_KEY")

# Create the OpenAI client using the retrieved key
client = openai.OpenAI(api_key=secret_value_0)


import base64

openai_model = "gpt-4o"

# Convert the image to base64 format to send it to the OpenAI API as a data URL
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode()

# Compose the prompt and send the image to OpenAI's GPT-4o for vision analysis:

# Define the detailed prompt
prompt_text = """
Analyze the image and describe any visible patterns that may suggest archaeological features. Specifically, look for:
- Rectilinear or circular clearings
- Straight lines or unnatural geometric boundaries
- Clusters of cleared land or soil differences
Report what you see, and include a confidence score (0 to 1) in whether these could be man-made features.
"""

# Compose and send the request to OpenAI
response = client.chat.completions.create(
    model= openai_model,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text.strip()
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=500
)

# Display the result returned by GPT-4o
print(response.choices[0].message.content)


print("Model version:", openai_model)
print("DatasetID: COPERNICUS/S2_SR_HARMONIZED/20220630T144729_20220630T144731_T19LFJ")



# In a secure environment, run the following (Google Colab), or use the GEE Code Editor interface:
import ee
import requests

# Initialize Earth Engine
# ee.Initialize()  # For standard authenticated session
# ee.Initialize(project='your_project_name')  # Use this if you have a specific GCP project


