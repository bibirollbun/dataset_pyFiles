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
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)


#Install Required Packages
!pip install fastapi uvicorn nest-asyncio pyngrok openai gtts weasyprint gradio



#Set Up Map UI
!pip install folium


import folium
from folium import plugins

# Initialize the base map
mapobj = folium.Map(
    location=[-6.6370, -52.3518],  # São Félix do Xingu
    zoom_start=7,
    zoom_control=False
)

# Add a custom tile layer with required attribution
folium.TileLayer(
    tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
    attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
    name='Stamen Terrain'
).add_to(mapobj)

# Add drawing tools
plugins.Draw(export=True).add_to(mapobj)

# Add layer control
folium.LayerControl().add_to(mapobj)

# Add a simple marker
folium.Marker(
    location=[-6.6370, -52.3518],
    popup="São Félix do Xingu",
    tooltip="Center"
).add_to(mapobj)


# Define the HTML popup content BEFORE using it
html_popup = folium.Popup('''
<b>São Félix do Xingu</b><br>
<button onclick="alert('Research Clicked')">Research</button>
<button onclick="alert('Chat Clicked')">Chat</button>
''', max_width=300)


folium.Marker(
    location=[-6.6370, -52.3518],
    popup=html_popup,
    tooltip='Interactive Marker'
).add_to(mapobj)

# Display the map
mapobj







#Connect UI to Backend (FastAPI)
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/map")
def analyze_tile(lat: float, lon: float, goal: str):
    # Simulate processing
    summary = f"This is a research summary for the task: '{goal}' near {lat}, {lon}"
    return {
        "location": f"{lat}, {lon}",
        "goal": goal,
        "summary": summary
    }

# Simulated call (for testing purposes only)
if __name__ == "__main__":
    # normally wouldn't run this part in FastAPI directly
    result = analyze_tile(-6.63, -52.35, "Find ancient settlements")
    print(result["summary"])
    print(result["location"])  # the lat/lon string




# GPT Integration Disabled for Cost-Saving
# from openai import OpenAI

# client = OpenAI(api_key="sk-...")  # Key intentionally disabled

# def gpt_research_summary(lat: float, lon: float, goal: str):
#     prompt = f"""
#     You are an expert AI archaeologist. Given a location in the Amazon (Lat: {lat}, Lon: {lon}), and the user goal: '{goal}', write a research report on why this site might be archaeologically significant.
#     Use evidence such as known vegetation patterns, soil fertility, fire history, proximity to rivers, and prior indigenous activity in the area. Respond with a short, clear summary (300 words max).
#     """

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response.choices[0].message.content

# SIMULATED FUNCTION (offline use) 
def gpt_research_summary(lat: float, lon: float, goal: str):
    return f"""
    Simulated Research Summary for ({lat}, {lon}) — Goal: {goal}

    This region shows promising geo-ecological patterns for ancient settlement:
    - Vegetation anomalies detected via NDVI suggest past forest clearing.
    - Soil in this area is loamy and fertile, supporting sustained agriculture.
    - Located ~2km from a river with low recent fire activity ideal for preservation.
    - No overlap with documented archaeological sites suggesting novelty.

    Hypothesis: This site may have hosted pre-Columbian activity linked to regional Xingu traditions or Z-like settlements.
    """


# Build a Simple Gradio UI
import gradio as gr

def interactive_gpt(lat, lon, goal):
    return gpt_research_summary(lat, lon, goal)

gr.Interface(
    fn=interactive_gpt,
    inputs=[
        gr.Number(label="Latitude", value=-6.63),
        gr.Number(label="Longitude", value=-52.35),
        gr.Textbox(label="Research Goal", placeholder="Find ancient settlements")
    ],
    outputs=gr.Textbox(label="AI Research Summary"),
    title="AmazonExplorer AI",
    description="Choose a location and research goal. GPT-4 will summarize findings."
).launch()



from gtts import gTTS
from weasyprint import HTML

# Define sample inputs
lat = -6.63
lon = -52.35
goal = "Find ancient settlements"

try:
    # Try to get GPT summary
    summary = gpt_research_summary(lat, lon, goal)
    #summary = """
#Simulated Research Summary for (-6.63, -52.35) — Goal: Find ancient settlements

#This region shows promising geo-ecological patterns for ancient settlement...
#"""

    # Save voice note
    tts = gTTS(text=summary, lang='en')
    tts.save("summary.mp3")

    # Save PDF
    HTML(string=f"<h1>Amazon Research</h1><p>{summary}</p>").write_pdf("summary.pdf")

except Exception as e:
    summary = f"(Simulation) GPT call failed due to: {str(e)}"
    print(summary)




