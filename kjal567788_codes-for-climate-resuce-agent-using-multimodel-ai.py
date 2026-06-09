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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image



# Create a random 256x256 RGB image to simulate a satellite image
satellite_img = Image.fromarray(
    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
)
satellite_img



weather_data = pd.DataFrame({
    'rainfall_mm': [12, 18, 30, 42, 51]  # increasing — simulates rising risk
})
weather_data



social_posts = pd.DataFrame({
    'text': [
        "Area flooded heavily",
        "People trapped near the river",
        "Roads blocked, need help",
        "Everything is submerged",
        "Urgent rescue needed"
    ]
})
social_posts



def vision_agent(image):
    # Fake analysis output
    return {
        "flood_detected": True,
        "flood_extent_km2": round(np.random.uniform(20, 60), 2),
        "damaged_buildings_est": int(np.random.randint(50, 200))
    }



def weather_agent(df):
    last = df['rainfall_mm'].iloc[-1]
    rising = df['rainfall_mm'].diff().iloc[-1] > 0
    return {
        "rainfall": int(last),
        "risk_rising": bool(rising)
    }



def ground_text_agent(posts_df):
    if 'text' in posts_df.columns:
        urgent = [t for t in posts_df['text'].astype(str) if "trapped" in t.lower() or "urgent" in t.lower()]
    else:
        urgent = []

    return {
        "distress_reports": len(urgent),
        "sample_message": urgent[:1] if urgent else ["No urgent messages"]
    }



def fuse_data(vision, weather, text):
    score = 0
    if vision["flood_detected"]:
        score += 3

    if weather["risk_rising"]:
        score += 2

    score += min(5, text["distress_reports"])
    return score



def response_plan(score):
    if score >= 8:
        return "Deploy rescue boats, medical teams, and food supply kits."
    elif score >= 5:
        return "Send assessment team + essential supplies."
    else:
        return "Monitor and prepare standby resources."



vision_output = vision_agent(satellite_img)
weather_output = weather_agent(weather_data)
text_output = ground_text_agent(social_posts)

severity = fuse_data(vision_output, weather_output, text_output)
plan = response_plan(severity)

print("=== ClimateResQ Agent Output ===\n")
print("Vision Agent:", vision_output)
print("Weather Agent:", weather_output)
print("Text Agent:", text_output)
print("\nFinal Severity Score:", severity)
print("Recommended Action:", plan)


