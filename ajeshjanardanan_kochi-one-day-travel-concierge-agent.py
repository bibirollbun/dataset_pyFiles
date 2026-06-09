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


# Imports

import pandas as pd
import json
import math


# kochi pois csv file

csv_text = """name,category,time_needed,cost,best_time,lat,lng,description
Fort Kochi Beach,Beach,60,0,Morning,9.9650,76.2423,Scenic beach with iconic Chinese fishing nets.
Chinese Fishing Nets,Landmark,30,0,Sunset,9.9669,76.2433,Historic fishing nets perfect for photos.
Jew Town,Shopping,90,0,Afternoon,9.9579,76.2593,Antique shops and spice markets.
Mattancherry Palace,Heritage,60,30,Morning,9.9635,76.2608,Kerala murals and history museum.
St Francis Church,Heritage,30,10,Morning,9.9662,76.2425,Oldest European church in India.
Marine Drive,Scenic,60,0,Evening,9.9816,76.2750,Beautiful promenade for walking.
Lulu Mall,Shopping,120,0,Afternoon,10.0260,76.3083,Huge mall with branded shops.
Kerala Kathakali Centre,Cultural,90,300,Evening,9.9631,76.2422,Kathakali and Kalaripayattu performances.
Kashi Art Café,Cafe,45,200,Morning,9.9657,76.2429,Famous artistic café with continental food.
Bolgatty Palace,Heritage,60,40,Morning,9.9822,76.2804,Historic palace on an island.
"""

with open("kochi_pois.csv", "w") as f:
    f.write(csv_text)


# Load Dataset

pois = pd.read_csv("kochi_pois.csv")
pois


# Weather Tool

def weather_tool(day="today"):
    return {"day": day, "weather": "sunny"}


# Travel Time Tool

def travel_time_tool(lat1, lng1, lat2, lng2):
    dist = math.sqrt((lat1-lat2)**2 + (lng1-lng2)**2)
    minutes = max(10, int(dist * 200))
    return {"distance_km": round(dist*111, 2), "minutes": minutes}


# POI Tool

def poi_tool(preferences):
    df = pois.copy()
    prefs = preferences.lower()
    if "beach" in prefs:
        df = df[df['category'] == "Beach"]
    if "history" in prefs or "heritage" in prefs:
        df = df[df['category'].isin(["Heritage"])]
    if "food" in prefs:
        df = df[df['category'].isin(["Cafe"])]
    return df.to_dict(orient="records")


#Itinerary Generator

def build_itinerary(user_request):
    matching = poi_tool(user_request)
    weather = weather_tool()
    
    if not matching:
        matching = pois.sample(3).to_dict(orient="records")
        
    plan = []
    current = matching[0]
    plan.append(("Start", current))
    
    for next_spot in matching[1:4]:
        tt = travel_time_tool(current['lat'], current['lng'], next_spot['lat'], next_spot['lng'])
        plan.append((f"Next ({tt['minutes']} min travel)", next_spot))
        current = next_spot
    
    return {
        "weather": weather,
        "itinerary": plan
    }


# Demo Query

result = build_itinerary("I like beaches, history and photography")
result

