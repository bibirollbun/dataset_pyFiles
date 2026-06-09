import pandas as pd
import re
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time



pip install pymupdf


import fitz 

pdf_path = "/kaggle/input/keller-explorations/Keller_book.pdf"
pdf_document = fitz.open(pdf_path)

# Extract text from each page
extracted_text = ""
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    extracted_text += page.get_text()


print(extracted_text[:1000])


# Define regular expressions for various anthropogenic features with location names
geometric_features_pattern = r"([a-zA-Z\s]+(?:geometric\s?earthworks?|geoglyphs?|circles?|rectangles?|patterns?))"
embankments_pattern = r"([a-zA-Z\s]+(?:raised\s?embankments?|terrains?|terraces?))"
ditches_canals_pattern = r"([a-zA-Z\s]+(?:ditches?|canals?|water\s?management\s?features?))"

# Search for matches in the extracted text
geometric_features = re.findall(geometric_features_pattern, extracted_text, flags=re.IGNORECASE)
embankments = re.findall(embankments_pattern, extracted_text, flags=re.IGNORECASE)
ditches_canals = re.findall(ditches_canals_pattern, extracted_text, flags=re.IGNORECASE)

# Print results for anthropogenic features found in the text
print("Geometric Features:", geometric_features)
print("Embankments/Terraces:", embankments)
print("Ditches/Canals:", ditches_canals)



from transformers import pipeline

# Load Hugging Face models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Question
question = "What are the concentric circles mentioned in the text?"

# Run the question-answering model
answer = qa_pipeline(question=question, context=extracted_text)

# Display the answer
print(f"Answer: {answer['answer']}")



from openai import OpenAI
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)



prompt = """
In the historical text, the term "stout palisades" is mentioned in relation to an archaeological feature. Could you explain what "stout palisades" typically refer to in the context of pre-Columbian archaeology? How might these palisades relate to geometric features, earthworks, or defensive structures used by ancient civilizations? Are they associated with fortifications, villages, or ceremonial sites?
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=500,
    temperature=0
)

print(response.choices[0].message.content.strip())



# Question
question = "What are the patterns mentioned in the text?"

# Run the question-answering model
answer = qa_pipeline(question=question, context=extracted_text)

# Display the answer
print(f"Answer: {answer['answer']}")


import time
from geopy.geocoders import Nominatim

# Initialize the geolocator with a custom user-agent
geolocator = Nominatim(user_agent="myApp")

# Example filtered historical references (replace with actual list)
historical_references = geometric_features

# Geocode each historical reference with a delay between requests
for reference in historical_references:
    location = geolocator.geocode(reference)
    
    if location:
        print(f"Geocoded location: {reference}")
        print(f"Latitude: {location.latitude}, Longitude: {location.longitude}")
    else:
        print(f"Could not geocode the location: {reference}")

    # Adding a delay between requests to avoid hitting rate limits
    time.sleep(1)  # Sleep for 1 second between requests



# Extract locations and matching sentences with spaCy

nlp = spacy.load("en_core_web_sm")

locations = []
features = []

for sent in nlp(extracted_text).sents:
    locs = [ent.text for ent in sent.ents if ent.label_ in ["GPE", "LOC"]]
    if locs and any(word in sent.text.lower() for word in ["circle", "terrace", "ditch", "embankment"]):
        for loc in locs:
            locations.append(loc)
            features.append(sent.text)

location_df = pd.DataFrame({"location_name": locations, "description": features})
print(f"Extracted {len(location_df)} location-feature pairs from text.")
display(location_df.head())


geolocator = Nominatim(user_agent="geoApp")
results = []

for loc_name in location_df['location_name'].unique():
    try:
        geo = geolocator.geocode(loc_name, timeout=10)
        if geo:
            results.append({"location_name": loc_name, "lat": geo.latitude, "lon": geo.longitude})
        else:
            results.append({"location_name": loc_name, "lat": None, "lon": None})
        time.sleep(1)  # respect Nominatim rate limits
    except Exception as e:
        print(f"⚠ Failed on '{loc_name}': {e}")
        results.append({"location_name": loc_name, "lat": None, "lon": None})

geo_df = pd.DataFrame(results)
print(f"Geocoded {len(geo_df.dropna())} locations successfully.")



full_df = pd.merge(location_df, geo_df, on="location_name", how="left")
display(full_df.head())


geolocator = Nominatim(user_agent="geoApp")
results = []

for loc_name in location_df['location_name'].unique():
    try:
        geo = geolocator.geocode(loc_name, timeout=10)
        if geo:
            results.append({"location_name": loc_name, "lat": geo.latitude, "lon": geo.longitude})
        else:
            results.append({"location_name": loc_name, "lat": None, "lon": None})
        time.sleep(1)  # respect Nominatim rate limits
    except Exception as e:
        print(f"⚠ Failed on '{loc_name}': {e}")
        results.append({"location_name": loc_name, "lat": None, "lon": None})

geo_df = pd.DataFrame(results)
print(f"Geocoded {len(geo_df.dropna())} locations successfully.")

# Merge back to full location-feature sentences
full_df = pd.merge(location_df, geo_df, on="location_name", how="left")
display(full_df.head())

# Compare to your anomaly locations
# Assume you have `top5_anomalies` dataframe with 'lat' and 'lon' columns
# and perhaps 'id' or index for each anomaly.
top5_anomalies= pd.read_csv("/kaggle/input/anomaly-points/anomaly_points.csv")

comparisons = []
for _, hist_row in full_df.dropna(subset=["lat", "lon"]).iterrows():
    for idx, anomaly in top5_anomalies.iterrows():
        dist_km = geodesic((hist_row["lat"], hist_row["lon"]), (anomaly["lat"], anomaly["lon"])).km
        comparisons.append({
            "anomaly_id": idx+1,
            "anomaly_lat": anomaly["lat"],
            "anomaly_lon": anomaly["lon"],
            "historical_loc": hist_row["location_name"],
            "historical_desc": hist_row["description"],
            "historical_lat": hist_row["lat"],
            "historical_lon": hist_row["lon"],
            "distance_km": dist_km
        })

comp_df = pd.DataFrame(comparisons)


nearest_df = comp_df.loc[comp_df.groupby("anomaly_id")["distance_km"].idxmin()]

print("Closest historical references to each anomaly:")
display(nearest_df[["anomaly_id", "historical_loc", "distance_km", "historical_desc"]])

# You can also sort to see most striking nearby matches
nearest_df_sorted = nearest_df.sort_values("distance_km")
display(nearest_df_sorted.head(10))


nearest_df.to_csv("nearest_df.csv")

