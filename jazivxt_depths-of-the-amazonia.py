from IPython.display import YouTubeVideo
YouTubeVideo('wwjtdOqTmrA',width=600, height=400)


"""!pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 
import torch
from transformers import AutoTokenizer
from transformers.models.gemma3 import Gemma3ForCausalLM
from IPython.display import display, Markdown, Latex

gp = '/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1'
processor = AutoTokenizer.from_pretrained(gp)
model = Gemma3ForCausalLM.from_pretrained(gp).to("cpu") #cuda
prompt = "<start_of_turn>user\nTell us about the lost city of El Dorado in South America<end_of_turn>\n<start_of_turn>model"
input_ids = processor(text=prompt, return_tensors="pt").to(device)
outputs = model.generate(**input_ids, max_new_tokens=512)
text = processor.batch_decode(outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)
display(Markdown(text[0]))"""
print('Output from Gemma3 1B')


#https://giscarta.com/atlas/amazon-river-map
#layer mapbiomass
import folium, requests
import geopandas as gpd
import pandas as pd

#Example, need full world listing
pyramids = 'https://raw.githubusercontent.com/LSIND/map-of-Ancient-Egypt/refs/heads/master/pyramids.csv'
arch = '/kaggle/input/archaeological-sites-with-maya-inscriptions/scrapedData.csv'

#df = gpd.read_file('Amazon River.geojson')
#df = df.to_crs(epsg='4326')
#js = df.to_json()

m = folium.Map(location=[-3.4653, -62.2159], zoom_start=2, tiles="CartoDB positron")
#folium.GeoJson(js, name='Amazon Rivers').add_to(m)

df = pd.read_csv(pyramids)
for i in range(0,len(df)):
   folium.Marker(location=[df.iloc[i]['Latitude'], df.iloc[i]['Longitude']], popup=df.iloc[i]['Modern name'],).add_to(m)

df = pd.read_csv(arch, encoding='latin-1')
df = df[pd.to_numeric(df['Latitude'], errors='coerce').notnull()]
df = df[pd.to_numeric(df['Longitude'], errors='coerce').notnull()]
for i in range(0,len(df)):
   folium.Marker(location=[df.iloc[i]['Latitude'], df.iloc[i]['Longitude']], popup=df.iloc[i]['Name'],).add_to(m)

#Add Inca Sites
folium.Marker(location=[-13.16308, -72.54525], popup='Machu Picchu - Inca',).add_to(m)

folium.LayerControl().add_to(m)
m


YouTubeVideo('qhWItvjk9Yg',width=600, height=400)

