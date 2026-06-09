# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import plotly
plotly.offline.init_notebook_mode(connected=True)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_excel('/kaggle/input/archaeological-sites-map/Brazilian radiocarbon bioarchaeological samples.xlsx')
df.tail()


#StackOverflow https://stackoverflow.com/questions/34076177/matplotlib-horizontal-bar-chart-barh-is-upside-down

df["Region"].value_counts().plot.barh(color=['blue', '#f5005a'], title='Brazilian Archaeological Regions')
plt.gca().invert_yaxis();


#By Marília Prata on Kaggle https://www.kaggle.com/code/mpwolke/airports-maps
#StackOverFlow https://stackoverflow.com/questions/25328003/how-can-i-change-the-font-size-using-seaborn-facetgrid

sns.set(font_scale=3) 

plt.figure(figsize=(20,12))
ax = plt.gca()
ax.set_title("Brazilian Archaeological Regions")

g = sns.scatterplot(x='Latitude (wgs 84)', y='Longitude (wgs 84)', data=df, hue='Type of Coordinates')
g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1);


#installation
!pip install folium


#https://medium.com/data-science/using-python-to-create-a-world-map-from-a-list-of-country-names-cd7480d03b10

# Create a world map to show distributions of users 
import folium
from folium.plugins import MarkerCluster
#empty map
bra_map= folium.Map(location=(-33, -60),tiles="cartodbpositron", zoom_start= 3)
marker_cluster = MarkerCluster().add_to(bra_map)
#for each coordinate, create circlemarker of altitude (m)
for i in range(len(df)):
        lat = df.iloc[i]['Latitude (wgs 84)']
        long = df.iloc[i]['Longitude (wgs 84)']
        radius=5
        popup_text = """Region : {}<br>
                    Altitude (m) : {}<br>"""
        popup_text = popup_text.format(df.iloc[i]['Region'],
                                   df.iloc[i]['Altitude (m)']
                                   )
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
#show the map
bra_map


df.info()


radiocarbon = pd.read_excel('/kaggle/input/archaeological-sites-map/individuals.xlsx')
radiocarbon.tail()


# Sort data by 14C 2σ Calibrated Date - Lower Limit in Descending order 
radiocarbon_reset = radiocarbon.reset_index()
radiocarbon_reset[["Relative Age - Lower Limit", "Age System", "14C SD (±σ)", "14C 2σ Calibrated Date - Lower Limit"]].sort_values(by = '14C 2σ Calibrated Date - Lower Limit', ascending=False).head(10)


funerals = pd.read_excel('/kaggle/input/archaeological-sites-map/human_individuals_funerary.xlsx')
funerals.head()


ax = funerals['Disposal Type'].value_counts().plot.barh(figsize=(8, 4), color='green')
ax.set_title('Body Disposal Type', size=18, color='orange')
ax.set_ylabel('Disposal Type', size=10)
ax.set_xlabel('Count', size=10)
plt.gca().invert_yaxis();


skeletal = pd.read_excel('/kaggle/input/archaeological-sites-map/sampled_skeletal_part.xlsx')
skeletal.head()


#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

sample_proportion = skeletal['Sample Type'].value_counts()/skeletal['Sample Type'].value_counts().sum()
colormap = plt.cm.tab10(range(0, len(sample_proportion)))

#Use logx=True cause the bar didn't appear due to small values

skeletal["Sample Type"].value_counts().plot.barh(logx=True, color=colormap, title='Sample Type')
plt.gca().invert_yaxis();


sample_proportion = skeletal['Sampled Skeletal Part'].value_counts()/skeletal['Sampled Skeletal Part'].value_counts().sum()
colormap = plt.cm.tab10(range(0, len(sample_proportion)))

#Use logx=True cause the bar didn't appear due to small values

skeletal["Sampled Skeletal Part"].value_counts().plot.barh(figsize=(18, 12),color=colormap, title='Sampled Skeletal Part')
plt.gca().invert_yaxis();

