import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


po_df = pd.read_csv("/kaggle/input/geolifeclef-2025/GLC25_P0_metadata_train.csv")
pa_df = pd.read_csv("/kaggle/input/geolifeclef-2025/GLC25_PA_metadata_test.csv")


print("PO Metadata: ", len(po_df))
print("PA Metadata: ", len(pa_df))


print("PO Missing Values: ", po_df.isna().sum().sum())
print("PA Missing Values: ", pa_df.isna().sum().sum())


po_df.head()


publishers = po_df["publisher"].value_counts(normalize=True)

plt.rcParams["figure.figsize"] = [10.00, 10.00]

fig, labels = plt.pie(
    publishers.values
)

labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(publishers.index, 100.*publishers.values/publishers.values.sum())]
plt.legend(fig, labels, loc='lower left', bbox_to_anchor=(-0.05, 0.05), fontsize=8)
plt.show()


# Note: Data in po_df is truncated by 2000 entities, to reduce lag.
f, axes = plt.subplots(1, 3, figsize=(30, 15))

sns.histplot(data=po_df.head(2000), x="year", ax=axes[0])
sns.histplot(data=po_df.head(2000), x="month", ax=axes[1])
sns.histplot(data=po_df.head(2000), x="dayOfYear", ax=axes[2])



taxons = po_df["taxonRank"].value_counts(normalize=True)

fig = plt.pie(
    taxons.values, 
    labels=taxons.index
)
plt.show()


# Note: Data in po_df is truncated by 2000 entities, to reduce lag.
sns.histplot(data=po_df.head(2000), x="geoUncertaintyInM")


# Source: mpwolke from https://www.kaggle.com/code/mpwolke/geolifeclef25-maps#Install-Folium-Matplotlib-Mapclassify
!pip install folium matplotlib mapclassify
from shapely.geometry import Polygon, LineString, Point
import geopandas as gpd
import tqdm

po_geo_df = po_df.drop_duplicates('surveyId').sample(n=2000, random_state=42)
po_geo_df.index = range(len(po_geo_df))

# make Point vector 
point_list = []
for i in tqdm.tqdm(range(len(po_geo_df))):
    x,y = po_df.loc[i, ['lon', 'lat']]
    poind_i = Point(x,y)
    point_list.append(poind_i)

po_geo_df.loc[:,'geometry'] = point_list


# Source: mpwolke from https://www.kaggle.com/code/mpwolke/geolifeclef25-maps#Install-Folium-Matplotlib-Mapclassify
vis_geo_po = gpd.GeoDataFrame(po_geo_df, geometry = 'geometry')
vis_geo_po.crs = ('EPSG:4326')
vis_geo_po.drop_duplicates(['lon', 'lat']).explore(color = 'green')


import folium
from folium import plugins

map = folium.Map(location = [52,9], tiles='Cartodb dark_matter', zoom_start = 3.5)
heat_data = [[point.xy[1][0], point.xy[0][0]] for point in po_geo_df.geometry ]
plugins.HeatMap(heat_data).add_to(map)
map

