import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
display(df[0:2].T)


df2=df[['latitude','longitude','common_name']]
print(len(df2))

df3 = df2[df2['latitude']!='None'][df2['longitude']!='None']
print(len(df3))

df3['latitude']=df3['latitude'].astype(float)
df3['longitude']=df3['longitude'].astype(float)

display(df3.info())


names = df3['common_name'].unique().tolist()
print(len(names))


name_df=names.copy()
for i,name in enumerate(names):
    name_df[i]=df3[df3['common_name']==name].reset_index(drop=True)


geometry = [Point(xy) for xy in zip(df3['longitude'], df3['latitude'])]
gdf = GeoDataFrame(df3, geometry = geometry)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(12,12)), color='orange', markersize=10)
plt.title('All')
plt.show()


for i in range(len(names)):
    datai=name_df[i]
    name=names[i]
    geometry = [Point(xy) for xy in zip(datai['longitude'], datai['latitude'])]
    gdf = GeoDataFrame(datai, geometry = geometry)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(12,12)), color='orange', markersize=15)
    plt.title(name)
    plt.show()







