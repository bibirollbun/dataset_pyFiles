# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

from tqdm import tqdm
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
from plotly.offline import iplot

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


#By Go Byeonggeon https://www.kaggle.com/code/gobyeonggeon/preprocess-visualize-spatial-data-eda-xgb/notebook

from glob import glob
import os
# from netCDF4 import Dataset
import pandas as pd
import geopandas as gpd #
from shapely.geometry import Polygon, LineString, Point
#import rasterio #No module named 'rasterio'
#from rasterio.plot import show
# from rasterio.transform import from_origin
#from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import cv2
import numpy as np
import geopandas as gpd
import tqdm

root_path = '/kaggle/input'


#By Go Byeonggeon https://www.kaggle.com/code/gobyeonggeon/preprocess-visualize-spatial-data-eda-xgb/notebook

train_meta = pd.read_csv(root_path + "/geolifeclef-2025/GLC25_PA_metadata_train.csv")
train_meta.head()

sub_train_meta = train_meta.drop_duplicates('surveyId').sample(n=1000, random_state=42)
sub_train_meta.index = range(len(sub_train_meta))

# make Point vector 
point_list = []
for i in tqdm.tqdm(range(len(sub_train_meta))):
    x,y = sub_train_meta.loc[i, ['lon', 'lat']]
    poind_i = Point(x,y)
    point_list.append(poind_i)

sub_train_meta.loc[:,'geometry'] = point_list

# Read meta data
test_meta = pd.read_csv(root_path + "/geolifeclef-2025/GLC25_PA_metadata_test.csv")
test_meta.head()

sub_test_meta = test_meta.drop_duplicates('surveyId').sample(n=1000, random_state=42)
sub_test_meta.index = range(len(sub_test_meta))

# make Point vector 
point_list = []
for i in tqdm.tqdm(range(len(sub_test_meta))):
    x,y = sub_test_meta.loc[i, ['lon', 'lat']]
    poind_i = Point(x,y)
    point_list.append(poind_i)

sub_test_meta.loc[:,'geometry'] = point_list


!pip install folium matplotlib mapclassify


#By Go Byeonggeon https://www.kaggle.com/code/gobyeonggeon/preprocess-visualize-spatial-data-eda-xgb/notebook

train_sub_meta_gdf = gpd.GeoDataFrame(sub_train_meta, geometry = 'geometry')
#train_sub_meta_gdf.crs = {'init':'epsg:4326'}#'+init=<authority>:<code>' syntax is deprecated.
train_sub_meta_gdf.crs = ('EPSG:4326')
train_sub_meta_gdf.head()

test_sub_meta_gdf = gpd.GeoDataFrame(sub_test_meta, geometry = 'geometry')
#test_sub_meta_gdf.crs = {'init':'epsg:4326'} #'+init=<authority>:<code>' syntax is deprecated.
test_sub_meta_gdf.crs = ('EPSG:4326') 
test_sub_meta_gdf.head()

# visualize each monitoring site
m = train_sub_meta_gdf.drop_duplicates(['lon', 'lat']).explore(color = 'green')
test_sub_meta_gdf.drop_duplicates(['lon', 'lat']).explore(m=m, color = 'red')


!pip install rasterio


#By Go Byeonggeon https://www.kaggle.com/code/gobyeonggeon/preprocess-visualize-spatial-data-eda-xgb/notebook

# check missing values
missing_value_col_list = []
rows_with_missing_values = {}
for column in train_meta.columns:
    missing_rows = train_meta.index[train_meta[column].isnull()].tolist()
    if len(missing_rows) > 0:
        rows_with_missing_values[column] = len(missing_rows)
        missing_value_col_list.append(column)

print(rows_with_missing_values)


#By Go Byeonggeon https://www.kaggle.com/code/gobyeonggeon/preprocess-visualize-spatial-data-eda-xgb/notebook

target_colnames = 'geoUncertaintyInM'
sub_df = pd.merge(train_meta.loc[:, ['surveyId', target_colnames]], train_meta.loc[:, ['lon', 'lat', 'surveyId']].drop_duplicates("surveyId"), on='surveyId')
na_ind = sub_df.loc[:,target_colnames].isna().values

sub_df_train = sub_df.loc[~na_ind,:]
sub_df_target = sub_df.loc[na_ind,:]
x,y,z = sub_df_train['lon'].values, sub_df_train['lat'].values, sub_df_train[target_colnames].values
x_target,y_target,z_target = sub_df_target['lon'].values, sub_df_target['lat'].values, np.zeros(sub_df_target[target_colnames].shape)

plt.figure(figsize=(12, 6))
plt.scatter(x, y, marker='o', color='green', label='Non Missing Data')
plt.scatter(x_target, y_target, marker='o', color='red', label='Missing Data', s = 5)

plt.legend()
plt.show()


#By Go Byeonggeon https://www.kaggle.com/code/gobyeonggeon/preprocess-visualize-spatial-data-eda-xgb/notebook

target_colnames = 'areaInM2'
sub_df = pd.merge(train_meta.loc[:, ['surveyId', target_colnames]], train_meta.loc[:, ['lon', 'lat', 'surveyId']].drop_duplicates("surveyId"), on='surveyId')
na_ind = sub_df.loc[:,target_colnames].isna().values

sub_df_train = sub_df.loc[~na_ind,:]
sub_df_target = sub_df.loc[na_ind,:]
x,y,z = sub_df_train['lon'].values, sub_df_train['lat'].values, sub_df_train[target_colnames].values
x_target,y_target,z_target = sub_df_target['lon'].values, sub_df_target['lat'].values, np.zeros(sub_df_target[target_colnames].shape)

plt.figure(figsize=(12, 6))
plt.scatter(x, y, marker='o', color='green', label='Non Missing Data')
plt.scatter(x_target, y_target, marker='o', color='red', label='Missing Data', s = 5)

plt.legend()
plt.show()

