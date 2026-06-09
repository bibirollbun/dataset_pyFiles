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


import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv("/kaggle/input/oilgas-field-prediction/train_oil.csv")


df.shape


pd.set_option('display.max_columns', None)


df.sample(5)


df['Onshore/Offshore'].value_counts()


df['Onshore/Offshore'] = df['Onshore/Offshore'].apply(lambda x: 1 if x == 'ONSHORE' else 0)


y=df['Onshore/Offshore']


# Trick to print all code in jupyter, else it runs only shows last one only
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


df2=pd.read_csv('/kaggle/input/oilgas-field-prediction/oil_test.csv')


df2.head()


df.isna().sum()


df.describe()
df.info()


df2.describe()
df2.info()


df['Field name'].value_counts()


df['Region'].value_counts()


df['Country'].value_counts().head(10)


df['Country'].value_counts().head(10)


for country in df['Region'].unique():
    print(f"Region: {country}")
    counts = df[df['Region'] == country]['Onshore/Offshore'].value_counts()
    print(counts)
    print('-' * 20)



df['Basin name'].value_counts()


df.columns


len(df['Tectonic regime'].value_counts())


df['Hydrocarbon type'].value_counts()


for hydro_type in df['Hydrocarbon type'].unique():
    print(f"Hydrocarbon type: {hydro_type}")
    subset = df[df['Hydrocarbon type'] == hydro_type]
    counts = subset['Onshore/Offshore'].value_counts()
    print(counts)
    print('-' * 20)



df['Lithology'].value_counts()
df2['Lithology'].value_counts()


# Count the occurrences of each category in 'Lithology'
counts = df['Lithology'].value_counts()
counts1 = df2['Lithology'].value_counts()
# Identify categories with counts less than 10
rare_categories = counts[counts < 10].index
rare_categories1 = counts1[counts1 < 10].index
# Replace rare categories with 'Other'
df['Lithology'] = df['Lithology'].replace(rare_categories, 'Other')
df2['Lithology'] = df2['Lithology'].replace(rare_categories1, 'Other')



df['Lithology'].value_counts()
df2['Lithology'].value_counts()


df['Basin name'].values


df.columns


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="MyApp")
location = geolocator.geocode("WEST SEMINOLE")
if location is not None:
    print(location.latitude, location.longitude)
else:
    print("Location not found")



df2.columns


df2.isna().sum()


missing_lat_lon = df[df['Latitude'].isna() | df['Longitude'].isna()]
missing_lat_lon2 = df2[df2['Latitude'].isna() | df2['Longitude'].isna()]


missing_lat_lon.index


missing_lat_lon.shape


pd.set_option('display.max_rows', None)


missing_lat_lon


field_names_array = missing_lat_lon['Field name'].values
# or equivalently
field_names_array = missing_lat_lon['Field name'].to_numpy()
field_names_array


field_names_array2 = missing_lat_lon2['Field name'].values
# or equivalently
field_names_array2 = missing_lat_lon2['Field name'].to_numpy()
field_names_array2


from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import time

geolocator = Nominatim(user_agent="my_app")

for idx, row in missing_lat_lon.iterrows():
    field_name = row['Field name']
    lat_missing = pd.isna(row['Latitude'])
    lon_missing = pd.isna(row['Longitude'])
    
    # Only query if Latitude or Longitude is missing
    if lat_missing or lon_missing:
        location = geolocator.geocode(field_name, timeout=10)
        time.sleep(1)  # To respect Nominatim usage policy
        
        if location is not None:
            # Update Latitude if missing
            if lat_missing:
                missing_lat_lon.at[idx, 'Latitude'] = location.latitude
            
            # Update Longitude if missing
            if lon_missing:
                missing_lat_lon.at[idx, 'Longitude'] = location.longitude
            
            print(f"Updated {field_name}: {location.latitude}, {location.longitude}")
        else:
            print(f"Location not found for: {field_name}")
    else:
        print(f"Coordinates already present for: {field_name}")



from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import time

geolocator = Nominatim(user_agent="my_app")

for idx, row in missing_lat_lon2.iterrows():
    Reservoir	 = row['Reservoir unit']
    lat_missing = pd.isna(row['Latitude'])
    lon_missing = pd.isna(row['Longitude'])
    
    # Only query if Latitude or Longitude is missing
    if lat_missing or lon_missing:
        augmented_name = f"{Reservoir} Oil Field"
        location = geolocator.geocode(augmented_name, timeout=10)
        time.sleep(1)  # To respect Nominatim usage policy
        
        if location is not None:
            # Update Latitude if missing
            if lat_missing:
                missing_lat_lon2.at[idx, 'Latitude'] = location.latitude
            
            # Update Longitude if missing
            if lon_missing:
                missing_lat_lon2.at[idx, 'Longitude'] = location.longitude
            
            print(f"Updated {Reservoir}: {location.latitude}, {location.longitude}")
        else:
            print(f"Location not found for: {Reservoir}")
    else:
        print(f"Coordinates already present for: {Reservoir}")




missing_lat_lon2.index


df2.loc[5]


df2.loc[5, 'Latitude'] = 5.0542
df2.loc[5, 'Longitude'] = 97.25
df2.loc[11, 'Latitude'] = 21
df2.loc[11, 'Longitude'] = 71.3
df2.loc[13, 'Latitude'] = 28.3
df2.loc[13, 'Longitude'] = 33.3
df2.loc[27, 'Latitude'] = 32.3
df2.loc[27, 'Longitude'] = 92.3
df2.loc[35, 'Latitude'] = 49
df2.loc[35, 'Longitude'] = -112


df2.loc[46, 'Latitude'] = -1.5
df2.loc[46, 'Longitude'] = 103.0

df2.loc[55, 'Latitude'] = 28.5
df2.loc[55, 'Longitude'] = 30.5

df2.loc[60, 'Latitude'] = 29.5
df2.loc[60, 'Longitude'] = 103.0

df2.loc[61, 'Latitude'] = 29.5
df2.loc[61, 'Longitude'] = 103.0

df2.loc[71, 'Latitude'] = 30.75
df2.loc[71, 'Longitude'] = 27.12

df2.loc[78, 'Latitude'] = 30.75
df2.loc[78, 'Longitude'] = 27.12

df2.loc[87, 'Latitude'] = 28.22
df2.loc[87, 'Longitude'] = 19.13

df2.loc[91, 'Latitude'] = 25.43
df2.loc[91, 'Longitude'] = 49.62

df2.loc[94, 'Latitude'] = 25.43
df2.loc[94, 'Longitude'] = 49.62

df2.loc[117, 'Latitude'] = -20.0
df2.loc[117, 'Longitude'] = 115.0

df2.loc[123, 'Latitude'] = -3.5
df2.loc[123, 'Longitude'] = 104.0



df2['Lithology'].value_counts()



for i in missing_lat_lon.index:
    print(df.loc[i]['Latitude'],df.loc[i]['Longitude'])
    print(missing_lat_lon.loc[i]['Latitude'],missing_lat_lon.loc[i]['Longitude'])
    df.loc[i, 'Latitude'] = missing_lat_lon.loc[i, 'Latitude']
    df.loc[i, 'Longitude'] = missing_lat_lon.loc[i, 'Longitude']



df.shape


to_drop = df[df['Latitude'].isna() | df['Longitude'].isna()].index
print(to_drop)


y.loc[[55, 89, 130, 168]]


y.drop(index=[55, 89, 130, 168], inplace=True)



df = df.dropna(subset=['Latitude', 'Longitude'])



df2.isna().sum()


df2.isna().sum()


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="MyApp")
location = geolocator.geocode("BATURAJA")
if location is not None:
    print(location.latitude, location.longitude)
else:
    print("Location not found")



df = df[['Longitude', 'Latitude', 'Depth', 'Lithology', 'Thickness (gross average ft)', 'Porosity', 'Permeability']]



df2 = df2[['Longitude', 'Latitude', 'Depth', 'Lithology', 'Thickness (gross average ft)', 'Porosity', 'Permeability']]


from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder(drop='first',sparse=False,dtype=np.int32)


train = ohe.fit_transform(df[['Lithology']])


train.shape


df.columns


train1=np.hstack((df[['Longitude','Latitude','Depth','Thickness (gross average ft)', 'Porosity', 'Permeability']].values,train))


test = ohe.fit_transform(df2[['Lithology']])


df['Lithology'].value_counts()
df2['Lithology'].value_counts()


test1=np.hstack((df2[['Longitude','Latitude','Depth','Thickness (gross average ft)', 'Porosity', 'Permeability']].values,test))


train1.shape
train.shape


test1.shape
test.shape


from sklearn.tree import DecisionTreeClassifier


y.value_counts()


clf=DecisionTreeClassifier()


clf.fit(train1,y)


y_pred=clf.predict(test1)


df3=pd.DataFrame({"Onshore/Offshore":y_pred});df3.index.name="index"


df3["Onshore/Offshore"]=df3["Onshore/Offshore"].replace({0:"OFFSHORE",1:"ONSHORE"});df3.head()


df3.sample(2)


df3.to_csv("arpit_model.csv")

