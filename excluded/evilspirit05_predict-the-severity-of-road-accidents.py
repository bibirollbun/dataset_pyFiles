import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTGBM_VERBOSE"] = "0"
warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns",None)

# Load train files
accidents_train = pd.read_csv('/kaggle/input/etiq-roadsense/accidents_train.csv')
places_train = pd.read_csv('/kaggle/input/etiq-roadsense/places_train.csv')
users_train = pd.read_csv('/kaggle/input/etiq-roadsense/users_train.csv')
vehicles_train = pd.read_csv('/kaggle/input/etiq-roadsense/vehicles_train.csv')


print("#"*150)
print(f"accident_train shape: {accidents_train.shape}")
print(f"accident_train check null values:\n{accidents_train.isnull().sum()}")
print(f"accident_train info: {accidents_train.info()}")
mode=accidents_train["Gravity"].mode()[0]
accidents_train["Gravity"].fillna(mode,inplace=True)
mode=accidents_train["PostalAddress"].mode()[0]
accidents_train["PostalAddress"].fillna(mode,inplace=True)
mode=accidents_train["GPSCode"].mode()[0]
accidents_train["GPSCode"].fillna(mode,inplace=True)

mode=accidents_train["Weather"].mode()[0]
accidents_train["Weather"].fillna(mode,inplace=True)

mode=accidents_train["CollisionType"].mode()[0]
accidents_train["CollisionType"].fillna(mode,inplace=True)

accidents_train["Latitude"] = accidents_train["Latitude"].fillna(accidents_train["Latitude"].median())
accidents_train["Longitude"] = accidents_train["Longitude"].fillna(accidents_train["Longitude"].median())
print(f"accident_train check null values:\n{accidents_train.isnull().sum()}")
print("#"*150)
print("\n\n")
print(f"Places_train shape: {places_train.shape}")
print(f"Places_train check null values:\n{places_train.isnull().sum()}")
print(f"Places_train info: {places_train.info()}")

mode=places_train["RoadNumber"].mode()[0]
places_train["RoadNumber"].fillna(mode,inplace=True)
mean=places_train["RoadSecNumber"].mean()
places_train["RoadSecNumber"].fillna(mean,inplace=True)
mode=places_train["RoadLetter"].mode()[0]
places_train["RoadLetter"].fillna(mode,inplace=True)
mode=places_train["Circulation"].mode()[0]
places_train["Circulation"].fillna(mode,inplace=True)
mode=places_train["LaneNumber"].mode()[0]
places_train["LaneNumber"].fillna(mode,inplace=True)

valid_lanes = ["Reserved", "SeparatedBike", "Bike", "Modified"]
mode_val = "Reserved"
places_train["SpecialLane"] = places_train["SpecialLane"].astype(str).str.strip()
places_train["SpecialLane"] = places_train["SpecialLane"].replace({"50503": mode_val, "0": mode_val, "nan": mode_val, "": mode_val})
places_train["SpecialLane"] = places_train["SpecialLane"].fillna(mode_val)
mode=places_train["Slope"].mode()[0]
places_train["Slope"].fillna(mode,inplace=True)

mode=places_train["RoadMarkerId"].mode()[0]
places_train["RoadMarkerId"].fillna(mode,inplace=True)

mean=places_train["RoadMarkerDistance"].mean()
places_train["RoadMarkerDistance"].fillna(mean,inplace=True)

mode=places_train["Layout"].mode()[0]
places_train["Layout"].fillna(mode,inplace=True)

mean=places_train["StripWidth"].mean()
places_train["StripWidth"].fillna(mean,inplace=True)

mean=places_train["LaneWidth"].mean()
places_train["LaneWidth"].fillna(mean,inplace=True)

mode=places_train["SurfaceCondition"].mode()[0]
places_train["SurfaceCondition"].fillna(mode,inplace=True)

places_train["Infrastructure"] = places_train["Infrastructure"].astype(str).str.strip()
places_train["Infrastructure"] = places_train["Infrastructure"].replace({"0": "Unknown", "-": "Unknown", "nan": "Unknown", "": "Unknown"})
places_train["Infrastructure"] = places_train["Infrastructure"].fillna("Unknown")

mode=places_train["Localization"].mode()[0]
places_train["Localization"].fillna(mode,inplace=True)


places_train["SchoolNear"] = places_train["SchoolNear"].replace({99.0: 2.0, 3.0: 1.0, 0.0: 0.0})
places_train["SchoolNear"] = np.where(np.isfinite(places_train["SchoolNear"]), places_train["SchoolNear"], 0)
places_train["SchoolNear"] = places_train["SchoolNear"].fillna(0).astype(int)
print(f"Places_train check null values:\n{places_train.isnull().sum()}")

print("#"*150)
print("\n\n")
print(f"Users_train shape: {users_train.shape}")
print(f"Users_train check null values:\n{users_train.isnull().sum()}")
print(f"Users_train info: {users_train.info()}")

mode=users_train["Seat"].mode()[0]
users_train["Seat"].fillna(mode,inplace=True)

def clean_gender(x):
    x = str(x).strip().lower()
    if x in ["h", "homme", "m", "1", "male"]:
        return "Male"
    elif x in ["f", "female", "femme", "0"]:
        return "Female"
    else:
        return "Male"

users_train["Gender"] = users_train["Gender"].apply(clean_gender)


mode=users_train["TripReason"].mode()[0]
users_train["TripReason"].fillna(mode,inplace=True)


mode=users_train["SafetyDevice"].mode()[0]
users_train["SafetyDevice"].fillna(mode,inplace=True)

def clean_safety(x):
    x = str(x).strip().lower()
    if x in ["y","yes","1","oui","true"]:
        return "Yes"
    elif x in ["n","no","0","non","false"]:
        return "No"
    else:
        return "Yes"

users_train["SafetyDeviceUsed"] = users_train["SafetyDeviceUsed"].apply(clean_safety)

mode=users_train["PedestrianLocation"].mode()[0]
users_train["PedestrianLocation"].fillna(mode,inplace=True)

mode=users_train["PedestrianAction"].mode()[0]
users_train["PedestrianAction"].fillna(mode,inplace=True)

mode=users_train["PedestrianCompany"].mode()[0]
users_train["PedestrianCompany"].fillna(mode,inplace=True)


mode=users_train["BirthYear"].mode()[0]
users_train["BirthYear"].fillna(mode,inplace=True)

print(f"accident_train check null values:\n{vehicles_train.isnull().sum()}")
print("#"*150)
print("\n\n")
print(f"Vehicles_train shape: {vehicles_train.shape}")
print(f"Vehicles_train check null values:\n{vehicles_train.isnull().sum()}")
print(f"Vehicles_train info: {vehicles_train.info()}")

def clean_direction(x):
    x = str(x).strip().lower()
    if x in ["increasing"]:
        return "Increasing"
    elif x in ["decreasing"]:
        return "Decreasing"
    else:
        return "Increasing"

vehicles_train["Direction"] = vehicles_train["Direction"].apply(clean_direction)

mode=vehicles_train["FixedObstacle"].mode()[0]
vehicles_train["FixedObstacle"].fillna(mode,inplace=True)

mode=vehicles_train["MobileObstacle"].mode()[0]
vehicles_train["MobileObstacle"].fillna(mode,inplace=True)

mode=vehicles_train["ImpactPoint"].mode()[0]
vehicles_train["ImpactPoint"].fillna(mode,inplace=True)

mode=vehicles_train["Maneuver"].mode()[0]
vehicles_train["Maneuver"].fillna(mode,inplace=True)

print(f"Vehicles_train check null values:\n{vehicles_train.isnull().sum()}")


accidents_train["AccidentId"] = accidents_train["AccidentId"].astype(str)
places_train["AccidentId"] = places_train["AccidentId"].astype(str)
users_train["AccidentId"] = users_train["AccidentId"].astype(str)
vehicles_train["AccidentId"] = vehicles_train["AccidentId"].astype(str)

# Merge all 4 files
train = accidents_train.merge(places_train, on="AccidentId", how="left")
train = train.merge(users_train, on="AccidentId", how="left")
train = train.merge(vehicles_train, on="AccidentId", how="left")
train.drop(columns=["AccidentId"],axis=1,inplace=True)
train.head()


pd.set_option("display.max.rows",None)
train["Weather"] = train["Weather"].astype(str).str.strip().str.lower()
weather_map = {
    "normal": "Normal",
    "verygood": "VeryGood",
    "lightrain": "LightRain",
    "heavyrain": "HeavyRain",
    "overcast": "Overcast",
    "fogorsmoke": "FogOrSmoke",
    "snoworhail": "SnowOrHail",
    "strongwindorstorm": "StrongWindOrStorm",
    "other": "Other"
}
train["Weather"] = train["Weather"].map(weather_map)

train["CollisionType"] = train["CollisionType"].astype(str).str.strip().str.lower()
collision_map = {
    "2vehicles-side": "2Vehicles-Side",
    "other": "Other",
    "2vehicles-behind": "2Vehicles-Behind",
    "3+vehicles-chain": "3+Vehicles-Chain",
    "2vehicles-behindvehicles-frontal": "2Vehicles-BehindVehicles-Frontal",
    "3+vehicles-multiple": "3+Vehicles-Multiple",
    "nocollision": "NoCollision"
}
train["CollisionType"] = train["CollisionType"].map(collision_map)


train['Date'] = pd.to_datetime(train['Date'], errors='coerce', dayfirst=True)
train = train.dropna(subset=['Date'])
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day
train['Weekday'] = train['Date'].dt.weekday
train['DayOfYear'] = train['Date'].dt.dayofyear

train.drop(columns=["Date"],axis=1,inplace=True)

train['Hour'] = pd.to_datetime(train['Hour'], format='%H:%M:%S', errors='coerce').dt.time
train['Hour_num'] = pd.to_datetime(train['Hour'].astype(str)).dt.hour
train['Minute'] = pd.to_datetime(train['Hour'].astype(str)).dt.minute
train['Second'] = pd.to_datetime(train['Hour'].astype(str)).dt.second

def time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

train['TimeOfDay'] = train['Hour_num'].apply(time_of_day)
train.drop(columns=["Hour"],axis=1,inplace=True)
train["RoadType"] = train["RoadType"].replace({"Unknown": "Other"})




mode=train["RoadType"].mode()[0]
train["RoadType"].fillna(mode,inplace=True)


mode=train["RoadNumber"].mode()[0]
train["RoadNumber"].fillna(mode,inplace=True)

train.drop(columns=["RoadSecNumber"],axis=1,inplace=True)

mode=train["RoadLetter"].mode()[0]
train["RoadLetter"].fillna(mode,inplace=True)

mode=train["Circulation"].mode()[0]
train["Circulation"].fillna(mode,inplace=True)

counts = train['LaneNumber'].value_counts(normalize=True)
threshold = 0.05
rare_values = counts[counts < threshold].index

train['LaneNumber'] = train['LaneNumber'].apply(lambda x: 0 if x in rare_values else x)

mode=train["LaneNumber"].mode()[0]
train["LaneNumber"].fillna(mode,inplace=True)

mode=train["SpecialLane"].mode()[0]
train["SpecialLane"].fillna(mode,inplace=True)

mode=train["Slope"].mode()[0]
train["Slope"].fillna(mode,inplace=True)

# Replace NaN if any
train['RoadMarkerId'] = train['RoadMarkerId'].fillna(0)
train['RoadMarkerId'] = pd.qcut(train['RoadMarkerId'], q=11, labels=False, duplicates='drop')

# Copy the column
s = train["RoadMarkerDistance"].copy()

# Fill NaN if any
s = s.fillna(0)  # or s.dropna() depending on your strategy

# Calculate upper and lower limits based on IQR
Q1 = s.quantile(0.25)
Q3 = s.quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Clip the values to remove extreme outliers
s_fixed = s.clip(lower=lower_limit, upper=upper_limit)

# Replace in original dataframe
train["RoadMarkerDistance"] = s_fixed

# Count value frequencies
value_counts = train["RoadMarkerDistance"].value_counts()

# Define threshold for rarity
threshold = 5  # values appearing <=5 times are considered rare

# Find rare values
rare_values = value_counts[value_counts <= threshold].index

# Replace rare values in-place with median
median_value = train["RoadMarkerDistance"].median()
train["RoadMarkerDistance"].replace(rare_values, median_value, inplace=True)

# List of invalid/rare values
replace_values = ["Unknown", "-", "0"]

# Find the most frequent category (excluding rare/invalids)
most_frequent = train["Layout"].value_counts().idxmax()  # "Straight"

# Replace invalid/rare values in-place
train["Layout"].replace(replace_values, most_frequent, inplace=True)

# Define bins and labels for two categories
bins = [-1, 40, float('inf')]
labels = [0, 1]

train["StripWidth"] = pd.cut(train["StripWidth"], bins=bins, labels=labels)
train["StripWidth"] = train["StripWidth"].astype('Int64')


mode=train["Layout"].mode()[0]
train["Layout"].fillna(mode,inplace=True)

mode=train["StripWidth"].mode()[0]
train["StripWidth"].fillna(mode,inplace=True)


mode=train["LaneWidth"].mode()[0]
train["LaneWidth"].fillna(mode,inplace=True)

threshold = 5
value_counts = train["LaneWidth"].value_counts()
rare_values = value_counts[value_counts <= threshold].index
median_value = train["LaneWidth"].median()
train["LaneWidth"].replace(rare_values, median_value, inplace=True)




train["SurfaceCondition"].replace(["Unknown", "-", "0"], "Other", inplace=True)
mode=train["SurfaceCondition"].mode()[0]
train["SurfaceCondition"].fillna(mode,inplace=True)

mode_value = train["Infrastructure"].mode()[0]
train["Infrastructure"].fillna(mode_value, inplace=True)

mode_value = train["Localization"].mode()[0]
train["Localization"].fillna(mode_value, inplace=True)

median_value = train["SchoolNear"].median()
train["SchoolNear"].fillna(median_value, inplace=True)

# Define threshold for rare VehicleId_x
threshold = 50  

top_categories = ["A01", "B01", "C01"]
train["VehicleId_x"] = train["VehicleId_x"].apply(lambda x: x if x in top_categories else "C01")


mode_value = train["VehicleId_x"].mode()[0]
train["VehicleId_x"].fillna(mode_value, inplace=True)

# Keep only main Seat categories, replace others with mode
main_seats = ["Conducteur", "Pilote", "Front Left", "Driver"]
mode_seat = train["Seat"].mode()[0]
train["Seat"] = train["Seat"].apply(lambda x: x if x in main_seats else mode_seat)


main_categories = ["Car", "Auto", "Car<=3.5T", "Voiture", "Light Vehicle", "Utility", "Motorbike>125cm3", "Bicycle"]
mode_category = train["Category_x"].mode()[0]
train["Category_x"] = train["Category_x"].apply(lambda x: x if x in main_categories else mode_category)
train["Category_x"] = train["Category_x"].replace("Car<=3.5T","Car")
train["Category_x"] = train["Category_x"].replace("Motorbike>125cm3","Motorbike")


mode_value = train["Gender"].mode()[0]
train["Gender"].fillna(mode_value, inplace=True)


mode_value = train["TripReason"].mode()[0]
train["TripReason"].fillna(mode_value, inplace=True)

mode_value = train["SafetyDevice"].mode()[0]
train["SafetyDevice"].fillna(mode_value, inplace=True)

mode_value = train["SafetyDeviceUsed"].mode()[0]
train["SafetyDeviceUsed"].fillna(mode_value, inplace=True)


top_categories = ["OnCrossingWithLigths", "OnLane<=OnSidewalk0mCrossing"]
train["PedestrianLocation"] = train["PedestrianLocation"].apply(lambda x: x if x in top_categories else top_categories[0])

top_actions = ["Crossing", "Other"]
train["PedestrianAction"] = train["PedestrianAction"].apply(lambda x: x if x in top_actions else top_actions[0])

mode_value = train["PedestrianCompany"].mode()[0]
train["PedestrianCompany"].fillna(mode_value, inplace=True)

import numpy as np

def fix_birth_year(x):
    try:
        x = float(x)
        if x < 1900 or x > 2025:  # treat as invalid
            if 5 <= x <= 120:      # likely age, convert to birth year
                x = 2025 - x
            else:
                return np.nan
        return x
    except:
        return np.nan

train["BirthYear"] = train["BirthYear"].apply(fix_birth_year)
train["BirthYear"] = train["BirthYear"].apply(lambda x: x if 1920 <= x <= 2010 else np.nan)
mode_year = train["BirthYear"].mode()[0]
train["BirthYear"] = train["BirthYear"].fillna(mode_year)
top_categories = ["Î±01", "V01", "Car_01"]
map_to = ["A01", "B01", "C01"]

mapping_dict = dict(zip(top_categories, map_to))

train["VehicleId_y"] = train["VehicleId_y"].apply(lambda x: mapping_dict.get(x, "C01"))
mode_category = train["VehicleId_y"].mode()[0]
train["VehicleId_y"].fillna(mode_category, inplace=True)

mode_year = train["Direction"].mode()[0]
train["Direction"] = train["Direction"].fillna(mode_year)

top_categories = ["Car", "Motorbike>125cm3", "Utility", "Bicycle", "Moped"]
train["Category_y"] = train["Category_y"].replace("Car<=3.5T", "Car")
train["Category_y"] = train["Category_y"].replace("Motorbike>125cm3", "Motorbike")
train["Category_y"] = train["Category_y"].apply(lambda x: x if x in top_categories else top_categories[0])

train["PassengerNumber"] = train["PassengerNumber"].apply(lambda x: 0 if x == 0 else 1)

mode_year = train["FixedObstacle"].mode()[0]
train["FixedObstacle"] = train["FixedObstacle"].fillna(mode_year)


mode_year = train["MobileObstacle"].mode()[0]
train["MobileObstacle"] = train["MobileObstacle"].fillna(mode_year)

mode_year = train["ImpactPoint"].mode()[0]
train["ImpactPoint"] = train["ImpactPoint"].fillna(mode_year)

mode_year = train["Maneuver"].mode()[0]
train["Maneuver"] = train["Maneuver"].fillna(mode_year)

mode_year = train["Hour_num"].mode()[0]
train["Hour_num"] = train["Hour_num"].fillna(mode_year)

mode_year = train["Minute"].mode()[0]
train["Minute"] = train["Minute"].fillna(mode_year)

train.drop(columns=["Second"],axis=1,inplace=True)
train = train.drop(['PedestrianLocation', 'PedestrianAction'], axis=1)

top_5_addresses = train["PostalAddress"].value_counts().head(5).index
train["PostalAddress"] = train["PostalAddress"].apply(lambda x: x if x in top_5_addresses else "Other")
allowed_values = [0, 1, 2, 3, 4, 5]

# Convert RoadNumber to numeric if needed
train['RoadNumber'] = pd.to_numeric(train['RoadNumber'], errors='coerce')
train['RoadNumber'] = train['RoadNumber'].apply(lambda x: x if x in allowed_values else 6)

# Standardize Light column
train['Light'] = train['Light'].str.lower()  # convert everything to lowercase

# Map similar values to clean categories
train['Light'] = train['Light'].replace({
    'daylight': 'Daylight',
    'nightstreelightson': 'NightStreelightsOn',
    'nightnostreetlight': 'NightNoStreetLight',
    'twilightordawn': 'TwilightOrDawn',
    'nightstreelightsoff': 'NightStreelightsOff'
})

train['RoadType'] = train['RoadType'].replace({'Modified': 'Other'})
main_letters = ['A', 'B', 'C']
train['RoadLetter'] = train['RoadLetter'].str.upper()  
train['RoadLetter'] = train['RoadLetter'].apply(lambda x: x if x in main_letters else 'C')
train['Circulation'] = train['Circulation'].replace({'Modified': 'Unknown'})
train.drop(columns=["Seat"],axis=1,inplace=True)

# List of main categories
main_lanes = ['Reserved', 'SeparatedBike', 'Bike']

# Distribute Modified to Reserved (or any logic you prefer)
train['SpecialLane'] = train['SpecialLane'].replace({'Modified': 'Reserved'})

train['Slope'] = train['Slope'].replace({'Modified': 'Unknown'})
train['Layout'] = train['Layout'].replace({'Modified': 'Straight'})
train['SurfaceCondition'] = train['SurfaceCondition'].replace({'Modified': 'Other'})
train['Infrastructure'] = train['Infrastructure'].replace({'Modified': 'Unknown'})
train['Localization'] = train['Localization'].replace({'Modified': 'Unknown'})
allowed_categories = ["Car", "Utility", "Motorbike", "Bicycle"]
train["Category_x"] = train["Category_x"].apply(lambda x: x if x in allowed_categories else "Car")


print('#'*150)
print(f"train data shape: {train.shape}")
print('#'*150)

print(f"train data check null values:\n{train.isnull().sum()}")
print('#'*150)

print(f"train info: {train.info()}")
print('#'*150)



train.head()


# for col in train.select_dtypes(include=['object']).columns:
#     print(f"\n--- {col} ---")
#     print(train[col].value_counts())



train["VehicleId_y"].value_counts()


accidents_test = pd.read_csv('/kaggle/input/etiq-roadsense/accidents_test.csv')
places_test = pd.read_csv('/kaggle/input/etiq-roadsense/places_test.csv')
users_test = pd.read_csv('/kaggle/input/etiq-roadsense/users_test.csv')
vehicles_test = pd.read_csv('/kaggle/input/etiq-roadsense/vehicles_test.csv')

print("#"*150)
print(f"accident_test shape: {accidents_test.shape}")
print(f"accident_test check null values:\n{accidents_test.isnull().sum()}")
print(f"accident_test info: {accidents_test.info()}")

mode=accidents_test["PostalAddress"].mode()[0]
accidents_test["PostalAddress"].fillna(mode,inplace=True)
mode=accidents_test["GPSCode"].mode()[0]
accidents_test["GPSCode"].fillna(mode,inplace=True)

accidents_test["Latitude"] = accidents_test["Latitude"].fillna(accidents_test["Latitude"].median())
accidents_test["Longitude"] = accidents_test["Longitude"].fillna(accidents_test["Longitude"].median())

print(f"accident_train check null values:\n{accidents_test.isnull().sum()}")
print("#"*150)
print("\n\n")
print(f"Places_test shape: {places_test.shape}")
print(f"Places_test check null values:\n{places_test.isnull().sum()}")
print(f"Places_test info: {places_test.info()}")

mode=places_test["RoadNumber"].mode()[0]
places_test["RoadNumber"].fillna(mode,inplace=True)
mean=places_test["RoadSecNumber"].mean()
places_test["RoadSecNumber"].fillna(mean,inplace=True)

mode=places_test["RoadLetter"].mode()[0]
places_test["RoadLetter"].fillna(mode,inplace=True)
mode=places_test["Circulation"].mode()[0]
places_test["Circulation"].fillna(mode,inplace=True)
mode=places_test["LaneNumber"].mode()[0]
places_test["LaneNumber"].fillna(mode,inplace=True)

valid_lanes = ["Reserved", "SeparatedBike", "Bike", "Modified"]
mode_val = "Reserved"
places_test["SpecialLane"] = places_test["SpecialLane"].astype(str).str.strip()
places_test["SpecialLane"] = places_test["SpecialLane"].replace({"50503": mode_val, "0": mode_val, "nan": mode_val, "": mode_val})
places_test["SpecialLane"] = places_test["SpecialLane"].fillna(mode_val)

mode=places_test["Slope"].mode()[0]
places_test["Slope"].fillna(mode,inplace=True)

mode=places_test["RoadMarkerId"].mode()[0]
places_test["RoadMarkerId"].fillna(mode,inplace=True)

mean=places_test["RoadMarkerDistance"].mean()
places_test["RoadMarkerDistance"].fillna(mean,inplace=True)

mode=places_test["Layout"].mode()[0]
places_test["Layout"].fillna(mode,inplace=True)

mean=places_test["StripWidth"].mean()
places_test["StripWidth"].fillna(mean,inplace=True)

mean=places_test["LaneWidth"].mean()
places_test["LaneWidth"].fillna(mean,inplace=True)

mode=places_test["SurfaceCondition"].mode()[0]
places_test["SurfaceCondition"].fillna(mode,inplace=True)

places_test["Infrastructure"] = places_test["Infrastructure"].astype(str).str.strip()
places_test["Infrastructure"] = places_test["Infrastructure"].replace({"0": "Unknown", "-": "Unknown", "nan": "Unknown", "": "Unknown"})
places_test["Infrastructure"] = places_test["Infrastructure"].fillna("Unknown")

mode=places_test["Localization"].mode()[0]
places_test["Localization"].fillna(mode,inplace=True)


places_test["SchoolNear"] = places_test["SchoolNear"].replace({99.0: 2.0, 3.0: 1.0, 0.0: 0.0})
places_test["SchoolNear"] = np.where(np.isfinite(places_test["SchoolNear"]), places_test["SchoolNear"], 0)
places_test["SchoolNear"] = places_test["SchoolNear"].fillna(0).astype(int)

print(f"Places_test check null values:\n{places_test.isnull().sum()}")

print("#"*150)
print("\n\n")
print(f"Users_test shape: {users_test.shape}")
print(f"Users_test check null values:\n{users_test.isnull().sum()}")
print(f"Users_test info: {users_test.info()}")

mode=users_test["Seat"].mode()[0]
users_test["Seat"].fillna(mode,inplace=True)

def clean_gender(x):
    x = str(x).strip().lower()
    if x in ["h", "homme", "m", "1", "male"]:
        return "Male"
    elif x in ["f", "female", "femme", "0"]:
        return "Female"
    else:
        return "Male"

users_test["Gender"] = users_test["Gender"].apply(clean_gender)


mode=users_test["TripReason"].mode()[0]
users_test["TripReason"].fillna(mode,inplace=True)


mode=users_test["SafetyDevice"].mode()[0]
users_test["SafetyDevice"].fillna(mode,inplace=True)

def clean_safety(x):
    x = str(x).strip().lower()
    if x in ["y","yes","1","oui","true"]:
        return "Yes"
    elif x in ["n","no","0","non","false"]:
        return "No"
    else:
        return "Yes"

users_test["SafetyDeviceUsed"] = users_test["SafetyDeviceUsed"].apply(clean_safety)

mode=users_test["PedestrianLocation"].mode()[0]
users_test["PedestrianLocation"].fillna(mode,inplace=True)

mode=users_test["PedestrianAction"].mode()[0]
users_test["PedestrianAction"].fillna(mode,inplace=True)

mode=users_test["PedestrianCompany"].mode()[0]
users_test["PedestrianCompany"].fillna(mode,inplace=True)


mode=users_test["BirthYear"].mode()[0]
users_test["BirthYear"].fillna(mode,inplace=True)


print(f"Users_test check null values:\n{users_test.isnull().sum()}")

print("#"*150)
print("\n\n")

print(f"Vehicles_test shape: {vehicles_test.shape}")
print(f"Vehicles_test check null values:\n{vehicles_test.isnull().sum()}")
print(f"Vehicles_test info: {vehicles_test.info()}")

def clean_direction(x):
    x = str(x).strip().lower()
    if x in ["increasing"]:
        return "Increasing"
    elif x in ["decreasing"]:
        return "Decreasing"
    else:
        return "Increasing"

vehicles_test["Direction"] = vehicles_test["Direction"].apply(clean_direction)

mode=vehicles_test["FixedObstacle"].mode()[0]
vehicles_test["FixedObstacle"].fillna(mode,inplace=True)

mode=vehicles_test["MobileObstacle"].mode()[0]
vehicles_test["MobileObstacle"].fillna(mode,inplace=True)

mode=vehicles_test["ImpactPoint"].mode()[0]
vehicles_test["ImpactPoint"].fillna(mode,inplace=True)

mode=vehicles_test["Maneuver"].mode()[0]
vehicles_test["Maneuver"].fillna(mode,inplace=True)
print(f"Vehicles_test check null values:\n{vehicles_test.isnull().sum()}")


accidents_test["AccidentId"] = accidents_test["AccidentId"].astype(str)
places_test["AccidentId"] = places_test["AccidentId"].astype(str)
users_test["AccidentId"] = users_test["AccidentId"].astype(str)
vehicles_test["AccidentId"] = vehicles_test["AccidentId"].astype(str)

# Merge all 4 files
test = accidents_test.merge(places_test, on="AccidentId", how="left")
test = test.merge(users_test, on="AccidentId", how="left")
test = test.merge(vehicles_test, on="AccidentId", how="left")
Id=test.AccidentId
test.drop(columns=["AccidentId"],axis=1,inplace=True)
test.drop(columns=["RoadSecNumber"],axis=1,inplace=True)
test.head()



test["Weather"] = test["Weather"].astype(str).str.strip().str.lower()
weather_map = {
    "normal": "Normal",
    "verygood": "VeryGood",
    "lightrain": "LightRain",
    "heavyrain": "HeavyRain",
    "overcast": "Overcast",
    "fogorsmoke": "FogOrSmoke",
    "snoworhail": "SnowOrHail",
    "strongwindorstorm": "StrongWindOrStorm",
    "other": "Other"
}
test["Weather"] = test["Weather"].map(weather_map)

test["CollisionType"] = test["CollisionType"].astype(str).str.strip().str.lower()
collision_map = {
    "2vehicles-side": "2Vehicles-Side",
    "other": "Other",
    "2vehicles-behind": "2Vehicles-Behind",
    "3+vehicles-chain": "3+Vehicles-Chain",
    "2vehicles-behindvehicles-frontal": "2Vehicles-BehindVehicles-Frontal",
    "3+vehicles-multiple": "3+Vehicles-Multiple",
    "nocollision": "NoCollision"
}
test["CollisionType"] = test["CollisionType"].map(collision_map)

test['Date'] = pd.to_datetime(test['Date'], errors='coerce', dayfirst=True)
test = test.dropna(subset=['Date'])
test['Year'] = test['Date'].dt.year
test['Month'] = test['Date'].dt.month
test['Day'] = test['Date'].dt.day
test['Weekday'] = test['Date'].dt.weekday
test['DayOfYear'] = test['Date'].dt.dayofyear

test.drop(columns=["Date"],axis=1,inplace=True)

test['Hour'] = pd.to_datetime(test['Hour'], format='%H:%M:%S', errors='coerce').dt.time
test['Hour_num'] = pd.to_datetime(test['Hour'].astype(str)).dt.hour
test['Minute'] = pd.to_datetime(test['Hour'].astype(str)).dt.minute
test['Second'] = pd.to_datetime(test['Hour'].astype(str)).dt.second

def time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

test['TimeOfDay'] = test['Hour_num'].apply(time_of_day)
test.drop(columns=["Hour"],axis=1,inplace=True)
test["RoadType"] = test["RoadType"].replace({"Unknown": "Other"})


counts = test['LaneNumber'].value_counts(normalize=True)
threshold = 0.05
rare_values = counts[counts < threshold].index

test['LaneNumber'] = test['LaneNumber'].apply(lambda x: 0 if x in rare_values else x)

test['RoadMarkerId'] = pd.qcut(test['RoadMarkerId'], q=11, labels=False, duplicates='drop')



# Copy the column
s = test["RoadMarkerDistance"].copy()
# Calculate upper and lower limits based on IQR
Q1 = s.quantile(0.25)
Q3 = s.quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Clip the values to remove extreme outliers
s_fixed = s.clip(lower=lower_limit, upper=upper_limit)

# Replace in original dataframe
test["RoadMarkerDistance"] = s_fixed


# Count value frequencies
value_counts = test["RoadMarkerDistance"].value_counts()

# Define threshold for rarity
threshold = 5  # values appearing <=5 times are considered rare

# Find rare values
rare_values = value_counts[value_counts <= threshold].index

# Replace rare values in-place with median
median_value = test["RoadMarkerDistance"].median()
test["RoadMarkerDistance"].replace(rare_values, median_value, inplace=True)


# List of invalid/rare values
replace_values = ["Unknown", "-", "0"]

# Find the most frequent category (excluding rare/invalids)
most_frequent = test["Layout"].value_counts().idxmax()  # "Straight"

# Replace invalid/rare values in-place
test["Layout"].replace(replace_values, most_frequent, inplace=True)


# Define bins and labels for two categories
bins = [-1, 40, float('inf')]
labels = [0, 1]

test["StripWidth"] = pd.cut(test["StripWidth"], bins=bins, labels=labels)
test["StripWidth"] = test["StripWidth"].astype('Int64')


threshold = 5
value_counts = test["LaneWidth"].value_counts()
rare_values = value_counts[value_counts <= threshold].index
median_value = test["LaneWidth"].median()
test["LaneWidth"].replace(rare_values, median_value, inplace=True)


test["SurfaceCondition"].replace(["Unknown", "-", "0"], "Other", inplace=True)


# Define threshold for rare VehicleId_x
threshold = 50  

# Find rare VehicleId_x values


top_categories = ["A01", "B01", "C01"]
test["VehicleId_x"] = test["VehicleId_x"].apply(lambda x: x if x in top_categories else "C01")


test.drop(columns=["Seat"],axis=1,inplace=True)

main_categories = ["Car", "Auto", "Car<=3.5T", "Voiture", "Light Vehicle", "Utility", "Motorbike>125cm3", "Bicycle"]
mode_category = test["Category_x"].mode()[0]
test["Category_x"] = test["Category_x"].apply(lambda x: x if x in main_categories else mode_category)
test["Category_x"] = test["Category_x"].replace("Car<=3.5T","Car")
test["Category_x"] = test["Category_x"].replace("Motorbike>125cm3","Motorbike")

test = test.drop(['PedestrianLocation', 'PedestrianAction'], axis=1)


def fix_birth_year(x):
    try:
        x = float(x)
        if x < 1900 or x > 2025:  # treat as invalid
            if 5 <= x <= 120:      # likely age, convert to birth year
                x = 2025 - x
            else:
                return np.nan
        return x
    except:
        return np.nan

test["BirthYear"] = test["BirthYear"].apply(fix_birth_year)
test["BirthYear"] = test["BirthYear"].apply(lambda x: x if 1920 <= x <= 2010 else np.nan)
mode_year = test["BirthYear"].mode()[0]
test["BirthYear"] = test["BirthYear"].fillna(mode_year)

# top_vehicle_ids = ["Î±01", "V01", "C01"]
# test["VehicleId_y"] = test["VehicleId_y"].apply(lambda x: x if x in top_vehicle_ids else top_vehicle_ids[0])

top_categories = ["Car", "Motorbike>125cm3", "Utility", "Bicycle", "Moped"]
test["Category_y"] = test["Category_y"].replace("Car<=3.5T", "Car")
test["Category_y"] = test["Category_y"].replace("Motorbike>125cm3", "Motorbike")

test["Category_y"] = test["Category_y"].apply(lambda x: x if x in top_categories else top_categories[0])

test["PassengerNumber"] = test["PassengerNumber"].apply(lambda x: 0 if x == 0 else 1)
test.drop(columns=["Second"],axis=1,inplace=True)
top_5_addresses = ['AUTOROUTE A86', 'A4', 'AUTOROUTE A6', 'A13', 'AUTOROUTE A15']
test['PostalAddress']  = test['PostalAddress'].apply(lambda x: x if x in top_5_addresses else 'Other')
test['RoadNumber']  = pd.to_numeric(test['RoadNumber'], errors='coerce')
test['RoadNumber']  = test['RoadNumber'].apply(lambda x: x if x in allowed_values else 6)
test['RoadLetter'] = test['RoadLetter'].str.upper()
test['RoadLetter'] = test['RoadLetter'].apply(lambda x: x if x in main_letters else 'C')

main_categories = ["A01", "B01", "C01"]

# Get mode from train
mode_category = train["VehicleId_y"].mode()[0]

# Map function
def map_vehicle_test(val):
    if val in main_categories:
        return val
    return np.random.choice(main_categories)

# Apply mapping to test
test["VehicleId_y"] = test["VehicleId_y"].apply(map_vehicle_test)


valid_values = test["MobileObstacle"].unique()
train = train[train["MobileObstacle"].isin(valid_values)]

def keep_top_n_merge_others(df, column, n=3):
    # Find top n categories
    top_n = df[column].value_counts().nlargest(n).index
    # Replace other categories with 'Other'
    df[column] = df[column].apply(lambda x: x if x in top_n else 'Other')
    return df

# Apply to train and test
train = keep_top_n_merge_others(train, 'ImpactPoint', n=3)
test = keep_top_n_merge_others(test, 'ImpactPoint', n=3)

top_features = ['NoDirectionChange', 'SameDirectionOrLane', 'SwerveToLeft', 'Stopped', 'PassLeft']

# Map other categories to 'Other' in train and test
train['Maneuver'] = train['Maneuver'].apply(lambda x: x if x in top_features else 'Other')
test['Maneuver'] = test['Maneuver'].apply(lambda x: x if x in top_features else 'Other')


print('#'*150)
print(f"test data shape: {train.shape}")
print('#'*150)

print(f"test data check null values:\n{train.isnull().sum()}")
print('#'*150)

print(f"test info: {train.info()}")
print('#'*150)




# Compare categorical columns between train and test
for col in train.select_dtypes(include=['object']).columns:
    if col in test.columns:
        train_vals = set(train[col].dropna().unique())
        test_vals = set(test[col].dropna().unique())

        if train_vals != test_vals:
            print(f"\nâš ï¸� Mismatch found in column: {col}")
            print(f"Values in train not in test: {train_vals - test_vals}")
            print(f"Values in test not in train: {test_vals - train_vals}")
        else:
            print(f"\nâœ… {col}: All categorical values match.")
    else:
        print(f"\nğŸš« Column '{col}' not found in test dataset.")



train.head()


test.head()


# # Select only object/string columns
# str_cols = train.select_dtypes(include='object').columns

# # Loop through each string column and print value counts
# for col in str_cols:
#     print(f"Column: {col}")
#     print(train[col].value_counts(dropna=False))  # dropna=False to include NaNs
#     print("-"*50)



unordered_cols = ['IntersectionType', 'Weather', 'CollisionType', 'PostalAddress', 'GPSCode', 
                  'RoadType', 'RoadLetter', 'Circulation', 'SpecialLane', 'Layout', 
                  'SurfaceCondition', 'Infrastructure', 'Localization', 'VehicleId_x', 
                  'Category_x', 'TripReason', 'SafetyDevice', 'PedestrianCompany', 
                  'VehicleId_y', 'Category_y', 'FixedObstacle', 'MobileObstacle', 
                  'ImpactPoint', 'Maneuver', 'TimeOfDay','Light',"Gender","SafetyDeviceUsed",
                  "Direction","InAgglomeration"]



train = pd.get_dummies(train, columns=unordered_cols, drop_first=True)

bool_cols = train.select_dtypes(include='bool').columns
train[bool_cols] = train[bool_cols].astype(int)

slope_mapping = {
    'Flat': 0,
    'Uphill': 1,
    'TopHill': 2,
    'BottomHill': 3,
    'Unknown': -1  # or np.nan
}

train['Slope'] = train['Slope'].map(slope_mapping)
train['Gravity'] = train['Gravity'].map({'NonLethal': 0, 'Lethal': 1})

test = pd.get_dummies(test, columns=unordered_cols, drop_first=True)

bool_cols = test.select_dtypes(include='bool').columns
test[bool_cols] = test[bool_cols].astype(int)

slope_mapping = {
    'Flat': 0,
    'Uphill': 1,
    'TopHill': 2,
    'BottomHill': 3,
    'Unknown': -1  # or np.nan
}

test['Slope'] = test['Slope'].map(slope_mapping)

train.head()


test.head()


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
X = train.drop(columns=['Gravity']).values
y = train['Gravity'].values
X_test = test.values




# ==============================
# CV SETUP
# ==============================
n_splits = 15
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cat = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_lgb = np.zeros(len(y))
pred_cat = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))

# ==============================
# 1. CATBOOST
# ==============================
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{n_splits} - CatBoost")
    model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        eval_metric='F1',
        random_seed=42,
        verbose=False,
        early_stopping_rounds=100,
        thread_count=-1,
        task_type='GPU'           
    )
    model.fit(X[trn_idx], y[trn_idx],eval_set=(X[val_idx], y[val_idx]),use_best_model=True)
    oof_cat[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    pred_cat += model.predict_proba(X_test)[:, 1] / n_splits

print(f"CatBoost OOF F1: {f1_score(y, (oof_cat>0.5).astype(int), average='macro'):.5f}")



# ==============================
# 2. XGBOOST (GPU)
# ==============================
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor'
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{n_splits} - XGBoost GPU")
    dtrain = xgb.DMatrix(X[trn_idx], y[trn_idx])
    dval   = xgb.DMatrix(X[val_idx], y[val_idx])
    model = xgb.train(xgb_params, dtrain,
                      num_boost_round=3000,
                      evals=[(dval, 'val')],
                      early_stopping_rounds=100,
                      verbose_eval=False)
    oof_xgb[val_idx] = model.predict(dval)
    pred_xgb += model.predict(xgb.DMatrix(X_test)) / n_splits

print(f"XGBoost OOF F1: {f1_score(y, (oof_xgb>0.5).astype(int), average='macro'):.5f}")


# ==============================
# 3. LIGHTGBM (GPU)
# ==============================
lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 120,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary',
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'random_state': 42,
    'verbose':-1
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{n_splits} - LightGBM GPU")
    train_set = lgb.Dataset(X[trn_idx], y[trn_idx])
    val_set   = lgb.Dataset(X[val_idx], y[val_idx], reference=train_set)
    model = lgb.train(lgb_params, train_set,valid_sets=[val_set])
    oof_lgb[val_idx] = model.predict(X[val_idx])
    pred_lgb += model.predict(X_test) / n_splits

print(f"LightGBM OOF F1: {f1_score(y, (oof_lgb>0.5).astype(int), average='macro'):.5f}")



# ==============================
# ENSEMBLE
# ==============================
ensemble_oof = (oof_cat + oof_xgb + oof_lgb) / 3
ensemble_pred = (pred_cat + pred_xgb + pred_lgb) / 3
print(f"\nENSEMBLE OOF Macro F1: {f1_score(y, (ensemble_oof>0.5).astype(int), average='macro'):.5f}")

# ==============================
# SUBMISSION
# ==============================
submission = pd.DataFrame({'AccidentId': Id,'Gravity': np.where(ensemble_pred > 0.5, 'Lethal', 'NonLethal')})
submission = submission.drop_duplicates(subset='AccidentId', keep='first')
submission.to_csv('submission.csv', index=False)
print("\nsubmission.csv saved! (GPU + Fixed)")
submission.head()




