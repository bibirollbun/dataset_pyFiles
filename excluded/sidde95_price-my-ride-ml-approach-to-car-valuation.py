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


train_df = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
sample_df = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')


train_df.sample(4)


test_df.sample(2)


sample_df.sample(2)


print(f"There are {train_df.shape[0]} rows and {train_df.shape[1]} columns")


train_df.info()


(train_df.isnull().sum()/len(train_df))* 100


# Removing rows which have Null value in more than 1 columns 
train_df = train_df[train_df.isna().sum(axis = 1) <= 1]

(train_df.isnull().sum()/len(train_df))* 100


# Accident Column
# As the no of Null values in accident column are very less, so the columns can be dropped
train_df = train_df.dropna(subset = 'accident')
(train_df.isnull().sum()/len(train_df))* 100



train_df.fuel_type.unique()


# Fuel Type Column
# Extract word before "Fuel System"
train_df["fuel_type_extracted"] = train_df["engine"].str.extract(r"(\b\w+\b)(?=\s+Fuel System)")

# If no "Fuel System" present, check for known fuel words
train_df["fuel_type_extracted"] = train_df["fuel_type_extracted"].fillna(
    train_df["engine"].str.extract(r"(Electric|Petrol|Diesel|Hybrid|CNG|LPG|Gasoline|Battery)", expand=False)
)

# Fill existing fuel_type only where it's missing
train_df["fuel_type"] = train_df["fuel_type"].fillna(train_df["fuel_type_extracted"])

# Drop helper column if not needed
train_df = train_df.drop(columns="fuel_type_extracted")


train_df.fuel_type.isnull().sum()


# Dropping the rows having fuel_type as Null
train_df = train_df.dropna(subset = "fuel_type")

(train_df.isnull().sum()/len(train_df))* 100



# clean_title have no dependency on other columns, hence to replace the Null values we will use mode value
train_df.clean_title = train_df.clean_title.fillna(train_df.clean_title.mode()[0])


(train_df.isnull().sum()/len(train_df))* 100


# Extracting information from engine column
# Extract horsepower
train_df["horsepower"] = train_df["engine"].str.extract(r"(\d+(?:\.\d+)?)\s*HP", expand=False)

# Extract engine displacement 
train_df["engine_displacement"] = train_df["engine"].str.extract(r"(\d+(?:\.\d+)?)\s*(?:L|LITER)", expand=False)

# Extract cylinder count
train_df["cylinder_count"] = ( train_df["engine"].astype(str).str.upper().str.extract(
        r"(?:\b(\d+)\s*(?:CYL(?:INDER)?)\b)"      # e.g. "8 Cylinder"
        r"|(?:\bV(\d+)\b)"                        # e.g. "V8"
        r"|(?:\bI(\d+)\b)"                        # e.g. "I4"
        r"|(?:\b(?:STRAIGHT|INLINE|FLAT)\s*(\d+)\b)",  # e.g. "Straight 6"
        expand=True
    )
    .bfill(axis=1)        
    .iloc[:, 0]           
)

train_df["horsepower"] = pd.to_numeric(train_df["horsepower"], errors="coerce")
train_df["engine_displacement"] = pd.to_numeric(train_df["engine_displacement"], errors="coerce")
train_df["cylinder_count"] = pd.to_numeric(train_df["cylinder_count"], errors="coerce").astype("Int64")

train_df.sample(4)



train_df.isnull().sum()/len(train_df) *100


# Removing the rows containing Null values 

train_df = train_df.dropna()
train_df.isnull().sum()/len(train_df) *100


train_df.transmission.unique()


import re

def cleaned_transmission(value):
    val = str(value).upper().strip()

    if re.search(r"\bA/T\b", val) or "AUTOMATIC" in val or "DCT" in val or "DUAL SHIFT MODE" in val or "OVERDRIVE SWITCH" in val or "CVT" in val:
        return "Automatic"
    if re.search(r"\bM/T\b", val) or "MANUAL" in val:
        return "Manual"
    return "Unknown"


train_df.transmission = train_df.transmission.apply(cleaned_transmission)


train_df.fuel_type.unique()


def cleaned_fuel_type(value):
    val = str(value).strip().upper()

    if val in ["GASOLINE", "E85 FLEX FUEL"]:
        return "Gasoline"
    elif val in ["DIESEL"]:
        return "Diesel"
    elif val in ["HYBRID", "PLUG-IN HYBRID"]:
        return "Hybrid"
    elif val in ["ELECTRIC"]:
        return "Electric"
    else:
        return "Unknown"

train_df["fuel_type"] = train_df["fuel_type"].apply(cleaned_fuel_type)


train_df.accident.unique()


train_df.accident = train_df.accident.replace({'None reported': "No", "At least 1 accident or damage reported": "Yes"})


train_df.ext_col.unique()


def clean_exterior_color(value):
    val = str(value).strip().upper()

    if "BLACK" in val or "ONYX" in val or "RAVEN" in val or "EBONY" in val or "NERO" in val or "BELUGA" in val:
        return "Black"
    elif "WHITE" in val or "ALPINE" in val or "BIANCO" in val or "SNOW" in val or "CHALK" in val or "GLACIER" in val:
        return "White"
    elif "GRAY" in val or "GREY" in val or "GRAPHITE" in val or "STEEL" in val or "SLATE" in val or "THUNDER" in val:
        return "Gray"
    elif "SILVER" in val or "PLATINUM" in val:
        return "Silver"
    elif "RED" in val or "ROSSO" in val or "FLAME" in val or "MAROON" in val:
        return "Red"
    elif "BLUE" in val or "BLU" in val or "AQUA" in val or "NAVY" in val:
        return "Blue"
    elif "GREEN" in val:
        return "Green"
    elif "YELLOW" in val or "GOLD" in val:
        return "Yellow"
    elif "ORANGE" in val:
        return "Orange"
    elif "BROWN" in val or "BRONZE" in val:
        return "Brown"
    elif "BEIGE" in val or "TAN" in val or "CREAM" in val or "IVORY" in val:
        return "Beige"
    elif "PURPLE" in val or "PLUM" in val:
        return "Purple"
    elif "PINK" in val:
        return "Pink"
    elif val in ["â€“", "-", "NONE", "UNKNOWN", "NOT SUPPORTED"]:
        return "Unknown"
    else:
        return "Unknown"

train_df.ext_col = train_df.ext_col.apply(clean_exterior_color)


train_df.int_col.unique()


def clean_interior_color(value):
    val = str(value).strip().upper()

    if any(x in val for x in ["BLACK", "EBONY", "ONYX", "NERO", "CHARCOAL", "JET", "BLK"]):
        return "Black"
    elif any(x in val for x in ["GRAY", "SLATE", "GRAPHITE", "ASH", "ANTHRACITE", "STONE"]):
        return "Gray"
    elif any(x in val for x in ["BEIGE", "TAN", "CAMEL", "ALMOND", "PARCHMENT", "LINEN", "SAND", "MACCHIATO"]):
        return "Beige"
    elif any(x in val for x in ["BROWN", "CHESTNUT", "BRANDY", "WALNUT", "AMBER", "COGNAC", "MESA", "ARAGON"]):
        return "Brown"
    elif any(x in val for x in ["RED", "HOTSPUR", "MAGMA", "RIOJA"]):
        return "Red"
    elif any(x in val for x in ["BLUE", "NAVY"]):
        return "Blue"
    elif any(x in val for x in ["WHITE", "OYSTER", "ICE", "CLOUD", "PLATINUM"]):
        return "White"
    elif "GREEN" in val:
        return "Green"
    elif "ORANGE" in val:
        return "Orange"
    elif "YELLOW" in val:
        return "Yellow"
    elif "SILVER" in val:
        return "Silver"
    elif val in ["â€“", "-", "NONE", "UNKNOWN", "NOT SUPPORTED", "N/A"]:
        return "Unknown"
    else:
        return "Unknown"

train_df.int_col = train_df.int_col.apply(clean_interior_color)


# Removing less impacted fuel_type
train_df = train_df[train_df.fuel_type.isin(['Unknown', 'Electric']) == False]

# Removing impacted transmission type
train_df = train_df[train_df.transmission.isin(['Unknown']) == False]

# Removing exterior colors which are less than 20%
train_df = train_df[~train_df.ext_col.isin(['Purple', 'Unknown', 'Pink'])]

# Removing interior colors which are less than 20%
train_df = train_df[~train_df.int_col.isin(['Yellow', 'Green'])]


train_df = train_df.reset_index(drop = True)
train_df.head()


train_df = train_df.drop(['id', 'model', 'engine', 'clean_title'], axis = 1)


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


def countplot(dataframe):
    cat_cols = train_df.select_dtypes(include = 'O').columns
    for i, j in enumerate(cat_cols):
        plt.subplot(len(cat_cols), 1, i+1)
        sns.countplot(data = dataframe, x = j)
        plt.title(f"{j} plot")
        plt.xticks(rotation = 60)
    plt.tight_layout()

plt.figure(figsize = (15, 18))
countplot(train_df)


def boxplot(dataframe):
    num_cols = train_df.select_dtypes(exclude = 'O').columns
    for i, j in enumerate(num_cols):
        plt.subplot(len(num_cols)//3 + 1, 2, i+1)
        sns.boxplot(data = dataframe, x = j)
        plt.title(f"{j} plot")
    plt.tight_layout()

plt.figure(figsize = (15, 10))
boxplot(train_df)


# Removing outlier from the price column

train_df = train_df[train_df.price <= 1_000_000]

plt.figure(figsize = (15, 5))
sns.boxplot(data = train_df, x = "price")


def histplot(dataframe):
    num_cols = train_df.select_dtypes(exclude = 'O').columns
    for i, j in enumerate(num_cols):
        plt.subplot(len(num_cols)//3 + 1, 2, i+1)
        sns.histplot(data = dataframe, x = j, kde = True)
        plt.title(f"{j} plot")
    plt.tight_layout()

plt.figure(figsize = (15, 10))
histplot(train_df)


from sklearn.model_selection import train_test_split

X = train_df.drop('price', axis = 1)
y = train_df.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

scaler = StandardScaler()
ohe = OneHotEncoder(drop = 'first', sparse_output=False)

preprocessing = ColumnTransformer([
    ("OneHotEncoder", ohe, X_train.select_dtypes(include = 'O').columns), 
    ("StandardScaler", scaler, X_train.select_dtypes(exclude = 'O').columns)
])

X_train_scaled = preprocessing.fit_transform(X_train)
X_test_scaled = preprocessing.transform(X_test)


X_train_scaled[0]


from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "LassoCV": LassoCV(),
    "RidgeCV": RidgeCV(),
    "ElasticNet": ElasticNet(),
    "ElasticNetCV": ElasticNetCV(),
    "KNN": KNeighborsRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "GradientBoost": GradientBoostingRegressor(),
    "XGB": XGBRegressor()
}

result_list = []
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r_score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    result_dict= {
        "name" : model_name,
        "r2" : r_score,
        "mse": mse,
        "mae" : mae,
        "rmse" : rmse
    }
    result_list.append(result_dict)

metric_df = pd.DataFrame(result_list)
metric_df


metric_df.sort_values(['r2', 'mse', 'rmse'], ascending = False)


from sklearn.model_selection import KFold, GridSearchCV

cv = KFold(shuffle=True)

model = GradientBoostingRegressor()

gb_params = {
    "n_estimators": [100, 200],          
    "learning_rate": [0.05, 0.1],        
    "max_depth": [3, 4],                 
    "min_samples_split": [2, 5],         
    "min_samples_leaf": [1, 3],          
    "max_features": [None, "sqrt"]       
}

grid_cv = GridSearchCV(estimator=model, param_grid=gb_params, scoring="neg_root_mean_squared_error", cv= cv, n_jobs=-1, verbose=2)
grid_cv.fit(X_train_scaled, y_train)
y_pred = grid_cv.predict(X_test_scaled)

gb_best = grid_cv.best_estimator_
print("Best Parameters:", grid_cv.best_params_)
print("Best RMSE (CV):", -grid_cv.best_score_)


test_df.sample(5)


test_df.isnull().sum()


# Removing rows which have Null value in more than 1 columns 
test_df = test_df[test_df.isna().sum(axis = 1) <= 1]

# Accident Column
# As the no of Null values in accident column are very less, so the columns can be dropped
test_df = test_df.dropna(subset = 'accident')

# Fuel Type Column
# Extract word before "Fuel System"
test_df["fuel_type_extracted"] = test_df["engine"].str.extract(r"(\b\w+\b)(?=\s+Fuel System)")

# If no "Fuel System" present, check for known fuel words
test_df["fuel_type_extracted"] = test_df["fuel_type_extracted"].fillna(
    test_df["engine"].str.extract(r"(Electric|Petrol|Diesel|Hybrid|CNG|LPG|Gasoline|Battery)", expand=False)
)

# Fill existing fuel_type only where it's missing
test_df["fuel_type"] = test_df["fuel_type"].fillna(test_df["fuel_type_extracted"])

# Drop helper column if not needed
test_df = test_df.drop(columns="fuel_type_extracted")


test_df = test_df.dropna(subset = "fuel_type")

test_df.clean_title = test_df.clean_title.fillna(test_df.clean_title.mode()[0])

# Extracting information from engine column
# Extract horsepower
test_df["horsepower"] = test_df["engine"].str.extract(r"(\d+(?:\.\d+)?)\s*HP", expand=False)

# Extract engine displacement 
test_df["engine_displacement"] = test_df["engine"].str.extract(r"(\d+(?:\.\d+)?)\s*(?:L|LITER)", expand=False)

# Extract cylinder count
test_df["cylinder_count"] = ( test_df["engine"].astype(str).str.upper().str.extract(
        r"(?:\b(\d+)\s*(?:CYL(?:INDER)?)\b)"      # e.g. "8 Cylinder"
        r"|(?:\bV(\d+)\b)"                        # e.g. "V8"
        r"|(?:\bI(\d+)\b)"                        # e.g. "I4"
        r"|(?:\b(?:STRAIGHT|INLINE|FLAT)\s*(\d+)\b)",  # e.g. "Straight 6"
        expand=True
    )
    .bfill(axis=1)        
    .iloc[:, 0]           
)

test_df["horsepower"] = pd.to_numeric(test_df["horsepower"], errors="coerce")
test_df["engine_displacement"] = pd.to_numeric(test_df["engine_displacement"], errors="coerce")
test_df["cylinder_count"] = pd.to_numeric(test_df["cylinder_count"], errors="coerce").astype("Int64")

test_df = test_df.dropna()


test_df.isnull().sum()


test_df.sample(3)


import re
def cleaned_transmission(value):
    val = str(value).upper().strip()

    if re.search(r"\bA/T\b", val) or "AUTOMATIC" in val or "DCT" in val or "DUAL SHIFT MODE" in val or "OVERDRIVE SWITCH" in val or "CVT" in val:
        return "Automatic"
    if re.search(r"\bM/T\b", val) or "MANUAL" in val:
        return "Manual"
    return "Unknown"


test_df.transmission = test_df.transmission.apply(cleaned_transmission)

def cleaned_fuel_type(value):
    val = str(value).strip().upper()

    if val in ["GASOLINE", "E85 FLEX FUEL"]:
        return "Gasoline"
    elif val in ["DIESEL"]:
        return "Diesel"
    elif val in ["HYBRID", "PLUG-IN HYBRID"]:
        return "Hybrid"
    elif val in ["ELECTRIC"]:
        return "Electric"
    else:
        return "Unknown"

test_df["fuel_type"] = test_df["fuel_type"].apply(cleaned_fuel_type)

test_df.accident = test_df.accident.replace({'None reported': "No", "At least 1 accident or damage reported": "Yes"})

def clean_exterior_color(value):
    val = str(value).strip().upper()

    if "BLACK" in val or "ONYX" in val or "RAVEN" in val or "EBONY" in val or "NERO" in val or "BELUGA" in val:
        return "Black"
    elif "WHITE" in val or "ALPINE" in val or "BIANCO" in val or "SNOW" in val or "CHALK" in val or "GLACIER" in val:
        return "White"
    elif "GRAY" in val or "GREY" in val or "GRAPHITE" in val or "STEEL" in val or "SLATE" in val or "THUNDER" in val:
        return "Gray"
    elif "SILVER" in val or "PLATINUM" in val:
        return "Silver"
    elif "RED" in val or "ROSSO" in val or "FLAME" in val or "MAROON" in val:
        return "Red"
    elif "BLUE" in val or "BLU" in val or "AQUA" in val or "NAVY" in val:
        return "Blue"
    elif "GREEN" in val:
        return "Green"
    elif "YELLOW" in val or "GOLD" in val:
        return "Yellow"
    elif "ORANGE" in val:
        return "Orange"
    elif "BROWN" in val or "BRONZE" in val:
        return "Brown"
    elif "BEIGE" in val or "TAN" in val or "CREAM" in val or "IVORY" in val:
        return "Beige"
    elif "PURPLE" in val or "PLUM" in val:
        return "Purple"
    elif "PINK" in val:
        return "Pink"
    elif val in ["â€“", "-", "NONE", "UNKNOWN", "NOT SUPPORTED"]:
        return "Unknown"
    else:
        return "Unknown"

test_df.ext_col = test_df.ext_col.apply(clean_exterior_color)


def clean_interior_color(value):
    val = str(value).strip().upper()

    if any(x in val for x in ["BLACK", "EBONY", "ONYX", "NERO", "CHARCOAL", "JET", "BLK"]):
        return "Black"
    elif any(x in val for x in ["GRAY", "SLATE", "GRAPHITE", "ASH", "ANTHRACITE", "STONE"]):
        return "Gray"
    elif any(x in val for x in ["BEIGE", "TAN", "CAMEL", "ALMOND", "PARCHMENT", "LINEN", "SAND", "MACCHIATO"]):
        return "Beige"
    elif any(x in val for x in ["BROWN", "CHESTNUT", "BRANDY", "WALNUT", "AMBER", "COGNAC", "MESA", "ARAGON"]):
        return "Brown"
    elif any(x in val for x in ["RED", "HOTSPUR", "MAGMA", "RIOJA"]):
        return "Red"
    elif any(x in val for x in ["BLUE", "NAVY"]):
        return "Blue"
    elif any(x in val for x in ["WHITE", "OYSTER", "ICE", "CLOUD", "PLATINUM"]):
        return "White"
    elif "GREEN" in val:
        return "Green"
    elif "ORANGE" in val:
        return "Orange"
    elif "YELLOW" in val:
        return "Yellow"
    elif "SILVER" in val:
        return "Silver"
    elif val in ["â€“", "-", "NONE", "UNKNOWN", "NOT SUPPORTED", "N/A"]:
        return "Unknown"
    else:
        return "Unknown"

test_df.int_col = test_df.int_col.apply(clean_interior_color)

# Removing less impacted fuel_type
test_df = test_df[test_df.fuel_type.isin(['Unknown', 'Electric']) == False]

# Removing impacted transmission type
test_df = test_df[test_df.transmission.isin(['Unknown']) == False]

# Removing exterior colors which are less than 20%
test_df = test_df[~test_df.ext_col.isin(['Purple', 'Unknown', 'Pink'])]

# Removing interior colors which are less than 20%
test_df = test_df[~test_df.int_col.isin(['Yellow', 'Green'])]

test_df = test_df.reset_index(drop = True)


test_df_copy = test_df.drop(['id', 'model', 'engine', 'clean_title'], axis = 1)


test_df_copy.sample(3)


test_df_scaled = preprocessing.transform(test_df_copy)
test_df_scaled[0]


y_test_pred = gb_best.predict(test_df_scaled)
y_test_pred


submission = pd.DataFrame({ "id": test_df["id"], "price": y_test_pred})
submission.head()


submission.to_csv("submission.csv", index = False)
print("Submission saved!")


import pickle 

# Saving the data preprocessing pipeline 
with open('preprocessing.pkl', 'wb') as file:
    pickle.dump(preprocessing, file)

# Saving the model
with open('model.pkl', "wb") as file:
    pickle.dump(gb_best, file)

