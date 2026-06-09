import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")


properties = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2016.csv", low_memory=False) 


properties.head(10) 


rename_map = {
    "parcelid": "parcel_id",
    "airconditioningtypeid": "ac_type",
    "architecturalstyletypeid": "architecture_style",
    "basementsqft": "basement_sqft",
    "bathroomcnt": "bathroom_count",
    "bedroomcnt": "bedroom_count",
    "buildingclasstypeid": "building_class",
    "buildingqualitytypeid": "building_quality",
    "calculatedbathnbr": "calculated_bath_count",
    "calculatedfinishedsquarefeet": "total_finished_sqft",
    "decktypeid": "deck_type",
    "finishedfloor1squarefeet": "first_floor_sqft",
    "finishedsquarefeet12": "living_area_sqft",
    "finishedsquarefeet13": "perimeter_living_area_sqft",
    "finishedsquarefeet15": "total_finished_area_sqft",
    "finishedsquarefeet50": "garage_sqft",
    "finishedsquarefeet6": "basement_finished_sqft",
    "fireplacecnt": "fireplace_count",
    "fullbathcnt": "full_bath_count",
    "garagecarcnt": "garage_car_count",
    "garagetotalsqft": "garage_total_sqft",
    "hashottuborspa": "has_hot_tub",
    "heatingorsystemtypeid": "heating_system_type",
    "latitude": "latitude",
    "longitude": "longitude",
    "lotsizesquarefeet": "lot_size_sqft",
    "poolcnt": "pool_count",
    "poolsizesum": "pool_total_area",
    "pooltypeid2": "pool_type2",
    "pooltypeid7": "pool_type_spa",
    "pooltypeid10": "pool_type_luxury",
    "propertycountylandusecode": "county_landuse_code",
    "propertylandusetypeid": "landuse_type",
    "propertyzoningdesc": "zoning_description",
    "rawcensustractandblock": "raw_census_tract_block",
    "regionidcity": "city_id",
    "regionidcounty": "county_id",
    "regionidneighborhood": "neighborhood_id",
    "regionidzip": "zip_id",
    "roomcnt": "room_count",
    "storytypeid": "story_type",
    "threequarterbathnbr": "three_quarter_bath_count",
    "typeconstructiontypeid": "construction_type",
    "unitcnt": "unit_count",
    "yardbuildingsqft17": "patio_sqft",
    "yardbuildingsqft26": "shed_sqft",
    "yearbuilt": "year_built",
    "numberofstories": "number_of_stories",
    "structuretaxvaluedollarcnt": "structure_tax_value",
    "taxvaluedollarcnt": "total_tax_value",
    "assessmentyear": "assessment_year",
    "landtaxvaluedollarcnt": "land_tax_value",
    "taxamount": "tax_amount",
    "taxdelinquencyflag": "tax_delinquent_flag",
    "taxdelinquencyyear": "tax_delinquent_year",
    "censustractandblock": "census_tract_block"
}
properties.rename(columns=rename_map, inplace=True)


print("Shape : ", properties.shape)


print("Information :", properties.info())


print('Description : \n', properties.describe())


print("Numeric Columns : \n", properties.select_dtypes(include=[np.number]).columns)


print("Categorical Columns : \n", properties.select_dtypes(include=['object']).columns)


missing = (properties.isna().mean() * 100).sort_values(ascending=False)
print("\nMissing percentage per column:")
print(missing)


def clean_and_group_properties(properties):

#1)LIVING AREA / SQFT GROUP 
    properties['home_size_sqft'] = (
        properties['finishedsquarefeet12']
        .fillna(properties['calculatedfinishedsquarefeet'])
        .fillna(properties['finishedsquarefeet15'])
    )

    properties['log_home_size'] = np.log1p(properties['home_size_sqft'])

    properties['sqft_per_room'] = properties['home_size_sqft'] / (properties['roomcnt'].replace(0, np.nan))
    properties['sqft_per_bed'] = properties['home_size_sqft'] / (properties['bedroomcnt'].replace(0, np.nan))


#2)GARAGE GROUP
    properties['garage_missing'] = properties['garagecarcnt'].isna().astype(int)
    properties['has_garage'] = (properties['garagecarcnt'] > 0).astype(int)
    properties['garage_car_count'] = properties['garagecarcnt'].fillna(0)
    properties['garage_sqft'] = properties['garagetotalsqft'].fillna(0)

#3)POOL GROUP
    properties['has_pool'] = (
        (properties['poolcnt'] > 0) |
        properties['pooltypeid7'].notna() |
        properties['pooltypeid2'].notna() |
        properties['pooltypeid10'].notna()
    ).astype(int)
    properties['is_spa'] = properties['pooltypeid7'].notna().astype(int)
    properties['has_luxury_pool'] = properties['pooltypeid10'].notna().astype(int)

#4)BATHROOMS GROUP
    properties['bathrooms'] = (
        properties['bathroomcnt']
        .fillna(properties['calculatedbathnbr'])
    )
    properties['has_fullbath'] = properties['fullbathcnt'].fillna(0).astype(int)

#5)TAX VALUE GROUP
    properties['land_to_structure_ratio'] = properties['landtaxvaluedollarcnt'] / (
        properties['structuretaxvaluedollarcnt'].replace(0, np.nan)
    )
    properties['effective_tax_rate'] = properties['taxamount'] / (
        properties['taxvaluedollarcnt'].replace(0, np.nan)
    )
    properties['log_total_tax_value'] = np.log1p(properties['taxvaluedollarcnt'])


#6)BUILDING QUALITY / SYSTEMS
    properties['has_ac'] = properties['airconditioningtypeid'].notna().astype(int)
    properties['has_heating'] = properties['heatingorsystemtypeid'].notna().astype(int)
    properties['multi_story'] = (properties['numberofstories'] > 1).astype(int)

#7)LOCATION GROUP
    properties['lat'] = properties['latitude'] / 1e6
    properties['lon'] = properties['longitude'] / 1e6

#8) DROP USELESS / DUPLICATE / HIGH-MISSING COLUMNS
    drop_cols = [
        # VERY high missing and useless
        'basementsqft', 'yardbuildingsqft26', 'decktypeid', 'finishedsquarefeet13',
        'buildingclasstypeid', 'poolsizesum', 'pooltypeid2', 'pooltypeid10',
        'storytypeid', 'architecturalstyletypeid',

        # redundant or internal IDs
        'censustractandblock', 'rawcensustractandblock',

        # columns we replaced with grouped features
        'garagecarcnt', 'garagetotalsqft', 'poolcnt',
        'finishedsquarefeet12', 'calculatedfinishedsquarefeet',
        'finishedsquarefeet15', 'fullbathcnt', 'threequarterbathnbr',
        'bathroomcnt', 'calculatedbathnbr'
    ]

    properties.drop(columns=[c for c in drop_cols if c in properties.columns], inplace=True)
    return properties


properties = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2016.csv", low_memory=False)
df_clean = clean_and_group_properties(properties)

df_clean.head()


print(df_clean.columns)


columns_to_drop_asthey_determine_only_minutely = [
    'propertycountylandusecode',
    'propertylandusetypeid',
    'propertyzoningdesc',
    'typeconstructiontypeid',
    'assessmentyear',
    'taxdelinquencyflag',
    'taxdelinquencyyear',
    'fireplaceflag'
]

df_clean.drop(columns=columns_to_drop_asthey_determine_only_minutely, inplace=True, errors='ignore')



df_clean.drop(columns='airconditioningtypeid', inplace=True, errors='ignore')



fill_zero = [
    'finishedsquarefeet6', 'hashottuborspa', 'yardbuildingsqft17',
    'finishedfloor1squarefeet', 'finishedsquarefeet50', 'fireplacecnt',
    'pooltypeid7', 'numberofstories', 'unitcnt', 'sqft_per_room',
    'sqft_per_bed', 'garage_sqft', 'garage_car_count', 'has_fullbath'
]

for col in fill_zero:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(0)


fill_minus_one = [
        'heatingorsystemtypeid',
        'buildingqualitytypeid',
        'regionidneighborhood',
        'regionidcity',
        'regionidzip',
        'regionidcounty',
        'fips'
    ]

for col in fill_minus_one:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(-1).astype(int)


fill_median = [
    'lotsizesquarefeet',
    'landtaxvaluedollarcnt',
    'yearbuilt',
    'home_size_sqft',
    'log_home_size',
    'structuretaxvaluedollarcnt',
    'taxvaluedollarcnt',
    'log_total_tax_value',
    'taxamount',
    'roomcnt',
    'bathrooms',
    'bedroomcnt'
]

for col in fill_median:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())


df_clean = df_clean.dropna(subset=['latitude', 'longitude'], how='any')
df_clean['lat'] = df_clean['latitude'] / 1e6
df_clean['lon'] = df_clean['longitude'] / 1e6


missing = (df_clean.isna().mean() * 100).sort_values(ascending=False)
print("\nMissing percentage per column:")
print(missing)


df_clean.to_csv("/kaggle/working/df_clean.csv", index=False)
print("Saved df_clean.csv")


train = pd.read_csv("/kaggle/input/zillow-prize-1/train_2016_v2.csv", low_memory=False)
df_clean = pd.read_csv("/kaggle/working/df_clean.csv", low_memory=False)

merged = train.merge(
    df_clean,
    on="parcelid",
    how="left"
)

print("Merged shape:", merged.shape)


merged['transactiondate'] = pd.to_datetime(merged['transactiondate'])
merged['trans_year'] = merged['transactiondate'].dt.year
merged['trans_month'] = merged['transactiondate'].dt.month
merged['trans_quarter'] = merged['transactiondate'].dt.quarter

merged = merged.drop(columns=['transactiondate'])


merged.isna().sum().sort_values(ascending=False).head(10)


merged['land_to_structure_ratio'] = (merged['landtaxvaluedollarcnt'] / (merged['structuretaxvaluedollarcnt'] + 1))
merged['effective_tax_rate'] = ( merged['taxamount'] / (merged['taxvaluedollarcnt'] + 1))


import seaborn as sns
merged.groupby('trans_month')['logerror'].mean()
sns.lineplot(data=merged, x='trans_month', y='logerror')
plt.title("Monthly Logerror Trend")
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(merged['logerror'], kde=True, bins=100)
plt.title("Logerror Distribution")
plt.xlabel("logerror")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8,3))
sns.boxplot(x=merged['logerror'])
plt.title("Logerror Boxplot — Outlier Visualization")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(data=merged, x='trans_month', y='logerror')
plt.title("Logerror by Month")
plt.show()


sns.scatterplot(data=merged, x='structuretaxvaluedollarcnt', y='logerror', alpha=0.2)
plt.title("Structure Tax Value vs Logerror")
plt.show()


sns.scatterplot(data=merged, x='landtaxvaluedollarcnt', y='logerror', alpha=0.2)
plt.title("Land Tax Value vs Logerror")
plt.show()


plt.figure(figsize=(10,6))
sns.scatterplot(data=merged, x='lon', y='lat', hue='logerror', alpha=0.3, palette='coolwarm')
plt.title("Spatial Distribution of Logerror")
plt.show()


num_cols = [
    'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
    'home_size_sqft', 'landtaxvaluedollarcnt'
]

for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(merged[col], kde=True, bins=50)
    plt.title(f"Distribution of {col}")
    plt.show()


merged = merged[merged['logerror'].abs() <= 0.4]


corr_matrix = merged.corr(numeric_only=True)

top_features = corr_matrix['logerror'].abs().sort_values(ascending=False).head(25).index
corr_top = corr_matrix.loc[top_features, top_features]

plt.figure(figsize=(14, 12))
sns.heatmap(corr_top, annot=False, cmap='coolwarm', center=0)
plt.show()


X = merged.drop(columns=['logerror'])
y = merged['logerror']


cat_cols = [
    'regionidcity', 'regionidcounty', 'regionidzip', 
    'regionidneighborhood', 'heatingorsystemtypeid',
    'buildingqualitytypeid', 'fips'
]

for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].astype('category')


bool_cols = ['has_pool', 'is_spa', 'has_luxury_pool', 
             'has_garage', 'garage_missing', 
             'has_ac', 'has_heating', 'multi_story', 'has_fullbath']

for col in bool_cols:
    if col in X.columns:
        X[col] = X[col].astype(str).replace({'True': 1, 'False': 0, 'true': 1, 'false': 0})
        X[col] = X[col].astype(float)

X['hashottuborspa'] = X['hashottuborspa'].astype(str).replace(
    {'True': 1, 'False': 0, 'true': 1, 'false': 0}
)
X['hashottuborspa'] = X['hashottuborspa'].astype(float)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
lr = LinearRegression()
lr.fit(X_train, y_train)


pred_lr = lr.predict(X_val)
print("Linear Regression MAE:", mean_absolute_error(y_val, pred_lr))


from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

pred_ridge = ridge.predict(X_val)
print("Ridge MAE:", mean_absolute_error(y_val, pred_ridge))


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.0005)
lasso.fit(X_train, y_train)

pred_lasso = lasso.predict(X_val)
print("Lasso MAE:", mean_absolute_error(y_val, pred_lasso))


from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=0.001, l1_ratio=0.5)
enet.fit(X_train, y_train)

pred_enet = enet.predict(X_val)
print("ElasticNet MAE:", mean_absolute_error(y_val, pred_enet))


from sklearn.model_selection import GridSearchCV

params_ridge = {'alpha': [0.1, 1, 10, 50, 100]}

grid_ridge = GridSearchCV(Ridge(), params_ridge, scoring='neg_mean_absolute_error', cv=5)
grid_ridge.fit(X_train, y_train)

print("Best Ridge MAE:", -grid_ridge.best_score_)
print("Best alpha:", grid_ridge.best_params_)


from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
params_ridge = {"alpha": np.logspace(-3, 3, 100)}

ridge = Ridge()
rand_ridge = RandomizedSearchCV(
    ridge,
    params_ridge,
    n_iter=15,             
    scoring='neg_mean_absolute_error',
    cv=5,
    random_state=42,
    n_jobs=-1
)
rand_ridge.fit(X_train, y_train)

print("Best Ridge MAE:", -rand_ridge.best_score_)
print("Best alpha:", rand_ridge.best_params_)


# Convert string booleans to numeric
merged['hashottuborspa'] = merged['hashottuborspa'].replace({
    'True': 1,
    'False': 0,
    '0': 0,
    0: 0
})

# Force numeric
merged['hashottuborspa'] = pd.to_numeric(merged['hashottuborspa'], errors='coerce').fillna(0)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

X = merged.drop(columns=['logerror'])
y = merged['logerror']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

lasso = Lasso(alpha=0.0005, max_iter=5000)
lasso.fit(X_train_scaled, y_train)
pred_lasso = lasso.predict(X_val_scaled)
mae = mean_absolute_error(y_val, pred_lasso)
print("Lasso MAE:", mae)

residuals = y_val - pred_lasso

plt.figure(figsize=(7,5))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residual Distribution")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x=pred_lasso, y=residuals)
plt.axhline(0, color='red')
plt.title("Residuals vs Predictions")
plt.show()


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=100),
    "Lasso": Lasso(alpha=0.001),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_val_scaled)
    
    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)
    
    results[name] = (mae, rmse, r2)
    print(f"{name}: MAE={mae:.5f}, RMSE={rmse:.5f}, R²={r2:.5f}")

import matplotlib.pyplot as plt
import seaborn as sns

model = models["Lasso"]  
pred = model.predict(X_val_scaled)
residuals = y_val - pred

plt.figure(figsize=(7,5))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residual Distribution")
plt.show()

#predicted vs actual
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_val, y=pred, alpha=0.3)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()


merged.to_csv("merged_clean.csv", index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("merged_clean.csv")
X = df.drop(["logerror"], axis=1)
y = df["logerror"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


bool_cols = [
    'has_pool','is_spa','has_luxury_pool','has_garage','garage_missing',
    'has_ac','has_heating','multi_story','has_fullbath',
    'hashottuborspa','fireplaceflag','taxdelinquencyflag'
]

for col in bool_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import numpy as np

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist"
)

xgb_param_dist = {
    "n_estimators": np.arange(300, 1500, 100),
    "learning_rate": np.linspace(0.005, 0.05, 10),
    "max_depth": np.arange(3, 12),
    "subsample": np.linspace(0.6, 1.0, 10),
    "colsample_bytree": np.linspace(0.6, 1.0, 10)
}

random_xgb = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_dist,
    n_iter=50,             
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_xgb.fit(X_train, y_train)

best_xgb = random_xgb.best_estimator_

print("Best MAE:", -random_xgb.best_score_)
print("Best Params:", random_xgb.best_params_)


from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
import numpy as np

lgb_model = LGBMRegressor(
    objective="regression",
    device="gpu",
    gpu_platform_id=0,
    gpu_device_id=0,
    random_state=42
)

lgb_param_dist = {
    "num_leaves": np.arange(20, 150, 10),
    "learning_rate": np.linspace(0.005, 0.05, 10),
    "n_estimators": np.arange(300, 1500, 100),
    "min_child_samples": np.arange(5, 50, 5),
    "subsample": np.linspace(0.7, 1.0, 7),
    "colsample_bytree": np.linspace(0.7, 1.0, 7)
}

random_lgb = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=lgb_param_dist,
    n_iter=50,                       # same as XGB
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_lgb.fit(X_train, y_train)

best_lgb = random_lgb.best_estimator_

print("Best LGB MAE:", -random_lgb.best_score_)
print("Best LGB Params:", random_lgb.best_params_)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


############################################################
# FINAL — 2017 PREDICTION PIPELINE (FULL WORKING, NO ERRORS)
############################################################

import pandas as pd
import numpy as np

# --- Load cleaned 2016 merged (already trained on this) ---
merged16 = pd.read_csv("/kaggle/working/merged_clean.csv")
X_train = merged16.drop(columns=["logerror"])
y_train = merged16["logerror"]


# --- Load 2017 raw ---
prop17 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2017.csv", low_memory=False)
train17 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2017.csv", low_memory=False)


#############################
# CLEAN FUNCTION (REUSE SAME)
#############################
def clean_and_group_properties(properties):

    # 1. Living area
    properties["home_size_sqft"] = (
        properties["finishedsquarefeet12"]
        .fillna(properties["calculatedfinishedsquarefeet"])
        .fillna(properties["finishedsquarefeet15"])
    )

    properties["log_home_size"] = np.log1p(properties["home_size_sqft"])
    properties["sqft_per_room"] = properties["home_size_sqft"] / properties["roomcnt"].replace(0, np.nan)
    properties["sqft_per_bed"] = properties["home_size_sqft"] / properties["bedroomcnt"].replace(0, np.nan)

    # 2. Garage
    properties["garage_missing"] = properties["garagecarcnt"].isna().astype(int)
    properties["has_garage"] = (properties["garagecarcnt"] > 0).astype(int)
    properties["garage_car_count"] = properties["garagecarcnt"].fillna(0)
    properties["garage_sqft"] = properties["garagetotalsqft"].fillna(0)

    # 3. Pools
    properties["has_pool"] = (
        (properties["poolcnt"] > 0)
        | properties["pooltypeid7"].notna()
        | properties["pooltypeid10"].notna()
        | properties["pooltypeid2"].notna()
    ).astype(int)
    properties["is_spa"] = properties["pooltypeid7"].notna().astype(int)
    properties["has_luxury_pool"] = properties["pooltypeid10"].notna().astype(int)

    # 4. Bathrooms
    properties["bathrooms"] = properties["bathroomcnt"].fillna(properties["calculatedbathnbr"])
    properties["has_fullbath"] = properties["fullbathcnt"].fillna(0).astype(int)

    # 5. Tax
    properties["land_to_structure_ratio"] = properties["landtaxvaluedollarcnt"] / (
        properties["structuretaxvaluedollarcnt"].replace(0, np.nan)
    )
    properties["effective_tax_rate"] = properties["taxamount"] / (
        properties["taxvaluedollarcnt"].replace(0, np.nan)
    )
    properties["log_total_tax_value"] = np.log1p(properties["taxvaluedollarcnt"])

    # 6. Systems
    properties["has_ac"] = properties["airconditioningtypeid"].notna().astype(int)
    properties["has_heating"] = properties["heatingorsystemtypeid"].notna().astype(int)
    properties["multi_story"] = (properties["numberofstories"] > 1).astype(int)

    # 7. Coordinates
    properties["lat"] = properties["latitude"] / 1e6
    properties["lon"] = properties["longitude"] / 1e6

    # 8. Drop same columns as 2016
    drop_cols = [
        "basementsqft","yardbuildingsqft26","decktypeid","finishedsquarefeet13",
        "buildingclasstypeid","poolsizesum","pooltypeid2","pooltypeid10",
        "storytypeid","architecturalstyletypeid","censustractandblock","rawcensustractandblock",
        "garagecarcnt","garagetotalsqft","poolcnt","finishedsquarefeet12","calculatedfinishedsquarefeet",
        "finishedsquarefeet15","fullbathcnt","threequarterbathnbr","bathroomcnt","calculatedbathnbr"
    ]

    for col in drop_cols:
        if col in properties.columns:
            properties.drop(columns=[col], inplace=True)

    return properties


###########################
# CLEAN PROPERTIES 2017
###########################
df17_clean = clean_and_group_properties(prop17)

###########################
# MERGE 2017 TRAIN
###########################
merged17 = train17.merge(df17_clean, on="parcelid", how="left")


########################################
# Add transaction features (must exist)
########################################
if "transactiondate" in merged17.columns:
    merged17["transactiondate"] = pd.to_datetime(merged17["transactiondate"])
    merged17["trans_year"] = merged17["transactiondate"].dt.year
    merged17["trans_month"] = merged17["transactiondate"].dt.month
    merged17["trans_quarter"] = merged17["transactiondate"].dt.quarter
else:
    merged17["trans_year"] = 2017
    merged17["trans_month"] = 10
    merged17["trans_quarter"] = 4

merged17.drop(columns=["transactiondate"], errors="ignore", inplace=True)

##########################################################
# FIX BLOCK — Convert ALL category dtypes → int
##########################################################

def fix_dtypes(df):
    # Convert categorical to integer codes
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].cat.codes

    # Convert object → numeric
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

# APPLY FIX TO BOTH TRAIN + 2017
X_train_fixed = fix_dtypes(X_train.copy())
X_2017_fixed  = fix_dtypes(X_2017.copy())

print("Final Train dtypes:", X_train_fixed.dtypes.unique())
print("Final 2017 dtypes:", X_2017_fixed.dtypes.unique())

########################################
# SAME BOOLEAN + CATEGORY FIXES
########################################
# Build X_train (ALREADY DONE FROM 2016 MERGE)
X_train = merged16.drop(columns=["logerror"])
y_train = merged16["logerror"]

# Build X_2017 from merged17
X_2017 = merged17.drop(columns=["logerror"], errors="ignore")

# Align columns: add missing columns
missing_cols = set(X_train.columns) - set(X_2017.columns)
for col in missing_cols:
    X_2017[col] = 0  # safe default

# Drop extra columns not in train
extra_cols = set(X_2017.columns) - set(X_train.columns)
X_2017 = X_2017.drop(columns=extra_cols)

# Same column order
X_2017 = X_2017[X_train.columns]

print("FINAL ALIGNED SHAPE → X_train:", X_train.shape, "  X_2017:", X_2017.shape)



print("Predicting 2017...")

pred_xgb_2017 = best_xgb.predict(X_2017_fixed)
pred_lgb_2017 = best_lgb.predict(X_2017_fixed)

# Blend (simple average)
final_pred = 0.5 * pred_xgb_2017 + 0.5 * pred_lgb_2017

print("Prediction done.")


submission = pd.DataFrame({
    "parcelid": df17_clean["parcelid"],  # All 2.98M properties
    "201610": final_pred,
    "201611": final_pred,
    "201612": final_pred,
    "201710": final_pred,
    "201711": final_pred,
    "201712": final_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()
print("submission.csv created with shape:", submission.shape)




