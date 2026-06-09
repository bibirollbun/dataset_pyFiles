import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))





data_subm = pd.read_csv("/kaggle/input/playground-series-s4e9/sample_submission.csv")
data_train = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
data_test = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")
# data_subm.head()


data_train.head()


data_train.info()


# rename the mi-spelled columns
data_train.rename(columns={'milage':"mileage"}, inplace=True)
data_test.rename(columns={'milage':"mileage"}, inplace=True)


# checking for duplicates
ans = data_train.duplicated().sum()
print("duplicates in training data: ", ans)

ans = data_test.duplicated().sum()
print("duplicates in testing data: ", ans )


# null counts in every columns
data_train.isna().sum()


# checking unique values in fuel_type feature
data_train.fuel_type.unique()


def clean_and_impute_fuel_type(df):
    # replace all known placeholders with NaN
    df['fuel_type'] = df['fuel_type'].replace(['-', '–', 'not supported', 'Not Supported'], pd.NA)

    # replace nan based on engine content
    df.loc[df['fuel_type'].isna() & df['engine'].str.contains("Flex", case=False, na=False), 'fuel_type'] = 'E85 Flex Fuel'
    df.loc[df['fuel_type'].isna() & df['engine'].str.contains("Diesel", case=False, na=False), 'fuel_type'] = 'Diesel'
    df.loc[df['fuel_type'].isna() & df['engine'].str.contains("Gasoline", case=False, na=False), 'fuel_type'] = 'Gasoline'

    # replace nan as Electric based on brand
    electric_brands = ['Tesla', 'Polestar', 'Lucid', 'Rivian']
    pattern = '|'.join(electric_brands)
    df.loc[df['fuel_type'].isna() & df['brand'].str.contains(pattern, case=False, na=False), 'fuel_type'] = 'Electric'

    # replace nan as Hybrid from engine keywords
    hybrid_keywords = ['Electric', 'Dual', 'kW', 'Battery']
    hybrid_pattern = '|'.join(hybrid_keywords)
    df.loc[df['fuel_type'].isna() & df['engine'].str.contains(hybrid_pattern, case=False, na=False), 'fuel_type'] = 'Hybrid'

    # replace nan as Hybrid from model
    df.loc[df['fuel_type'].isna() & df['model'].str.contains("Hybrid", case=False, na=False), 'fuel_type'] = 'Hybrid'

    # replace Any leftover blanks → fill with Hybrid as default
    df['fuel_type'] = df['fuel_type'].fillna('Hybrid')

    return df

data_train = clean_and_impute_fuel_type(data_train)
data_test = clean_and_impute_fuel_type(data_test)

print("Unique fuel types:", data_train['fuel_type'].unique())
print("Remaining NaNs:", data_train['fuel_type'].isna().sum())



# checking for any nan in fuel_type
data_train['fuel_type'].isna().sum()


# chekcing unique values in brands
data_train.brand.unique()


# extracting a col called, brand_category

# premium brand list
premium_brands = [
    'Lexus', 'Acura', 'Volvo', 'MINI', 'INFINITI', 'Jeep', 'Lincoln', 'Genesis',
    'BMW', 'Mercedes-Benz', 'Audi', 'Cadillac', 'Tesla', 'Land', 'Porsche',
    'Polestar', 'Lucid', 'Alfa'
]

# Apply lambda to both datasets
for df in [data_train, data_test]:
    df['brand_category'] = df['brand'].apply(lambda x: 'premium' if x in premium_brands else 'normal')

# Show sample
data_train.brand_category.value_counts()


# checking unique values in model
data_train.model.nunique()


# types of unique values in model_year feature
data_train.model_year.unique()


# unique values in transmission col.
data_train.transmission.unique()


# extracting transmission_type as a broader category
def clean_transmission_type(x):
    x = str(x).lower().strip()

    if any(keyword in x for keyword in ['manual', 'm/t']):
        return 'Manual'
    elif any(keyword in x for keyword in ['automatic', 'a/t', 'cvt', 'auto-shift', 'dct', 'electronically controlled']):
        return 'Automatic'
    elif any(keyword in x for keyword in ['dual', 'dct']):
        return 'Dual'
    elif pd.isna(x) or x in ['-', 'not specified', 'nan']:
        return 'Other'
    else:
        return 'Other'

# Apply to both train and test
data_train['transmission_type'] = data_train['transmission'].apply(clean_transmission_type)
data_test['transmission_type'] = data_test['transmission'].apply(clean_transmission_type)

print(data_train['transmission_type'].unique())



# extracting a base external color
def map_base_ext_color(x):
    x = str(x).lower().strip()
    
    # Assign 'Gray' if value is exactly or nearly alone variant
    if x in ['matte', 'chalk', 'ice', 'gt silver', 'matte grey', 'matte gray', 'chalk grey', 'chalk gray']:
        return 'Gray'
    
    if 'black' in x:
        return 'Black'
    elif 'white' in x:
        return 'White'
    elif 'gray' in x or 'grey' in x:
        return 'Gray'
    elif 'silver' in x:
        return 'Silver'
    elif 'red' in x:
        return 'Red'
    elif 'blue' in x:
        return 'Blue'
    elif 'green' in x:
        return 'Green'
    elif 'yellow' in x or 'gold' in x:
        return 'Yellow'
    elif 'orange' in x:
        return 'Orange'
    elif 'purple' in x:
        return 'Purple'
    elif 'brown' in x or 'bronze' in x:
        return 'Brown'
    elif 'pink' in x:
        return 'Pink'
    else:
        return 'Other'

# Apply the function
data_train['base_ext_col'] = data_train['ext_col'].apply(map_base_ext_color)
data_test['base_ext_col'] = data_test['ext_col'].apply(map_base_ext_color)

data_train['base_ext_col'].unique()


# extracting a base interior color
def map_base_int_color(x):
    x = str(x).lower()
    
    if 'black' in x:
        return 'Black'
    if 'white' in x:
        return 'White'
    
    grey_keywords = ['grey', 'gray', 'silver', 'slate', 'graphite', 'charcoal', 'titanium', 'ash', 'smoke', 'pewter']
    for word in grey_keywords:
        if word in x:
            return 'Grey'
    
    return 'Other'

data_train['base_int_col'] = data_train['int_col'].apply(map_base_int_color)
data_test['base_int_col'] = data_test['int_col'].apply(map_base_int_color)

data_train['base_int_col'].unique()



# checking values in accident
data_train.accident.unique()


data_train['accident'].fillna('None reported',inplace=True)
data_test['accident'].fillna('None reported',inplace=True)


# assigning numeric values to accidnets
def assign_accident_status(x):
    if x == "None reported":
        return "No"
    elif x == "At least 1 accident or damage reported":
        return "Yes"
    else:
        return None

data_train['accident_status'] = data_train['accident'].apply(assign_accident_status)
data_test['accident_status'] = data_test['accident'].apply(assign_accident_status)

data_test['accident_status'].unique()



# checking for nan in accident even after assigning, which can be dropped as number is quite small
data_train.accident.isna().sum()


# checking all values in mileage feature
data_train.mileage.unique()


# extracting the age feature
def assign_age(x):
    return 2025-x

data_train['age'] = data_train['model_year'].apply(assign_age)
data_test['age'] = data_test['model_year'].apply(assign_age)

data_train.age.unique()


# extracting the km_per_year means, average used km.
def assign_km_per_year(x):
    if x['age'] == 0:
        return x['mileage']
    else:
        return x['mileage'] / x['age']

data_train['km_per_year'] = data_train.apply(assign_km_per_year, axis=1)
data_test['km_per_year'] = data_test.apply(assign_km_per_year, axis=1)


# checking all unique values of engine feature
data_train.engine.unique()


# extacting the features from engine column
import re

def extract_engine_features(row):
    engine = str(row).upper()

    # Horsepower
    hp_match = re.search(r'(\d+\.?\d*)HP', engine)
    hp = float(hp_match.group(1)) if hp_match else None

    # Engine Size in Liters
    size_match = re.search(r'(\d+\.?\d*)L', engine)
    size = float(size_match.group(1)) if size_match else None

    # Cylinders
    cyl_match = re.search(r'(\d+)\sCYLINDER', engine)
    cyl = int(cyl_match.group(1)) if cyl_match else None

    return pd.Series([hp, size, cyl])

# Apply to both train and test
for df in [data_train, data_test]:
    df[['horsepower', 'engine_size_l', 'cylinders']] = df['engine'].apply(extract_engine_features)

    # Fill missing values with column means
    df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
    df['engine_size_l'].fillna(df['engine_size_l'].mean(), inplace=True)
    df['cylinders'].fillna(round(df['cylinders'].mean()), inplace=True)

# Show result on training data
print(data_train[['horsepower', 'engine_size_l', 'cylinders']].head())



data_train.columns


data_test.columns





data_train.clean_title.unique()


# updating clean_title column
for df in [data_train, data_test]:
    # Fill NaN clean_title with 'No' if accident history is reported
    df.loc[df['clean_title'].isna() & (df['accident'] == 'At least 1 accident or damage reported'), 'clean_title'] = 'No'
    
    # Fill remaining NaN with 'Yes'
    df['clean_title'].fillna('Yes', inplace=True)


print(data_train['clean_title'].value_counts(dropna=False))



data_train.clean_title.isna().sum()





data_train.columns


# extracting X and y:

X = data_train.drop(columns=['id', 'price', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'brand'])
# X_train.columns

y = data_train['price']

# X_test = data_test.drop(columns=['id', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'brand'])



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)



print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

print()

print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


numeric_cols = X_train.select_dtypes(include=['number']).columns
print(numeric_cols)

print()

categorical_cols = X_train.select_dtypes(include=['object']).columns
print(categorical_cols)


# before encoding and scaling
X_train.head()


# labelling the categorical columns
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[categorical_cols] = oe.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = oe.transform(X_test[categorical_cols])


X_train.head()


# Scaling of numerical columns
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_sc = X_train.copy()
X_test_sc = X_test.copy()

X_train_sc[numeric_cols] = sc.fit_transform(X_train[numeric_cols])
X_test_sc[numeric_cols] = sc.transform(X_test[numeric_cols])


# after labelling and scaling
X_train.head()


X_train_sc.head()


X_train.isna().sum()


X_test.isna().sum()


col_index = list(categorical_cols).index("accident_status")
print("accident_status classes:", oe.categories_[col_index])
print(X_train.accident_status.unique())


X_train.accident_status.value_counts()





X_test


# Required imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Dictionary of models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "SVR (RBF)": SVR(kernel='rbf', C=100, gamma=0.1),
    "SVR (Linear)": SVR(kernel='linear', C=1),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
}

# Initialize empty results list
results = []



for name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
    model = models[name]
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R² Score": round(r2, 4)
    })



# remainning is: "SVR (RBF)", "SVR (Linear)", 
for name in ["ElasticNet", "KNN Regressor"]:
    model = models[name]
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R² Score": round(r2, 4)
    })



results_df = pd.DataFrame(results)
results_df.sort_values(by="R² Score", ascending=False, inplace=True)
print(results_df)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Define tree-based models
tree_models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, verbosity=0)
}

results_tree = []  # To store evaluation metrics



for name, model in tree_models.items():
    model.fit(X_train, y_train)  # Use encoded but **not scaled** data
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results_tree.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R² Score": round(r2, 4)
    })



results_df_tree = pd.DataFrame(results_tree)
results_df_tree.sort_values(by="R² Score", ascending=False, inplace=True)
print(results_df_tree)



print("Done")





from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold

# Use a subset for tuning (optional)
X_sample = X_train_sc[:50000]
y_sample = y_train[:50000]

# Define a small grid to avoid long runtime
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8]
}

# XGBoost model
xgb = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)

# 3-fold CV to save time
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# GridSearch
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           cv=cv,
                           scoring='r2',
                           verbose=1,
                           n_jobs=-1)

# Fit only on the sample data
grid_search.fit(X_sample, y_sample)



# Best params from GridSearch
print("Best Parameters:", grid_search.best_params_)

# Use best estimator on full training set
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train_sc, y_train)



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = best_xgb.predict(X_test_sc)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")



from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Base model
xgb = XGBRegressor(random_state=42, n_jobs=-1)

# Define param grid (keep it small to avoid hanging)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

# Randomized Search with 3-fold CV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=5,          # Try 5 random combinations of params
    cv=3,
    verbose=1,
    n_jobs=-1,
    scoring='r2',
    random_state=42
)

# Fit on training data
random_search.fit(X_train_sc, y_train)

# Best estimator
best_xgb = random_search.best_estimator_

# Predict
y_pred = best_xgb.predict(X_test_sc)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


