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


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv("/kaggle/input/bpackp/Backpack-price-prediction-main/dataset.csv")
train['Compartments'] = train['Compartments'].fillna(train['Compartments'].median())
features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_category = encoder.fit_transform(train[features])
encoded_df = pd.DataFrame(encoded_category, columns=encoder.get_feature_names_out(features))

train = train.drop(columns=features).reset_index(drop=True)
train = pd.concat([train, encoded_df], axis=1)
train.head()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# === 1. Încarcă datele ===
train = pd.read_csv("/kaggle/input/bpackp/Backpack-price-prediction-main/dataset.csv")

# === 2. Completează valorile lipsă în 'Compartments' (numerice) ===
train['Compartments'] = train['Compartments'].fillna(train['Compartments'].median())

# === 3. Selectează coloanele categorice pentru encoding ===
cat_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# === 4. One-hot encoding ===
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_category = encoder.fit_transform(train[cat_features])
encoded_df = pd.DataFrame(encoded_category, columns=encoder.get_feature_names_out(cat_features))

# === 5. Combină datele codificate cu restul ===
train = train.drop(columns=cat_features).reset_index(drop=True)
train = pd.concat([train, encoded_df], axis=1)

# === 6. Elimină rândurile fără 'Price' (valoarea țintă) ===
train = train.dropna(subset=['Price'])

# === 7. Separă X și y ===
X = train.drop(columns=['Price'])
y = train['Price']

# === 8. Imputează valorile lipsă din X ===
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# === 9. Împarte în seturi de antrenare/testare ===
xtr, xte, ytr, yte = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# === 10. Modelul Random Forest ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(xtr, ytr)

# === 11. Predicții și metrici ===
y_pred = model.predict(xte)

mse = mean_squared_error(yte, y_pred)
mae = mean_absolute_error(yte, y_pred)

print(f"✅ Mean Squared Error: {mse:.2f}")
print(f"✅ Mean Absolute Error: {mae:.2f}")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

train = pd.read_csv("/kaggle/input/bpackp/Backpack-price-prediction-main/dataset.csv")

train['Compartments'] = train['Compartments'].fillna(train['Compartments'].median())

train['Total_Compartments'] = train['Compartments'] + train['Laptop Compartment'].map({'Yes': 1, 'No': 0})

categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_category = encoder.fit_transform(train[categorical_features])
encoded_df = pd.DataFrame(encoded_category, columns=encoder.get_feature_names_out(categorical_features))

train = train.drop(columns=categorical_features).reset_index(drop=True)
train = pd.concat([train, encoded_df], axis=1)

train = train.dropna(subset=['Price'])

X = train.drop(columns=['Price'])
y = train['Price']

xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(xtr, ytr)

y_pred = model.predict(xte)

mse = mean_squared_error(yte, y_pred)
mae = mean_absolute_error(yte, y_pred)

print(f"Prediction: {y_pred[:50]}")

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
# feature_importances.nlargest(10).plot(kind='barh')
# plt.title("Top 10 Feature Importances")
# plt.show()



import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
extra_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# train_df.head()
# extra_df.head()
# test_df.head()

def missing_values(df):
    if df.isnull().sum().sum() > 0:
        print(df.isnull().sum())
    else:
        print("No missing values")

# missing_values(train_df)
# missing_values(extra_df)
# missing_values(test_df)



train = pd.concat([train_df, extra_df])

print(f"Obs: ", train.shape[0])
print(f"Features: " ,train.shape[1])


train.drop('id', axis=1, inplace=True)
train.drop_duplicates(inplace=True)
missing_values(train)


train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].median(), inplace=True)
test_df['Weight Capacity (kg)'].fillna(test_df['Weight Capacity (kg)'].median(), inplace=True)
# missing_values(train_df)
# missing_values(test_df)
train_df.fillna('Unknown', inplace=True)
test_df.fillna('Unknown', inplace=True)
# missing_values(train_df)
# missing_values(test_df)


print(f"Mean: ", train['Price'].mean())
print(f"Median: ", train['Price'].median())
print(f"Standard Deviation: ", train['Price'].std())
print(f"Minimum: ", train['Price'].min())
print(f"Maximum: ", train['Price'].max())
print(f"Change in Price: ", train['Price'].max() - train['Price'].min())


brand_price = train.groupby('Brand')['Price'].mean().sort_values(ascending=False).reset_index()
brand_count = train['Brand'].value_counts().reset_index()
brand_count.columns = ['Brand', 'Count']
brand_stats = pd.merge(brand_price, brand_count, on='Brand')
print(brand_stats)


compartment_price = train.groupby('Compartments')['Price'].mean().sort_values(ascending=False).reset_index()
compartment_count = train['Compartments'].value_counts().reset_index()
compartment_count.columns = ['Compartments', 'Count']
compartment_stats = pd.merge(compartment_price, compartment_count, on='Compartments')
print(compartment_stats)


material_price = train.groupby('Material')['Price'].mean().sort_values(ascending=False).reset_index()
material_count = train['Material'].value_counts().reset_index()
material_count.columns = ['Material', 'Count']
material_stats = pd.merge(material_price, material_count, on='Material')
print(material_stats)


size_price = train.groupby('Size')['Price'].mean().sort_values(ascending=False).reset_index()
size_count = train['Size'].value_counts().reset_index()
size_count.columns = ['Size', 'Count']
size_stats = pd.merge(size_price, size_count, on='Size')
print(size_stats)


laptop_compartment_price = train.groupby('Laptop Compartment')['Price'].mean().sort_values(ascending=False).reset_index()
laptop_compartment_count = train['Laptop Compartment'].value_counts().reset_index()
laptop_compartment_count.columns = ['Laptop Compartment', 'Count']
laptop_compartment_stats = pd.merge(laptop_compartment_price, laptop_compartment_count, on='Laptop Compartment')
print(laptop_compartment_stats)


waterproof_price = train.groupby('Waterproof')['Price'].mean().sort_values(ascending=False).reset_index()
waterproof_count = train['Waterproof'].value_counts().reset_index()
waterproof_count.columns = ['Waterproof', 'Count']
waterproof_stats = pd.merge(waterproof_price, waterproof_count, on='Waterproof')
print(waterproof_stats)


style_price = train.groupby('Style')['Price'].mean().sort_values(ascending=False).reset_index()
style_count = train['Style'].value_counts().reset_index()
style_count.columns = ['Style', 'Count']
style_stats = pd.merge(style_price, style_count, on='Style')
print(style_stats)


color_price = train.groupby('Color')['Price'].mean().sort_values(ascending=False).reset_index()
color_count = train['Color'].value_counts().reset_index()
color_count.columns = ['Color', 'Count']
color_stats = pd.merge(color_price, color_count, on='Color')
print(color_stats)


weight_capacity_price = train.groupby('Weight Capacity (kg)')['Price'].mean().sort_values(ascending=False).reset_index()
weight_capacity_count = train['Weight Capacity (kg)'].value_counts().reset_index()
weight_capacity_count.columns = ['Weight Capacity (kg)', 'Count']
weight_capacity_stats = pd.merge(weight_capacity_price, weight_capacity_count, on='Weight Capacity (kg)')
print(weight_capacity_stats)


train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)
# train_df.head() 
# test_df.head()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = train_df.drop('Price', axis=1)
y = train_df['Price']
X_train, X_val, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(X_train)
x_val_scaler = scaler.transform(X_val)



import numpy as np
from sklearn.metrics import r2_score
def model_evaluation(model):
    model.fit(x_train_scaler, y_train)
    y_pred = model.predict(x_train_scaler)
    y_value_pred = model.predict(x_val_scaler)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


decision_tree = DecisionTreeRegressor()
model_evaluation(decision_tree)



import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Separate input (X) and target variable (y)
X = train_df.drop('Price', axis=1)
y = train_df['Price']

# Ensure a proper train-test split (before scaling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features properly
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # Fit on training data
X_test_s = scaler.transform(X_test)  # Transform test data only

# Model evaluation function
def model_evaluation(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Predictions
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, mae, r2

# Train Decision Tree with constraints to prevent overfitting
decision_tree = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
dt_mse, dt_rmse, dt_mae, dt_r2 = model_evaluation(decision_tree, X_train_s, y_train, X_test_s, y_test)

# Print results with more decimal places
print(f"Decision Tree Mean Squared Error: {dt_mse:.7f}")
print(f"Decision Tree Root Mean Squared Error: {dt_rmse:.7f}")
print(f"Decision Tree Mean Absolute Error: {dt_mae:.7f}")
print(f"Decision Tree R² Score: {dt_r2:.7f}")



train.to_csv("processed_dataset.csv", index=False)


