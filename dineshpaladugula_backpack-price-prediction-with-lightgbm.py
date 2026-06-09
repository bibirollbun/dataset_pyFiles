import warnings
warnings.filterwarnings('ignore')


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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_tr_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


data_shapes = {
    "Train Shape": df_train.shape,
    "Extra Train Shape": df_tr_extra.shape,
    "Test Shape": df_test.shape
}

print(data_shapes)


df = pd.concat([df_train, df_tr_extra], axis=0, ignore_index=True)
df.head()


for col in df.columns:
    print(df[col].unique())


df.shape


df.columns


df.rename(columns={'Brand':'brand', 'Material':'material', 'Size':'size', 'Compartments':'compartments', 'Laptop Compartment':'laptop_compartment', 'Waterproof':'waterproof', 'Style':'style', 'Color':'color', 'Weight Capacity (kg)':'weight_capacity_kgs', 'Price':'price'}, inplace=True)


df_test.rename(columns={'Brand':'brand', 'Material':'material', 'Size':'size', 'Compartments':'compartments', 'Laptop Compartment':'laptop_compartment', 'Waterproof':'waterproof', 'Style':'style', 'Color':'color', 'Weight Capacity (kg)':'weight_capacity_kgs'}, inplace=True)


df.columns


df.dtypes


df.describe()


df.isnull().sum()


import numpy as np
features_with_na = [feature for feature in df.columns if df[feature].isnull().sum() > 1]
for feature in features_with_na:
    print(f"{feature}: {np.round(df[feature].isnull().mean() * 100, 4)}% missing values")


from matplotlib import pyplot as plt
import seaborn as sns

for feature in features_with_na:
    data = df.copy()
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    data.groupby(feature)['price'].median().plot.bar()
    plt.ylabel('Price')
    plt.title(feature)
    plt.xticks(ticks=[0, 1], labels=['No missing values', 'Missing values'], rotation=0)
    plt.show()


cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
cat_features


numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
print(f"Number of numerical features: {len(numerical_features)}")
df[numerical_features].head()


discrete_feature = [feature for feature in numerical_features if len(df[feature].unique()) < 25 and not feature == 'id']
print(f"Number of discrete features: {len(discrete_feature)}")


for feature in discrete_feature:
    data = df.copy()
    data.groupby(feature)['price'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('price')
    plt.title(feature)
    plt.show()


continuous_feature = [feature for feature in numerical_features if feature not in discrete_feature and not feature == 'id']
print(f"Number of continuous features: {len(continuous_feature)}")


for feature in continuous_feature:
    data = df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()


df['weight_capacity_kgs'].corr(df['price'])


print(df['laptop_compartment'].value_counts())
print(df['waterproof'].value_counts())


import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Laptop Compartment Distribution
sns.countplot(x=df['laptop_compartment'], ax=axes[0], hue = df['laptop_compartment'],palette="viridis")
axes[0].set_title('Laptop Compartment Distribution')
axes[0].set_xlabel('Laptop Compartment (Yes/No)')
axes[0].set_ylabel('Count')

# Waterproof Distribution
sns.countplot(x=df['waterproof'], ax=axes[1], hue=df['waterproof'],palette="viridis")
axes[1].set_title('Waterproof Distribution')
axes[1].set_xlabel('Waterproof (Yes/No)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_features, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=df[col], data=df, hue=df[col],palette='viridis')
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col], y=df['price'], data=df, palette='coolwarm')
    plt.xticks(rotation=45)
    plt.title(f"Price Distribution by {col}")

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_features, 1):
    plt.subplot(3, 3, i)
    avg_price = df.groupby(col)['price'].mean().sort_values()
    sns.barplot(x=avg_price.index, y=avg_price.values, palette='coolwarm')
    plt.xticks(rotation=45)
    plt.title(f"Average Price by {col}")

plt.tight_layout()
plt.show()


sns.boxplot(x=df['brand'], y=df['price'], hue=df['material'])
plt.xticks(rotation=45)
plt.title("Price Distribution by Brand & Material")
plt.show()


df_imputed = df.copy()
df_test_imputed = df_test.copy()


from sklearn.impute import SimpleImputer
def simple_imputer(df, categorical_strategy='most_frequent', numerical_strategy='median'):
    num_f = [feature for feature in df.columns if df[feature].dtypes != 'O']
    cat_f = [feature for feature in df.columns if df[feature].dtypes == 'O']

    if num_f:
        num_imp = SimpleImputer(strategy=numerical_strategy)
        df[num_f] = num_imp.fit_transform(df[num_f])
    if cat_f:    
        cat_imp = SimpleImputer(strategy=categorical_strategy)
        df[cat_f] = cat_imp.fit_transform(df[cat_f])
    return df


df_imputed = simple_imputer(df_imputed)
df_test_imputed = simple_imputer(df_test_imputed)


df_imputed.isnull().sum()


df_test_imputed.isnull().sum()


df_encoded = df_imputed.copy()
df_test_encoded = df_test_imputed.copy()


def binary_cols_imp(df):
    binary_cols = ['laptop_compartment', 'waterproof']
    binary_mapping = {'Yes': 1, 'No': 0}
    df[binary_cols] = df[binary_cols].replace(binary_mapping)
    return df
df_encoded = binary_cols_imp(df_encoded)
df_test_encoded = binary_cols_imp(df_test_encoded)
print('Binary Encoding Completed...')


target='price'
exclude_features=['laptop_compartment', 'waterproof']


from sklearn.model_selection import KFold
def kfold_target_encoding(df, feature, target, n_splits=5):
    encoding = df[feature].copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    target_mean_map = {}

    for train_idx, valid_idx in kf.split(df):
        train_data, valid_data = df.iloc[train_idx], df.iloc[valid_idx]
        target_mean = train_data.groupby(feature)[target].mean()
        target_mean_map[feature] = target_mean
        encoding.iloc[valid_idx] = valid_data[feature].map(target_mean)
    return encoding, target_mean_map
    
target_mean_map = {}
for feature in cat_features:
    if feature not in exclude_features:
        encoded_feature, target_mean_map_feature = kfold_target_encoding(df_encoded, feature, target)
        df_encoded[feature + '_encoded'] = encoded_feature
        target_mean_map[feature] = target_mean_map_feature[feature]
for feature in cat_features:
    if feature not in exclude_features:
        if feature in target_mean_map:
            df_test_encoded[feature + '_encoded'] = df_test_encoded[feature].map(target_mean_map[feature])
        else:
            print(f"Warning: '{feature}' not found in target_mean_map")




df_encoded.head()


df_test_encoded.head()


columns_to_drop = ['brand', 'material', 'size', 'style', 'color']
df_encoded = df_encoded.drop(columns=columns_to_drop)
df_test_encoded = df_test_encoded.drop(columns=columns_to_drop)


df_encoded.dtypes


encoded_columns = ['brand_encoded', 'material_encoded', 'size_encoded', 'style_encoded', 'color_encoded']
for col in encoded_columns:
    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
print(df_encoded.dtypes)


df_test_encoded.dtypes


from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=['price'])  # Features
y = df_encoded['price']  # Target variable

X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

models = {
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostRegressor(n_estimators=100, learning_rate=0.1, verbose=0, random_state=42)
}

rmse_results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('regressor', model)
    ])
    
    pipeline.fit(X_train_split, y_train_split)
    
    y_train_pred = pipeline.predict(X_train_split)
    y_valid_pred = pipeline.predict(X_valid_split)
    
    rmse_train = np.sqrt(mean_squared_error(y_train_split, y_train_pred))
    rmse_valid = np.sqrt(mean_squared_error(y_valid_split, y_valid_pred))
    
    rmse_results[name] = {"Train RMSE": rmse_train, "Validation RMSE": rmse_valid}
    
    print(f"{name}: Train RMSE = {rmse_train:.7f} | Validation RMSE = {rmse_valid:.7f}")

# Compare RMSE results
rmse_df = pd.DataFrame(rmse_results).T
print("\nðŸ“Š RMSE Comparison Table:\n", rmse_df)



best_model_name = min(rmse_results, key=lambda k: rmse_results[k]["Validation RMSE"])
best_model = models[best_model_name]

print(f"Best Model: {best_model_name} with Validation RMSE = {rmse_results[best_model_name]['Validation RMSE']:.7f}")


test_predictions = best_model.predict(df_test_encoded)


# Ensure 'id' column exists
submission = pd.DataFrame({'id': df_test_encoded['id'], 'price': test_predictions})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved successfully!")

