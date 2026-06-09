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


!pip install xgboost


!pip install pytorch-tabnet



# importing all the neccesary libraires 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import missingno as msno
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import scipy.stats as stats
from scipy.stats import skew
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor  
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pytorch_tabnet.tab_model import TabNetRegressor
%matplotlib inline


import warnings
warnings.filterwarnings('ignore')



# Loading the training dataset in chunks (only first 20,000 rows)
def load_data_in_chunks(file_path, chunk_size, max_rows):
    total_loaded = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk
        total_loaded += chunk.shape[0]
        if total_loaded >= max_rows:
            break

# Load first 20,000 rows
podcast_data = pd.concat(load_data_in_chunks('/kaggle/input/playground-series-s5e4/train.csv', 10000, 750000), ignore_index=True)
podcast_data.head(5)



#Data Shape
podcast_data.shape


# Data Types
podcast_data.dtypes


# Null values 
missing = podcast_data.isnull().sum()
missing


# percentage of Null values 
percentage_missing = (missing / len(podcast_data)) * 100
percentage_missing.sort_values(ascending=False)


# Target variable 
podcast_data['Listening_Time_minutes'].notnull().sum()


# Checking the min, max, mean of the dataset
podcast_data.describe(include='all')


# checking skweness 
stats.skew(podcast_data['Listening_Time_minutes'])


# Checking Kurtosis
stats.kurtosis(podcast_data['Listening_Time_minutes'])


# Summary of the numerical columns and correlation to the target variable
target = 'Listening_Time_minutes'
numerical_cols = podcast_data.select_dtypes(include=[np.number]).columns.drop(target)

stats = []

for col in numerical_cols:
    count_non_null = podcast_data[col].count()
    mean = podcast_data[col].mean()
    median = podcast_data[col].median()
    min_val = podcast_data[col].min()
    max_val = podcast_data[col].max()
    std_dev = podcast_data[col].std()
    missing_values = podcast_data[col].isnull().sum()
    skewness = skew(podcast_data[col].dropna())
    correlation = podcast_data[[col, target]].corr().iloc[0, 1]
    
    stats.append({
        'Column': col,
        'Count': count_non_null,
        'Mean': mean,
        'Median': median,
        'Min': min_val,
        'Max': max_val,
        'Std Dev': std_dev,
        'Missing Values': missing_values,
        'Skewness': skewness,
        'Correlation w/ Target': correlation
    })

summary_df = pd.DataFrame(stats)
summary_df


# List of potential ID columns
id_cols = ['id', 'Episode_Title', 'Podcast_Name']  # adjust as needed

for col in id_cols:
    print(f"Column: {col}")
    print("Is Unique?:", podcast_data[col].is_unique)
    print("Number of Unique Values:", podcast_data[col].nunique())
    print()


# Publication_Day
print("Unique Publication Days:", podcast_data['Publication_Day'].unique())
print("\nPublication Day Counts:\n", podcast_data['Publication_Day'].value_counts())

# Publication_Time
print("\nUnique Publication Times:", podcast_data['Publication_Time'].unique())
print("\nPublication Time Counts:\n", podcast_data['Publication_Time'].value_counts())


# 1. Missing percentage for each column
missing_percent = podcast_data.isnull().mean() * 100
print("Missing % per column:\n", missing_percent)

# 2. Columns with >20% missing
print("\nColumns with >20% missing:\n", missing_percent[missing_percent > 20])

# 3. Columns with >50% missing
print("\nColumns with >50% missing:\n", missing_percent[missing_percent > 50])

# 4. Check if rows missing Guest_Popularity_percentage also miss Episode_Length_minutes
both_missing = podcast_data[
    podcast_data['Guest_Popularity_percentage'].isnull() & 
    podcast_data['Episode_Length_minutes'].isnull()
]
print("\nRows missing both Guest_Popularity_percentage and Episode_Length_minutes:", len(both_missing))

# 5. Count rows with any missing value
rows_with_missing = podcast_data.isnull().any(axis=1).sum()
print("\nTotal rows with any missing value:", rows_with_missing)


# Choose columns with missing values
cols_with_missing = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']

# Create a correlation matrix of missing flags
missing_corr = podcast_data[cols_with_missing].isnull().astype(int).corr()

print("Missing Value Co-occurrence (correlation between missingness):")
print(missing_corr)


# List of categorical columns
categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre',
                    'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Print top 5 frequent values for each
for col in categorical_cols:
    print(f"\nTop 5 values for {col}:")
    print(podcast_data[col].value_counts())


# List of numerical columns
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                  'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']

# Dictionary to store outlier counts
z_outlier_counts = {}

# Loop through each column and compute Z-scores and outlier count
for col in numerical_cols:
    col_data = podcast_data[col].dropna()
    z = zscore(col_data)
    outliers = np.sum(np.abs(z) > 3)
    z_outlier_counts[col] = outliers

# Display result
z_outlier_counts


# setting up the graphs 
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


podcast_data['Log_Listen'] = np.log1p(podcast_data['Listening_Time_minutes'])


# Function to plot mean listening time by category
def plot_mean_listening_by_category(df, col_name, cmap='Blues'):
    grouped = df.groupby(col_name)['Listening_Time_minutes'].mean().reset_index()
    grouped = grouped.sort_values('Listening_Time_minutes', ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped[col_name], grouped['Listening_Time_minutes'],
                   color=plt.cm.get_cmap(cmap)(grouped['Listening_Time_minutes'] / grouped['Listening_Time_minutes'].max()))
    plt.xticks(rotation=90)
    plt.title(f'Mean Listening Time by {col_name}')
    plt.xlabel(col_name)
    plt.ylabel('Avg Listening Time (mins)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, round(yval, 2), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()




# Function to plot category distribution
def plot_category_distribution(df, col_name, cmap='Blues'):
    count_data = df[col_name].value_counts().reset_index()
    count_data.columns = [col_name, 'Count']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(count_data[col_name], count_data['Count'],
                   color=plt.cm.get_cmap(cmap)(count_data['Count'] / count_data['Count'].max()))
    plt.xticks(rotation=90)
    plt.title(f'Distribution of {col_name}')
    plt.xlabel(col_name)
    plt.ylabel('Count')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()



# Histogram + boxplot of Listening_Time_minutes
fig, ax = plt.subplots(nrows=2, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]})
sns.histplot(podcast_data['Listening_Time_minutes'], bins=70, kde=False, color='red', ax=ax[0])
ax[0].set_title('Distribution of Listening_Time_minutes')
ax[0].set_ylabel('Number of Records')
sns.boxplot(x=podcast_data['Listening_Time_minutes'], color='red', ax=ax[1])
plt.tight_layout()
plt.show()



# Box plot
plt.figure(figsize=(8, 4))
sns.boxplot(x=podcast_data['Listening_Time_minutes'], color='skyblue')
plt.title('Boxplot of Listening Time')
plt.show()


# KDE + Histogram
plt.figure(figsize=(10, 6))
sns.histplot(podcast_data['Listening_Time_minutes'], bins=100, kde=True, stat="density", color='skyblue')
plt.title('KDE + Histogram of Listening Time')
plt.show()


# Correlation Heatmap
numerical_cols = podcast_data.select_dtypes(include='number')
corr_matrix = numerical_cols.corr()

plt.figure(figsize=(12, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu', center=0)
plt.title('Correlation Heatmap (Numerical Features vs Target)')
plt.show()


# Apply reusable plotting functions
plot_mean_listening_by_category(podcast_data, 'Podcast_Name', cmap='viridis')
plot_mean_listening_by_category(podcast_data, 'Genre', cmap='cividis')
plot_mean_listening_by_category(podcast_data, 'Publication_Day', cmap='plasma')
plot_mean_listening_by_category(podcast_data, 'Publication_Time', cmap='magma')
plot_mean_listening_by_category(podcast_data, 'Episode_Sentiment', cmap='Greens')


plot_category_distribution(podcast_data, 'Genre')
plot_category_distribution(podcast_data, 'Podcast_Name')
plot_category_distribution(podcast_data, 'Publication_Day')
plot_category_distribution(podcast_data, 'Episode_Sentiment')
plot_category_distribution(podcast_data, 'Publication_Time')



# Log-transformed Listening Time Boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=podcast_data['Log_Listen'], color='#BE3D2A')
plt.title('Boxplot of Log Listening Time')
plt.show()


# KDE + Histogram of Log Listening Time
plt.figure(figsize=(10, 6))
sns.histplot(podcast_data['Log_Listen'], bins=100, kde=True, stat="density", color='gray')
plt.title('KDE + Histogram of Log Listening Time')
plt.show()




# RANDOM FOREST: RandomizedSearchCV
rf_model = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf_rand_search = RandomizedSearchCV(
    rf_model,
    rf_param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)
rf_rand_search.fit(X_train, y_train)

# GRID SEARCH
rf_best_params = rf_rand_search.best_params_
rf_grid_search = GridSearchCV(
    rf_model,
    {k: [v] for k, v in rf_best_params.items()},
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1
)
rf_grid_search.fit(X_train, y_train)



# XGBOOST: RandomizedSearchCV
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

xgb_rand_search = RandomizedSearchCV(
    xgb_model,
    xgb_param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)
xgb_rand_search.fit(X_train, y_train)

# GRID SEARCH
xgb_best_params = xgb_rand_search.best_params_
xgb_grid_search = GridSearchCV(
    xgb_model,
    {k: [v] for k, v in xgb_best_params.items()},
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1
)
xgb_grid_search.fit(X_train, y_train)



# GRADIENT BOOSTING: RandomizedSearchCV
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 1.0]
}

gbr_rand_search = RandomizedSearchCV(
    gbr_model,
    gbr_param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)
gbr_rand_search.fit(X_train, y_train)

# GRID SEARCH
gbr_best_params = gbr_rand_search.best_params_
gbr_grid_search = GridSearchCV(
    gbr_model,
    {k: [v] for k, v in gbr_best_params.items()},
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1
)
gbr_grid_search.fit(X_train, y_train)



# ADABOOST: RandomizedSearchCV
ada_model = AdaBoostRegressor(random_state=42)
ada_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 1.0]
}

ada_rand_search = RandomizedSearchCV(
    ada_model,
    ada_param_grid,
    n_iter=6,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)
ada_rand_search.fit(X_train, y_train)

# GRID SEARCH
ada_best_params = ada_rand_search.best_params_
ada_grid_search = GridSearchCV(
    ada_model,
    {k: [v] for k, v in ada_best_params.items()},
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1
)
ada_grid_search.fit(X_train, y_train)




# 1. Load test set
test_original = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test = test_original.copy()

# 2. Drop unnecessary column
test.drop(columns=['Episode_Title'], inplace=True)

# 3. Apply same preprocessing
test['Podcast_Name'] = test['Podcast_Name'].map(freq_map)
test['Episode_Length_minutes'].fillna(data['Episode_Length_minutes'].median(), inplace=True)
test['Guest_Popularity_percentage'].fillna(data['Guest_Popularity_percentage'].median(), inplace=True)
test['Number_of_Ads'].fillna(data['Number_of_Ads'].mode()[0], inplace=True)

# 4. One-hot encode categorical columns
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
categorical_cols = [col for col in categorical_cols if col in test.columns]
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# 5. Align columns with training data
for col in X_train.columns:
    if col not in test.columns:
        test[col] = 0
test = test[X_train.columns]  # Ensure correct column order

# 6. Scale numeric features
test[numeric_cols] = scaler.transform(test[numeric_cols])

# 7. Predict using best Random Forest model
best_model = rf_grid_search.best_estimator_
test_preds = best_model.predict(test)




# âœ… 7. Create submission file
submission = pd.DataFrame({
    'id': test_original['id'],
    'Listening_Time_minutes': test_preds
})
submission.to_csv("submission.csv", index=False)


# 1. Load test set
test_original = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test = test_original.copy()

# 2. Drop unnecessary column
test.drop(columns=['Episode_Title'], inplace=True)

# 3. Apply same preprocessing
test['Podcast_Name'] = test['Podcast_Name'].map(freq_map)
test['Episode_Length_minutes'].fillna(data['Episode_Length_minutes'].median(), inplace=True)
test['Guest_Popularity_percentage'].fillna(data['Guest_Popularity_percentage'].median(), inplace=True)
test['Number_of_Ads'].fillna(data['Number_of_Ads'].mode()[0], inplace=True)

# 4. One-hot encode categorical columns
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
categorical_cols = [col for col in categorical_cols if col in test.columns]
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# 5. Align with training columns
for col in X_train.columns:
    if col not in test.columns:
        test[col] = 0
test = test[X_train.columns]

# 6. Scale numeric columns
test[numeric_cols] = scaler.transform(test[numeric_cols])






from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=-1, random_state=42)
lgb_model.fit(X_train, y_train)

lgb_preds = lgb_model.predict(test)

submission_lgb = pd.DataFrame({
    'id': test_original['id'],
    'Listening_Time_minutes': lgb_preds
})
submission_lgb.to_csv("submission_lgb.csv", index=False)



from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(verbose=0, iterations=500, learning_rate=0.1, depth=6, random_state=42)
cat_model.fit(X_train, y_train)

cat_preds = cat_model.predict(test)

submission_cat = pd.DataFrame({
    'id': test_original['id'],
    'Listening_Time_minutes': cat_preds
})
submission_cat.to_csv("submission_cat.csv", index=False)



from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge


estimators = [
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=50, max_depth=6, objective='reg:squarederror', random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42))
]

stack_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
stack_model.fit(X_train, y_train)

stack_preds = stack_model.predict(test)

submission_stack = pd.DataFrame({
    'id': test_original['id'],
    'Listening_Time_minutes': stack_preds
})
submission_stack.to_csv("submission_stack.csv", index=False)



xgb_preds = xgb_grid_search.best_estimator_.predict(test)
rf_preds = rf_grid_search.best_estimator_.predict(test)

avg_preds = (xgb_preds + rf_preds) / 2

submission_avg = pd.DataFrame({
    'id': test_original['id'],
    'Listening_Time_minutes': avg_preds
})
submission_avg.to_csv("submission_avg.csv", index=False)



# Predict on training data
train_preds = cat_model.predict(X_train)  # Replace 'model' with your model variable

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_r2 = r2_score(y_train, train_preds)

print(f"ğŸ§  Training RMSE: {train_rmse:.4f}")
print(f"ğŸ§  Training RÂ² Score: {train_r2:.4f}")


val_preds = cat_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
val_r2 = r2_score(y_val, val_preds)

print(f"ğŸ§ª Validation RMSE: {val_rmse:.4f}")
print(f"ğŸ§ª Validation RÂ² Score: {val_r2:.4f}")



# 1. Load predictions
cat = pd.read_csv("submission_cat.csv")
stack = pd.read_csv("submission_stack.csv")
test_original = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")  # for IDs

# 2. Average the predictions
avg_preds = (cat["Listening_Time_minutes"] + stack["Listening_Time_minutes"]) / 2

# 3. Clip (optional but recommended to avoid negatives)
avg_preds = np.clip(avg_preds, 0, None)

# 4. Create submission file
submission_avg = pd.DataFrame({
    'id': test_original['id'],
    'Listening_Time_minutes': avg_preds
})

# 5. Save to CSV
submission_avg.to_csv("submission_avg_cat_stack.csv", index=False)
print("âœ… submission_avg_cat_stack.csv saved.")



# Load the dataset
data = podcast_data.copy()

# Drop ID and Episode_Title
data.drop(columns=['id', 'Episode_Title'], inplace=True)

# âœ… Feature Engineering

# 1. Log transform for skewed numerical data
data["log_episode_length"] = np.log1p(data["Episode_Length_minutes"])

# 2. Interaction: Host Ã— Guest Popularity
data["host_guest_interaction"] = data["Host_Popularity_percentage"] * data["Guest_Popularity_percentage"]

# 3. Target encode Podcast_Name
podcast_target = data.groupby("Podcast_Name")["Listening_Time_minutes"].mean()
data["podcast_encoded"] = data["Podcast_Name"].map(podcast_target)

# 4. Weekend flag
data["Is_Weekend"] = data["Publication_Day"].isin(["Saturday", "Sunday"]).astype(int)

# Impute missing values
data["Episode_Length_minutes"].fillna(data["Episode_Length_minutes"].median(), inplace=True)
data["Guest_Popularity_percentage"].fillna(data["Guest_Popularity_percentage"].median(), inplace=True)
data["Number_of_Ads"].fillna(data["Number_of_Ads"].mode()[0], inplace=True)

# One-hot encode categoricals
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Final features and target
X = data.drop(columns=["Listening_Time_minutes", "Podcast_Name"])
y = data["Listening_Time_minutes"]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


cat_model = CatBoostRegressor(verbose=0, iterations=500, learning_rate=0.1, depth=6, random_state=42)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_val)


lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_val)



xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)


# Ensure there are no NaNs in the final train set
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)



ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_preds = ridge_model.predict(X_val)


def print_metrics(name, true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    print(f"ğŸ”� {name} â†’ RMSE: {rmse:.4f} | RÂ²: {r2:.4f}")

print_metrics("CatBoost", y_val, cat_preds)
print_metrics("LightGBM", y_val, lgb_preds)
print_metrics("XGBoost", y_val, xgb_preds)
print_metrics("Ridge", y_val, ridge_preds)



meta_X_train = np.column_stack((cat_preds, lgb_preds, xgb_preds, ridge_preds))
meta_model = Ridge()
meta_model.fit(meta_X_train, y_val)  # Note: using val preds to simulate holdout

# For test later:
# meta_X_test = np.column_stack((cat_test_preds, lgb_test_preds, etc.))
# final_preds = meta_model.predict(meta_X_test)



avg_preds = (cat_preds + lgb_preds + xgb_preds + ridge_preds) / 4
print_metrics("Simple Averaging", y_val, avg_preds)




# 1. Load test set
test_original = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test = test_original.copy()

# 2. Drop unnecessary column
test.drop(columns=['Episode_Title'], inplace=True)

# 3. Feature Engineering (same as training)
test["log_episode_length"] = np.log1p(test["Episode_Length_minutes"])
test["host_guest_interaction"] = test["Host_Popularity_percentage"] * test["Guest_Popularity_percentage"]

# 4. Target Encoding for Podcast_Name using training means
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
podcast_target = train.groupby("Podcast_Name")["Listening_Time_minutes"].mean()
test["podcast_encoded"] = test["Podcast_Name"].map(podcast_target)

# 5. Weekend indicator
test["Is_Weekend"] = test["Publication_Day"].isin(["Saturday", "Sunday"]).astype(int)

# 6. Impute missing values
test["Episode_Length_minutes"].fillna(train["Episode_Length_minutes"].median(), inplace=True)
test["Guest_Popularity_percentage"].fillna(train["Guest_Popularity_percentage"].median(), inplace=True)
test["Number_of_Ads"].fillna(train["Number_of_Ads"].mode()[0], inplace=True)

# 7. One-hot encode categorical variables
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# 8. Drop Podcast_Name (already encoded via target encoding)
test.drop(columns=["Podcast_Name"], inplace=True)

# 9. Align test columns with training columns (assumes `X.columns` from training)
missing_cols = set(X.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0
test = test[X.columns]  # ensure same column order

# 10. Final cleanup
test.fillna(0, inplace=True)  # safety for models like Ridge

# âœ… `test` is now fully preprocessed and ready for prediction
# âœ… use test_original["id"] for submission file later


# ğŸ”® Predict on test set
cat_test_preds   = cat_model.predict(test)
lgb_test_preds   = lgb_model.predict(test)
xgb_test_preds   = xgb_model.predict(test)
ridge_test_preds = ridge_model.predict(test)



# Simple average of all four models
avg_preds = (cat_test_preds + lgb_test_preds + xgb_test_preds) / 3

# Create submission for average
pd.DataFrame({
    "id": test_original["id"],
    "Listening_Time_minutes": avg_preds
}).to_csv("submission_avg_of_ensembelers_second.csv", index=False)



cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=200,
    early_stopping_rounds=50
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_preds = cat_model.predict(X_val)


lgb_model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(X_train, y_train)

xgb_model = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
xgb_model.fit(X_train, y_train)


stack_model = StackingRegressor(
    estimators=[
        ("cat", cat_model),
        ("lgb", lgb_model),
        ("xgb", xgb_model)
    ],
    final_estimator=Ridge(alpha=1.0),
    passthrough=True,
    n_jobs=-1
)
stack_model.fit(X_train, y_train)
stack_preds = stack_model.predict(X_val)


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"ğŸ”� {name} â†’ RMSE: {rmse:.4f} | RÂ²: {r2:.4f}")

evaluate("CatBoost", y_val, cat_preds)
evaluate("Stacking", y_val, stack_preds)


stack_test_preds = stack_model.predict(test)


pd.DataFrame({
    "id": test_original["id"],
    "Listening_Time_minutes": stack_test_preds
}).to_csv("submission_stacking_of_3.csv", index=False)


# Ensure all are float32 and no object dtypes remain
X_train = X_train.select_dtypes(exclude='object').astype(np.float32)
X_val = X_val.select_dtypes(exclude='object').astype(np.float32)




X_train_np = X_train.to_numpy()
X_val_np = X_val.to_numpy()
y_train_np = y_train.to_numpy().reshape(-1, 1)
y_val_np = y_val.to_numpy().reshape(-1, 1)


# 2ï¸�âƒ£ Define TabNet Model
tabnet_model = TabNetRegressor(
    n_d=16, n_a=16,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_params=dict(lr=2e-2),
    seed=42,
    verbose=10
)

# 3ï¸�âƒ£ Train the model
tabnet_model.fit(
    X_train_np, y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    eval_metric=['rmse'],
    max_epochs=200,
    patience=30,
    batch_size=1024,
    virtual_batch_size=128
)



# 4ï¸�âƒ£ Predict and evaluate
tabnet_preds = tabnet_model.predict(X_val_np).flatten()
rmse = mean_squared_error(y_val_np, tabnet_preds, squared=False)
r2 = r2_score(y_val_np, tabnet_preds)

print(f"ğŸ”� TabNet â†’ RMSE: {rmse:.4f} | RÂ²: {r2:.4f}")

