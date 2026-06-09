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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor if regression
from sklearn.metrics import accuracy_score  # or mean_squared_error, etc.

# Load datasets
train = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
sample_submission = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv")



print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain Info:")
print(train.info())

print("\nMissing values:\n", train.isnull().sum())

print("\nSummary statistics:")
print(train.describe())

# Plotting target distribution (for classification)
#sns.countplot(x='Lap_Time_Seconds', data=train)
#plt.title("Target Distribution")
#plt.show()



# Loading the CSV into a DataFrame
df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')

# Create a new column for Success Rate (with points per start)
df['Success_Rate'] = df['with_points'] / df['starts']

# Create a new column for Average Starts per Year
df['Average_Starts_Per_Year'] = df['starts'] / df['years_active']


print("DataFrame with new metrics:")
print(df.head())  



df['Composite_Performance'] = (
    df['points'] + 
    df['Championship_Points'] - 
    df['position'] - 
    df['Championship_Position']
)

# View the cleaned-up DataFrame
print("Cleaned-up DataFrame:")
print(df.head())

# Confirm that the composite score is present
print("\nColumns in cleaned-up DataFrame:")
print(df.columns)



weather_score = {"Sunny": 1, "Clear": 1, "Partly cloudy": 2, "Cloudy": 2, "Raining": 3, "Overcast": 2}
track_score = {"Dry": 1, "Wet": 2}
tire_score = {"Soft": 3, "Medium": 2, "Hard": 1}


df["Track_Surface_Index"] = df["track"].map(track_score) * df["Track_Condition"].map(track_score)

df["Tire_Grip_Score"] = df["Tire_Compound_Front"].map(tire_score) + df["Tire_Compound_Rear"].map(tire_score)


df[["weather", "track", "Track_Condition", "Tire_Compound_Front", "Tire_Compound_Rear", "Track_Surface_Index", "Tire_Grip_Score"]].head()


df.head()


penalty_score = {
    "DNS": 0,
    "DNF": 0,
    "+3s": 3,
    "+5s": 5,
    "Ride Through": 22,
    None: 0, 
}

df["Penalty_Severity"] = df["Penalty"].map(penalty_score).fillna(0)


df.head()


df.drop(['points', 'Championship_Points', 'position', 'Championship_Position'], axis=1, inplace=True)
df.drop(['with_points', 'finishes', 'years_active'], axis=1, inplace=True)
df.drop(["track", "Track_Condition", "Tire_Compound_Front", "Tire_Compound_Rear"], axis=1)

# list of columns we don't want
drop_columns = [
    'Unique ID', 'Rider_ID', 'rider_name', 'team_name', 'bike_name',
    'circuit_name', 'shortname', 'min_year', 'max_year', 'Track_Temperature_Celsius'
    , 'podiums', 'wins', 'Penalty'
]

# Perform the drop
df.drop(drop_columns, axis=1, inplace = True)
df.head()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv('final_cleaned_dataset3.csv')

# 1️⃣ General Data Overview
print("Dataset Shape (rows, columns):", df.shape)
print("\nColumn Data Types:")
print(df.dtypes)
print("\nCount of missing values per column:")
print(df.isnull().sum())


# 2️⃣ Univariate Analysis
# Box Plot, Histogram, Pie Chart with Annotations
numerical_df = df.select_dtypes(['number'])

for col in numerical_df.columns:
    mean_val = numerical_df[col].mean()
    max_val = numerical_df[col].max()
    min_val = numerical_df[col].min()

    # Box Plot with Annotations
    box = numerical_df.boxplot(column=col)
    box.annotate(f'mean: {mean_val:.2f}', (1, mean_val), color='red')
    box.annotate(f'max: {max_val:.2f}', (1, max_val), color='green')
    box.annotate(f'min: {min_val:.2f}', (1, min_val), color='blue')
    box.set_title(f'Box Plot of {col}')

    plt.show()

    # Histogram with Mean
    numerical_df[col].hist(bins=20)
    plt.axvline(mean_val, color='red', label='Mean')
    plt.legend()
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# 3️⃣ Bivariate Analysis
# Pair Plot for Numerical Columns
if len(numerical_df.columns) > 5:
    sample_columns = numerical_df.columns[:5]
else:
    sample_columns = numerical_df.columns

sns.pairplot(df[sample_columns]) 
plt.suptitle('Pair Plot of Numerical Columns')
plt.show()


# Calculate correlation matrix
correlation_matrix = df.select_dtypes(['number']).corr()

# Plotting the Heatmap with cleaned-up annotations
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    annot_kws={"size": 8},
    square=True
)

ax.set_title('Correlation Heatmap', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from tqdm import tqdm


df.dtypes


def feature_engineering(df):
    df = df.copy()
    
    weather_score = {"Sunny": 1, "Clear": 1, "Partly cloudy": 2, "Cloudy": 2, "Raining": 3, "Overcast": 2}
    track_score = {"Dry": 1, "Wet": 2}
    tire_score = {"Soft": 3, "Medium": 2, "Hard": 1}
    penalty_score = {"DNS": 0, "DNF": 0, "+3s": 3, "+5s": 5, "Ride Through": 22, None: 0}
    
    df["Track_Surface_Index"] = df["track"].map(track_score) * df["Track_Condition"].map(track_score)
    df["Tire_Grip_Score"] = df["Tire_Compound_Front"].map(tire_score) + df["Tire_Compound_Rear"].map(tire_score)
    df['Composite_Performance'] = df['points'] + df['Championship_Points'] - df['position'] - df['Championship_Position']
    df['Success_Rate'] = df['with_points'] / df['starts']
    df['Average_Starts_Per_Year'] = df['starts'] / df['years_active']
    df["Penalty_Severity"] = df["Penalty"].map(penalty_score).fillna(0)
    
    drop_columns = [
        'points', 'Championship_Points', 'position', 'Championship_Position',
        'with_points', 'finishes', 'years_active',
        "track", "Track_Condition", "Tire_Compound_Front", "Tire_Compound_Rear",
        'Unique ID', 'Rider_ID', 'rider_name', 'team_name', 'bike_name',
        'circuit_name', 'shortname', 'min_year', 'max_year', 'Track_Temperature_Celsius',
        'podiums', 'wins', 'Penalty'
    ]
    df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
    return df


#df = feature_engineering(df)

X = df.drop(columns=["Lap_Time_Seconds"])
y = df["Lap_Time_Seconds"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess 
preprocessor.fit(X_train)
X_train_trans = preprocessor.transform(X_train)
X_val_trans = preprocessor.transform(X_val)

train_data = lgb.Dataset(X_train_trans, label=y_train)
val_data = lgb.Dataset(X_val_trans, label=y_val, reference=train_data)

# 3. Training LightGBM Model with Verbose Logging
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': 7,
    'verbosity': -1
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000
)

# Evaluation
y_pred = model.predict(X_val_trans, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"\n✅ Final RMSE: {rmse:.4f}")


val_df = feature_engineering(val)

# Split features and target
X_val_final = val_df.drop(columns=["Lap_Time_Seconds"])
y_val_final = val_df["Lap_Time_Seconds"]

X_val_final_trans = preprocessor.fit_transform(X_val_final)

# Predict using trained model
y_val_pred = model.predict(X_val_final_trans, num_iteration=model.best_iteration)

# RMSE
val_rmse = np.sqrt(mean_squared_error(y_val_final, y_val_pred))
print(f"\n RMSE for Validation Set: {val_rmse:.4f}")


test_df = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")

# 2. Unique ID for submission later 
submission_ids = test_df["Unique ID"]

test_df = feature_engineering(test_df)
X_test = test_df.copy()

X_test_trans = preprocessor.transform(X_test)
y_test_pred = model.predict(X_test_trans, num_iteration=model.best_iteration)

# 7. Creating submission DataFrame
submission_df = pd.DataFrame({
    "Unique ID": submission_ids,
    "Lap_Time_Seconds": y_test_pred
})

# 8. Saving to CSV
submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv file created successfully!")




