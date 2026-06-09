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


# Data manipulation and analysis
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning models
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import time



# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv') 
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv') 
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

# Show basic info
print("Train Shape:", train_data.shape)
print("Test Shape:", test_data.shape)
print(train_data.info())
print(train_data.isnull().sum())



train_data.head()


# Get the shape of each dataset
print(f'Train Data Shape: {train_data.shape}')
print(f'Test Data Shape: {test_data.shape}')


# Get information about the train dataset
print(train_data.info())

# Check for missing values in train dataset
print(train_data.isnull().sum())


# Get a statistical summary of the train dataset

train_data.describe()


# Plot distributions of numerical features
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories'] 

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train_data[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# # Create age bins
# bins = [0, 20, 30, 40, 50, 60, 70, 80]
# labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
# train_data['Age_Group'] = pd.cut(train_data['Age'], bins=bins, labels=labels)

# # Group by age and calculate average calories
# age_group_calories = train_data.groupby('Age_Group')['Calories'].mean().reset_index()

# plt.figure(figsize=(10, 6))
# sns.barplot(x='Age_Group', y='Calories', data=age_group_calories)
# plt.title('Average Calories Burned by Age Group')
# plt.ylabel('Average Calories')
# plt.xlabel('Age Group')
# plt.show()


# Pairplot to visualize relationships between features
sns.pairplot(train_data[numerical_features])
plt.show()


# Box plots to detect outliers
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=train_data[feature])
    plt.title(f'Box plot of {feature}')
plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to the 'Sex' column in train_data
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])

# Apply label encoding to the 'Sex' column in test_data
test_data['Sex'] = label_encoder.transform(test_data['Sex'])

# Check the updated DataFrames
print(train_data.head())
print(test_data.head())



# Feature Engineering: Create cross terms between numerical features
numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

def add_feature_cross_terms(df, features):
    df_new = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            name = f"{features[i]}_x_{features[j]}"
            df_new[name] = df_new[features[i]] * df_new[features[j]]
    return df_new

train_data = add_feature_cross_terms(train_data, numerical_features)
test_data = add_feature_cross_terms(test_data, numerical_features)


# Prepare training and testing data
X = train_data.drop(columns=["id", "Calories"])
y = np.log1p(train_data["Calories"])  # log-transform the target
X_test = test_data.drop(columns=["id"])


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train_data))
pred = np.zeros(len(test_data))

for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#' * 10} Fold {i+1} {'#' * 10}")

    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = XGBRegressor(
        max_depth=10,
        colsample_bytree=0.7,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=0.01,
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True,
        tree_method="hist"  # ✅ Use CPU-based training
    )

    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)

    oof[valid_idx] = model.predict(x_valid)
    pred += model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")



# Average predictions from folds
pred /= FOLDS

# Final validation score
final_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\n✅ Final CV RMSE: {final_rmse:.4f}")

# Exponentiate and clip predictions
y_preds = np.expm1(pred)
y_preds = np.clip(y_preds, 1, 314)

# Save submission
submission["Calories"] = y_preds
submission.to_csv("submission.csv", index=False)
submission.head()


