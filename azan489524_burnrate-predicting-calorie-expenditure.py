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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler



df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_sample = df.sample(frac=0.3, random_state=42).reset_index(drop=True)  # using 30% of the data


print(df_sample.info())
print(df_sample.describe())

# Select only numeric columns for correlation
numeric_df = df_sample.select_dtypes(include=['number'])

# Plot the correlation heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()



le = LabelEncoder()
df_sample['Sex'] = le.fit_transform(df_sample['Sex'])  # male=1, female=0


df_sample['BMI'] = df_sample['Weight'] / ((df_sample['Height']/100) ** 2)
df_sample['Temp_Diff'] = df_sample['Body_Temp'] - 36.5



X = df_sample.drop(['id', 'Calories'], axis=1)
y = df_sample['Calories']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    print(f"{model.__class__.__name__} CV RMSE: {-scores.mean():.3f}")



models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "Ridge": Ridge(alpha=1.0)
}


for name, model in models.items():
    evaluate_model(model, X_train_scaled, y_train)


final_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)
y_pred = np.maximum(y_pred, 0)  # clip negatives to 0final_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)
y_pred = np.maximum(y_pred, 0)  # clip negatives to 0


rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Final RMSE on hold-out test set:", round(rmse, 3))


# âœ… Final Submission Creation

# Step 1: Load test data
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# Step 2: Encode and engineer features like training set
test_df['Sex'] = le.transform(test_df['Sex'])  # same encoder used earlier
test_df['BMI'] = test_df['Weight'] / ((test_df['Height']/100) ** 2)
test_df['Temp_Diff'] = test_df['Body_Temp'] - 36.5

# Step 3: Save 'id' column, drop it for prediction
test_ids = test_df['id']
X_submission = test_df.drop(columns=['id'])

# Step 4: Scale features
X_submission_scaled = scaler.transform(X_submission)

# Step 5: Predict calories and clip negatives
calorie_preds = np.maximum(final_model.predict(X_submission_scaled), 0)

# Step 6: Create submission DataFrame
submission_df = pd.DataFrame({
    'id': test_ids,
    'Calories': calorie_preds
})

# Step 7: Save to CSV
submission_df.to_csv("submission.csv", index=False)
print("âœ… submission.csv has been saved.")


