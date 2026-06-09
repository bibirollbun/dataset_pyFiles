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


# Basic Libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")


# Load training and test data
df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Drop User ID and duplicates
df.drop(columns='id', inplace=True)
df_test.drop(columns='id', inplace=True)
df.drop_duplicates(inplace=True)

# Preview
df.head()


# Shape and datatypes
print("Dataset shape:", df.shape)
print(df.dtypes)

# Check for null values
print(df.isnull().sum())

# Basic statistics
df.describe()



df.info()


# Gender distribution
sns.countplot(data=df, x='Sex')
plt.title('Gender Distribution')
plt.show()

# Pairwise relationships
sns.pairplot(df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']])
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()


# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Encode Gender
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # Male=1, Female=0
df_test['Sex'] = le.transform(df_test['Sex'])

# Features and Target
X = df.drop('Calories', axis=1)
y = df['Calories']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(df_test)



X_train.where(X_train<0).sum()


from sklearn.metrics import mean_squared_log_error
import numpy as np

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    # "SVR": SVR(),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred=np.maximum(0, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    # rmsle=rmsle(y_val,y_pred)
    
    results.append([name, mse, rmse, r2])

# Display results
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'R²'])
results_df.sort_values(by='RMSE')



best_model = XGBRegressor(random_state=42, n_estimators=100)
best_model.fit(X_train, y_train)

test_preds = best_model.predict(df_test)

submission = pd.DataFrame({
    "Calories": test_preds
})

submission.to_csv("calories_predictions.csv", index=False)


best_model = XGBRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)

plt.figure(figsize=(10,5))
plt.scatter(y_val, y_pred, alpha=0.5, color='teal')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories (Random Forest)")
plt.show()


# Predict on test data
final_preds = best_model.predict(df_test)

# Create submission file if required
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
final_preds=np.maximum(0., final_preds)
submission['Calories'] = final_preds
submission.to_csv('submission.csv', index=False)




