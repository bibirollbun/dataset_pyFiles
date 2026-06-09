import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import warnings
import pickle  # <-- Add this at the top



# Ignore warnings
warnings.filterwarnings("ignore")


# Load Data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')




# Exploratory Data Analysis (EDA)
print(df_train.info())
print(df_train.describe())


# Visualizing Data Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_train['Price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.show()



# Checking Missing Values
plt.figure(figsize=(12, 6))
sns.heatmap(df_train.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title('Missing Values Heatmap')
plt.show()


# Pairplot to visualize relationships
sns.pairplot(df_train)
plt.show()


# Handling Missing Values
df_train.fillna(df_train.select_dtypes(include=['number']).mean(), inplace=True)
df_test.fillna(df_test.select_dtypes(include=['number']).mean(), inplace=True)



# Label Encoding for Categorical Features
label_encoders = {}
for col in df_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le


# Feature Selection
X = df_train.drop(columns=['id', 'Price'])
y = df_train['Price']
X_test = df_test.drop(columns=['id'])


# Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



# Scaling Numerical Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# Model Training & Comparison
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression()
}

rmse_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    rmse_scores[name] = rmse
    print(f'{name} RMSE: {rmse}')


# Best Model Selection
best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]
y_test_pred = best_model.predict(X_test)


# Prepare Submission
submission = pd.DataFrame({'id': df_test['id'], 'Price': y_test_pred})
submission.to_csv('submission.csv', index=False)




print(f"Best model: {best_model_name}")
print("Submission file created!")


# Save the Best Model
joblib.dump(best_model, 'best_model.pkl')
print(f"Best model saved: {best_model_name}")



from sklearn.preprocessing import StandardScaler

# Assuming X_train is your training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler on training data

# Save the fitted scaler
import pickle
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)



# Identify categorical columns
categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", categorical_columns)



label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])  # Fit and transform on training data
    df_test[col] = le.transform(df_test[col])  # Transform test data using same encoder
    label_encoders[col] = le

# Save the label encoders
with open("label_encoders.pkl", "wb") as le_file:
    pickle.dump(label_encoders, le_file)












