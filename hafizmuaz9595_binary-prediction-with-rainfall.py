import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score


import pandas as pd

# Load the dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df2=pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

# Display first few rows
print(df.head())



# Check dataset shape
print(f"Dataset Shape: {df.shape}")

# Check for missing values
print(df.isnull().sum())

# Check column data types
print(df.info())



df_filtered = df[['id', 'rainfall']]
df_sampled = df_filtered.sample(n=730, random_state=42)


print(df_sampled.shape)  # Expected Output: (730, 2)
print(df_sampled.head())  # Display first few rows


# Drop ID column
df = df.drop(columns=['id'])



df.columns


# Define features and target
X = df.drop(columns=['rainfall'])  # Features
y = df['rainfall']  # Target variable

# Print shapes
print(f"Features Shape: {X.shape}, Target Shape: {y.shape}")



from sklearn.model_selection import train_test_split

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Data Shape: {X_train.shape}, Testing Data Shape: {X_test.shape}")



from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)  # Ensure sufficient iterations
model.fit(X_train, y_train)  # Train the model


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)



from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)



from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=500, max_depth=None)  # Maximize depth
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Define input shape here
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
])


y_pred = model.predict(X_test) 


y_pred = (y_pred > 0.5).astype(int) 

