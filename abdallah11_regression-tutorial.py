import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



train = pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')



train.head()


test.head()


train.isna().sum()


train['Sex'].unique()


# Plot distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
columns = ["Length", "Diameter", "Height", "Whole weight", "Shell weight"]
for i, col in enumerate(columns):
    sns.histplot(train[col], kde=True, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x="Sex", y="Whole weight", data=train)
plt.title("Distribution of Whole Weight by Sex")
plt.show()


numerical_cols = ["Length", "Diameter", "Height", "Whole weight", "Shell weight"]

Q1 = train[numerical_cols].quantile(0.25)
Q3 = train[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df_clean = train[~((train[numerical_cols] < lower_bound) | (train[numerical_cols] > upper_bound)).any(axis=1)]

print(f"Original dataset: {train.shape[0]} rows")
print(f"After removing outliers: {df_clean.shape[0]} rows")




plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean[numerical_cols])
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()



df = pd.get_dummies(train, columns=['Sex'], drop_first=True)  # One-hot encoding



test = pd.get_dummies(test, columns=['Sex'], drop_first=True)


scaler = StandardScaler()  # Or MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


X = df.drop(["Rings", "id"], axis=1)  # Dropping both Rings (target) and ID
y = df["Rings"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train a Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Make Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)



# Compute Accuracy Metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)


train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}")
print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")


# Visualization of Accuracy (Train vs Validation)
labels = ['Train', 'Validation']
r2_scores = [train_r2, test_r2]

plt.figure(figsize=(8, 5))
plt.bar(labels, r2_scores, color=['blue', 'orange'])
plt.xlabel("Dataset")
plt.ylabel("R² Score")
plt.title("Model Accuracy (Training vs Validation)")
plt.ylim(0, 1)
plt.show()






test_d = test.drop( "id", axis=1) 
y_pred = model.predict(test_d)

# Ensure ID column is included
test_results = pd.DataFrame({
    "id": test["id"],  # Assuming 'test_df' is your test dataset with ID
    "Rings": y_pred.flatten()  # Flatten in case predictions are in (n,1) shape
})

# Save to Excel
test_results.to_csv("submission.csv", index=False)

print("Predictions saved successfully as submission.xlsx")











