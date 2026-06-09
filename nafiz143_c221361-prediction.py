import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier


df_train = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")


cat_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']



# Fill missing values (CatBoost handles categorical NaNs internally)

df_train.fillna(-999, inplace=True)


# Initialize CatBoost model

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    verbose=0,
    random_state=42
)


# Prepare training features and labels

X = df_train.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = df_train['satisfaction']

# Train-test split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model

model.fit(X_tr, y_tr)

# Predict on validation set

val_preds = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")

# Load test data

df_test = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")
df_test.fillna(-999, inplace=True)

# Predict on test set

X_test = df_test.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
df_test['satisfaction'] = model.predict(X_test)



# Create the solution DataFrame from df_test

solution = df_test[['id', 'satisfaction']].copy()

# Rename the column

solution.rename(columns={'id': 'ID'}, inplace=True)

# Save to submission.csv

solution.to_csv("submission.csv", index=False)


# Create solution DataFrame for submission

solution = df_test[['id', 'satisfaction']].copy()
solution.rename(columns={'id': 'ID'}, inplace=True)

# Preview the solution

print(solution.head())


# Load training dataset
df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Get summary of dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())


import seaborn as sns
import matplotlib.pyplot as plt

# Count plot for satisfaction
sns.countplot(x='satisfaction', data=df)
plt.title("Distribution of Passenger Satisfaction")
plt.xlabel("Satisfaction Level")
plt.ylabel("Count")
plt.show()

# Print value counts
print(df['satisfaction'].value_counts())


# Only numerical columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
correlation = numeric_df.corr()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


# Categorical: Gender

sns.countplot(x='Gender', hue='satisfaction', data=df)
plt.title("Satisfaction by Gender")
plt.show()



# Categorical: Type of Travel
sns.countplot(x='Type of Travel', hue='satisfaction', data=df)
plt.title("Satisfaction by Type of Travel")
plt.show()


# Categorical: Class
sns.countplot(x='Class', hue='satisfaction', data=df)
plt.title("Satisfaction by Class")
plt.show()


# Distribution of flight distance
sns.histplot(data=df, x='Flight Distance', hue='satisfaction', kde=True, bins=30)
plt.title("Flight Distance Distribution by Satisfaction")
plt.show()


# Example: Inflight wifi service by satisfaction
sns.boxplot(x='satisfaction', y='Inflight wifi service', data=df)
plt.title("Inflight WiFi Rating by Satisfaction")
plt.show()


# Plot feature importance
import matplotlib.pyplot as plt

feature_importances = model.get_feature_importance(prettified=True)
feature_importances.plot(kind='barh', x='Feature Id', y='Importances', figsize=(10,8), legend=False)
plt.title("CatBoost Feature Importance")
plt.show()


solution = df_test[['id', 'satisfaction']].copy()
solution.rename(columns={'id': 'ID'}, inplace=True)



# Save predictions
df_test[['id', 'satisfaction']].to_csv("submission_catboost.csv", index=False)


# Preview the solution
solution.head()




