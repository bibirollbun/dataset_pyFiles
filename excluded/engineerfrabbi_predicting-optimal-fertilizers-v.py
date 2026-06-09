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

train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)


print("\nTrain data info:")
train_df.info()


print("\nTest data info:")
test_df.info()


print("\nTrain data head:")
train_df.head()


print("\nTest data head:")
test_df.head()


print("\nDescriptive statistics for numerical features in Train data:")
train_df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']].describe()


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Numerical Features', fontsize=16)

sns.histplot(train_df['Temparature'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Temparature Distribution')

sns.histplot(train_df['Humidity'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Humidity Distribution')

sns.histplot(train_df['Moisture'], kde=True, ax=axes[0, 2])
axes[0, 2].set_title('Moisture Distribution')

sns.histplot(train_df['Nitrogen'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Nitrogen Distribution')

sns.histplot(train_df['Potassium'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Potassium Distribution')

sns.histplot(train_df['Phosphorous'], kde=True, ax=axes[1, 2])
axes[1, 2].set_title('Phosphorous Distribution')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


print("\nValue counts for Soil Type:")
print(train_df['Soil Type'].value_counts())

print("\nValue counts for Crop Type:")
print(train_df['Crop Type'].value_counts())

print("\nValue counts for Fertilizer Name:")
print(train_df['Fertilizer Name'].value_counts())


fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Distribution of Categorical Features', fontsize=16)

sns.countplot(data=train_df, x='Soil Type', ax=axes[0])
axes[0].set_title('Soil Type Distribution')
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(data=train_df, x='Crop Type', ax=axes[1])
axes[1].set_title('Crop Type Distribution')
axes[1].tick_params(axis='x', rotation=45)

sns.countplot(data=train_df, x='Fertilizer Name', ax=axes[2])
axes[2].set_title('Fertilizer Name Distribution')
axes[2].tick_params(axis='x', rotation=90)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Numerical Features vs. Fertilizer Name', fontsize=16)

sns.boxplot(data=train_df, x='Fertilizer Name', y='Temparature', ax=axes[0, 0])
axes[0, 0].set_title('Temparature vs. Fertilizer Name')
axes[0, 0].tick_params(axis='x', rotation=90)

sns.boxplot(data=train_df, x='Fertilizer Name', y='Humidity', ax=axes[0, 1])
axes[0, 1].set_title('Humidity vs. Fertilizer Name')
axes[0, 1].tick_params(axis='x', rotation=90)

sns.boxplot(data=train_df, x='Fertilizer Name', y='Moisture', ax=axes[0, 2])
axes[0, 2].set_title('Moisture vs. Fertilizer Name')
axes[0, 2].tick_params(axis='x', rotation=90)

sns.boxplot(data=train_df, x='Fertilizer Name', y='Nitrogen', ax=axes[1, 0])
axes[1, 0].set_title('Nitrogen vs. Fertilizer Name')
axes[1, 0].tick_params(axis='x', rotation=90)

sns.boxplot(data=train_df, x='Fertilizer Name', y='Potassium', ax=axes[1, 1])
axes[1, 1].set_title('Potassium vs. Fertilizer Name')
axes[1, 1].tick_params(axis='x', rotation=90)

sns.boxplot(data=train_df, x='Fertilizer Name', y='Phosphorous', ax=axes[1, 2])
axes[1, 2].set_title('Phosphorous vs. Fertilizer Name')
axes[1, 2].tick_params(axis='x', rotation=90)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Soil Type vs. Fertilizer Name
soil_fertilizer_crosstab = pd.crosstab(train_df['Soil Type'], train_df['Fertilizer Name'])
plt.figure(figsize=(12, 6))
sns.heatmap(soil_fertilizer_crosstab, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Soil Type vs. Fertilizer Name')
plt.show()

# Crop Type vs. Fertilizer Name
crop_fertilizer_crosstab = pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name'])
plt.figure(figsize=(15, 8))
sns.heatmap(crop_fertilizer_crosstab, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Crop Type vs. Fertilizer Name')
plt.show()


print('\nMissing values in Train data:')
print(train_df.isnull().sum())

print('\nMissing values in Test data:')
print(test_df.isnull().sum())


# Feature Engineering: Adding interaction terms and polynomial features
train_df["Temp_Humid_Interaction"] = train_df["Temparature"] * train_df["Humidity"]
test_df["Temp_Humid_Interaction"] = test_df["Temparature"] * test_df["Humidity"]

train_df["N_P_K_Sum"] = train_df["Nitrogen"] + train_df["Phosphorous"] + train_df["Potassium"]
test_df["N_P_K_Sum"] = test_df["Nitrogen"] + test_df["Phosphorous"] + test_df["Potassium"]

train_df["Temp_Moisture_Ratio"] = train_df["Temparature"] / (train_df["Moisture"] + 1e-6)
test_df["Temp_Moisture_Ratio"] = test_df["Temparature"] / (test_df["Moisture"] + 1e-6)

print('\nTrain data head after feature engineering:')
print(train_df.head())
print('\nTest data head after feature engineering:')
print(test_df.head())


from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

combined_df = pd.concat([train_df.drop("Fertilizer Name", axis=1), test_df], ignore_index=True)

# Identify numerical and categorical columns for scaling and encoding
numerical_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous", "Temp_Humid_Interaction", "N_P_K_Sum", "Temp_Moisture_Ratio"]
categorical_cols = ["Soil Type", "Crop Type"]

# One-hot encode categorical features
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])

train_ids = train_df["id"]
test_ids = test_df["id"]

train_processed = combined_df[combined_df["id"].isin(train_ids)].copy()
test_processed = combined_df[combined_df["id"].isin(test_ids)].copy()

mlb = MultiLabelBinarizer()
y_train_mlb = mlb.fit_transform(train_df["Fertilizer Name"].apply(lambda x: [x]))

train_processed = train_processed.drop(columns=["id"])
test_processed = test_processed.drop(columns=["id"])

X_train = train_processed
y_train = y_train_mlb
X_test = test_processed

print('\nShape of X_train:')
print(X_train.shape)
print('\nShape of y_train:')
print(y_train.shape)
print('\nShape of X_test:')
print(X_test.shape)

print('\nX_train head:')
print(X_train.head())

print('\ny_train head (first 5 rows of binarized labels):')
print(y_train[:5])


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

# Using GradientBoostingClassifier with MultiOutputClassifier
# Parameters are chosen to balance performance and memory usage
model = MultiOutputClassifier(estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
model.fit(X_train, y_train)

print(f"Trained {len(model.estimators_)} Gradient Boosting models.")


test_pred_proba = np.array([estimator.predict_proba(X_test)[:, 1] for estimator in model.estimators_]).T

# Get top 3 predicted classes for each sample in the test set
test_top_3_predictions_indices = np.argsort(test_pred_proba, axis=1)[:, -3:][:, ::-1]

# Convert indices back to fertilizer names
final_predicted_fertilizer_names = []
for indices in test_top_3_predictions_indices:
    final_predicted_fertilizer_names.append([mlb.classes_[idx] for idx in indices])

print("Generated predictions for the test set.")


submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(names) for names in final_predicted_fertilizer_names]
})

submission_df.to_csv('submission_v2.csv', index=False)

print('\nSubmission file created: submission.csv')
print(submission_df.head())

