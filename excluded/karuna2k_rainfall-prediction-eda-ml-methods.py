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
import matplotlib.pyplot as plt
import seaborn as sns
!pip install imbalanced-learn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.ensemble import RandomForestClassifier


test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


print(train_df.head())
train_df.info()



# Display basic information about the test dataset
print(test_df.head())
test_df.info()



# Check for duplicate rows in train and test datasets
duplicate_train = train_df.duplicated().sum()
duplicate_test = test_df.duplicated().sum()

# Check the distribution of the target variable (rainfall)
rainfall_distribution = train_df["rainfall"].value_counts(normalize=True) * 100

# Display results
duplicate_train, duplicate_test, rainfall_distribution





# Set plot style
plt.style.use("ggplot")

# Plot distribution of numerical features
train_df.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()





# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()


# Fill missing values in test data (winddirection)
imputer = SimpleImputer(strategy="median")
test_df["winddirection"] = imputer.fit_transform(test_df[["winddirection"]])



# Feature Engineering: Convert 'day' into seasonal categories
def get_season(day):
    if 1 <= day <= 90:
        return "Winter"
    elif 91 <= day <= 180:
        return "Spring"
    elif 181 <= day <= 270:
        return "Summer"
    else:
        return "Autumn"

train_df["season"] = train_df["day"].apply(get_season)
test_df["season"] = test_df["day"].apply(get_season)


# One-hot encode 'season'
train_df = pd.get_dummies(train_df, columns=["season"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["season"], drop_first=True)

# Drop 'day' since we've extracted season information
train_df.drop(columns=["day"], inplace=True)
test_df.drop(columns=["day"], inplace=True)





# Define features and target
X = train_df.drop(columns=["rainfall"])
y = train_df["rainfall"]

# Handle class imbalance using SMOTE (Oversampling the minority class)
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)




# Normalize **AFTER** resampling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)  # Use X_resampled instead of original X
test_scaled = scaler.transform(test_df)


# Train the model (Random Forest) using the resampled data
model = RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=200)
model.fit(X_train_scaled, y_resampled)  # Use y_resampled instead of y

# Predict on the test dataset
test_predictions = model.predict(test_scaled)



# Ensure test_df still has the original 'id' column
test_ids = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")["id"]  # Load original test IDs

# Create the submission DataFrame with the correct IDs
submission = pd.DataFrame({"id": test_ids, "rainfall": test_predictions})

# Save the corrected submission file
submission.to_csv("/kaggle/working/predicted_rainfall.csv", index=False)

print("File saved successfully with correct test IDs!")


