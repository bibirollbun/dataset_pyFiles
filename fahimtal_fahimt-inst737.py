# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error


warnings.filterwarnings("ignore", category=FutureWarning)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


# Replace inf/-inf with NaN, then drop
train.replace([np.inf, -np.inf], np.nan, inplace=True)
train.dropna(inplace=True)
train.isnull().sum()


train.info()


train['BMI'] = train['Weight'] / (train['Height'] / 100) ** 2



# List of numerical features, including target
train_num_outlier = train[['Calories', 'Age', 'Height', 'Weight', 
                           'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']]

# Plot
plt.figure(figsize=(18, 16))
for i, col in enumerate(train_num_outlier.columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train_num_outlier[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


train['Sex'].unique()


sns.countplot(x='Sex', data=train)
plt.title('Distribution of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()



sns.boxplot(x='Sex', y='Calories', data=train)
plt.title('Calories Burned by Sex')
plt.xlabel('Sex')
plt.ylabel('Calories')
plt.show()



# Drop the ID column — not useful for prediction
train = train.drop(columns=["id"])



# Convert categorical to numeric (male = 0, female = 1)
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})


# First split: 80% train_val, 20% test
train_val, test_set = train_test_split(train, test_size=0.2, random_state=42)

# Second split: 80% train, 20% val (of the remaining 80%)
train_set, val_set = train_test_split(train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


# Define features to scale
features_to_scale = ['Age', 'Height', 'Weight', 'Duration',
                     'Heart_Rate', 'Body_Temp', 'BMI']

# Separate features and target
X_train = train_set.drop(columns=['Calories'])
y_train = train_set['Calories']

X_val = val_set.drop(columns=['Calories'])
y_val = val_set['Calories']

X_test = test_set.drop(columns=['Calories'])
y_test = test_set['Calories']

# Standardize features
scaler = StandardScaler()
X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
X_val[features_to_scale] = scaler.transform(X_val[features_to_scale])
X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# Predict and clip negative predictions
val_preds = lr_model.predict(X_val)
val_preds = np.clip(val_preds, 0, None)

# Evaluate performance
rmsle = np.sqrt(mean_squared_log_error(y_val, val_preds))
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
mae = mean_absolute_error(y_val, val_preds)

print(f"RMSLE: {rmsle:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")


# Predicted vs Actual
sns.scatterplot(x=y_val, y=val_preds, alpha=0.3)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Predicted vs Actual Calories")
plt.show()



# Residual Plot
residuals = y_val - val_preds

sns.histplot(residuals, bins=50, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.show()


# Apply the preprocessing
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
test["BMI"] = test["Weight"] / (test["Height"] / 100) ** 2

# Drop ID temporarily and scale features
ids = test["id"]
test = test.drop(columns=["id"])

# Scale test features using the SAME scaler used during training
test[features_to_scale] = scaler.transform(test[features_to_scale])

# Predict calories
final_preds = lr_model.predict(test)
final_preds = np.clip(final_preds, 0, None)  # clip negatives

# Prepare submission DataFrame
submission = pd.DataFrame({
    "id": ids,
    "Calories": final_preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)


