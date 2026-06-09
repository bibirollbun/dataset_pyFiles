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


# ====================================================
# ğŸ“¦ 1ï¸�âƒ£ Import Libraries
# ====================================================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ====================================================
# ğŸ“‚ 2ï¸�âƒ£ Load Dataset
# ====================================================
base_dir = "/kaggle/input"
folders = os.listdir(base_dir)
print("ğŸ“� Datasets found:", folders)

data_dir = f"{base_dir}/playground-series-s5e5"
print(f"âœ… Dataset directory set to: {data_dir}")

train_df = pd.read_csv(f"{data_dir}/train.csv")
test_df = pd.read_csv(f"{data_dir}/test.csv")
sub_df = pd.read_csv(f"{data_dir}/sample_submission.csv")

# Clean up any infinities or NaNs
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

print(f"Train Shape â�œ {train_df.shape}")
print(f"Test Shape â�œ {test_df.shape}")
print("Columns:", list(train_df.columns))
display(train_df.head())

# ====================================================
# ğŸ“Š 3ï¸�âƒ£ Exploratory Data Analysis (EDA)
# ====================================================

# Null values visualization
plt.figure(figsize=(8, 4))
sns.heatmap(train_df.isnull(), cbar=False, cmap="Reds")
plt.title("ğŸ©¸ Missing Values in Train Data")
plt.show()

# Calories distribution
plt.figure(figsize=(7, 4))
sns.histplot(train_df["Calories"], kde=True, bins=40, color="#4CAF50")
plt.title("Distribution of Calories Burned")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.show()

# Boxplot by gender
plt.figure(figsize=(6, 4))
sns.boxplot(x="Sex", y="Calories", data=train_df, palette="pastel")
plt.title("Calories Burned by Gender")
plt.show()

# Correlation map
plt.figure(figsize=(8, 6))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ====================================================
# âš™ 4ï¸�âƒ£ Preprocessing
# ====================================================
target_col = "Calories"
feature_cols = [col for col in train_df.columns if col not in ["id", target_col]]

# Encode 'Sex'
for df in [train_df, test_df]:
    df["Sex"] = df["Sex"].astype(str).str.lower()

encoder = LabelEncoder()
encoder.fit(pd.concat([train_df["Sex"], test_df["Sex"]]))
train_df["Sex"] = encoder.transform(train_df["Sex"])
test_df["Sex"] = encoder.transform(test_df["Sex"])

X = train_df[feature_cols]
y = train_df[target_col]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=123)

# ====================================================
# ğŸ¤– 5ï¸�âƒ£ Model Training (Gradient Boosting)
# ====================================================
model = GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=4,
    random_state=123
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
r2 = r2_score(y_valid, y_pred)

print(f"ğŸ“‰ RMSE: {rmse:.4f}")
print(f"ğŸ“ˆ RÂ² Score: {r2:.4f}")

# ====================================================
# ğŸ“ˆ 6ï¸�âƒ£ Visualizing Model Performance
# ====================================================
plt.figure(figsize=(6, 6))
plt.scatter(y_valid, y_pred, alpha=0.4, color="#FF7043")
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Predicted vs Actual Calories (Validation)")
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], "b--")
plt.show()

# Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
feat_imp.plot(kind="barh", color="#81D4FA")
plt.title("Feature Importance (Gradient Boosting)")
plt.show()

# ====================================================
# ğŸ�� 7ï¸�âƒ£ Submission
# ====================================================
test_pred = model.predict(test_df[feature_cols])
sub_df["Calories"] = test_pred
sub_df.to_csv("submission.csv", index=False)

print("âœ… Submission file 'submission.csv' generated successfully!")
display(sub_df.head())

