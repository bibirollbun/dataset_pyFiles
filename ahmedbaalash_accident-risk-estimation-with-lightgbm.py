import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import os

import warnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")
warnings.filterwarnings(
    "ignore",
    message="unique with argument that is not not a Series"
)


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

train.head()


print("Train shape :")
print(train.shape , "\n")
print("Test shape :")
print(test.shape)


print("Train info :")
print(train.info())
print("-"*50)
print("Test info :")
print(test.info())


print("Train nulls :")
print(train.isna().sum())
print("-"*50)
print("Test nulls :")
print(test.isna().sum())


print("Train duplicates :")
print(train.duplicated().sum())
print("-"*50)
print("Test duplicates :")
print(test.duplicated().sum())


train["speed_limit"].value_counts()


plt.figure(figsize=(10,6))
plt.title("Distribution of Accident Risk")
sns.histplot(train["accident_risk"],kde=True)
plt.show()


# List of categorical columns to visualize
counts = [
    "road_type", "num_lanes", "speed_limit", "lighting", "weather",
    "road_signs_present", "public_road", "time_of_day",
    "holiday", "school_season", "num_reported_accidents"
]

# Set a consistent Seaborn style and color palette
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

# Loop through each column and create a countplot
for col in counts:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=train, x=col, palette=palette)
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.show()



# A box plot to see the outliers in the datasets
# There is not much in this dataset except the Target variable
plt.figure(figsize=(16,6))
sns.boxplot(train.drop(["id","speed_limit"] , axis=1))
plt.show()


train = train.copy()
test = test.copy()

# Feature groups
scale_features = ["curvature"]
label_encode_features = [
    "speed_limit", "road_signs_present", "public_road",
    "time_of_day", "holiday", "school_season"
]
one_hot_encode_features = ["road_type", "lighting", "weather"]

# ---- 1. Scale numeric features ----
scaler = StandardScaler()
train[scale_features] = scaler.fit_transform(train[scale_features])
test[scale_features] = scaler.transform(test[scale_features])

# ---- 2. Label encode categorical features ----
for col in label_encode_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# ---- 3. One-hot encode categorical features using pandas ----

combined = pd.concat([train, test], axis=0)
combined = pd.get_dummies(combined, columns=one_hot_encode_features, drop_first=False , dtype=int)

# Split back into train/test with aligned columns
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].copy()



X = train.drop(["id","accident_risk"],axis=1)
Y = train["accident_risk"]

models = {
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
}

r2_results = []
rmse_results = []
predictions = {}

# split the data to train and test
x_train,x_test,y_train,y_test = train_test_split(X,Y , test_size = 0.2 , random_state = 42)

for name, model in models.items():

    # Fit the model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calculate RMSE and R2
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Append the results to visualize them later
    rmse_results.append(rmse)
    r2_results.append(r2)
    predictions[name] = y_pred

    print(f"Training {name} is Done.")


# Combining the results to compare and visualize
results_df = pd.DataFrame({
    "Model": list(models.keys()),
    "RMSE": rmse_results,
    "R2": r2_results
}).sort_values(by="R2")

print("\n Model Performance:")
print(results_df)


# Visualize the performance of the models by R2
plt.figure(figsize=(12,6))
sns.barplot(x=list(models.keys()),y=r2_results)
plt.show()



plt.figure(figsize=(12,6))

# Sort test values for better visual comparison
sorted_idx = np.argsort(y_test.values)
y_sorted = y_test.values[sorted_idx]

plt.plot(y_sorted, label="Actual", color='black', linewidth=2)

for name, y_pred in predictions.items():
    plt.plot(y_pred[sorted_idx], label=name, alpha=0.6)

plt.title("Model Predictions vs Actual Values")
plt.xlabel("Sample Index (sorted by true value)")
plt.ylabel("Target Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


X = train.drop(["id","accident_risk"],axis=1)
Y = train["accident_risk"]

test_id = test["id"].reset_index(drop=True)
test_values = test.drop(["id","accident_risk"],axis=1)

model = LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X, Y)
y_pred = model.predict(test_values)

submission = pd.DataFrame({
    "id": test_id,
    "accident_risk": y_pred
}).set_index("id")


submission.to_csv("submission.csv")

