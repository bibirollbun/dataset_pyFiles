import pandas as pd
import numpy as np
import os


# Kaggle competition input directory
DATA_DIR = "/kaggle/input/playground-series-s5e5"

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))



print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample shape:", sample.shape)


# Basic info
train.info()
train.describe().T


# Missing values
train.isna().sum()


# Target distribution (Calories)
import matplotlib.pyplot as plt

plt.hist(train["Calories"], bins=50, color="skyblue", edgecolor="black")
plt.title("Target Distribution: Calories")
plt.xlabel("Calories burned")
plt.ylabel("Count")
plt.show()


# Correlation heatmap (numerical features)
import seaborn as sns

num_cols = train.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(8,6))
sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()



# Boxplots for key features vs Calories
for col in ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]:
    plt.figure(figsize=(5,4))
    sns.scatterplot(data=train, x=col, y="Calories", alpha=0.3)
    plt.title(f"{col} vs Calories")
    plt.show()



# Distribution of categorical feature (Sex)
sns.countplot(data=train, x="Sex")
plt.title("Distribution of Sex")
plt.show()



# ğŸ› ï¸� Feature Engineering â€“ creating new columns

# Body Mass Index
train["BMI"] = train["Weight"] / ((train["Height"] / 100) ** 2)
test["BMI"] = test["Weight"] / ((test["Height"] / 100) ** 2)

# Duration relative to heart rate (efficiency measure)
train["Duration_per_Heart"] = train["Duration"] / train["Heart_Rate"]
test["Duration_per_Heart"] = test["Duration"] / test["Heart_Rate"]

# Exercise intensity: interaction between heart rate and body temperature
train["Intensity"] = train["Heart_Rate"] * train["Body_Temp"]
test["Intensity"] = test["Heart_Rate"] * test["Body_Temp"]

# Body temperature per minute of exercise
train["Temp_per_Minute"] = train["Body_Temp"] / train["Duration"]
test["Temp_per_Minute"] = test["Body_Temp"] / test["Duration"]



test.shape


train.head()


test.head()


train.info()


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(train["Calories"], bins=100, color="skyblue", edgecolor="black")
plt.title("Distribution of Calories")
plt.xlabel("Calories burned")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



# Grouped statistics by Sex (mean, median, count)
stats_by_sex = (train
                .groupby("Sex")["Calories"]
                .agg(["mean","median","count"])
                .round(2)
                .sort_values("mean", ascending=False))
display(stats_by_sex)



# Boxplot: Calories by Sex
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.boxplot(data=train, x="Sex", y="Calories")
plt.title("Calories by Sex")
plt.xlabel("Sex")
plt.ylabel("Calories")
plt.show()



# Sorted correlation with target only (clean & focused)
num_cols = train.select_dtypes(include=["int64","float64"]).columns
corr_to_target = (train[num_cols]
                  .corr(numeric_only=True)["Calories"]
                  .drop("Calories")
                  .sort_values(ascending=False))
display(corr_to_target.to_frame("corr_with_Calories").round(3))



# Heatmap (numerical only)
plt.figure(figsize=(9,7))
sns.heatmap(train[num_cols].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Numerical Features)")
plt.show()



# Pick top 3 numeric correlates to Calories for scatterplots
top3 = corr_to_target.index[:3].tolist()

for col in top3:
    plt.figure(figsize=(5,4))
    sns.scatterplot(data=train, x=col, y="Calories", alpha=0.35)
    plt.title(f"{col} vs Calories")
    plt.show()



plt.figure(figsize=(6,4))
plt.hist(np.log1p(train["Calories"]), bins=60, edgecolor="black")
plt.title("log1p(Calories) Distribution")
plt.xlabel("log1p(Calories)")
plt.ylabel("Frequency")
plt.show()



Q1, Q3 = train["Calories"].quantile([0.25, 0.75])
IQR = Q3 - Q1
upper_cap = Q3 + 1.5*IQR
share_outliers = (train["Calories"] > upper_cap).mean()
print(f"Share of potential outliers (> Q3 + 1.5*IQR): {share_outliers:.2%}")



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ğŸ”¢ Encode 'Sex' column as numeric
le = LabelEncoder()
train["Sex"] = le.fit_transform(train["Sex"])   # e.g., female=0, male=1 (alphabetical order)
test["Sex"]  = le.transform(test["Sex"])

# ğŸ�¯ Separate features and target
X = train.drop(columns=["id", "Calories"])   # keep engineered features
y = train["Calories"]

# Prepare test features (drop ID only)
X_test = test.drop(columns=["id"])

# âœ‚ï¸� Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Shapes
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)



# ğŸ› ï¸� Feature Engineering
train["BMI"] = train["Weight"] / ((train["Height"] / 100) ** 2)
test["BMI"] = test["Weight"] / ((test["Height"] / 100) ** 2)

train["Duration_per_Heart"] = train["Duration"] / train["Heart_Rate"]
test["Duration_per_Heart"] = test["Duration"] / test["Heart_Rate"]

train["Intensity"] = train["Heart_Rate"] * train["Body_Temp"]
test["Intensity"] = test["Heart_Rate"] * test["Body_Temp"]

train["Temp_per_Minute"] = train["Body_Temp"] / train["Duration"]
test["Temp_per_Minute"] = test["Body_Temp"] / test["Duration"]

# ğŸ”¢ Encode 'Sex' as numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["Sex"] = le.fit_transform(train["Sex"])
test["Sex"]  = le.transform(test["Sex"])

# ğŸ�¯ Define features and target
X = train.drop(columns=["id", "Calories"])
y = train["Calories"]
X_test = test.drop(columns=["id"])

# âœ‚ï¸� Split into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)



import numpy as np
from sklearn.metrics import mean_squared_log_error

# âš¡ Try XGBoost; fall back to RandomForest if not available
try:
    import xgboost as xgb
    USE_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    USE_XGB = False

RANDOM_STATE = 42

# ---- Model setup
if USE_XGB:
    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1
    )
else:
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# ---- Fit on training fold
model.fit(X_train, y_train)

# ---- Validate with RMSLE
pred_val = model.predict(X_valid)
pred_val = np.clip(pred_val, 0, None)  # avoid negatives for RMSLE
rmsle = mean_squared_log_error(y_valid, pred_val, squared=False)
print(f"âœ… Valid RMSLE: {rmsle:.5f}")

# ---- Train on full training data
model.fit(X, y)

# ---- Predict test & build submission
pred_test = model.predict(X_test)
pred_test = np.clip(pred_test, 0, None)

submission = sample.copy()            # must contain ['id','Calories']
submission["Calories"] = pred_test
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv created!")
submission.head()



import joblib
joblib.dump(model, "model.pkl")
print("âœ… model.pkl successfully saved!")





