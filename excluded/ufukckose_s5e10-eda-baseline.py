# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


num_cols = df.select_dtypes("number").columns
num_cols = num_cols.drop(["id", "accident_risk"])
num_cols


# Data Visualization

# Histograms for numeric features
plt.figure(figsize=(12,6))
df[num_cols].hist(bins = 20 ,figsize = (15, 15), color="mediumorchid", edgecolor="purple", linewidth=1.2)
plt.suptitle("Histograms of the Numeric Columns", fontsize=30, fontweight="bold")
plt.show()


# Accident Risk Distribution (target)
plt.figure(figsize=(10, 5))
sns.histplot(df["accident_risk"], kde=True, color="mediumorchid", edgecolor="purple", linewidth=1.2)
plt.title("Distribution of Accident Risk", fontweight = "bold" , fontsize=30)
plt.ylabel("Frequency")
plt.xlabel("Accident Risk")
plt.show()


plt.figure(figsize=(10, 5))
# Compute correlation matrix
corr = df.corr(numeric_only=True)
# Plot heatmap
sns.heatmap(corr, annot=True, cmap="Purples", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# ------------------------------
# Mean accident_risk by road_type adn weather
# ------------------------------
pivot_rw = df.pivot_table(
    values="accident_risk",
    index="road_type",
    columns="weather",
    aggfunc="mean"
)

plt.figure(figsize=(6,4))
sns.heatmap(pivot_rw, annot=True, fmt=".2f", cmap="Purples")
plt.title("Mean Accident Risk by Road Type & Weather")
plt.show()


def ftr_eng(X) :
    """
    This function adds secondary features for the model
    
    Sources:
    (1) https://www.kaggle.com/code/imaadmahmood/road-accident-risk-prediction
    (2) https://www.kaggle.com/code/ravi20076/playgrounds5e10-public-baseline-v1
    """
    
    # Copy input
    df = X.copy()
    
    ordinal_features = ["lighting"]
    boolean_features = ["road_signs_present", "public_road", "holiday", "school_season"]
    categorical_features = ["road_type", "weather", "time_of_day"]


    # Interaction
    df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
    df["speed_accident"] = df["speed_limit"] * df["num_reported_accidents"]
    df["curvature_speed"] = df["curvature"] * df["speed_limit"]
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-5)
    df["lanes_accidents"] = df["num_lanes"] * df["num_reported_accidents"]
    df["curvature_per_lane"] = df["curvature"] / (df["num_lanes"] + 1e-5)
    df["risky_conditions"] = ((df["curvature"] > 0.5) & (df["speed_limit"] > 50) & (df["num_reported_accidents"] > 0)).astype(int)
    df["weather_time"] = df["weather"] + "_" + df["time_of_day"]
    df["lighting_weather"] = df["lighting"] + "_" + df["weather"]

    # Binning
    df["speed_bin"] = pd.cut(df["speed_limit"], bins=[0, 35, 50, 70], labels=["low", "medium", "high"]
    )
    
    df["curvature_bin"] = pd.qcut(df["curvature"], q=4, labels=["very_low", "low", "high", "very_high"]
    )
    
    # ratio and log
    df["log_accidents"]    = np.log1p(df["num_reported_accidents"])
    df["accident_density"] = df["num_reported_accidents"] / (df["speed_limit"] * df["num_lanes"] + 1e-5)
    
    # Ordinal Encoding
    lighting_order = {"daylight": 2, "dim": 1, "night": 0}
    df["lighting"] = df["lighting"].map(lighting_order)
        
    return df

df = ftr_eng(df)
test = ftr_eng(test)


# imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


X = df.drop(columns=["id", "accident_risk"])
y = df["accident_risk"]
X_test = test.drop(columns=["id"])

numeric_features = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents",
                    "lanes_speed", "speed_accident", "curvature_speed", 
                    "accidents_per_lane", "lanes_accidents", "curvature_per_lane",
                    "log_accidents", "accident_density"]

categorical_features = ["road_type", "lighting", "weather", "time_of_day",
                        "weather_time", "lighting_weather", "speed_bin", "curvature_bin"]

boolean_features = ["road_signs_present", "public_road", "holiday", "school_season", "risky_conditions"]

# Column transformer: scale numeric, one-hot encode categorical, pass boolean features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("bool", "passthrough", boolean_features)
    ]
)


models = {
    "XGBoost": XGBRegressor(
        subsample=0.8,
        reg_lambda=2,
        reg_alpha=1,
        n_estimators=1000,
        max_depth=9,
        random_state=42,
        learning_rate=0.01,
        colsample_bytree=0.6
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=-1, 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    ),
    "CatBoost": CatBoostRegressor(
        n_estimators=200, 
        learning_rate=0.1, 
        depth=6, 
        random_state=42, 
        verbose=0
    )
}


kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Store RMSE for each fold
    fold_rmse = []

    # 5-Fold CV
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        
        rmse = mean_squared_error(y_val, preds, squared=False)
        fold_rmse.append(rmse)
    
    print(f"{name} mean RMSE: {np.mean(fold_rmse):.4f}")
    
    pipeline.fit(X, y)
    test_preds = pipeline.predict(X_test)
    
    submission = pd.DataFrame({
        "id": test["id"],
        "accident_risk": test_preds
    })
    
    submission.to_csv(f"submission_{name}.csv", index=False)
    print(f"Saved submission_{name}.csv")

