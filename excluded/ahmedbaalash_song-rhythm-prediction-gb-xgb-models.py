import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

import warnings
warnings.filterwarnings(
    "ignore", 
    message=".*The default of observed=False is deprecated.*"
)
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

train.head()


train.info()


print("Check NaN values:")
print("Train: \n",train.isna().sum())
print("-"*50)
print("Test: \n",test.isna().sum())


import matplotlib.pyplot as plt
import seaborn as sns

features = [
    "RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality",
    "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore",
    "TrackDurationMs", "Energy", "BeatsPerMinute"
]

plt.figure(figsize=(16,12))
for i, col in enumerate(features, 1):
    plt.subplot(4, 3, i)
    sns.histplot(train[col], bins=50, kde=True, color="royalblue")
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()



# Create BPM categories
train["BPM_bin"] = pd.cut(train["BeatsPerMinute"], bins=[0,80,120,160,200], 
                          labels=["Slow","Medium","Fast","Very Fast"])

plt.figure(figsize=(16,12))
for i, col in enumerate(features[:-1], 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=train.sample(50000, random_state=42), 
                x="BPM_bin", y=col, palette="viridis")
    plt.title(f"{col} across BPM bins")
plt.tight_layout()
plt.show()


def clip_outliers_iqr(df, cols):
    df_clipped = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clipped[col] = np.clip(df[col], lower, upper)
    return df_clipped
    
features = [
    "RhythmScore","AudioLoudness","VocalContent","AcousticQuality",
    "InstrumentalScore","LivePerformanceLikelihood","MoodScore",
    "TrackDurationMs","Energy","BeatsPerMinute"
]

train_clipped = clip_outliers_iqr(train, features)


train["vocal_vs_instrumental"] = train["VocalContent"] / (train["InstrumentalScore"] + 1e-6)
train["danceability"] = train["RhythmScore"] * train["Energy"]
train["emotional_energy"] = train["MoodScore"] * train["Energy"]
train["vocal_minus_instrumental"] = train["VocalContent"] - train["InstrumentalScore"]
train["energy_minus_acoustic"] = train["Energy"] - train["AcousticQuality"]
train["log_duration"] = np.log1p(train["TrackDurationMs"])

test["vocal_vs_instrumental"] = test["VocalContent"] / (test["InstrumentalScore"] + 1e-6)
test["danceability"] = test["RhythmScore"] * test["Energy"]
test["emotional_energy"] = test["MoodScore"] * test["Energy"]
test["vocal_minus_instrumental"] = test["VocalContent"] - test["InstrumentalScore"]
test["energy_minus_acoustic"] = test["Energy"] - test["AcousticQuality"]
test["log_duration"] = np.log1p(test["TrackDurationMs"])


plt.figure(figsize=(14,10))
sns.heatmap(train.corr(numeric_only = True) , annot = True , fmt=".2g" , cmap = sns.cubehelix_palette(as_cmap=True))
plt.show()


from sklearn.preprocessing import StandardScaler

train_numeric_columns = train.drop(["id","BeatsPerMinute"],axis= 1).select_dtypes("float64").columns
test_numeric_columns = test.drop("id",axis= 1).select_dtypes("float64").columns

scaler = StandardScaler()
train[train_numeric_columns] = scaler.fit_transform(train[train_numeric_columns])
test[test_numeric_columns] = scaler.transform(test[test_numeric_columns])


train.drop(["id","BPM_bin"],axis=1 , inplace = True)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = train.drop("BeatsPerMinute" , axis=1)
Y = train["BeatsPerMinute"]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state=42)


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_models(models, x_train, y_train, x_test, y_test):
    results = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results.append({"Model": name, "R2": r2, "MSE": mse})
        print(f"{name} â†’ R2: {r2:.4f}, MSE: {mse:.4f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Plot R2 scores
    plt.figure(figsize=(8,6))
    plt.bar(results_df["Model"], results_df["R2"], color="skyblue")
    plt.ylabel("R2 Score")
    plt.title("Model Comparison (R2 Scores)")
    plt.xticks(rotation=45)
    plt.show()

    return results_df


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = {
    "LinearRegression": LinearRegression(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.05),
    "LightGBM": LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.05)
}

results_df = evaluate_models(models, x_train, y_train, x_test, y_test)


import pandas as pd
from xgboost import XGBRegressor

# Train model
reg = XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.05)
reg.fit(X, Y)

# Predict
y_pred = reg.predict(test.drop("id", axis=1))

# Create dataframe with id + predictions
pred_df = pd.DataFrame({
    "id": test["id"],
    "Predicted_BPM": y_pred
})

# Save to CSV
pred_df.to_csv("xgb_predictions.csv", index=False)

print("âœ… Predictions saved as xgb_predictions.csv")


