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


bpm_data = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
print(bpm_data.head())


# Compute pairwise correlation of features (default method = Pearson)
correlation_matrix = bpm_data.corr()

print(correlation_matrix["BeatsPerMinute"].sort_values(ascending=False))  # Correlation with target



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()



#Checking Spearman and kendall correlations with BPM
print(bpm_data.corr(method="spearman")["BeatsPerMinute"].sort_values(ascending=False))
print(bpm_data.corr(method="kendall")["BeatsPerMinute"].sort_values(ascending=False))


#Creating scatter plots to investigate relationships 
#for col in bpm_data.columns:
    #if col != "BeatsPerMinute" or "id":
       # sns.scatterplot(data=bpm_data, x=col, y="BeatsPerMinute", alpha=0.3, s=10)
       # sns.regplot(data=bpm_data, x=col, y="BeatsPerMinute", scatter=False, color="red")
       # plt.title(f"{col} vs BeatsPerMinute")
       # plt.show()



#from itertools import combinations

#def add_interactions(df, cols):
   # for (a, b) in combinations(cols, 2):
        #df[f"{a}_x_{b}"] = df[a] * df[b]
        #df[f"{a}_div_{b}"] = df[a] / (df[b] + 1e-5)   # safe divide
        #df[f"{a}_minus_{b}"] = df[a] - df[b]
    #return df

#bpm_data_interactions = add_interactions(bpm_data, ["TrackDurationMs", "MoodScore", "RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality", "InstrumentalScore", "LivePerformanceLikelihood", "Energy"])
#corrs = bpm_data_interactions.corr(method="spearman")["BeatsPerMinute"].sort_values(ascending=False)
#print(corrs.head(15))   # top correlated features
#print(corrs.tail(15))


from sklearn.feature_selection import mutual_info_regression

#X = bpm_data.drop(columns=["BeatsPerMinute", "id"])
#y = bpm_data["BeatsPerMinute"]

#mi = mutual_info_regression(X, y, random_state=42)
#mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

#print(mi_series.head(15))



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Features + target
X = bpm_data.drop(columns=["BeatsPerMinute", "id"])  # drop target + ID
y = bpm_data["BeatsPerMinute"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=200,      # number of trees
    max_depth=20,        # let trees expand fully
    min_samples_split=2,   # splits down to pure leaves
    random_state=42,
    n_jobs=-1,              # use all CPU cores
    max_features = "sqrt"
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R^2:", r2)



import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# DMatrix is an optimized data format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters (tweak these later with tuning)
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,        # try [4, 6, 8, 10]
    "eta": 0.05,           # lower learning rate
    "subsample": 0.8,      # try [0.6, 0.8, 1.0]
    "colsample_bytree": 0.8, # try [0.6, 0.8, 1.0]
    "min_child_weight": 5,   # helps prevent overfitting
    "lambda": 1.0,         # L2 regularization
    "alpha": 0.0           # L1 regularization
}


# Train with early stopping
evals = [(dtrain, "train"), (dtest, "eval")]
xgb_model = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=20, verbose_eval=50)

# Predictions
y_pred = xgb_model.predict(dtest)

print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R^2:", r2_score(y_test, y_pred))



import pandas as pd
import xgboost as xgb

# Load the test data
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# Separate IDs and features
test_ids = test_df["id"]
X_test = test_df.drop(columns=["id"])

# Convert to DMatrix for XGBoost
dtest = xgb.DMatrix(X_test)

# Predict with your trained model
y_pred = xgb_model.predict(dtest)

# Build the submission dataframe
submission = pd.DataFrame({
    "id": test_ids,
    "BeatsPerMinute": y_pred
})

# Save to CSV
submission.to_csv("submission.csv", index=False, header=True)

