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


s = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
s


data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
data


data.isna().sum()


data.dtypes


def value_count(col , data):
    return data[col].value_counts()
for col in data.columns:
    print(f"{col}:- {data[col].nunique()}")


categorical = [col for col in data.columns if data[col].dtype == "object"]
numerical = [col for col in data.columns if (data[col].dtype == "float" or data[col].dtype == "int")]


data["school_season"] = data["school_season"].astype(int)


road_type_risk = data.groupby("road_type")["accident_risk"].mean().reset_index(name="avg_risk")
speed_limit_risk = data.groupby("speed_limit")["accident_risk"].mean().reset_index(name="avg_risk")
weather_risk = data.groupby("weather")["accident_risk"].mean().reset_index(name="avg_risk")
time_risk = data.groupby("time_of_day")["accident_risk"].mean().reset_index(name="avg_risk")
lighting_risk = data.groupby("lighting")["accident_risk"].mean().reset_index(name="avg_risk")
num_reported_accidents_risk = data.groupby("num_reported_accidents")["accident_risk"].mean().reset_index(name="avg_risk")


import matplotlib.pyplot as plt

# Store DataFrames in a dictionary
risk_dfs = {
    "Road Type": road_type_risk,
    "Speed Limit": speed_limit_risk,
    "Weather": weather_risk,
    "Time of Day": time_risk,
    "Lighting": lighting_risk,
    "reported":num_reported_accidents_risk,
}

fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 cols
axs = axs.flatten()  # flatten to make iteration easier

for i, (name, df) in enumerate(risk_dfs.items()):
    axs[i].bar(df.iloc[:, 0], df["avg_risk"], color='skyblue')
    axs[i].set_title(name)
    axs[i].set_xlabel(df.columns[0])
    axs[i].set_ylabel("Average Accident Risk")
    axs[i].tick_params(axis='x', rotation=45)

# Remove the empty subplot (since 2x3=6 and we have 5 plots)
# fig.delaxes(axs[-2])

plt.tight_layout()
plt.show()



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Assume 'categorical_cols' is a list of categorical column names

# ColumnTransformer: one-hot encode categorical columns, leave rest as is
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ],
    remainder="passthrough"  # keep other numeric columns unchanged
)
# Fit transformer
data_encoded = preprocessor.fit_transform(data.drop(columns="accident_risk"))

# Get one-hot column names
ohe_columns = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical)

# Get numeric columns that were passed through
numeric_cols = [col for col in data.columns if col not in categorical and col !="accident_risk"]

# Combine all column names
all_columns = list(ohe_columns) + numeric_cols

# Convert to DataFrame
d = pd.DataFrame(data_encoded, columns=all_columns)
d


d[["road_signs_present","public_road","holiday", "school_season"]] = d[["road_signs_present","public_road","holiday", "school_season"]].astype(int)


d["f1"] = d["speed_limit"]*d["curvature"]






correlation_matrix = d.drop(columns=["id"]).corr()


import seaborn as sns
plt.figure(figsize=(8,8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.tight_layout()
plt.show()



for col in d.columns:
    if d[col].dtypes == "object":
        try:
            # Try converting to int
            d[col] = d[col].astype(int)
        except ValueError:
            print(f"Column '{col}' cannot be converted to int")  # for non-numeric strings



d.columns


data.columns


X = d.drop(columns=["id"])
y = data["accident_risk"]


import matplotlib.pyplot as pltt
fig , axes = pltt.subplots(figsize=(8,8))
axes.hist(y, bins=50)
pltt.xlim(0, np.percentile(y, 95))
pltt.tight_layout()
pltt.show()


# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error, make_scorer
# import numpy as np
# import lightgbm as lgb

# # Create the LightGBM model
# # These are some good starting parameters
# model = lgb.LGBMRegressor(
#     objective='regression_l1',
#     metric='rmse',
#     n_estimators=5000,       # allow more trees
#     learning_rate=0.01,      # smaller steps for smoother learning
#     num_leaves=40,           # slightly more complex leaves
#     max_depth=-1,
#     min_child_samples=15,    # allow smaller leaves to capture nuances
#     subsample=0.9,           # more data per tree
#     colsample_bytree=0.9,    # more features per tree
#     random_state=42,
#     n_jobs=-1
# )

# rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))
# cv_scores = cross_val_score(model, X, y, cv=5, scoring=rmse_scorer)
# print("RMSE for each fold:", cv_scores)
# print("Average RMSE:", np.mean(cv_scores))
# model.fit(X,y)


from sklearn.model_selection import GridSearchCV
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, make_scorer

# RMSE scorer
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# Initialize LightGBM with GPU
model = lgb.LGBMRegressor(
    objective='regression_l1',
    n_estimators=5000,
    learning_rate=0.01,
    random_state=42,
    device='gpu',  # Use GPU
    n_jobs=-1
)

# Hyperparameter grid
param_grid = {
    'num_leaves': [31, 40, 50],
    'max_depth': [-1, 10, 20],
    'min_child_samples': [10, 15, 20],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'learning_rate': [0.01, 0.05]
}

# GridSearchCV setup
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Fit on data
grid_search.fit(X, y)

# Best parameters and RMSE
print("Best parameters found:", grid_search.best_params_)
print("Best CV RMSE:", -grid_search.best_score_)

# Best model
best_model = grid_search.best_estimator_



print("RMSE for each fold:", cv_scores)
print("Average RMSE:", np.mean(cv_scores))


X


import pandas as pd

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance.head(20))  # top 20



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.gca().invert_yaxis()  # most important at top
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (LightGBM)")
plt.show()



import matplotlib.pyplot as pltt
fig , axes = pltt.subplots(figsize=(8,8))
axes.hist(y, bins=50)
pltt.xlim(0, np.percentile(y, 95))
pltt.tight_layout()
pltt.show()


test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

test["school_season"] = test["school_season"].astype(int)
test["f1"] = test["speed_limit"]*test["curvature"]


X_test_processed = preprocessor.transform(test)

# Get one-hot column names
ohe_columns = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical)

# Get numeric columns that were passed through
numeric_cols = [col for col in data.columns if col not in categorical and col !="accident_risk"]

# Combine all column names
all_columns = list(ohe_columns) + numeric_cols

# Convert to DataFrame
X_test = pd.DataFrame(X_test_processed, columns=all_columns)
X_test


for col in X_test.columns:
    if X_test[col].dtypes == "object":
        try:
            # Try converting to int
            X_test[col] = X_test[col].astype(int)
        except ValueError:
            print(f"Column '{col}' cannot be converted to int")  # for non-numeric strings



X_test[["road_signs_present","public_road","holiday", "school_season"]] = X_test[["road_signs_present","public_road","holiday", "school_season"]].astype(int)


X_test.isna().sum()
X_test.dtypes


y_test = model.predict(X_test)


y_test.shape


submission = pd.DataFrame({
    "id": test["id"].values,        # 1D array
    "accident_risk": y_test.ravel() # flatten to 1D
})



submission.to_csv("submission.csv", index=False)





submission




