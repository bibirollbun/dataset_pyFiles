# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# go to the input dir
os.chdir("/kaggle/input/playground-series-s5e5");
os.getcwd()


# Get the data
data = pd.read_csv("train.csv")


data.info()


data.describe()


sns.set()


sns.countplot(x="Sex", data=data)


# distribution of age
sns.histplot(x="Age", data=data, bins=20)


data.replace({"Sex": {"male": 1, "female": 0}}, inplace=True)


sns.countplot(x="Sex", data=data)


data.describe()


data[data["Calories"] < 5]


correlation = data.corr()


plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={"size":8}, cmap="Blues")


data.info()


data["Duration_HeartRate"] = data["Duration"] * data["Heart_Rate"]
data["Duration_BodyTemp"] = data["Duration"] * data["Body_Temp"]


# Separate features from target
# X = data[["Duration", "Heart_Rate", "Body_Temp"]]
X = data.drop(columns=["id", "Calories"])
y = data["Calories"]


X.head()


y.head()


print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# # Random Forest
# rf_model = RandomForestRegressor(random_state=21, n_jobs=-1)
# rf_model.fit(X_train, y_train)

# # Decision Tree
# dt_model = DecisionTreeRegressor(random_state=21)
# dt_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBRegressor(
    random_state=21,
    tree_method='hist',   # now use 'hist'
    device='cuda',        # NEW way to activate GPU
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)


models = {
    # "Random Forest": rf_model,
    # "Decision Tree": dt_model,
    "XGBoost": xgb_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    
    # Ensure no negative predictions (RMSLE requirement)
    y_pred = np.maximum(y_pred, 0)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    
    # Display
    print(f"{name}:")
    print(f"  R² Score  : {r2:.4f}")
    print(f"  MAE       : {mae:.2f}")
    print(f"  RMSLE     : {rmsle:.4f}\n")


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')  # ideal line
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title(f'{model_name} - Actual vs Predicted Calories')
    plt.grid(True)
    plt.show()


# y_pred_rf = rf_model.predict(X_test)
# y_pred_dt = dt_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest")
# plot_actual_vs_predicted(y_test, y_pred_dt, "Decision Tree")
plot_actual_vs_predicted(y_test, y_pred_xgb, "XGBoost")


def plot_feature_importances(model, feature_names, model_name):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=feat_imp, y=feat_imp.index)
    plt.title(f"{model_name} - Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.grid(True)
    plt.show()


feature_names = X_test.columns  # or X_train.columns if you're using that
# plot_feature_importances(rf_model, feature_names, "Random Forest")
# plot_feature_importances(dt_model, feature_names, "Decision Tree")


import xgboost as xgb

# Option 1: Native XGBoost plot
xgb.plot_importance(xgb_model)
plt.title("XGBoost - Feature Importances")
plt.show()

# Option 2: Use same barplot style
plot_feature_importances(xgb_model, feature_names, "XGBoost")


os.getcwd()


# Random Forest
# rf_model_final = RandomForestRegressor(random_state=21, n_jobs=-1)
# rf_model_final.fit(X, y)


# XGBoost
xgb_model_final = XGBRegressor(
    random_state=21,
    tree_method='hist',   # now use 'hist'
    device='cuda',        # NEW way to activate GPU
    n_jobs=-1
)
xgb_model_final.fit(X, y)


test_data = pd.read_csv("test.csv")


test_data.replace({"Sex": {"male": 1, "female": 0}}, inplace=True)
test_data.head()


test_data[test_data['id'] == 989850]




test_data["Duration_HeartRate"] = test_data["Duration"] * test_data["Heart_Rate"]
test_data["Duration_BodyTemp"] = test_data["Duration"] * test_data["Body_Temp"]


test_X = test_data.drop(columns=["id"])


test_X.head()


test_y_pred = xgb_model_final.predict(test_X)


test_y_pred


test_y_pred = np.maximum(test_y_pred, 0)



submission_df = pd.DataFrame({
    "id": test_data["id"],
    "Calories": test_y_pred
})

os.chdir("/kaggle/working/")
submission_df.to_csv("submission.csv", index=False)


check = pd.read_csv("submission.csv")
check.head()

