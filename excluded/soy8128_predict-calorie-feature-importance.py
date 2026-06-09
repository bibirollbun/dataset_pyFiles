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


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/predict-calorie-data/train.csv")
test = pd.read_csv("/kaggle/input/predict-calorie-data/test.csv")
submission = pd.read_csv("/kaggle/input/predict-calorie-data/sample_submission.csv")


le = LabelEncoder()
train["Sex"] = le.fit_transform(train["Sex"])
test["Sex"] = le.transform(test["Sex"])
train["Sex"] = train["Sex"].astype(int)
test["Sex"] = test["Sex"].astype(int)  


def add_interactions_onehot(df, features, gender_col="Sex"):
    df["Male"] = df[gender_col]
    df["Female"] = 1 - df[gender_col]

    for feat in features:
        df[f"{feat}_x_Male"] = df[feat] * df["Male"]
        df[f"{feat}_x_Female"] = df[feat] * df["Female"]

    df.drop(["Male", "Female"], axis=1, inplace=True)
    return df

train = add_interactions_onehot(train, features=["Duration", "Heart_Rate", "Body_Temp"])
test = add_interactions_onehot(test, features=["Duration", "Heart_Rate", "Body_Temp"])


numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
for df in [train, test]:
    df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
    df["Intensity"] = df["Heart_Rate"] / df["Duration"]
    df["HR_Duration"]= df["Heart_Rate"] * df["Duration"]
    df["Temp_Duration"] = df["Body_Temp"] * df["Duration"]
    df['BT_Group'] = pd.qcut(df['Body_Temp'], q=4, labels=[0, 1, 2, 3])
    df['BT_Group'] = df['BT_Group'].astype(float)
    df['Max_HR'] = 220 - df['Age']
    df['HR_Ratio'] = df['Heart_Rate'] / df['Max_HR']
    for col in numerical_features:
        df[f"{col}_squared"] = df[col] ** 2
        df[f"{col}_sqrt"] = np.sqrt(df[col])


def add_cross_terms(df, features):
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

cross_features = numerical_features + ["BMI"] 
train = add_cross_terms(train, cross_features)
test = add_cross_terms(test, cross_features)


X = train.drop(columns=["Calories","id"])
y = np.log1p(train["Calories"])
X_test = test.drop(columns=["id"])


X.shape


X_test.shape


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

models = {
    "CatBoost": CatBoostRegressor(verbose=0, random_seed=42, cat_features=["Sex"],
                                  early_stopping_rounds=100),
    "XGBoost": XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9,
                            n_estimators=2000, learning_rate=0.02, gamma=0.01,
                            max_delta_step=2, early_stopping_rounds=100,
                            eval_metric="rmse", enable_categorical=True, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10,
                             colsample_bytree=0.7, subsample=0.9, random_state=42, verbose=-1)
}
results = {name: {"pred": np.zeros(len(X_test)), "rmsle": []} for name in models}



for name, model in models.items():
    print(f"\n Training {name}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        if name == "XGBoost":
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
            booster = model.get_booster()
            fscore = booster.get_score(importance_type="total_gain")
            fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
            print(f"\n {name} importance")
            print(fscore[:20])
        elif name == "CatBoost":
            model.fit(x_train, y_train, eval_set=(x_val, y_val))
            feature_names = model.feature_names_
            importances = model.get_feature_importance(type="PredictionValuesChange")
            fscore = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            print(f"\n {name} importance")
            print(fscore[:20])
        else:
            model.fit(x_train, y_train)
            importances = model.booster_.feature_importance(importance_type="gain")
            fscore = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            print(f"\n {name} importance")
            print(fscore[:20])


        y_pred_val = model.predict(x_val)
        y_pred_test = model.predict(X_test)
        results[name]["pred"] += y_pred_test / FOLDS
        score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_val)))
        results[name]["rmsle"].append(score)
        print(f"Fold {fold + 1} RMSLE: {score:.5f}")
        features, scores = zip(*fscore[:20])
        plt.figure(figsize=(10, 5))
        plt.barh(features, scores)
        plt.xlabel("Total Gain")
        plt.title(f"{name} Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


blend_preds = (
    0.4 * np.expm1(results["XGBoost"]["pred"]) + 
    0.3 * np.expm1(results["CatBoost"]["pred"]) + 
    0.3 * np.expm1(results["LightGBM"]["pred"])
)

submission["Calories"] = np.clip(blend_preds, 1, 314)
submission.to_csv("/kaggle/working/submission_blend_bmi_squared.csv", index=False)


for name in models:
    scores = results[name]["rmsle"]
    print(f"{name} Mean RMSLE: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
print("\n submission_blend_bmi_squared.csv is saved.")

