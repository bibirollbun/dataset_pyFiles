import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
train_extra = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
test_ids = test["id"]
train.drop("id", inplace = True, axis = 1)
test.drop("id", inplace = True, axis = 1)


train_extra.columns = train_extra.columns.str.replace(" ","")
train_extra["rainfall"] = train_extra["rainfall"].map({"no" : 0, "yes" : 1})
train_extra["humidity"] = train_extra["humidity"].astype(float)
train_extra["cloud"] = train_extra["cloud"].astype(float)
train_extra = train_extra[train.columns]


train = pd.concat([train, train_extra], axis = 0, ignore_index = True)
train = train.drop_duplicates()
print("Train Shape:", train.shape)


train["winddirection"] = train["winddirection"].fillna(train["winddirection"].mean())
train["windspeed"] = train["windspeed"].fillna(train["windspeed"].mean())
test["winddirection"] = test["winddirection"].fillna(test["winddirection"].mean())
test["windspeed"] = test["windspeed"].fillna(test["windspeed"].mean()) 


def addfeatures(df):
    df["temp_range"] = df["maxtemp"] = df["mintemp"]
    df["avg_temp"] = (df["maxtemp"] + df["mintemp"]) / 2
    df["humidity_cloud"] = df["humidity"] * df["cloud"]
    df["cloud_sunshine_ratio"] = df["cloud"] / (df["sunshine"] + 0.0001)
    df["wind_x"] = df["windspeed"] * np.cos(np.radians(df["winddirection"]))
    df["wind_y"] = df["windspeed"] * np.sin(np.radians(df["winddirection"]))
    df["dewpoint_diff"] = df["temparature"] - df["dewpoint"]
    df["pressure_change"] = df["pressure"] - df["pressure"].shift(1)
    df["humidity_clud_interaction"] = df["humidity"] * df["cloud"]
    df["cloud_sunshine"] = df["cloud"] * df["sunshine"]
    def calc_saturation_vapor_pressure(temp):
        return 6.11 * np.exp((17.27 * temp) / (temp + 237.3))
    df["e_s_temp"] = calc_saturation_vapor_pressure(df["temparature"])
    df["e_s_dewpoint"] = calc_saturation_vapor_pressure(df["dewpoint"])
    df["vapor_pressure_deficit"] = df["e_s_temp"] - df["e_s_dewpoint"]
    def calc_wet_bulb(T, RH):
        return T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
               np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
               0.00391838 * RH**(3/2) * np.arctan(0.023101 * RH) - 4.686035
    df["wet_bulb_temp"] = calc_wet_bulb(df["temparature"], df["humidity"])
    return df

train = addfeatures(train)
test = addfeatures(test)


X = train.drop("rainfall", axis = 1)
Y = train["rainfall"]
X_test = test


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
print(f"{X_scaled.shape} , {X_test_scaled.shape}")


X_train, X_val, y_train, y_val = train_test_split(X_scaled, Y, test_size = 0.2, random_state = 42)


xgb = XGBClassifier(random_state = 42, eval_metric = "auc")


param_grid = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.005, 0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 400],
    "scale_pos_weight": [2, 3, 4, 5, 6],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

grid = GridSearchCV(xgb, param_grid, cv = 5, scoring = "roc_auc", n_jobs = 1)
grid.fit(X_scaled, Y)
print("Best Params:", grid.best_params_)


best_xgb = grid.best_estimator_


val_probs = best_xgb.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_probs)
print(f"Val AUC : {val_auc:.4f}")


test_probs = best_xgb.predict_proba(X_test_scaled)[:, 1]
submission = pd.DataFrame({"id": test_ids, "rainfall": test_probs})
submission.to_csv("submission.csv", index=False)
print("Test Prob Mean:", test_probs.mean())




