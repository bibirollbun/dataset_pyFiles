!pip install -q lofo-importance


import numpy as np
import pandas as pd

df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df


target = "Listening_Time_minutes"
features = [col for col in df.columns if col not in {target, "id"}]
len(features)


from sklearn.preprocessing import LabelEncoder

for col in features:
    if df[col].dtype == object:
        print(col)
        df[col] = LabelEncoder().fit_transform(df[col].fillna("nan").astype(str))
        df[col] = df[col].astype("category")


from xgboost import XGBRegressor

model = XGBRegressor(
    device="cuda",
    max_depth=3,  
    colsample_bytree=0.5, 
    subsample=0.8, 
    n_estimators=400,  
    learning_rate=0.1,
    objective='reg:squarederror',
    enable_categorical=True,
    min_child_weight=5
)


import lofo

ds = lofo.Dataset(df, target=target, features=features, auto_group_threshold=0.85)


lofo_imp = lofo.LOFOImportance(ds, cv=4, scoring="neg_mean_squared_error",
                               model=model, n_jobs=1)
imp_df = lofo_imp.get_importance()


lofo.plot_importance(imp_df, figsize=(12, 12))


df["t"] = (df["Listening_Time_minutes"] + 1) / (df["Episode_Length_minutes"] + 1)

df = df[df["t"].notnull()].reset_index(drop=True)

df["t"] = df["t"].clip(0, 1)
df["t"].hist(bins=50)


ds = lofo.Dataset(df, target="t", 
                  features=[f for f in features if f != "Episode_Length_minutes"], 
                  auto_group_threshold=0.5)


lofo_imp = lofo.LOFOImportance(ds, cv=4, scoring="neg_mean_absolute_error",
                               model=model, n_jobs=1)
imp_df = lofo_imp.get_importance()


lofo.plot_importance(imp_df, figsize=(12, 12))

