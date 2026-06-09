import pandas as pd
import numpy as np 


train=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train



# Feature Engineering Function

def add_features(df):
    
    df["speed_per_lane"] = df["speed_limit"] / (df["num_lanes"] + 1e-3)

    
    df["curv_speed"] = df["curvature"] * df["speed_limit"]

    
    df["high_risk_condition"] = (
        ((df["lighting"] != "daylight") & (df["weather"] != "clear")).astype(int)
    )

    
    mapping_time = {
        "morning": 8,
        "afternoon": 14,
        "evening": 19,
        "night": 23,
    }
    df["time_num"] = df["time_of_day"].map(mapping_time).fillna(12)
    df["time_sin"] = np.sin(2 * np.pi * df["time_num"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["time_num"] / 24)

    
    df["accident_density"] = df["num_reported_accidents"] / (
        (df["num_lanes"] + 1) * (df["speed_limit"] + 1)
    )

    return df

# --------------------------

# --------------------------
train = add_features(train.copy())
test = add_features(test.copy())




train=train.drop(columns ="id")
test=test.drop(columns ="id")


train=pd.get_dummies(train)
test=pd.get_dummies(test)


X=train.drop(columns ="accident_risk")
y=train["accident_risk"]


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# --------------------------
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# --------------------------
model = XGBRegressor(
    n_estimators=500,      
    learning_rate=0.05,    
    max_depth=6,           
    subsample=0.8,         
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"     
)
model.fit(X_train, y_train)

# --------------------------
# --------------------------
preds = model.predict(X_test)

# --------------------------

# --------------------------
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("✅ Train/Test Split Results (80/20) - XGBoost:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


pre=model.predict(test)


su=pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


su["accident_risk"]=pre


su["accident_risk"] = su["accident_risk"].clip(lower=0)


su.to_csv("road.csv",index =False)

