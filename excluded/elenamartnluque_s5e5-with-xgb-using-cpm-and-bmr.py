import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


def add_columns(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df["BMR"] = (1-df["Sex"]) * ((13.4 * df["Weight"]) + (4.8 * df["Height"]) - (5.677 * df["Age"]) + 88.362) + df["Sex"] * ((9.25 * df["Weight"]) + (3.1 * df["Height"]) - (4.33 * df["Age"]) - 447.59)
    df["BMRxdur"] =  df["Duration"] * df["BMR"] / (24*60)

    return df


df_train = add_columns(df_train)
df_test = add_columns(df_test)
df_train["Calories_per_minute_1"] = df_train["Calories"] / df_train["Duration"]
df_train["Calories_per_minute_2"] = (df_train["Calories"] - df_train["BMRxdur"]) / df_train["Duration"]





models = []
scores = []
predictions = []
# I don't want create a new class for this experiment, so I did that.
drop_columns = ["id","BMR","BMRxdur","Calories","Calories_per_minute_1", "Calories_per_minute_2"]
X_train = df_train.drop(columns=drop_columns, inplace=False)
X_test = df_test.drop(columns=["id","BMR","BMRxdur"])
for cpm in [0,1,2]:
    if cpm == 1: 
        y_train = df_train["Calories_per_minute_1"]
    elif cpm == 2:
        y_train = df_train["Calories_per_minute_2"]
    else:
        y_train = df_train["Calories"]

    # Training model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=250, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    if cpm == 1:
        y_train_pred = model.predict(X_train)*df_train["Duration"]
        y_train_pred = [0 if i < 0 else i for i in y_train_pred]
        rmsle_train = mean_squared_log_error(y_train_pred, df_train["Calories"])
        y_test_pred = model.predict(X_test) * df_test["Duration"]
    elif cpm == 2:
        y_train_pred = model.predict(X_train)*df_train["Duration"] + df_train["BMRxdur"]
        y_train_pred = [0 if i < 0 else i for i in y_train_pred]
        rmsle_train = mean_squared_log_error(y_train_pred, df_train["Calories"])
        y_test_pred = model.predict(X_test) * df_test["Duration"] + df_test["BMRxdur"]
    else:
        y_train_pred = model.predict(X_train)
        y_train_pred = [0 if i < 0 else i for i in y_train_pred]
        rmsle_train = mean_squared_log_error(y_train_pred, df_train["Calories"])
        y_test_pred = model.predict(X_test)

    models.append(model)
    scores.append(rmsle_train)
    predictions.append(y_test_pred)


print("Train RMSLE:")
print(f"Base: {scores[0]}")
print(f"Using calories per minute: {scores[1]}")
print(f"Using BMR: {scores[2]}")


y_test_pred = [0 if i < 0 else i for i in y_test_pred]
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
df_sub["Calories"] = y_test_pred  #predictions[2]
df_sub.to_csv('submission.csv', index=False)

