import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from xgboost import XGBRegressor 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")



# Drop ID column
train = train.drop("id", axis=1)


print("Train shape:", train.shape) 
print("Test shape:", test.shape) 
print("Sample Submission shape:", sample_submission.shape)


X = train.drop("accident_risk", axis=1)
y = train["accident_risk"]


# Save test IDs for submission 
test_ids = test["id"] 
X_test_final = test.drop(columns=["id"])


categorical = ["road_type", "lighting", "weather", "time_of_day"]
boolean = ["road_signs_present", "public_road", "holiday", "school_season"]



preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("bool", FunctionTransformer(lambda x: x.astype(int)), boolean)
    ],
    remainder="passthrough"
)



model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




model.fit(X_train, y_train)



y_pred = model.predict(X_test)


print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))


test_preds = model.predict(X_test_final)


test_preds = np.round(test_preds, 3)


submission = pd.DataFrame({ "id": test_ids, "accident_risk": test_preds })


submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


submission.head()





