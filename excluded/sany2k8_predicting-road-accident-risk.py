import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv", low_memory=False)
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv", low_memory=False)
print("Train shape:", train.shape)
print("Test shape:", test.shape)



train.head()


test.head()


train.info()
train.describe()
print(train.isnull().sum())



import matplotlib.pyplot as plt
plt.hist(train["accident_risk"], bins=30)
plt.title("Accident Risk Distribution")
plt.xlabel("accident_risk")
plt.ylabel("Count")
plt.show()



data = pd.concat([train.drop(columns=["accident_risk"]), test], axis=0)


data["curvature_per_lane"] = data["curvature"] / (data["num_lanes"] + 1e-3)
data["speed_curvature_ratio"] = data["speed_limit"] / (data["curvature"] + 0.01)
data["is_dark"] = data["lighting"].isin(["night", "dim"]).astype(int)
data["rainy_night"] = ((data["lighting"] == "night") & (data["weather"] == "rainy")).astype(int)


from sklearn.preprocessing import LabelEncoder

# Convert booleans to 0/1
bool_cols = data.select_dtypes("bool").columns
data[bool_cols] = data[bool_cols].astype(int)

# Label Encode object columns and convert to numeric
for col in data.select_dtypes("object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col]).astype(int) # Convert to int after encoding


train_prep = data.iloc[:len(train)]
test_prep = data.iloc[len(train):]

X_train = train_prep.drop(columns=["id"])
y_train = train["accident_risk"]
X_test = test_prep.drop(columns=["id"])



import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)



from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
rmse = np.sqrt(-scores.mean())
print("CV RMSE:", rmse)



preds = model.predict(X_test)
preds = preds.clip(0, 1)  # ensure predictions stay in [0,1]



submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": preds
})
submission.to_csv("sample_submission.csv", index=False)



print("submission shape:", submission.shape)


submission.head()

