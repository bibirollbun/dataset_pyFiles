import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
test = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")

y = train["Cover_Type"]
X = train.drop(columns=["Id", "Cover_Type"])
X_test = test.drop(columns=["Id"])

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

print("Обучение RandomForest...")
model.fit(X, y)

test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test["Id"],
    "Cover_Type": test_preds.astype(int)
})

submission.to_csv("submission.csv", index=False)
print("Готово. Файл submission.csv сохранен.")

