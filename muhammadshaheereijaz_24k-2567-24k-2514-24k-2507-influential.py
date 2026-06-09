import pandas as pd
from lightgbm import LGBMClassifier

train_path = "/kaggle/input/predict-who-is-more-influential-in-a-social-network/train.csv"
test_path = "/kaggle/input/predict-who-is-more-influential-in-a-social-network/test.csv"
sample_path = "/kaggle/input/predict-who-is-more-influential-in-a-social-network/sample_predictions.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

target = "Choice"
features = [c for c in train.columns if c != target]

X_train = train[features]
y_train = train[target]
X_test = test[features]

model = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

submission = pd.DataFrame({
    "Id": sample_submission["Id"],
    "Choice": preds
})

submission.to_csv("/kaggle/working/submission_influential.csv", index=False)
submission.head()


