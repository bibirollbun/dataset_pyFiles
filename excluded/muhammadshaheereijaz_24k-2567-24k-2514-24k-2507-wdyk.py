import pandas as pd

test_path = "/kaggle/input/WhatDoYouKnow/test.csv.gz"
test = pd.read_csv(test_path, compression='gzip')

print(test.columns)
test.head()



import pandas as pd

train_path = "/kaggle/input/WhatDoYouKnow/training.csv.gz"
train = pd.read_csv(train_path, compression='gzip')

print(train.columns)
train.head()



import pandas as pd
from lightgbm import LGBMClassifier

train_path = "/kaggle/input/WhatDoYouKnow/training.csv.gz"
test_path = "/kaggle/input/WhatDoYouKnow/test.csv.gz"

train = pd.read_csv(train_path, compression='gzip')
test = pd.read_csv(test_path, compression='gzip')

# drop non-numeric / target / ID columns
drop_cols = [
    "correct", "user_id", "question_id", "date_of_test",
    "tag_string", "round_started_at", "answered_at", "deactivated_at"
]

train_features = [c for c in train.columns if c not in drop_cols and train[c].dtype in ["int64","float64"]]
test_features = [c for c in train_features if c in test.columns]

X_train = train[test_features]
y_train = train["correct"]
X_test = test[test_features]

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)

# use user_id + question_id as unique identifier for submission
submission = pd.DataFrame({
    "user_id": test["user_id"],
    "question_id": test["question_id"],
    "correct": preds
})

submission.to_csv("submission_whatdoyouknow.csv", index=False)
submission.head()


