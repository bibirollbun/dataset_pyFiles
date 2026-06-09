import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

DATA_PATH = Path("/kaggle/input/deceptive-points-aicc-round-0")
OUT_PATH = Path("/kaggle/working/")
TRAIN_PATH = DATA_PATH / "train.csv"
TEST_PATH = DATA_PATH / "test.csv"
SUB_PATH = OUT_PATH / "submission.csv"
SOL_PATH = DATA_PATH / "solution.csv" # Not available during the competition

train_df = pd.read_csv(TRAIN_PATH)
X_train = train_df[["feature1","feature2","feature3","feature4"]].values
y_train = train_df["target"].values

test_df = pd.read_csv(TEST_PATH)
X_test = test_df[["feature1","feature2","feature3","feature4"]].values
test_ids = test_df["ID"].values

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

submission_df = pd.DataFrame({
    "ID": test_ids,
    "Target": y_pred
})
submission_df.to_csv(SUB_PATH, index=False)
print("Saved")


import pandas as pd
from sklearn.metrics import mean_squared_error

submission_df = pd.read_csv(SUB_PATH)
y_pred = submission_df["Target"].values

"""
This is how a submission will be evaluated:

solution_df = pd.read_csv(SOL_PATH)
y_teacher = solution_df["Target"].values

mse = mean_squared_error(y_teacher, y_pred)
print(f"MSE on teacher-only test set: {mse:.3f}")
"""

