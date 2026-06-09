import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
x_train = train.drop(['id', 'Calories'], axis=1)
y_train = train['Calories']


temp_df = x_train.drop('Sex', axis=1)
min_ = temp_df.min()
max_ = temp_df.max()
temp_df = (temp_df - min_) / (max_ - min_)
x_train = pd.concat([x_train['Sex'], temp_df], axis=1)


x_train.head()


y_train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
x_test = test.drop('id', axis=1)
temp_df = x_test.drop('Sex', axis=1)
temp_df = (temp_df - min_) / (max_ - min_)
x_test = pd.concat([x_test['Sex'], temp_df], axis=1)

x_test.head()


# Initialize the regressor
regressor = TabPFNRegressor(
    model_path="/kaggle/input/tabpfn/pytorch/regressor/1/tabpfn-v2-regressor.ckpt",
    ignore_pretraining_limits=True,
    # n_estimators=32,                # (int) Number of estimators in the ensemble for robustness.
    inference_config={
        "SUBSAMPLE_SAMPLES": 10000  # (int) Maximum number of samples per inference step to manage memory usage.
    },
)


def get_batched_predictions(x_test, batch_size=25000):
    predictions = []
    num_batches = len(x_test) // batch_size
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x_test))
        predictions.append(regressor.predict(x_test[start_idx:end_idx]))

    return np.concatenate(predictions)


kf = KFold(random_state=0, n_splits=10, shuffle=True)


scores = []
for i, (train_idxs, test_idxs) in enumerate(kf.split(x_train)):
    cv_train_x = x_train.iloc[train_idxs]
    cv_train_y = y_train.iloc[train_idxs]
    cv_test_x = x_train.iloc[test_idxs]
    cv_test_y = y_train.iloc[test_idxs]

    regressor.fit(cv_train_x, cv_train_y)
    predictions = get_batched_predictions(cv_test_x)
    rmsle = root_mean_squared_log_error(cv_test_y, predictions)
    scores.append(rmsle)
    print(f"Root Mean Squared Logarithmic Error (fold {i}): {rmsle}")

print(f"Average score: {np.mean(scores)}")


# Fit the data
regressor.fit(x_train, y_train)


# Predict on the test set
predictions = get_batched_predictions(x_test)


assert len(predictions) == len(x_test), "Number of predictions doesn't match the number of samples!"


submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission["Calories"] = predictions
submission.head()
submission.to_csv("submission.csv", index=False)

