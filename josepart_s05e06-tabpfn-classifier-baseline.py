import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Any, Union
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
from tabpfn import TabPFNClassifier


cat_features = ['Soil Type', 'Crop Type']
num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
target_col = 'Fertilizer Name'


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# Training set features
x_train = train.drop(['id', target_col], axis=1)
# Training set target
y_train = train[target_col]


x_train.head()


y_train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
x_test = test.drop('id', axis=1)

x_test.head()


train_full = x_train.copy()
train_full[target_col] = y_train
for col in cat_features + [target_col]:
    train_full[col] = train_full[col].map({k: v for v, k in enumerate(set(train_full[col]))})
coor_matrix = train_full.corr()


coor_matrix[target_col].sort_values(ascending=False)


figs = scatter_matrix(train_full.sample(25000, replace=False, random_state=42, axis=0), figsize=(16, 12))


# Note that I cannot use varargs when defining the extended class. See error below:
# RuntimeError: scikit-learn estimators should always specify their parameters in the
# signature of their __init__ (no varargs). <class '__main__.TabPFNClassifierExtended'>
# with constructor (self, *args, **kwargs) doesn't  follow this convention.

class TabPFNClassifierExtended(TabPFNClassifier):
    def __init__(
        self,
        model_path,
        ignore_pretraining_limits,
        inference_config,
    ):
        super().__init__(
            model_path=model_path,
            ignore_pretraining_limits=ignore_pretraining_limits,
            inference_config=inference_config,
        )


    def predict(self, X: Any) -> np.ndarray:
        """Predict the class labels for the provided input samples.

        Args:
            X: The input samples.

        Returns:
            The predicted class labels.
        """
        proba = self.predict_proba(X)
        predictions = np.argsort(-proba, axis=1)
        return [self.label_encoder_.inverse_transform(prediction) for prediction in predictions]


# Initialize the classifier
classifier = TabPFNClassifierExtended(
    model_path="/kaggle/input/tabpfn/pytorch/classifier/1/tabpfn-v2-classifier.ckpt",
    ignore_pretraining_limits=True,
    # n_estimators=32,                # (int) Number of estimators in the ensemble for robustness.
    inference_config={
        "SUBSAMPLE_SAMPLES": 10000  # (int) Maximum number of samples per inference step to manage memory usage.
    },
)


def get_batched_predictions(x_test: pd.DataFrame, batch_size: int=25000):
    predictions = []
    num_batches = len(x_test) // batch_size
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x_test))
        predictions.append(classifier.predict(x_test[start_idx:end_idx]))

    return np.concatenate(predictions)


def mean_average_precision_at_k(
    y_true: Union[list, np.ndarray, pd.Series],
    y_pred: Union[list, np.ndarray],
    k: int=3,
):
    """
    Computes the Mean Average Precision @ k (MAP@k) for a classification task.

    Args:
        y_true (list or numpy array or pd.Series): A list where each inner list/row
            contains the true labels for a given sample.
        y_pred (list of lists or numpy array): A list where each inner list
            contains the predicted labels ordered according to their scores.
        k (int): The number of top predictions to consider for precision calculation.

    Returns:
        float: The Mean Average Precision @ k.
    """
    if not isinstance(y_true, (list, np.ndarray, pd.Series)):
        raise TypeError("y_true must be a list or a numpy array or a pandas series.")
    if not isinstance(y_pred, (list, np.ndarray)):
        raise TypeError("y_pred must be a list or a numpy arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of samples.")

    if not isinstance(y_true, list):
        y_true_list = y_true.tolist()
    else:
        y_true_list = y_true

    average_precisions = []
    for i in range(len(y_true_list)):
        # Consider only the top k predictions
        top_k_pred = y_pred[i][:k]
        precision = -1
        for n, pred in enumerate(top_k_pred):
            # Check if the predicted class is the true label
            if pred == y_true_list[i]:
                precision = 1 / (n + 1)
                break

        if precision < 0:
            average_precisions.append(0.0)  # No relevant items found in top k
        else:
            average_precisions.append(precision)

    return np.mean(average_precisions)


kf = KFold(random_state=0, n_splits=5, shuffle=True)


scores = []
for i, (train_idxs, test_idxs) in enumerate(kf.split(x_train)):
    cv_train_x = x_train.iloc[train_idxs]
    cv_train_y = y_train.iloc[train_idxs]
    cv_test_x = x_train.iloc[test_idxs]
    cv_test_y = y_train.iloc[test_idxs]

    classifier.fit(cv_train_x, cv_train_y)
    predictions = get_batched_predictions(cv_test_x)
    score = mean_average_precision_at_k(cv_test_y, predictions, 3)
    scores.append(score)
    print(f"MAP@3 (fold {i}): {score}")

print(f"\nAverage CV score: {np.mean(scores)}")


# Fit the data
classifier.fit(x_train, y_train)


# Predict on the test set
predictions = get_batched_predictions(x_test)


assert len(predictions) == len(x_test), "Number of predictions doesn't match the number of samples!"


submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission[target_col] = [" ".join(preds[:3]) for preds in predictions]
submission.to_csv("submission.csv", index=False)
submission.head()

