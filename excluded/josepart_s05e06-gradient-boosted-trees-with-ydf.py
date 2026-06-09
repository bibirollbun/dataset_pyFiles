# Upgrade library to have access to the latest features
!pip install -U ydf


import os
import numpy as np
import pandas as pd

from typing import Union
from sklearn.model_selection import KFold
from ydf import GradientBoostedTreesLearner


cat_features = ['Soil Type', 'Crop Type']
num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
target_col = 'Fertilizer Name'


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
x_train = train.drop('id', axis=1)

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
    x_train = x_train.sample(25000, replace=False, random_state=42, axis=0)

x_train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
x_test = test.drop('id', axis=1)

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
    x_test = x_test[:25000]

x_test.head()


learner = GradientBoostedTreesLearner(label=target_col, validation_ratio=0.0, num_trees=1000, max_depth=8)


learner.hyperparameters


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


def convert_predictions(predictions, labels, k=3):
    sorted_idxs = np.argsort(-predictions, axis=1)[:, :k]
    labels_np = np.array(labels)
    return list(map(" ".join, labels_np[sorted_idxs]))


kf = KFold(random_state=0, n_splits=5, shuffle=True)


scores = []
models = []
for i, (train_idxs, test_idxs) in enumerate(kf.split(x_train)):
    cv_train_x = x_train.iloc[train_idxs]
    cv_train_y = cv_train_x[target_col]
    cv_test_x = x_train.iloc[test_idxs]
    cv_test_y = cv_test_x[target_col]

    model = learner.train(cv_train_x)
    models.append(model)
    predictions = model.predict(cv_test_x)
    predictions = convert_predictions(predictions, model.label_classes())
    score = mean_average_precision_at_k(cv_test_y, list(map(str.split, predictions)), 3)
    scores.append(score)
    print(f"\nMAP@3 (fold {i}): {score}\n")

print(f"Average CV score: {np.mean(scores)}")


# Make sure that the order of labels is the same for every model
for model in models:
    print(model.label_classes())

assert all(model.label_classes() == models[0].label_classes() for model in models[1:]), "The order of labels doesn't match!"


models[0].describe()


models[1].describe()


models[2].describe()


predictions = []
for model in models:
    predictions.append(model.predict(x_test))

# I found that ensembling the models trained on different folds
# slightly improves the public LB score. This can be seen by
# comparing the scores of Version 1 and Version 9:

# Version 1    |    0.33531
# Version 9    |    0.33795

predictions = np.mean(predictions, axis=0)
print(predictions.shape)


labels = models[0].label_classes()
print(labels)


submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
    submission = submission[:25000]
submission[target_col] = convert_predictions(predictions, labels)
submission.to_csv("submission.csv", index=False)
submission.head()

