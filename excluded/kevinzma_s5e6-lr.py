# important stuff that needs to be configured first
from __future__ import annotations

import os
import multiprocessing
from threadpoolctl import threadpool_limits

# optimize joblib to use all cores
total = multiprocessing.cpu_count()
n_jobs = 7  # 7 labels = 7 models in parallel = 7 processes (hardcoded)
per_proc = total // n_jobs
os.environ["OMP_NUM_THREADS"] = str(per_proc)
os.environ["MKL_NUM_THREADS"] = str(per_proc)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

from joblib import Parallel, delayed  # parallelization

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.sparse import issparse, hstack
from scipy.sparse._matrix import spmatrix

# import seaborn as sns
# from collections import Counter

from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif

import itertools
import time
from datetime import datetime


def df_to_sparse_matrix(df: pd.DataFrame) -> spmatrix | np.ndarray:
    dense_cols, sparse_cols = [], []

    for col in df.columns:
        if isinstance(df[col].dtype, pd.SparseDtype):
            sparse_cols.append(col)
        else:
            dense_cols.append(col)
    
    dense_block = df[dense_cols].to_numpy()
    if sparse_cols:
        sparse_block = df[sparse_cols].sparse.to_coo().tocsr()
        X_csr = hstack([dense_block, sparse_block], format="csr")
    else:
        X_csr = dense_block

    return X_csr


SEED = 43


! python3 --version


SUBMISSION_MODE = False

COMP = "input"
if os.path.exists("/kaggle/input"):
    COMP = "/kaggle/input/playground-series-s5e6"
elif os.path.exists("/home/jovyan/work"):
    COMP = "/home/jovyan/work"

train_original = pd.read_csv(f"{COMP}/train.csv")
test = pd.read_csv(f"{COMP}/test.csv")
sample = pd.read_csv(f"{COMP}/sample_submission.csv")

if SUBMISSION_MODE:
    # consistent feature engineering
    test_with_placeholder = test.copy()
    test_with_placeholder["Fertilizer Name"] = "UNKNOWN"
    combined_data = pd.concat(
        [
            train_original.assign(Source="train"),
            test_with_placeholder.assign(Source="test"),
        ],
        ignore_index=True,
    )
    train = combined_data
else:
    train = train_original.copy()
    train["Source"] = "train"
    test["Source"] = "test"


train["Fertilizer Name"].unique()


quantitative = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]
qualitative = ["Soil Type", "Crop Type"]

f_interact_quant = []
f_group_normed = []
f_transformed = []


for f in quantitative:
    train[f"{f}^2"] = train[f] ** 2
    train[f"{f}_sqrt"] = np.sqrt(train[f])
    train[f"{f}_log"] = np.log1p(train[f])
    f_transformed += [f"{f}^2", f"{f}_sqrt", f"{f}_log"]  # 6x3 = 18 more features
    

# do interactions before normalization to better capture
for f1, f2 in list(itertools.combinations(quantitative, 2)):
    train[f"{f1} x {f2}"] = train[f1] * train[f2]
    train[f"{f1} / {f2}"] = train[f1] / (train[f2] + 1)
    train[f"{f2} / {f1}"] = train[f2] / (train[f1] + 1)
    f_interact_quant += [f"{f1} x {f2}", f"{f1} / {f2}", f"{f2} / {f1}"]

# z-score normalize within groups (e.g. soil type, crop type)
for cat in qualitative:
    grouped = train.groupby(cat)[quantitative]
    scaled = grouped.transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )  # normalization already occurs here
    train[[f"{col}_vs_{cat}" for col in quantitative]] = scaled
    f_group_normed += [f"{col}_vs_{cat}" for col in quantitative]


continuous_features = quantitative + f_interact_quant + f_transformed

# quantitative quartiles/bins to categorical
all_dummies: list[pd.DataFrame] = [train]

# bins for transformed are basically same as original features
BINS = 150
for feat in quantitative + f_interact_quant + f_group_normed + f_transformed:  # NOTE: adding group_normed boosted from .316 -> .322
    quartiles = pd.qcut(train[feat], q=BINS, labels=False, duplicates="drop")

    # convert to one-hot
    quartile_dummies = pd.get_dummies(quartiles, prefix=f"{feat}_Q", dtype=int, sparse=True)

    # temporary store in list to prevent dataframe copying every iteration
    all_dummies.append(quartile_dummies)

    # # new approach: each unique integer value is a "category" -> one-hot encode
    # unique_values = (train[feat]).astype(int)
    # categorical_dummies = pd.get_dummies(
    #     unique_values, prefix=f"{feat}_VAL", dtype=int, sparse=True
    # )
    # all_dummies.append(categorical_dummies)

train = pd.concat(all_dummies, axis=1)  # 42s -> 5.6s on M3 Ultra

# def create_feature_dummies(feat_data: pd.Series, feat_name: str) -> pd.DataFrame:
#     unique_values = feat_data.astype(int)
#     return pd.get_dummies(unique_values, prefix=f"{feat_name}_VAL", dtype=int)

# # Process features in parallel - took even longer
# dummy_results: list[pd.DataFrame] = Parallel(n_jobs=-1)(
#     delayed(create_feature_dummies)(train[feat], feat)
#     for feat in quantitative + interact_quant
# )

# train = pd.concat([train] + dummy_results, axis=1)


# finally, normalize original quantitative features and interactions
means = train[continuous_features].mean()
stds = train[continuous_features].std(ddof=0)
train[continuous_features] = (
    train[continuous_features] - means
) / stds

# one-hot encode qualitative features
# TODO (fixed): could be encoded differently in test dataset
train = pd.get_dummies(train, columns=qualitative, dtype=int)


print("Quantitative interactions:", len(f_interact_quant))
print("Categorial normalizations:", len(f_group_normed))


def train_info() -> None:
    print(f"Number of columns in train: {len(train.columns)}")
    memory_bytes = train.memory_usage(deep=True).sum()
    if memory_bytes >= 1024**3:
        print(f"Memory usage: {memory_bytes / 1024**3:.2f} GB")
    else:
        print(f"Memory usage: {memory_bytes / 1024**2:.2f} MB")

train_info()


dense_cols = [c for c in train.columns if not isinstance(train[c].dtype, pd.SparseDtype)]
sparse_cols = [c for c in train.columns if c not in dense_cols]

print(f"{len(dense_cols)=}, {len(sparse_cols)=}")


train.head(5)


class LogisticRegressionClassifier:
    def __init__(
        self,
        *,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        lambda_: float = 0.1,
    ):
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.lambda_: float = lambda_

    def get_params(self) -> dict:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "lambda_": self.lambda_,
        }

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        #
        # $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
        #
        # return 1.0 / (1.0 + np.exp(-z))  # can overflow for large z
        computed = np.piecewise(  # https://stackoverflow.com/a/64717799
            z,
            [z > 0],
            [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
        )
        return np.clip(computed, 1e-15, 1 - 1e-15)

    def compute_loss(self, X: np.ndarray | spmatrix, y: np.ndarray) -> float:
        # we want to minimize this function

        # shape of X: (n_observations, n_features)
        # $$L_i = - \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
        #
        # $$ J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n L_i $$
        #
        z = X @ self.weights + self.bias  # shape: (n_observations, )
        y_hat = self.sigmoid(z)

        # average loss over all observations
        J = np.mean(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))

        # L2 regularization
        #
        # $$ J_{reg} = J + \lambda \sum_{j=1}^n w_j^2 $$
        #
        J_reg = J + self.lambda_ * np.sum(self.weights**2)

        return J_reg

    def compute_gradients(
        self, X: np.ndarray | spmatrix, y: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Compute the direction (gradient) to minimize the loss function,
        which depends on the weights and bias.

        Notation:
        - i: index of observation
        - j: index of feature
        """
        n_observations = y.shape[0]
        # $$ z_i(\mathbf{w}, b) = \mathbf{w} \cdot \mathbf{x}_i + b $$
        z: np.ndarray = X @ self.weights + self.bias  # shape: (n_observations, )
        # $$ \hat{y}_i(\mathbf{w}, b) = \sigma(z_i(\mathbf{w}, b)) $$
        y_hat: np.ndarray = self.sigmoid(z)  # shape: (n_observations, )

        error = y_hat - y  # shape: (n_observations, )

        # if y=1 (target class): weight is (1 - target_proportion)
        # if y=0 (non-target): weight is target_proportion
        class_weights = (
            y * (1 - self.target_proportion) + (1 - y) * self.target_proportion
        )  # shape: (n_observations, )
        weighted_error = error  # * class_weights  # shape: (n_observations, )

        # differentiate J_reg
        #
        # $$ \frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i) x_j^{(i)} $$
        #
        # vectorized:
        #
        # $$ \nabla_w J = \frac{1}{n} \mathbf{X}^T (\mathbf{\hat{y}} - \mathbf{y}) $$
        #
        grad_w_J = X.T @ weighted_error / n_observations  # shape: (n_features, )

        #
        # $$ \nabla_w J_{reg} = \nabla_w J + 2 \lambda \mathbf{w} $$
        #
        grad_w_J_reg = (
            grad_w_J + 2 * self.lambda_ * self.weights
        )  # shape: (n_features, )

        # $$ \frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i) $$
        #
        dJ_db = np.mean(error)
        return grad_w_J_reg, dJ_db

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray) -> None:
        if not (isinstance(X, np.ndarray) or issparse(X)) or not isinstance(
            y, np.ndarray
        ):
            raise ValueError(
                "X must be NumPy array or sparse matrix, y must be NumPy array"
            )
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X, y must be 2D, 1D arrays with matching n_observations")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0s and 1s")

        n_features = X.shape[1]
        self.weights = np.zeros(
            n_features
        )  # np.random.normal(0, 0.1, n_features)  # gaussian
        self.bias = 0.0
        self.target_proportion = y.mean()

        for i in range(self.n_iterations):
            grad_w_J_reg, db = self.compute_gradients(X, y)

            # near local minimum
            gradient_norm = np.linalg.norm(grad_w_J_reg) + np.abs(db)

            # if i % 200 == 0:
            #     print(f"Iter {i}: grad_norm = {gradient_norm:.2e}")

            if gradient_norm < 1e-5:
                # print(
                #     f"Converged at iteration {i}: gradient_norm = {gradient_norm:.2e}"
                # )
                break

            # decayed_learning_rate = self.learning_rate / (1 + i * self.learning_rate)

            # $$ \mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_w J_{reg} $$
            #
            self.weights -= self.learning_rate * grad_w_J_reg

            # $$ b \leftarrow b - \frac{\partial J}{\partial b} $$
            #
            self.bias -= db
        print(f"Final gradient norm: {gradient_norm:.2e}")
        print(f"Final loss: {self.compute_loss(X, y):.2e}")

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        z = X @ self.weights + self.bias  # shape: (n_observations, )
        y_hat = self.sigmoid(z)  # shape: (n_observations, )
        # return (y_hat >= 0.127*1.5).astype(int)  # shape: (n_observations, )
        return y_hat


class OneVsAllClassifier:
    def __init__(
        self, *, learning_rate=0.01, n_iterations=1000, lambda_=0.1, n_jobs=-1
    ):
        self.models: list[LogisticRegressionClassifier] = []
        self.classes: np.ndarray | None = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.n_jobs = n_jobs  # -1 for all cores

    def _train_single_model(
        self, X: np.ndarray | spmatrix, y: np.ndarray, label: str
    ) -> tuple[LogisticRegressionClassifier, np.ndarray]:
        y_binary = (y == label).astype(int)
        model = LogisticRegressionClassifier(
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            lambda_=self.lambda_,
        )
        model.fit(X, y_binary)
        print(
            f"[{datetime.now()}] Trained {label} model"
        )  # isn't printing during parallel
        return model, label

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray) -> None:
        classes = np.unique(y)
        
        with threadpool_limits(limits=per_proc, user_api='blas'):
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._train_single_model)(X, y, label) for label in classes
            )

        # results = []
        # for label in classes:
        #     model, label_result = self._train_single_model(X, y, label)
        #     results.append((model, label_result))
        
        self.models = [model for model, _ in results]
        self.classes = np.array([label for _, label in results])

    def predict(
        self, X: np.ndarray | spmatrix, *, top_k: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        # e.g. [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] for models A, B; for observations 0, 1, 2
        observations_per_model = [
            model.predict(X) for model in self.models
        ]  # shape: (n_models, n_observations)

        # we want top 3 for each observation
        # target shape: (n_observations, n_models) -> (n_observations, top_k)
        # e.g. [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]] for observations 0, 1, 2; for labels A, B
        models_per_observation = np.column_stack(observations_per_model)

        top_classes_idx = np.argsort(models_per_observation, axis=1)[:, ::-1][
            :, :top_k
        ]  # shape: (n_observations, top_k)

        # use fancy indexing
        return (
            self.classes[top_classes_idx],  # shape: (n_observations, top_k)
            models_per_observation[top_classes_idx],  # shape: (n_observations, top_k)
        )


if False:
    # Test with small dataset generated from Grok
    X = np.array(
        [
            [5, 7, 2],  # study hrs, sleep hrs, practice tests
            [2, 6, 1],
            [4, 8, 3],
            [3, 5, 1],
            [6, 7, 4],
        ]
    )
    y = np.array([1, 0, 1, 0, 1])  # pass/fail

    print("Study | Sleep | Tests | Pass")
    for i in range(len(X)):
        print(f"{X[i, 0]:>5} | {X[i, 1]:>5} | {X[i, 2]:>5} | {y[i]:>4}")

    model = LogisticRegressionClassifier(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.sigmoid(np.dot(X, model.weights) + model.bias)

    print(f"\nWeights: {model.weights}")
    print(f"Bias: {model.bias:.3f}")
    print(f"Accuracy: {np.mean(preds == y):.3f}")

    # Test on new data
    new_X = np.array([[5, 6, 3], [2, 4, 1]])
    new_pred = model.predict(new_X)
    new_prob = model.sigmoid(np.dot(new_X, model.weights) + model.bias)
    print(f"New student [5,6,3]: {new_pred[0]} (prob: {new_prob[0]:.3f})")
    print(f"New student [2,4,1]: {new_pred[1]} (prob: {new_prob[1]:.3f})")


train["Fertilizer Name"].value_counts()


train_info()


# sample = train.sample(n=50000, random_state=SEED)
sample = train.copy()

# features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
features = [
    col for col in train.columns if col not in ["id", "Source", "Fertilizer Name"]
]


def MAP_K(y_true, y_pred, K=3) -> float:
    score = 0
    U = len(y_true)
    for true_label, preds in zip(y_true, y_pred):
        for k, pred_label in enumerate(preds[:K]):
            if pred_label == true_label:
                score += 1.0 / (k + 1)
                break  # don't count duplicates
    return score / U


if False:  # single model test
    X = sample[features][:90000].to_numpy()
    X_test = sample[features][90000:].to_numpy()

    y_is_dap = sample["Fertilizer Name"] == "DAP"

    y = y_is_dap[:90000]
    y_test = y_is_dap[90000:]

    print(f"DAP samples: {y_is_dap.sum()}")
    print(f"Other samples: {len(y_is_dap) - y_is_dap.sum()}")
    print(f"DAP ratio: {y_is_dap.sum() / len(y_is_dap):.3f}")

    model = LogisticRegressionClassifier(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    print(model.get_params())

    preds = model.predict(X)
    probs = model.sigmoid(np.dot(X, model.weights) + model.bias)

    print(f"Predicted DAP ratio: {preds.mean():.3f}")
    print(f"Probability for actual DAP: {probs[y == 1].mean():.3f}")
    print(f"Probability for actual non-DAP: {probs[y == 0].mean():.3f}")
    print(f"Accuracy: {(preds == y).mean():.3f}")

    pred_test = model.predict(X_test)
    probs_test = model.sigmoid(np.dot(X_test, model.weights) + model.bias)

    print(f"Predicted DAP ratio: {pred_test.mean():.3f}")
    print(f"Probability for actual DAP: {probs_test[y_test == 1].mean():.3f}")
    print(f"Probability for actual non-DAP: {probs_test[y_test == 0].mean():.3f}")
    print(f"Accuracy: {(pred_test == y_test).mean():.3f}")
    print(probs_test[:10])
elif not SUBMISSION_MODE:
    learning_rates = [0.07]
    iterations = [2000]
    lambdas = [0]

    best_score = 0
    associated_train_score = 0
    best_params = (None, None, None)
    
    # there are sparse columns, we need scipy sparse matrix
    time_start_sparse = time.time()
    print("Converting to sparse matrix...")
    X_csr = df_to_sparse_matrix(sample[features])
    time_end_sparse = time.time()
    print(f"Time to convert to sparse matrix: {time_end_sparse - time_start_sparse:.2f}s")

    grid = itertools.product(learning_rates, iterations, lambdas)
    for learning_rate, iteration, lambda_val in grid:

        print(
            f"learning_rate={learning_rate}, iterations={iteration}, lambda={lambda_val}"
        )

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

        train_scores = []
        test_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(sample)):
            print(f"Fold {fold + 1} of {kf.n_splits}")
            time_start = time.time()
            
            X_train = X_csr[train_idx]
            X_test = X_csr[test_idx]

            # labels (can stay as numpy array)
            y_train = sample["Fertilizer Name"].iloc[train_idx].to_numpy()
            y_test = sample["Fertilizer Name"].iloc[test_idx].to_numpy()

            # selector = SelectKBest(f_classif, k=50)
            # X_train = selector.fit_transform(X_train, y_train)
            # X_test = selector.transform(X_test)

            ova_model = OneVsAllClassifier(
                learning_rate=learning_rate, n_iterations=iteration, lambda_=lambda_val
            )
            ova_model.fit(X_train, y_train)

            pred_train, probs_train = ova_model.predict(X_train, top_k=3)
            train_score = MAP_K(y_train, pred_train, K=3)
            train_scores.append(train_score)

            pred_test, probs_test = ova_model.predict(X_test, top_k=3)
            test_score = MAP_K(y_test, pred_test, K=3)
            test_scores.append(test_score)

            time_end = time.time()
            print(
                f"Fold {fold + 1} - Training MAP@3: {train_score:.5f}, Test MAP@3: {test_score:.5f}, time: {time_end - time_start:.2f}s"
            )

        avg_test_score = np.mean(test_scores)
        print(
            f"\nAverage Training MAP@3: {np.mean(train_scores):.5f} Â± {np.std(train_scores):.5f}"
        )
        print(f"Average Test MAP@3: {avg_test_score:.5f} Â± {np.std(test_scores):.5f}")

        if avg_test_score > best_score:
            best_score = avg_test_score
            best_score_std = np.std(test_scores)
            associated_train_score = np.mean(train_scores)
            best_params = (learning_rate, iteration, ova_model.lambda_)
        print(
            f"ğŸ�† Highest MAP@3 so far: (train={associated_train_score:.5f}, test={best_score:.5f} Â± {best_score_std:.5f})"
            f" with params: {best_params}, sample={len(sample)}"
        )
        print("-" * 25)





if SUBMISSION_MODE:
    sample = train[train["Source"] == "train"]

    features = [
        col for col in train.columns if col not in ["id", "Source", "Fertilizer Name"]
    ]

    X_train = df_to_sparse_matrix(sample[features])
    X_test = df_to_sparse_matrix(train[train["Source"] == "test"][features])
    
    y_train = sample["Fertilizer Name"].to_numpy()
    test_ids = train[train["Source"] == "test"]["id"].to_numpy()
    # no y_test for real submission

    ova_model = OneVsAllClassifier(learning_rate=0.07, n_iterations=2000, lambda_=0)
    ova_model.fit(X_train, y_train)

    pred_train, probs_train = ova_model.predict(X_train, top_k=3)
    train_score = MAP_K(y_train, pred_train, K=3)
    print(f"Final training MAP@3: {train_score:.5f}")

    pred_test, probs_test = ova_model.predict(X_test, top_k=3)

    submission_data = []
    for i, test_id in enumerate(test_ids):
        top_3_preds = pred_test[i]
        pred_str = " ".join(top_3_preds)
        submission_data.append([test_id, pred_str])

    submission_df = pd.DataFrame(submission_data, columns=["id", "Fertilizer Name"])

    submission_df.to_csv("submission.csv", index=False)
    print(f"Submission saved to submission.csv with {len(submission_df)} predictions")

