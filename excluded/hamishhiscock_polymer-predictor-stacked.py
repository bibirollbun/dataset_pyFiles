!pip install tabpfn
!pip install rdkit


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
from functools import partial

from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tabpfn import TabPFNRegressor
from tabpfn.utils import meta_dataset_collator
from tabpfn.finetune_utils import clone_model_for_evaluation


# configure data

csv_path = "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
data = pd.read_csv(csv_path)


# represent using 512 bits so closer to tabpfn pretraining limit of 500 (this could be experimented with)
def _smiles_to_fingerprint(smiles: str, radius: int=2, nBits: int=512) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    return np.array(generator.GetFingerprint(mol))


def configure_data(df: pd.DataFrame, test_size: float=0.2, targets: list[str]=targets) -> (np.ndarray, np.ndarray):
    dfs = {}
    features = {}
    y = {}
    train_val_df, dfs["test"] = train_test_split(
        data,
        test_size=test_size,
        random_state=42,
        shuffle=True
    )
    dfs["train"], dfs["val"] = train_test_split(
        train_val_df,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )
    for fold, df in dfs.items():
        features[fold] = np.vstack(df["SMILES"].apply(_smiles_to_fingerprint).values)
        y[fold] = df[targets].to_numpy()
    features["train_val"] = np.vstack((features["train"], features["val"]))
    y["train_val"] = np.vstack((y["train"], y["val"]))
    return features, y
    

features, y = configure_data(data)


def multitask_evaluation_function(
    y_true: np.array, y_predict: np.array, task_names: list[str],
) -> dict[str, float]:
    num_tasks = len(task_names)
    assert num_tasks <= y_true.shape[1], "More targets than data"
    ranges = {}
    num_samples = {}
    mae = {}
    for i, target in enumerate(task_names):
        ranges[target] = np.nanmax(y_predict[:, i]) - np.nanmin(y_predict[:, i])
        num_samples[target] = (~np.isnan(y_true[:, i])).sum()
        mae[target] = np.nanmean(np.abs(y_true[:, i] - y_predict[:, i]))

    num_samples_normalisation = sum([
        1 / np.sqrt(n) for n in num_samples.values()
    ])
    for target in task_names:
        mae["overall"] = mae.get("overall", 0) + (
            (num_tasks * (1 / np.sqrt(num_samples[target])))
            / (ranges[target] * num_samples_normalisation)
        ) * num_samples[target] * mae[target]
    mae["overall"] /= y_true.shape[0]
    return mae


def fit_target_models(model: type, X: np.ndarray, y: np.ndarray, targets: list[str]=targets, **model_kwargs) -> dict[str, type]:
    models = {}
    for i, target in enumerate(targets):
        individual_y = y[:, i]
        mask  = ~np.isnan(individual_y)
        models[target] = model(**model_kwargs)
        models[target].fit(X[mask], individual_y[mask])
    return models


def evaluate_models(models: dict[str, type], X: np.ndarray, y: np.ndarray, targets: list[str]=targets) -> dict[str, float]:
    y_predict = np.zeros_like(y)
    for i, target in enumerate(targets):
        y_predict[:, i] = models[target].predict(X)
    return multitask_evaluation_function(y, y_predict, targets)


random_forest_models = fit_target_models(RandomForestRegressor, features["train_val"], y["train_val"], n_estimators=100, random_state=42)
print("Random Forest baseline:")
evaluate_models(random_forest_models, features['test'], y['test'])


tabpfn_models = fit_target_models(
    TabPFNRegressor, features["train_val"], y["train_val"],
    n_estimators=1, random_state=42, ignore_pretraining_limits=True,
)
print("Out-of-the-box TabPFN:")
evaluate_models(tabpfn_models, features['test'], y['test'])


def append_predictions_to_features(models: dict[str, type], X: np.ndarray) -> np.ndarray:
    appended_X = X.copy()
    for model in models.values():
        predictions = model.predict(X)
        appended_X = np.hstack((appended_X, predictions.reshape(-1, 1)))
    return appended_X


first_layer_models = fit_target_models(
    TabPFNRegressor, features["train"], y["train"],
    n_estimators=1, random_state=42, ignore_pretraining_limits=True,
)
appended_val_features = append_predictions_to_features(first_layer_models, features["val"])
appended_test_features = append_predictions_to_features(first_layer_models, features["test"])
second_layer_models = fit_target_models(
    TabPFNRegressor, appended_val_features, y["val"],
    n_estimators=1, random_state=42, ignore_pretraining_limits=True,
)
print("Stacked TabPFN model")
evaluate_models(second_layer_models, appended_test_features, y["test"])


evaluate_models(first_layer_models, features["test"], y["test"])


num_target_nans = np.sum(np.isnan(y["train_val"]), axis=0)
print({target: num_target_nans[i] for i, target in enumerate(targets)})


single_target_ix = 0
single_target = targets[single_target_ix]

X = features["train_val"]
single_y = y["train_val"][:, single_target_ix]
mask = ~np.isnan(single_y)
X = X[mask]
single_y = single_y[mask]
X_test = features["test"]
single_y_test = y["test"][:, single_target_ix]
mask = ~np.isnan(single_y_test)
X_test = X_test[mask]
single_y_test = single_y_test[mask]


num_epochs = 20
config = {
    "n_estimators": 1,
    "random_state": 42,
    "ignore_pretraining_limits": True,
    "device": "cuda",
    "inference_precision": torch.float16,
    #"inference_config": {"SUBSAMPLE_SAMPLES": X.shape[0]}
}
regressor = TabPFNRegressor(
    **config, fit_mode="batched", differentiable_input=False,
)
splitter = partial(train_test_split, test_size=0.3)
training_datasets = regressor.get_preprocessed_datasets(
    X, single_y, splitter,# max_data_size=int(0.7 * X.shape[0]),
)
dataloader = DataLoader(training_datasets, batch_size=1, collate_fn=meta_dataset_collator)
optimizer = Adam(regressor.model_.parameters(), lr=1.0e-6)


def evaluate_model(
    model: TabPFNRegressor, config: dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> float:
    eval_model = clone_model_for_evaluation(regressor, config, TabPFNRegressor)
    eval_model.fit(X_train, y_train)
    predictions = eval_model.predict(X_test)
    return mean_absolute_error(y_test, predictions)


def _train_epoch(model, optimizer, batch):
    optimizer.zero_grad()
    (
        X_trains_p,
        X_tests_p,
        y_trains_p,
        y_test_std,
        cat_ixs,
        confs,
        norm_bardist,
        bardist,
        _,
        batch_y_test_raw,
    ) = batch

    model.normalized_bardist_ = norm_bardist[0]
    model.fit_from_preprocessed(X_trains_p, y_trains_p, cat_ixs, confs)
    logits, _, _ = regressor.forward(X_tests_p)

    # For regression, the loss function is part of the preprocessed data
    loss_fn = norm_bardist[0]
    y_target = y_test_std

    loss = loss_fn(logits, y_target.to(config["device"])).mean()
    loss.backward()
    optimizer.step()


print(f"Initial MAE: {evaluate_model(regressor, config, X, single_y, X_test, single_y_test)}")
for epoch in range(num_epochs):
    for batch in dataloader:
        _train_epoch(regressor, optimizer, batch)
        print(f"Epoch: {epoch}, MAE: {evaluate_model(regressor, config, X, single_y, X_test, single_y_test)}")




