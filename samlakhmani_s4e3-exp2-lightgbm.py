# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train = pd.read_csv("/kaggle/input/playground-series-s4e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e3/test.csv")


def feature_engineering(data):

    data['Ratio_Length_Thickness'] = data['Length_of_Conveyer'] / data['Steel_Plate_Thickness']
    data['Normalized_Steel_Thickness'] = (data['Steel_Plate_Thickness'] -data['Steel_Plate_Thickness'].min()) / (data['Steel_Plate_Thickness'].max() - data['Steel_Plate_Thickness'].min())
    data['X_Range*Pixels_Areas'] = (data['X_Maximum'] - data['X_Minimum']) * data['Pixels_Areas']

    # features_to_drop = ['Y_Minimum', 'Steel_Plate_Thickness', 'Sum_of_Luminosity', 'Edges_X_Index', 'SigmoidOfAreas', 'Luminosity_Index', 'TypeOfSteel_A300']
    # data = data.drop(features_to_drop,axis=1)

    return data


train = feature_engineering(train)
test = feature_engineering(test)


target_list = [
    'Pastry', 
    'Z_Scratch', 
    'K_Scatch', 
    'Stains',
    'Dirtiness', 
    'Bumps', 
    'Other_Faults'
]


import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import optuna.visualization as vis
from sklearn.utils.class_weight import compute_class_weight


# ----- Prepare data -----
# Assuming your dataframes `train`, `test`, and `target_list` are already defined
# Use original dataset (not scaled)
X = train.drop(columns=target_list + ['id'], errors='ignore')
y = train[target_list].values  # shape: [n_samples, n_labels]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_labels = y_train.shape[1]
print(f"Number of labels: {n_labels}")


# ----- Compute class weights -----
class_weights = []
for i in range(n_labels):
    classes = np.unique(y_train[:, i])
    if len(classes) == 1:
        # Avoid division by zero when all labels are the same
        weights = {classes[0]: 1.0}
    else:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train[:, i]
        )
        weights = {cls: w for cls, w in zip(classes, cw)}
    class_weights.append(weights)


# ----- Objective function -----
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    preds_all = np.zeros_like(y_val, dtype=float)

    # Train one binary LightGBM model per label
    for i in range(n_labels):
        # Apply class weights to the training data
        weights_train = np.array([class_weights[i][cls] for cls in y_train[:, i]])

        dtrain = lgb.Dataset(X_train, label=y_train[:, i], weight=weights_train)
        dval = lgb.Dataset(X_val, label=y_val[:, i], reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(0),  # silence logging
            ],
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        preds_all[:, i] = preds

    # Compute MICRO-average ROC-AUC
    auc_micro = roc_auc_score(y_val, preds_all, average="micro")

    return auc_micro


# ----- Run Optuna study -----
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)



# ----- Print best trial -----
print("\nBest Trial:")
best_trial = study.best_trial
print(f"ROC-AUC (micro): {best_trial.value:.4f}")
print("Best hyperparameters:")
for k, v in best_trial.params.items():
    print(f"  {k}: {v}")

# ----- Visualization -----
vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
vis.plot_slice(study).show()



study.best_trial.params


# ----- Load test data -----
X_test = test.drop(columns=['id'], errors='ignore')  # drop id if present


# --- 1ï¸�âƒ£ Combine train + validation for final training ---
X_full = pd.concat([X_train, X_val], axis=0)
y_full = np.vstack([y_train, y_val])

n_labels = y_full.shape[1]

# --- 2ï¸�âƒ£ Extract best parameters from Optuna ---
best_params = study.best_trial.params.copy()
# Add static parameters that were not tuned
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_estimators": 500,
})


# --- 3ï¸�âƒ£ Train final LightGBM models for each label ---
final_models = []
preds_all = np.zeros((X_test.shape[0], 7), dtype=float)

for i in range(n_labels):
    print(f"\nğŸ”¹ Training model for label {i+1}/{n_labels} ...")

    # Compute final class weights for the full dataset
    classes = np.unique(y_full[:, i])
    if len(classes) == 1:
        weights_full = np.ones_like(y_full[:, i], dtype=float)
    else:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_full[:, i]
        )
        weights_full = np.array([dict(zip(classes, cw))[cls] for cls in y_full[:, i]])

    dtrain_full = lgb.Dataset(X_full, label=y_full[:, i], weight=weights_full)
    dval = lgb.Dataset(X_val, label=y_val[:, i])

    model = lgb.train(
        best_params,
        dtrain_full,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    final_models.append(model)
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    preds_all[:, i] = preds


test 


preds_all


len(preds_all), len(preds_all[0])


test[target_list] = preds_all


test[['id']+target_list].to_csv('submission.csv',index=False)

