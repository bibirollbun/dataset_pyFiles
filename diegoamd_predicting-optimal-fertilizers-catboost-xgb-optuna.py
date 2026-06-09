import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import optuna


train_set = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
train_set.head()


train_set.info()


X = train_set.drop(columns = ["id", "Fertilizer Name"])
y = train_set["Fertilizer Name"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 19)


numerical_columns = X_train.select_dtypes(include = ["int", "float"]).columns
categorical_columns = X_train.select_dtypes(include = ["object"]).columns


ct = ColumnTransformer([
    ("scaler", StandardScaler(), numerical_columns),
    ("encoder", OneHotEncoder(handle_unknown = "ignore"), categorical_columns)
])


X_train_preprocessed = ct.fit_transform(X_train)
X_test_preprocessed = ct.transform(X_test)


le = LabelEncoder()


y_train_labeled = le.fit_transform(y_train)
y_test_labeled = le.transform(y_test)


def map_at_3(y_true, y_pred_top3):
    """
    Compute MAP@3 score for top-3 encoded label predictions.
    Scoring: 1 point for rank 1, 1/2 for rank 2, 1/3 for rank 3, 0 otherwise.
    Args:
        y_true: List or array of encoded true labels (e.g., [0, 1, ...])
        y_pred_top3: List or array of top-3 predicted labels (e.g., [[0, 1, 2], ...])
    Returns:
        float: Mean Average Precision at 3 (MAP@3)
    """
    scores = []
    for true_label, pred_top3 in zip(y_true, y_pred_top3):
        if len(pred_top3) != 3:
            raise ValueError(f"Prediction must contain exactly 3 labels, got: {pred_top3}")
        if pred_top3[0] == true_label:
            scores.append(1.0)
        elif pred_top3[1] == true_label:
            scores.append(0.5)
        elif pred_top3[2] == true_label:
            scores.append(1/3)
        else:
            scores.append(0.0)
    return np.mean(scores)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'task_type': 'GPU',
        'verbose': 0
    }

    kf = KFold(n_splits = 3, shuffle = True, random_state = 19)
    scores = []

    for train_idx, val_idx in kf.split(X_train_preprocessed):
        X_train_fold = X_train_preprocessed[train_idx]
        X_val_fold = X_train_preprocessed[val_idx]
        y_train_fold = y_train_labeled[train_idx]
        y_val_fold = y_train_labeled[val_idx]

        # Train model
        model = CatBoostClassifier(**params, cat_features=[])
        model.fit(X_train_fold, y_train_fold)

        # Get top-3 predictions
        probs = model.predict_proba(X_val_fold)
        y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in probs]
        fold_score = map_at_3(y_val_fold, y_pred_top3)
        scores.append(fold_score)

    return np.mean(scores)


# Run optimization
study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 50)


# Print best parameters
print("Best parameters:", study.best_params)
print("Best MAP@3 score:", study.best_value)

# Train final model
best_params = study.best_params
best_params["verbose"] = 0
best_params["task_type"] = "GPU"

opt_cat = CatBoostClassifier(**best_params)
opt_cat.fit(X_train_preprocessed, y_train_labeled)


# Save model
opt_cat.save_model("catboost_best_map_at_3.cbm")


# Get Test MAP@3
probs = opt_cat.predict_proba(X_test_preprocessed)
y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in probs]
test_score = map_at_3(y_test_labeled, y_pred_top3)
print("Test MAP@3 score:", test_score)


test_set = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test_set.head()


test_features = test_set.drop(columns = ["id"])


test_features_preprocessed = ct.transform(test_features)


probs = opt_cat.predict_proba(test_features_preprocessed)
y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in probs]
y_pred_strings = [' '.join([le.inverse_transform([idx])[0] for idx in top3]) for top3 in y_pred_top3]


submission = pd.DataFrame({"id": test_set["id"], "Fertilizer Name": y_pred_strings})
submission.head()


submission.to_csv("sub2.csv", index = False)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'tree_method': 'gpu_hist',  # GPU support
        'verbosity': 0
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=19)
    scores = []

    for train_idx, val_idx in kf.split(X_train_preprocessed):
        X_train_fold = X_train_preprocessed[train_idx]
        X_val_fold = X_train_preprocessed[val_idx]
        y_train_fold = y_train_labeled[train_idx]
        y_val_fold = y_train_labeled[val_idx]

        # Train model
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train_fold, y_train_fold)

        # Get top-3 predictions
        probs = model.predict_proba(X_val_fold)
        y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in probs]
        fold_score = map_at_3(y_val_fold, y_pred_top3)
        scores.append(fold_score)

    return np.mean(scores)


# Run optimization
study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 50)


# Print best parameters
print("Best parameters:", study.best_params)
print("Best MAP@3 score:", study.best_value)

# Train final model
best_params = study.best_params
best_params["tree_method"] = "gpu_hist"
best_params["verbosity"] = 0
best_params["use_label_encoder"] = False
best_params["eval_metric"] = "mlogloss"

opt_xgb = XGBClassifier(**best_params)
opt_xgb.fit(X_train_preprocessed, y_train_labeled)


# Save model
opt_xgb.save_model("xgboost_best_map_at_3.json")


# Get Test MAP@3
probs = opt_xgb.predict_proba(X_test_preprocessed)
y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in probs]
test_score = map_at_3(y_test_labeled, y_pred_top3)
print("Test MAP@3 score:", test_score)


probs = opt_cat.predict_proba(test_features_preprocessed)
y_pred_top3 = [np.argsort(prob_row)[::-1][:3] for prob_row in probs]
y_pred_strings = [' '.join([le.inverse_transform([idx])[0] for idx in top3]) for top3 in y_pred_top3]


submission = pd.DataFrame({"id": test_set["id"], "Fertilizer Name": y_pred_strings})
submission.head()


submission.to_csv("sub3.csv", index = False)




