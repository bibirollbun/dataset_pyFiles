import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier

import pickle
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


with open("fertilizer_data_splits.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
X_val = data["X_val"]
y_train = data["y_train"]
y_val = data["y_val"]
X_test = data["X_test"]

X_train_nocategorical = data["X_train_nocat"]
X_val_nocategorical = data["X_val_nocat"]
X_test_nocategorical = data["X_test_nocat"]

target_encoder = data["target_encoder"]
target_label_mapping = data["target_label_mapping"]

print("âœ… All data splits loaded (including _nocategorical)")


def grid_search_model(model, param_grid, model_name):
    grid = HalvingGridSearchCV(
        model, param_grid,
        scoring='neg_log_loss',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"ğŸ”� Best params for {model_name}:", grid.best_params_)
    print(f"âœ… Best CV Log Loss for {model_name}: {grid.best_score_:.4f}")
    return grid.best_estimator_


rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}


lgbm_params = {
    'n_estimators': [100, 300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [-1, 10, 20, 30],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}


xgb_params = {
    'n_estimators': [100, 300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'min_child_weight': [1, 5, 10],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0]
}


rf_best = grid_search_model(RandomForestClassifier(random_state=888), rf_params, "Random Forest")


rf_best


lgbm_params = {
    'n_estimators': 500,                 # 100 is often too small, 1000 adds training time
    'learning_rate': 0.05,               # 0.05â€“0.1 is the sweet spot for boosted trees
    'max_depth': 20,                      # -1 = unlimited; too risky, so removed
    'num_leaves': 63,                     # Smaller values for less overfitting
    'min_child_samples': 10,                  # Default 20 is often fine, 10 gives flexibility
    'subsample': 0.8,                         # Fix at 0.8 (widely optimal)
    'colsample_bytree': 0.8,                  # Fix at 0.8 too
    'reg_alpha': 0,                           # Drop regularization unless overfitting is seen
    'reg_lambda': 0.5                         # Keep light regularization
}


xgb_params = {
    'n_estimators': 600,                 # 100 is too shallow, 1000 slows down a lot
    'learning_rate': 0.1,
    'max_depth': 10,                       # 6 is default, 10 for more complexity
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,                             # Helps reduce overfitting, default is usually fine
    'min_child_weight': 5,                    # 1 is risky for overfitting
    'reg_alpha': 0,
    'reg_lambda': 1.0,
    'eval_metric': 'mlogloss'
}


# Train LightGBM
lgbm_best = LGBMClassifier(random_state=888, **lgbm_params)
lgbm_best.fit(X_train, y_train)

# Train XGBoost
xgb_best = XGBClassifier(random_state=888, **xgb_params)
xgb_best.fit(X_train, y_train)


ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_best),
        ('lgbm', lgbm_best),
        ('xgb', xgb_best)
    ],
    voting='soft',
    n_jobs=-1
)
ensemble.fit(X_train, y_train)

# Evaluate on validation set
val_preds = ensemble.predict(X_val)
val_probs = ensemble.predict_proba(X_val)
val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]


def mapk(y_true, y_pred, k=3):
    def apk(actual, predicted, k):
        predicted = list(predicted)  # âœ… Convert to list
        if actual in predicted[:k]:
            return 1 / (predicted[:k].index(actual) + 1)
        return 0

    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])


print(f"âœ… Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")
print(f"ğŸ�¯ MAP@3: {mapk(y_val, val_top3, k=3):.4f}")


test_df = pd.read_csv('test.csv')


# Predict test set
test_probs = ensemble.predict_proba(X_test)
top3_test = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top3_labels = target_encoder.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)

# Create submission
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})

submission_df.to_csv("submission_champion_ensemble.csv", index=False)
print("ğŸ“� Saved: submission_champion_ensemble.csv")

