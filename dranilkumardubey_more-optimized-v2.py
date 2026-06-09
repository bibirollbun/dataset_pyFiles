pip install optuna-integration[xgboost]


# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
import optuna
from optuna.integration import XGBoostPruningCallback
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


# Step 2: Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train.head()


test.head()


# Step 3: Label Encode categorical features
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


# Step 4: Feature Interaction Engineering
train['Soil_Crop_Interaction'] = train['Soil Type'] * train['Crop Type']
test['Soil_Crop_Interaction'] = test['Soil Type'] * test['Crop Type']


# Step 5: Define target and features
target = 'Fertilizer Name'
features = [col for col in train.columns if col not in ['ID', target]]
X = train[features]
y = train[target]
X_test = test[features]

# Encode target for classification
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
num_classes = len(le_target.classes_)


# Step 6: Define Optuna tuning for XGBoost
def optuna_objective(trial):
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": num_classes,
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    model = XGBClassifier(**params, use_label_encoder=False)
    scores = cross_val_score(model, X, y_encoded, cv=3, scoring='accuracy')
    return np.mean(scores)


# Step 7: Run Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(optuna_objective, n_trials=20)

best_params = study.best_params
best_params.update({
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": num_classes,
    "tree_method": "hist",
    "use_label_encoder": False
})


# Step 8: Train model with Stratified K-Fold & multiple seeds
oof_preds = np.zeros((X.shape[0], num_classes))
test_preds = np.zeros((X_test.shape[0], num_classes))
seeds = [0, 42, 2025]

for seed in seeds:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train_idx, valid_idx in skf.split(X, y_encoded):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_encoded[train_idx], y_encoded[valid_idx]

        model = XGBClassifier(**best_params, random_state=seed)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds=20, verbose=False)

        oof_preds[valid_idx] += model.predict_proba(X_valid) / len(seeds)
        test_preds += model.predict_proba(X_test) / (len(seeds) * 10)


# Step 9: Final Predictions
final_oof = np.argmax(oof_preds, axis=1)
final_test = np.argmax(test_preds, axis=1)

# Step 10: Evaluation
accuracy = accuracy_score(y_encoded, final_oof)
recall = recall_score(y_encoded, final_oof, average='weighted')
precision = precision_score(y_encoded, final_oof, average='weighted')
f1 = f1_score(y_encoded, final_oof, average='weighted')


# Step 11: Show result table
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Recall", "Precision", "F1 Score"],
    "Score": [accuracy, recall, precision, f1]
})
print("Evaluation Results:")
display(metrics_df)


# Step 12: Prepare Submission
submission["Fertilizer Name"] = le_target.inverse_transform(final_test)
submission.to_csv("submission.csv", index=False)




