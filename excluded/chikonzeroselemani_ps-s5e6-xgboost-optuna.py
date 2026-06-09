# !pip install optuna-integration[xgboost]
# from optuna.integration import XGBoostPruningCallback
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings("ignore")


test_path = "/kaggle/input/playground-series-s5e6/test.csv"
train_path = "/kaggle/input/playground-series-s5e6/train.csv"
external_data_path = "/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv"

train = pd.read_csv(train_path, index_col="id")
test = pd.read_csv(test_path, index_col="id")
external = pd.read_csv(external_data_path)

overall_train = pd.concat([train,external], ignore_index=True)
overall_train = overall_train.drop_duplicates()

overall_train


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

x = overall_train.drop(columns=["Fertilizer Name"])
y = overall_train["Fertilizer Name"]

encoder = LabelEncoder()
targetEncoder = LabelEncoder()

categorical = x.select_dtypes(include=['object']).columns

y = targetEncoder.fit_transform(y)
for cat in categorical:
    x[cat] = encoder.fit_transform(x[cat])
    test[cat] = encoder.transform(test[cat])



x


# Compute mutual information between each feature and the target
mi_scores = mutual_info_classif(x, y, discrete_features='auto', random_state=42)
    
    # Create a DataFrame of features and their MI scores
mi_df = pd.DataFrame({
        "feature": x.columns,
        "mi_score": mi_scores
    }).sort_values(by="mi_score", ascending=False).reset_index(drop=True)
    
    # Display the MI scores
display(mi_df)
    
    # Plot MI scores as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(mi_df["feature"], mi_df["mi_score"], color='skyblue')
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Mutual Information Score")
plt.title("Mutual Information Between Features and Target")
plt.tight_layout()
plt.show()



def map_at_3(y_true, y_pred_proba, k=3):

    map_score = 0.0
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true  # Convert Series to NumPy array
    for i in range(len(y_true)):
        top_k_preds = np.argsort(y_pred_proba[i])[-k:][::-1]  # Get top k predictions
        if y_true[i] in top_k_preds:
            rank = np.where(top_k_preds == y_true[i])[0][0] + 1
            map_score += 1.0 / rank
    return map_score / len(y_true)


# Split data with stratification
X_train, X_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",  # Faster and memory-efficient
        "enable_categorical": False,
        "early_stopping_rounds": None,
        "device" :"cuda"
    }
    pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
    
    model = XGBClassifier(**params)
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=0,
        callbacks=[pruning_callback]
    )

    # Get best iteration from early stopping
    best_iter = model.best_iteration
    if best_iter is None:
        best_iter = params["n_estimators"]
    
    # Predict using best model
    preds = model.predict_proba(X_val, iteration_range=(0, best_iter))
    return map_at_3(y_val, preds)

# # Create study with TPE sampler and pruning
# study = optuna.create_study(
#     direction="maximize",
#     sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
#     pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
# )
# study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour timeout

# print("Best trial:")
# trial = study.best_trial
# print(f"  Value (accuracy): {trial.value:.5f}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")



# Cross-validation# Prepare data containers
n_classes=7
oof_proba = np.zeros((len(x), n_classes))
test_preds = np.zeros((len(test), n_classes))
fold_counter = np.zeros(len(x))

best_params = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'max_depth': 16,
    'learning_rate': 0.02,
    'n_estimators': 10_000,
    'reg_alpha': 3,
    'reg_lambda': 1.4,
    'gamma': 0.26,
    'max_delta_step': 5,
    'subsample': 0.86,
    'colsample_bytree': 0.4,
    'min_child_weight': 5,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
    'enable_categorical': True,
    'device': "cuda"
}


n_splits = 7
n_repeats = 2
seed = 42

cv = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=seed
)

print(f"Starting {n_splits}-fold CV with {n_repeats} repeats")
fold_scores = []

for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y)):
    print(f"\n=== Fold {fold+1}/{n_splits * n_repeats} ===")
    
    X_train, X_valid = x.iloc[train_idx], x.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    # Initialize model
    model = XGBClassifier(**best_params)
    model.fit(
        X_train, 
        y_train,
        early_stopping_rounds=50,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    
    # Get validation predictions
    valid_proba = model.predict_proba(X_valid)
    oof_proba[valid_idx] += valid_proba
    fold_counter[valid_idx] += 1
    
    # Calculate MAP@3 for this fold
    fold_map = map_at_3(y_valid, valid_proba)
    fold_scores.append(fold_map)
    print(f"Fold {fold+1} MAP@3: {fold_map:.5f}")
    
    # Predict test set
    test_preds += model.predict_proba(test) / (n_splits * n_repeats)



# Finalize OOF predictions
oof_proba /= fold_counter[:, np.newaxis]

# Calculate overall metrics
cv_map = np.mean(fold_scores)
oof_logloss = log_loss(y, oof_proba)

print("\n" + "=" * 50)
print(f"CV MAP@3: {cv_map:.5f}")
print(f"OOF Log Loss: {oof_logloss:.5f}")
print(f"Individual Fold MAP@3 Scores: {[f'{s:.5f}' for s in fold_scores]}")

# Generate submission
top3_indices = np.argsort(-test_preds, axis=1)[:, :3]  # Get top 3 predictions
top3_labels = targetEncoder.inverse_transform(top3_indices.ravel())
top3_labels = top3_labels.reshape(len(test), 3)

id = pd.read_csv(test_path)["id"]

submission = pd.DataFrame({
    "id": id,
    "Fertilizer Name": [" ".join(row) for row in top3_labels]
})
submission.to_csv("submission.csv", index=False)
print("\nâœ” Submission file created: submission.csv")
submission.head()

