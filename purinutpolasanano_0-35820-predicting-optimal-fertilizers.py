!pip install optuna-integration[xgboost]


import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


# Data Loading and Preparation
def load_and_prepare_data():
    """Load and prepare the training and test datasets."""
    train_path = "/kaggle/input/playground-series-s5e6/train.csv"
    test_path = "/kaggle/input/playground-series-s5e6/test.csv"
    external_data_path = "/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv"
    
    # Load datasets
    train = pd.read_csv(train_path, index_col="id")
    test = pd.read_csv(test_path, index_col="id")
    external = pd.read_csv(external_data_path)
    
    # Combine train and external data, remove duplicates
    overall_train = pd.concat([train, external], ignore_index=True)
    overall_train = overall_train.drop_duplicates()
    
    return overall_train, test


# Feature Engineering
def encode_features(train_df, test_df):
    """Encode categorical features using LabelEncoder."""
    encoder = LabelEncoder()
    target_encoder = LabelEncoder()
    
    x = train_df.drop(columns=["Fertilizer Name"])
    y = train_df["Fertilizer Name"]
    
    # Encode target variable
    y_encoded = target_encoder.fit_transform(y)
    
    # Encode categorical features
    categorical = x.select_dtypes(include=['object']).columns
    for cat in categorical:
        x[cat] = encoder.fit_transform(x[cat])
        test_df[cat] = encoder.transform(test_df[cat])
    
    return x, y_encoded, test_df, target_encoder


# Feature Importance Analysis
def analyze_feature_importance(X, y):
    """Analyze feature importance using mutual information."""
    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=SEED)
    
    # Create a DataFrame of features and their MI scores
    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi_scores
    }).sort_values(by="mi_score", ascending=False).reset_index(drop=True)
    
    # Plot MI scores
    plt.figure(figsize=(12, 6))
    plt.bar(mi_df["feature"], mi_df["mi_score"], color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Mutual Information Score")
    plt.title("Mutual Information Between Features and Target")
    plt.tight_layout()
    plt.show()
    
    return mi_df


# Evaluation Metric
def map_at_3(y_true, y_pred_proba, k=3):
    """Compute Mean Average Precision at 3 (MAP@3)."""
    map_score = 0.0
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    
    for i in range(len(y_true)):
        top_k_preds = np.argsort(y_pred_proba[i])[-k:][::-1]
        if y_true[i] in top_k_preds:
            rank = np.where(top_k_preds == y_true[i])[0][0] + 1
            map_score += 1.0 / rank
    
    return map_score / len(y_true)


# Hyperparameter Optimization
def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100):
    """Optimize XGBoost hyperparameters using Optuna."""
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
            "random_state": SEED,
            "n_jobs": -1,
            "tree_method": "hist",
            "enable_categorical": False,
            "early_stopping_rounds": None,
            "device": "cuda"
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

        best_iter = model.best_iteration if model.best_iteration else params["n_estimators"]
        preds = model.predict_proba(X_val, iteration_range=(0, best_iter))
        return map_at_3(y_val, preds)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=3600)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (MAP@3): {trial.value:.5f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params


# Cross-Validation and Model Training
def cross_validate_and_predict(X, y, test_df, best_params, n_splits=10, n_repeats=2):
    """Perform repeated stratified K-fold cross-validation."""
    n_classes = len(np.unique(y))
    oof_proba = np.zeros((len(X), n_classes))
    test_preds = np.zeros((len(test_df), n_classes))
    fold_counter = np.zeros(len(X))
    
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=SEED
    )
    
    print(f"Starting {n_splits}-fold CV with {n_repeats} repeats")
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"\n=== Fold {fold+1}/{n_splits * n_repeats} ===")
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        model = XGBClassifier(**best_params)
        model.fit(
            X_train, 
            y_train,
            early_stopping_rounds=50,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        
        valid_proba = model.predict_proba(X_valid)
        oof_proba[valid_idx] += valid_proba
        fold_counter[valid_idx] += 1
        
        fold_map = map_at_3(y_valid, valid_proba)
        fold_scores.append(fold_map)
        print(f"Fold {fold+1} MAP@3: {fold_map:.5f}")
        
        test_preds += model.predict_proba(test_df) / (n_splits * n_repeats)
    
    # Normalize OOF predictions by fold count
    # This fixes the bug where fold_counter was used outside the function
    oof_proba = oof_proba / fold_counter.reshape(-1, 1)
    
    return oof_proba, test_preds, fold_scores, fold_counter


# Create Submission
def create_submission(test_preds, test_df, target_encoder):
    """Create submission file with top 3 predictions."""
    top_3_preds = []
    for pred in test_preds:
        top_3 = np.argsort(pred)[-3:][::-1]
        top_3_labels = target_encoder.inverse_transform(top_3)
        top_3_preds.append(" ".join(top_3_labels))
    
    submission = pd.DataFrame({
        "id": test_df.index,
        "Fertilizer Name": top_3_preds
    })
    
    submission.to_csv("submission.csv", index=False)
    return submission


# Main Execution
def main():
    # Load and prepare data
    train_df, test_df = load_and_prepare_data()
    X, y, test_df, target_encoder = encode_features(train_df, test_df)
    
    # Analyze feature importance
    mi_df = analyze_feature_importance(X, y)
    
    # Split data for hyperparameter optimization
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Hyperparameter optimization (commented out to save time)
    # best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100)
    
    # Best parameters from previous optimization
    best_params = {
        'n_estimators': 1755,
        'learning_rate': 0.023978418824816997,
        'max_depth': 11,
        'min_child_weight': 0.2854006820895279,
        'subsample': 0.6019709318151113,
        'colsample_bytree': 0.6061190867138214,
        'gamma': 0.06717613642780118,
        'reg_alpha': 1.5722436436906744,
        'reg_lambda': 0.13933386527890473,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'random_state': SEED,
        "tree_method": "hist",
        "device": "cuda"
    }
    
    # Cross-validation and prediction
    oof_proba, test_preds, fold_scores, fold_counter = cross_validate_and_predict(
        X, y, test_df, best_params, n_splits=10, n_repeats=2
    )
    
    # Calculate metrics
    cv_map = np.mean(fold_scores)
    oof_loss = log_loss(y, oof_proba)
    
    print("\n==================================================")
    print(f"CV MAP@3: {cv_map:.5f}")
    print(f"OOF Log Loss: {oof_loss:.5f}")
    print(f"Individual Fold MAP@3 Scores: {[f'{score:.5f}' for score in fold_scores]}")
    
    # Create submission
    submission = create_submission(test_preds, test_df, target_encoder)
    print("\n✔ Submission file created: submission.csv")
    print("\nSubmission preview:")
    print(submission.head())

if __name__ == "__main__":
    main()

