import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

def add_subgrade_feature(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the numeric subgrade value from the alphanumeric string column."""

    train = train.copy()
    test = test.copy()
    train['subgrade'] = train['grade_subgrade'].str[1:].astype(int)
    test['subgrade'] = test['grade_subgrade'].str[1:].astype(int)
    return train, test

def create_frequency_features(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    columns: list[str],
    numeric_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create frequency features for all columns in the dataframe
    - Categorical column: frequency = how often the value appears in the column
    - Numeric column: frequency = the bin in which the value falls into (3 types of bins: 5, 10, 15 bins)
    """
    numeric_set = set(numeric_columns)

    for col in columns:
        if col not in df.columns or col not in df_test.columns:
            continue
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        fill_value = freq.mean() if len(freq) else 0.0
        df_test[f"{col}_freq"] = df_test[col].map(freq).fillna(fill_value)

        # --- Quantile binning for numeric columns ---
        if col in numeric_set:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    df[f"{col}_bin{q}"] = train_bins
                    df_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    df[f"{col}_bin{q}"] = 0
                    df_test[f"{col}_bin{q}"] = 0

    return df, df_test


def target_encoding(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    n_splits: int = 7,
    smooth: float | str = 'auto',
    fallback_smooth: float = 10.0, # fallback value for smoothing
    SEED: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add K-Fold target mean encoded features to train and predict datasets.
    
    Parameters:
    - train: training DataFrame
    - predict: prediction/test DataFrame
    - target: name of the target column
    - n_splits: number of folds for K-Fold encoding
    - smooth : 'auto' | float -> use empirical Bayes smooth factor m = variance_within / variance_between
        - 'auto' computes an empirical Bayes smooth factor m = variance_within / variance_between
        - float uses a fixed smoothing value
        - fallback_smooth : Used when auto computation is unstable (e.g., variance_between == 0).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    mean_features_train = pd.DataFrame(index=train.index)
    mean_features_test = pd.DataFrame(index=predict.index)

    def _smooth_stats(stat_df: pd.DataFrame, global_mean: float) -> pd.Series:
        stats = stat_df.copy()
        stats['var'] = stats['var'].fillna(0.0)

        if smooth == 'auto':
            variance_between = stats['mean'].var(ddof=0)
            avg_variance_within = stats['var'].mean()
            if pd.isna(avg_variance_within):
                avg_variance_within = 0.0
            if pd.isna(variance_between) or variance_between <= 0:
                m_value = fallback_smooth
            else:
                m_value = avg_variance_within / variance_between if variance_between > 0 else fallback_smooth
                if np.isnan(m_value) or m_value < 0:
                    m_value = fallback_smooth
        else:
            m_value = smooth

        if m_value == 0:
            stats['smooth_mean'] = stats['mean']
        else:
            stats['smooth_mean'] = (
                stats['count'] * stats['mean'] + m_value * global_mean
            ) / (stats['count'] + m_value)

        return stats['smooth_mean']

    for col in feature_columns:
        if col not in train.columns or col not in predict.columns:
            continue
        # --- K-Fold Target Mean Encoding ---
        mean_encoded = np.zeros(len(train))
        for train_idx, val_idx in kf.split(train):
            train_fold = train.iloc[train_idx]
            val_fold = train.iloc[val_idx]
            global_mean_fold = train_fold[target_col].mean()

            stats = train_fold.groupby(col)[target_col].agg(['mean', 'count', 'var'])
            mean_map = _smooth_stats(stats, global_mean_fold)
            fold_encoded = val_fold[col].map(mean_map)
            mean_encoded[val_idx] = fold_encoded.fillna(global_mean_fold)

        mean_features_train[f'mean_{col}'] = mean_encoded

        # --- Apply global mean mapping to prediction/test data ---
        overall_mean = train[target_col].mean()
        full_stats = train.groupby(col)[target_col].agg(['mean', 'count', 'var'])
        global_mean = _smooth_stats(full_stats, overall_mean)
        mean_features_train[f'mean_{col}'] = mean_features_train[f'mean_{col}'].fillna(overall_mean)
        mean_features_test[f'mean_{col}'] = predict[col].map(global_mean).fillna(overall_mean)

    # --- Concatenate new features at once to avoid fragmentation ---
    train = pd.concat([train, mean_features_train], axis=1)
    predict = pd.concat([predict, mean_features_test], axis=1)

    # Defragment
    train = train.copy()
    predict = predict.copy()
    return train, predict

def run_xgb_training(
    features: pd.DataFrame,
    labels: pd.Series,
    test_features: pd.DataFrame,
    params: dict,
    folds: int = 7,
    SEED: int = 42
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """K-Fold training and return OOF predictions, test predictions, and best iterations."""

    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof_predictions = np.zeros(len(features))
    test_predictions = np.zeros(len(test_features))
    best_iterations = []

    dtest = xgb.DMatrix(test_features, enable_categorical=True)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(features)):
        print("#" * 50)
        print(f"### Fold {fold + 1}/{kf.n_splits} ###")
        print("#" * 50)

        X_train = features.iloc[train_idx].copy()
        y_train = labels.iloc[train_idx]
        X_valid = features.iloc[valid_idx].copy()
        y_valid = labels.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100_000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=200,
            verbose_eval=500,
        )

        fold_oof = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        oof_predictions[valid_idx] = fold_oof

        fold_test = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        test_predictions += fold_test / kf.n_splits
        best_iterations.append(model.best_iteration)

        fold_auc = roc_auc_score(y_valid, fold_oof)
        print(f"Fold {fold + 1} - Best iteration: {model.best_iteration} | Fold AUC: {fold_auc:.6f}\n")

    return oof_predictions, test_predictions, best_iterations


def main() -> None:
    train_path = "/kaggle/input/playground-series-s5e11/train.csv"
    test_path = "/kaggle/input/playground-series-s5e11/test.csv"
    FOLDS=7
    SEED=42

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target_col = train_df.columns.tolist()[-1]

    train_df, test_df = add_subgrade_feature(train_df, test_df)

    base_features = [col for col in train_df.columns if col not in [target_col, "id"]]
    categorical_cols = [c for c in base_features if train_df[c].dtype in ["object", "category"]]
    numeric_cols = [c for c in base_features if train_df[c].dtype not in ["object", "category", "bool"]]

    train_df, test_df = target_encoding(train_df, test_df, base_features, target_col, n_splits=FOLDS, SEED=SEED)
    train_df, test_df = create_frequency_features(train_df, test_df, base_features, numeric_cols)

    if categorical_cols:
        train_df[categorical_cols] = train_df[categorical_cols].astype("category")
        test_df[categorical_cols] = test_df[categorical_cols].astype("category")

    drop_candidates = [
        "education_level",
        "loan_purpose",
        "grade_subgrade",
        "interest_rate",
        "marital_status",
        "gender",
        "employment_status_freq",
        "credit_score_bin5",
        "loan_amount_bin5",
        "debt_to_income_ratio_bin5",
    ]
    train_df = train_df.drop(columns=[col for col in drop_candidates if col in train_df.columns])
    test_df = test_df.drop(columns=[col for col in drop_candidates if col in test_df.columns])

    train_df = train_df.reset_index(drop=True)
    train_ids = train_df['id'].reset_index(drop=True)

    duplicate_mask = ~train_df.drop(columns=['id']).duplicated()
    train_df = train_df.loc[duplicate_mask].reset_index(drop=True)
    train_ids = train_ids.loc[duplicate_mask].reset_index(drop=True)

    train_df = train_df.drop(columns="id")

    feature_columns = [col for col in train_df.columns if col != target_col]
    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_col].reset_index(drop=True)
    X_test = test_df[feature_columns].copy()

    for col in feature_columns:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    xgb_params = {
        'device': 'cuda',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.01,
        'random_state': SEED,
        'min_child_weight': 80, 
        'max_leaves': 6,
        'reg_alpha': 1.4,
        'reg_lambda': 5.9,
    }

    oof_preds, test_preds, best_iterations = run_xgb_training(
        features=X_train,
        labels=y_train,
        test_features=X_test,
        params=xgb_params,
        folds=FOLDS,
        SEED=SEED,
    )

    oof_binary = (oof_preds >= 0.5).astype(int)
    cv_auc = roc_auc_score(y_train, oof_preds)
    cv_accuracy = accuracy_score(y_train, oof_binary)

    print("=" * 50)
    print(f"OOF AUC: {cv_auc:.6f} | OOF Accuracy: {cv_accuracy:.6f}")
    print(f"Average best iteration: {np.mean(best_iterations):.2f}")
    print("=" * 50)

    if not os.path.exists('xgb'):
        os.makedirs('xgb')

    submission = pd.DataFrame({
        'id': test_df['id'],
        'loan_paid_back': test_preds,
    })
    submission.to_csv('xgb/submission_xgb.csv', index=False)

    oof_df = pd.DataFrame({
        'id': train_ids,
        'loan_paid_back_oof': oof_preds,
    })
    oof_df.to_csv('xgb/oof_xgb.csv', index=False)

    print("Submission saved to: xgb/submission_xgb.csv")
    print("OOF predictions saved to: xgb/oof_xgb.csv")


if __name__ == "__main__":
    main()





