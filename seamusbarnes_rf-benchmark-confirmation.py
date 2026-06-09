import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, log_loss

import xgboost as xgb


PATH_TO_TRAIN = "/kaggle/input/playground-series-s5e7/train.csv"
PATH_TO_TEST = "/kaggle/input/playground-series-s5e7/test.csv"
PATH_TO_SS = "/kaggle/input/playground-series-s5e7/sample_submission.csv"

df_train = pd.read_csv(PATH_TO_TRAIN)
df_test = pd.read_csv(PATH_TO_TEST)
df_ss = pd.read_csv(PATH_TO_SS)


# X = df_train.drop(["id", "Personality"], axis=1)
# y = df_train["Personality"]

# print("Feature types:")
# print(X.dtypes)

# print("\nTarget type:")
# print(y.dtypes)


def encode_cat_cols(
    df_train,
    df_test,
    target_col="Personality",
    id_col="id",
    encoder_type="ordinal"
):
    # Split features and target
    X = df_train.drop([id_col, target_col], axis=1).copy()
    y = df_train[target_col].copy()
    X_test = df_test.drop([id_col], axis=1).copy()
    
    # Identify categorical columns
    cat_cols = [col for col in X.columns if X[col].dtype == "object"]
    
    if encoder_type == "label":
        # Use LabelEncoder for each column
        for col in cat_cols:
            le = LabelEncoder()
            # Fit on both train and test to avoid unknown labels at predict time
            le.fit(list(X[col].astype(str)) + list(X_test[col].astype(str)))
            X[col] = le.transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    elif encoder_type == "ordinal":
        # Use OrdinalEncoder for all columns together
        combined = pd.concat([X, X_test], ignore_index=True)
        encoder = OrdinalEncoder()
        combined[cat_cols] = encoder.fit_transform(combined[cat_cols].astype(str))
        # Split back
        X = combined.iloc[:len(X)].reset_index(drop=True)
        X_test = combined.iloc[len(X):].reset_index(drop=True)
    else:
        raise ValueError("encoder_type must be 'label' or 'ordinal'")
    
    return X, y, X_test


# print(X.isnull().sum().sort_values(ascending=False).head(10))
# print(X_test.isnull().sum().sort_values(ascending=False).head(10))


def train_and_inference_random_forest(X, y, X_test, df_test, submission_filename="submission.csv", verbose=True):
    # Label encode target (if needed)
    target_enc = LabelEncoder()
    y_encoded = target_enc.fit_transform(y)
    
    # Impute missing values, if any
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_imputed, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=200, random_state=None, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_val_pred = rf.predict(X_val)
    
    if verbose:
        print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
        print(classification_report(y_val, y_val_pred, target_names=target_enc.classes_))
    
    # Retrain and do test prediction
    rf.fit(X_imputed, y_encoded)
    test_preds = rf.predict(X_test_imputed)
    test_preds_labels = target_enc.inverse_transform(test_preds)
    
    submission = df_test[['id']].copy()
    submission['Personality'] = test_preds_labels
    submission.to_csv(submission_filename, index=False)

    return submission

# X, y, X_test = encode_cat_cols(df_train, df_test, encoder_type="label")
# submission = train_and_inference_random_forest(X, y, X_test, df_test)
# submission.head()


def train_and_inference_xgboost(X, y, X_test, df_test, submission_filename="submission.csv", verbose=True):
    """
    Train XGBoostClassifier, print validation results, produce submission using encoded X, y, X_test.
    
    Args:
        X: pd.DataFrame (features, already encoded)
        y: pd.Series or np.ndarray (target labels, **not yet label-encoded**)
        X_test: pd.DataFrame (test features, already encoded)
        df_test: pd.DataFrame (true test, used for 'id' column in submission)
        submission_filename: str
        verbose: bool
    
    Returns:
        pd.DataFrame: submission DataFrame
    """
    # Label encode the target
    target_enc = LabelEncoder()
    y_encoded = target_enc.fit_transform(y)

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X_imputed, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    clf_xgb = xgb.XGBClassifier(
        n_estimators=500,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    clf_xgb.fit(X_train, y_train)

    y_val_pred = clf_xgb.predict(X_val)
    if verbose:
        print("XGBoost Validation Accuracy:", accuracy_score(y_val, y_val_pred))
        print(classification_report(y_val, y_val_pred, target_names=target_enc.classes_))
    
    # Retrain on all data
    clf_xgb.fit(X_imputed, y_encoded)
    test_preds_xgb = clf_xgb.predict(X_test_imputed)
    test_preds_labels_xgb = target_enc.inverse_transform(test_preds_xgb)
    
    # Submission
    submission_xgb = df_test[['id']].copy()
    submission_xgb['Personality'] = test_preds_labels_xgb
    submission_xgb.to_csv(submission_filename, index=False)

    return submission_xgb

# X, y, X_test = encode_cat_cols(df_train, df_test, encoder_type="ordinal")
# submission = train_and_inference_xgboost(X, y, X_test, df_test, submission_filename="submission.csv", verbose=True)
# submission.head()


def ensemble_xgboost_cv(X, y, X_test, df_test, submission_filename='submission.csv',
                        N_SEEDS=3, N_FOLDS=3, verbose=True, plot_curve=True):
    """
    Train ensemble of XGBoost classifiers with cross-validation and ensembling.

    X, y, X_test: encoded features/labels.
    df_test: for 'id' column in submission.
    """
    import matplotlib.pyplot as plt

    # Label encode the target
    target_enc = LabelEncoder()
    y_encoded = target_enc.fit_transform(y)

    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    n_classes = len(np.unique(y_encoded))
    test_preds_sum = np.zeros((X_test_imputed.shape[0], n_classes))
    val_preds_sum = np.zeros((X_imputed.shape[0], n_classes))

    oof_accs = []            # For tracking OOF accuracy as models are added
    oof_loglosses = []
    models_trained = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    model_count = 0

    for seed in range(N_SEEDS):
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_imputed, y_encoded)):
            model_count += 1

            ### PARAMETERS 01 ###
            # clf_xgb = xgb.XGBClassifier(
            #     n_estimators=1000,
            #     max_depth=4,
            #     learning_rate=0.1,
            #     colsample_bytree=0.8,
            #     subsample=0.8,
            #     use_label_encoder=False,
            #     eval_metric="logloss",
            #     objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            #     random_state=seed * 1000 + fold,
            #     n_jobs=-1,
            #     early_stopping_rounds=10,
            # )
            ### ------------- ###
            
            ### PARAMETERS 02 ###
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_leaves': 25,
                'min_child_weight': np.float64(0.003440906647223279),
                'learning_rate': np.float64(0.09470087254583547),
                'n_estimators': 10000,
                'subsample': np.float64(0.8025291728808135),
                'colsample_bylevel': np.float64(0.8360122952647302),
                'colsample_bytree': np.float64(0.87329448975438),
                'reg_alpha': np.float64(0.002926163798802797),
                'reg_lambda': np.float64(27.126259438996986),
                'random_state': 42,
                'tree_method': 'hist',
                'early_stopping_rounds': 50
            }
            clf_xgb = xgb.XGBClassifier(**xgb_params)
            ### ------------- ###
            
            clf_xgb.fit(
                X_imputed.iloc[train_idx], y_encoded[train_idx],
                eval_set=[(X_imputed.iloc[val_idx], y_encoded[val_idx])],
                verbose=False,
            )
            val_pred_proba = clf_xgb.predict_proba(X_imputed.iloc[val_idx])
            test_pred_proba = clf_xgb.predict_proba(X_test_imputed)
            
            val_preds_sum[val_idx] += val_pred_proba
            test_preds_sum += test_pred_proba

            # Current average OOF predictions so far
            val_preds_avg_current = val_preds_sum / model_count
            val_preds_labels_current = np.argmax(val_preds_avg_current, axis=1)
            oof_acc = accuracy_score(y_encoded, val_preds_labels_current)
            oof_accs.append(oof_acc)
            models_trained.append(model_count)
            logl = log_loss(y_encoded, val_preds_avg_current)
            oof_loglosses.append(logl)
            if verbose:
                print(f"After model {model_count} (seed={seed}, fold={fold}): "
                      f"Running OOF accuracy = {oof_acc:.5f}, logloss={logl:.5f}")

    # Final averaging after all models
    val_preds_avg = val_preds_sum / model_count
    val_preds_labels = np.argmax(val_preds_avg, axis=1)
    print("\n== FINAL ENSEMBLE ==")
    print("Total models trained:", model_count)
    print("Out-of-fold XGB kfold+seed ensemble accuracy: {:.5f}".format(
        accuracy_score(y_encoded, val_preds_labels)))
    print(classification_report(y_encoded, val_preds_labels, target_names=target_enc.classes_))

    # Test set: average and decode
    test_preds_avg = test_preds_sum / model_count
    test_preds_labels = target_enc.inverse_transform(np.argmax(test_preds_avg, axis=1))
    submission = df_test[['id']].copy()
    submission['Personality'] = test_preds_labels
    submission.to_csv(submission_filename, index=False)

    # Visualization: OOF accuracy and log loss as models are added
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(models_trained, oof_accs, marker='o')
    plt.grid()
    plt.title("OOF Ensemble Accuracy")
    plt.xlabel("Ensembled models")
    plt.ylabel("Validation accuracy")

    plt.subplot(1,2,2)
    plt.plot(models_trained, oof_loglosses, marker='o', color='tomato')
    plt.grid()
    plt.title("OOF Ensemble Log Loss")
    plt.xlabel("Ensembled models")
    plt.ylabel("Log Loss")

    plt.tight_layout()
    plt.show()

    return submission

X, y, X_test = encode_cat_cols(df_train, df_test, encoder_type="ordinal")
submission = ensemble_xgboost_cv(X, y, X_test, df_test, N_SEEDS=5, N_FOLDS=5, submission_filename='submission.csv')

submission.head()

