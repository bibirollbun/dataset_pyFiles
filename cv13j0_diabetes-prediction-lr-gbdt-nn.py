!pip -q install -U "tensorflow[and-cuda]"


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


# Load the information into a pandas dataframe.
trn_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
tst_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sub_df = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
org_df = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')


# Display the first few rows of the dataset.
trn_df.head().T


# Display the first few rows of the dataset.
org_df.head().T


trn_df.describe().T


trn_df = trn_df.drop(columns=['id'])
tst_df = tst_df.drop(columns=['id'])
trn_df = pd.concat([trn_df, org_df[trn_df.columns]], axis=0, ignore_index=True)


trn_df.isnull().sum()


tst_df.isnull().sum()


categorical = ['gender', 'ethnicity', 'education_level', 'income_level','smoking_status', 'employment_status']
numerical = [feat for feat in trn_df.columns if feat not in categorical]
target = ['diagnosed_diabetes']
numerical = [feat for feat in numerical if feat not in target]


from sklearn.preprocessing import StandardScaler

def scale_numerical_features(train_df, test_df, num_cols):
    """
    Scales numerical features using StandardScaler.
    Fits on Train, Transforms both Train and Test.
    """
    scaler = StandardScaler()
    
    # Create copies to avoid SettingWithCopy warnings
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    # 1. Fit on TRAIN data only
    scaler.fit(train_df[num_cols])
    
    # 2. Transform both
    train_scaled[num_cols] = scaler.transform(train_df[num_cols])
    test_scaled[num_cols] = scaler.transform(test_df[num_cols])
    
    return train_scaled, test_scaled

# --- Example Usage ---
# num_cols = ['Glucose', 'BloodPressure', 'BMI', 'Age']
# train_scaled, test_scaled = scale_numerical_features(train, test, num_cols)


train_scaled, test_scaled = scale_numerical_features(trn_df, tst_df, num_cols = numerical)


import pandas as pd

def encode_categorical_features(train_df, test_df, cat_cols):
    """
    One-Hot Encodes categorical features for both Train and Test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        cat_cols: List of column names to encode
        
    Returns:
        train_encoded, test_encoded
    """
    
    # 1. Mark the split point so we can separate them later
    train_len = len(train_df)
    
    # 2. Concatenate temporarily to ensure column alignment
    # (This ensures 'Gender_Male' exists in both, even if one set has no males)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # 3. Apply One-Hot Encoding
    # drop_first=True avoids the "Dummy Variable Trap" (multicollinearity), 
    # which is important for linear models like Logistic Regression.
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    
    # 4. Split back into Train and Test
    train_encoded = combined_encoded.iloc[:train_len].copy()
    test_encoded = combined_encoded.iloc[train_len:].reset_index(drop=True).copy()
    
    return train_encoded, test_encoded

# --- Example Usage ---
# cat_cols = ['Gender', 'Location', 'Smoking_Status']
# train_enc, test_enc = encode_categorical_features(train, test, cat_cols)


train_encoded, test_encoded = encode_categorical_features(train_scaled, test_scaled, categorical)
test_encoded.drop(columns = ['diagnosed_diabetes'], inplace = True)


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def run_cv_logistic_regression(train_df, test_df, target_col, feature_cols, n_splits=5, seed=42):
    """
    Trains Logistic Regression using Stratified K-Fold Cross-Validation.
    
    Returns:
        test_preds: The averaged probability predictions for the test set across all folds.
    """
    
    # 1. Setup Data
    X = train_df[feature_cols]
    y = train_df[target_col]
    X_test = test_df[feature_cols]
    
    # 2. Initialize Cross-Validation and Storage
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Arrays to store results
    oof_preds = np.zeros(len(train_df))       # Out-of-Fold predictions for the training set
    test_preds = np.zeros(len(test_df))       # Final predictions for the test set
    fold_aucs = []                            # Store AUC scores for each fold

    print(f"--- Starting {n_splits}-Fold Cross-Validation ---")

    # 3. CV Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Split data for this fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Initialize and Train
        model = LogisticRegression(solver='liblinear', random_state=seed)
        model.fit(X_train, y_train)
        
        # Validation Inference (OOF)
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        
        # Test Inference (Accumulate for averaging)
        # We add a fraction of the predictions (1/5th if 5 folds)
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        
        # Calculate Fold Metric
        fold_auc = roc_auc_score(y_val, val_probs)
        fold_aucs.append(fold_auc)
        
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

    # 4. Final Metrics
    overall_auc = roc_auc_score(y, oof_preds) # Calculate AUC on the full OOF predictions
    mean_auc = np.mean(fold_aucs)
    
    print(f"-----------------------------------------")
    print(f"Mean Fold AUC : {mean_auc:.4f}")
    print(f"Overall OOF AUC: {overall_auc:.4f}")
    print(f"-----------------------------------------")
    
    return test_preds

# --- Example Usage ---
# predictions = run_cv_logistic_regression(train, test, 'Outcome', ['Glucose', 'BMI', 'Age'])


feature_cols = [feat for feat in train_encoded.columns if feat not in target]
print(feature_cols)


lr_predictions = run_cv_logistic_regression(train_encoded, test_encoded, 'diagnosed_diabetes', feature_cols, seed = 159)


sub_df['diagnosed_diabetes'] = lr_predictions
sub_df.to_csv('submission.csv', index = False)


def train_model(train_df, test_df, target_column, feature_columns=None, model_type="xgboost", param_file=None, 
                n_splits=5, categorical_features=None):
    """
    Train a machine learning classifier using the provided training and test datasets with K-Fold cross-validation,
    utilizing GPU support where applicable, and return predicted probabilities.

    Parameters:
        train_df (pandas.DataFrame): Training DataFrame.
        test_df (pandas.DataFrame): Testing DataFrame.
        target_column (str): Name of the target column.
        feature_columns (list): List of feature column names to use. If None, all columns except target_column are used.
        model_type (str): "xgboost", "catboost", "lgbm", or "hgb".
        param_file (dict): Dictionary of hyperparameters for the model.
        n_splits (int): Number of folds for cross-validation.
        categorical_features (list): List of categorical column names. If None, they are inferred from the features.

    Returns:
        tuple: (test_probabilities, oof_predictions, model)
            - test_probabilities: Final predicted probabilities for the test set.
            - oof_predictions: Out-of-fold predicted probabilities for the training set.
            - model: The final fitted model from the last fold.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
    import numpy as np

    # Determine feature set: use specified feature_columns if provided, otherwise use all except target_column
    if feature_columns is not None:
        # Ensure target_column is not included in feature_columns
        feature_columns = [col for col in feature_columns if col != target_column]
        X = train_df[feature_columns].copy()
        X_test = test_df[feature_columns].copy()
    else:
        X = train_df.drop(columns=[target_column])
        X_test = test_df.copy()

    # Extract target values
    y = train_df[target_column]

    # Infer categorical features from the selected features if not provided
    if categorical_features is None:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Import model-specific classifier classes
    if model_type == "xgboost":
        from xgboost import XGBClassifier as Model
    elif model_type == "catboost":
        from catboost import CatBoostClassifier as Model
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier as Model
    elif model_type == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier as Model
    else:
        raise ValueError("Unsupported model_type. Choose from 'xgboost', 'catboost', 'lgbm', or 'hgb'.")

    # Set default parameters if none are provided
    if param_file is None:
        if model_type == "xgboost":
            param_file = {
                'n_estimators': 4096,
                'learning_rate': 0.01,
                'max_depth': 5,
                'subsample': 0.10,
                'colsample_bytree': 0.60,
                'min_child_weight': 80,
                'alpha': 1.5,
                'tree_method': 'hist',
                'device': 'cuda',
                'eval_metric':'auc',
                'random_state': 42
            }
        elif model_type == "catboost":
            param_file = {
                'iterations': 2048,
                'learning_rate': 0.01,
                'depth': 6,
                'task_type': 'GPU',
                'random_seed': 42,
                'verbose': False
            }
        elif model_type == "lgbm":
            param_file = {
                'n_estimators': 4096,
                'learning_rate': 0.01,
                'max_depth': -1,
                'subsample': 0.75,
                'colsample_bytree': 0.50,
                'min_child_weight': 80,
                'device': 'gpu',
                'random_state': 42
            }
        elif model_type == "hgb":
            param_file = {
                'max_iter': 200,
                'learning_rate': 0.03,
                'max_leaf_nodes': 31,
                'early_stopping': True,
                'random_state': 42
            }

    # Initialize the model with the specified parameters
    model = Model(**param_file)

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store metrics and test-set predictions
    accuracy_scores = []
    f1_scores = []
    auc_scores = []
    logloss_scores = []
    test_pred_probs = []  # To collect test-set probabilities from each fold
    oof_predictions = None  # Will be initialized upon first fold

    for train_index, val_index in kf.split(X):
        # Create fold-specific training and validation data
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        X_test_fold = X_test.copy()

        # --- PROCESS CATEGORICAL FEATURES ---
        if model_type == "catboost":
            # CatBoost handles categorical features internally if passed as strings.
            X_train[categorical_features] = X_train[categorical_features].fillna('Missing').astype(str)
            X_val[categorical_features] = X_val[categorical_features].fillna('Missing').astype(str)
            X_test_fold[categorical_features] = X_test_fold[categorical_features].fillna('Missing').astype(str)
            model.fit(X_train, y_train, cat_features=categorical_features)
        else:
            # For other models, perform label encoding for categorical features.
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train[categorical_features] = encoder.fit_transform(
                X_train[categorical_features].fillna('Missing').astype(str)
            )
            X_val[categorical_features] = encoder.transform(
                X_val[categorical_features].fillna('Missing').astype(str)
            )
            X_test_fold[categorical_features] = encoder.transform(
                X_test_fold[categorical_features].fillna('Missing').astype(str)
            )
            model.fit(X_train, y_train)

        # --- EVALUATION ON VALIDATION SET ---
        # Attempt to get predicted probabilities; if not available, fall back to class predictions.
        try:
            y_val_pred_proba = model.predict_proba(X_val)
        except AttributeError:
            y_val_pred_proba = None

        if y_val_pred_proba is not None:
            # Initialize oof_predictions array on first fold
            if oof_predictions is None:
                n_classes = y_val_pred_proba.shape[1]
                oof_predictions = np.zeros((len(train_df), n_classes))
            oof_predictions[val_index] = y_val_pred_proba
        else:
            # Fallback: use predicted classes (note: these are not probabilities)
            y_val_pred = model.predict(X_val)
            if oof_predictions is None:
                oof_predictions = np.empty((len(train_df),), dtype=object)
            oof_predictions[val_index] = y_val_pred

        # For computing metrics, use class predictions.
        y_val_pred_classes = model.predict(X_val)
        accuracy_scores.append(accuracy_score(y_val, y_val_pred_classes))
        f1_scores.append(f1_score(y_val, y_val_pred_classes, average='macro'))
        try:
            if y_val_pred_proba is not None:
                if len(np.unique(y)) == 2:
                    auc_scores.append(roc_auc_score(y_val, y_val_pred_proba[:, 1]))
                logloss_scores.append(log_loss(y_val, y_val_pred_proba))
        except Exception:
            pass

        # --- PREDICTION ON TEST SET ---
        if target_column in X_test_fold.columns:
            X_test_fold = X_test_fold.drop(columns=[target_column], errors='ignore')
        try:
            test_pred_probs.append(model.predict_proba(X_test_fold))
        except AttributeError:
            test_pred_probs.append(model.predict(X_test_fold))

    # Calculate and print average metrics
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    print("Model Performance Metrics (Cross-Validation):")
    print("..................")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average F1 Score (macro): {avg_f1:.4f}")
    if auc_scores:
        avg_auc = np.mean(auc_scores)
        print(f"Average ROC-AUC: {avg_auc:.4f}")
    if logloss_scores:
        avg_logloss = np.mean(logloss_scores)
        print(f"Average Log Loss: {avg_logloss:.4f}")

    # Average the test-set predicted probabilities from each fold
    test_pred_probs = np.array(test_pred_probs)
    mean_probs = np.mean(test_pred_probs, axis=0)

    # Optionally, print the feature names used in the last training fold
    print("Features used in the last fold:")
    print(X_train.columns)

    return mean_probs, oof_predictions, model


xgb_mean_probs, xgb_oof_predictions, xgb_model = train_model(train_encoded, 
                                                             test_encoded, 
                                                             feature_columns = feature_cols , 
                                                             target_column = 'diagnosed_diabetes', 
                                                             model_type = "xgboost", param_file = None, 
                                                             n_splits = 5, 
                                                             categorical_features = None)


sub_df['diagnosed_diabetes'] = xgb_mean_probs[:, 1]
sub_df.to_csv('submission_xgb.csv', index = False)


cat_mean_probs, cat_oof_predictions, cat_model = train_model(train_encoded, test_encoded, feature_columns = feature_cols , target_column = 'diagnosed_diabetes', model_type = "catboost", param_file = None, n_splits = 5, categorical_features = None)


sub_df['diagnosed_diabetes'] = cat_mean_probs[:, 1]
sub_df.to_csv('submission_cbc.csv', index = False)


lgb_mean_probs, lgb_oof_predictions, lgb_model = train_model(train_encoded, test_encoded, feature_columns = feature_cols , target_column = 'diagnosed_diabetes', model_type = "lgbm", param_file = None, n_splits = 5, categorical_features = None)


sub_df['diagnosed_diabetes'] = lgb_mean_probs[:, 1]
sub_df.to_csv('submission_lgb.csv', index = False)


hgb_mean_probs, hgb_oof_predictions, hgb_model = train_model(train_encoded, test_encoded, feature_columns = feature_cols , target_column = 'diagnosed_diabetes', model_type = "hgb", param_file = None, n_splits = 5, categorical_features = None)


sub_df['diagnosed_diabetes'] = lgb_mean_probs[:, 1]
sub_df.to_csv('submission_hgb.csv', index = False)





import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def keras_cv_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target,
    build_model_fn,
    features=None,
    n_splits=5,
    epochs=30,
    batch_size=256,
    random_state=42,
    verbose=0,
):
    """
    Cross-validated Keras training using pandas DataFrames.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame
    target   : str (column name in train_df) OR pd.Series/np.ndarray of labels
    build_model_fn : function(num_classes, input_shape) -> compiled tf.keras.Model
    features : list of feature columns to use (optional)
    Returns
    -------
    oof_proba, test_proba, fold_metrics, overall_metrics
    """

    # ---- Extract y ----
    if isinstance(target, str):
        y = train_df[target].values
        if features is None:
            X_train_df = train_df.drop(columns=[target])
        else:
            X_train_df = train_df[features]
    else:
        y = np.asarray(target)
        X_train_df = train_df if features is None else train_df[features]

    X_test_df = test_df if features is None else test_df[features]

    # ---- Convert to numeric numpy (Keras needs numbers) ----
    X_train = X_train_df.to_numpy(dtype=np.float32)
    X_test = X_test_df.to_numpy(dtype=np.float32)
    y = np.asarray(y)

    # Basic safety check: no object dtypes sneaking in
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        # You can replace this with your preferred imputation
        raise ValueError("NaNs found in features. Impute/fill missing values before training.")

    num_classes = len(np.unique(y))
    input_shape = X_train.shape[1:]

    proba_dim = 1 if num_classes == 2 else num_classes
    oof_proba = np.zeros((len(X_train), proba_dim), dtype=np.float32)
    test_proba = np.zeros((len(X_test), proba_dim), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), start=1):
        X_tr, y_tr = X_train[tr_idx], y[tr_idx]
        X_va, y_va = X_train[va_idx], y[va_idx]

        tf.keras.backend.clear_session()
        model = build_model_fn(num_classes, input_shape)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]

        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        va_pred = model.predict(X_va, batch_size=batch_size, verbose=0)
        te_pred = model.predict(X_test, batch_size=batch_size, verbose=0)

        if num_classes == 2:
            va_pred = va_pred.reshape(-1, 1)
            te_pred = te_pred.reshape(-1, 1)

        oof_proba[va_idx] = va_pred
        test_proba += te_pred / n_splits

        # Metrics per fold
        if num_classes == 2:
            va_pos = va_pred[:, 0]
            va_label = (va_pos >= 0.5).astype(int)
            fm = {
                "fold": fold,
                "accuracy": float(accuracy_score(y_va, va_label)),
                "auc": float(roc_auc_score(y_va, va_pos)),
                "log_loss": float(log_loss(y_va, va_pos)),
            }
        else:
            va_label = np.argmax(va_pred, axis=1)
            fm = {
                "fold": fold,
                "accuracy": float(accuracy_score(y_va, va_label)),
                "auc": float(roc_auc_score(y_va, va_pred, multi_class="ovr", average="macro")),
                "log_loss": float(log_loss(y_va, va_pred)),
            }

        fold_metrics.append(fm)
        print(f"Fold {fold}: {fm}")

    # Overall OOF metrics
    if num_classes == 2:
        oof_pos = oof_proba[:, 0]
        oof_label = (oof_pos >= 0.5).astype(int)
        overall = {
            "accuracy": float(accuracy_score(y, oof_label)),
            "auc": float(roc_auc_score(y, oof_pos)),
            "log_loss": float(log_loss(y, oof_pos)),
        }
    else:
        oof_label = np.argmax(oof_proba, axis=1)
        overall = {
            "accuracy": float(accuracy_score(y, oof_label)),
            "auc": float(roc_auc_score(y, oof_proba, multi_class="ovr", average="macro")),
            "log_loss": float(log_loss(y, oof_proba)),
        }

    print("Overall OOF:", overall)
    return oof_proba, test_proba, fold_metrics, overall



# Example model builder (simple MLP)
def build_mlp(num_classes, input_shape):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(256, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    if num_classes == 2:
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss)
    return model


net_oof_predictions, net_test_probs, fold_metrics, overall = keras_cv_df(
    train_df=train_encoded,
    test_df=test_encoded,
    target="diagnosed_diabetes",
    build_model_fn=build_mlp,
    features=None,     # or a list like ["f1","f2",...]
    n_splits=5,
    epochs=128,
    batch_size=512,
    verbose=0,
)


net_oof_predictions[:,-1]


hgb_oof_predictions[:,1]


sub_df['diagnosed_diabetes'] = net_test_probs
sub_df.to_csv('submission_nn.csv', index = False)


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

seed = 42

def meta_model(
    oof_train_df,         # DataFrame containing the OOF predictions (features) for training
    target,               # Series (or 1d array) with the ground truth values
    oof_test_df=None,     # (Optional) DataFrame with OOF predictions for test set
    params=None,          # (Optional) Dictionary of LightGBM parameters
    n_splits=5,           # Number of folds for cross-validation
    num_boost_round=1000,
    early_stopping_rounds=50,
    random_state=42,
    verbose_eval=False
):
    """
    Trains a meta model using LightGBM as the gradient boosted decision tree on the provided OOF features.
    
    Parameters:
      oof_train_df (pd.DataFrame): DataFrame with out-of-fold predictions from base models for training.
      target (pd.Series or array-like): Target variable corresponding to oof_train_df.
      oof_test_df (pd.DataFrame, optional): DataFrame with OOF predictions for the test set.
      params (dict, optional): Parameters for the LightGBM model.
      n_splits (int): Number of folds to use in KFold cross-validation.
      num_boost_round (int): Maximum number of boosting rounds.
      early_stopping_rounds (int): Early stopping rounds for validation.
      random_state (int): Random seed for reproducibility.
      verbose_eval (bool or int): Whether to print evaluation messages during training.
    
    Returns:
      If oof_test_df is provided, returns a tuple:
          (oof_predictions, test_predictions)
      Otherwise, returns the cross-validated OOF predictions.
    """
    
    # Default LightGBM parameters (for regression; change objective if needed)
    if params is None:
        params = {
            'objective': 'regression',
            'learning_rate': 0.01,
            'num_leaves': 16,
            'metric': 'rmse',
            'seed': random_state,
            'verbosity': -1,
        }
    
    # Convert inputs to NumPy arrays if they aren't already
    X = oof_train_df.values if isinstance(oof_train_df, pd.DataFrame) else oof_train_df
    y = target.values if isinstance(target, pd.Series) else target
    
    # Initialize arrays for out-of-fold predictions and (if applicable) test predictions
    oof_preds = np.zeros(X.shape[0])
    if oof_test_df is not None:
        X_test = oof_test_df.values if isinstance(oof_test_df, pd.DataFrame) else oof_test_df
        test_preds = np.zeros(X_test.shape[0])
    
    # Set up KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # List to hold the RMSE for each fold
    fold_rmse_scores = []
    
    # Loop over each fold
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold+1}/{n_splits}...")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
        
        # Train the LightGBM model
        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            #early_stopping_rounds=early_stopping_rounds,
            #verbose_eval=verbose_eval
        )
        
        # Get predictions for the validation fold
        fold_preds = model.predict(X_valid, num_iteration=model.best_iteration)
        oof_preds[valid_idx] = fold_preds
        
        # Calculate and print RMSE for the fold
        fold_rmse = np.sqrt(mean_squared_error(y_valid, fold_preds))
        fold_rmse_scores.append(fold_rmse)
        print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")
        
        # If a test set is provided, average predictions over folds
        if oof_test_df is not None:
            test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

    # Calculate and print overall performance
    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nOverall CV RMSE: {overall_rmse:.4f}")
    print(f"Average Fold RMSE: {np.mean(fold_rmse_scores):.4f}")

    if oof_test_df is not None:
        return oof_preds, test_preds
    else:
        return oof_preds

# ===========================
# Example Usage
# ===========================

# Suppose you have a training dataframe "df_train" that includes the target variable
# and OOF predictions from three base models in columns: 'model1_oof', 'model2_oof', 'model3_oof'.

# Also, suppose you have a test dataframe "df_test" with the corresponding OOF prediction columns.

# Prepare the meta features:

# For training:
df_train = pd.DataFrame({
    'model1_oof': xgb_oof_predictions[:,1],
    'model2_oof': lgb_oof_predictions[:,1],
    'model3_oof': cat_oof_predictions[:,1],
    'model4_oof': hgb_oof_predictions[:,1],
    'model5_oof': net_oof_predictions[:,-1]
})

df_test = pd.DataFrame({
    'model1_oof': xgb_mean_probs[:,1],
    'model2_oof': lgb_mean_probs[:,1],
    'model3_oof': cat_mean_probs[:,1],
    'model4_oof': hgb_mean_probs[:,1],
    'model5_oof': net_test_probs[:,-1]
})

meta_features_train = df_train[['model1_oof', 
                                'model2_oof', 
                                'model3_oof', 
                                'model4_oof',
                                'model5_oof'
                               ]]
target = train_encoded['diagnosed_diabetes']

# For test (if available):
meta_features_test = df_test[['model1_oof', 
                              'model2_oof', 
                              'model3_oof', 
                              'model4_oof',
                              'model5_oof'
                             ]]

# Train the meta model and generate final predictions:
oof_predictions, final_test_predictions = meta_model(
    oof_train_df=meta_features_train,
    target=target,
    oof_test_df=meta_features_test,
    params=None,  # Uses default parameters; feel free to adjust if needed
    n_splits=5,
    num_boost_round=2048,
    #early_stopping_rounds=50,
    random_state=seed,
    #verbose_eval=100  # Adjust as needed, or set to True/False
)

# Now, "oof_predictions" contains the cross-validated predictions on the training data,
# and "final_test_predictions" contains the aggregated predictions on the test set.


sub_df['diagnosed_diabetes'] = final_test_predictions
sub_df.to_csv('submission_meta.csv', index = False)




