import pandas as pd
import numpy as np
import warnings
import joblib
import optuna
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, log_evaluation, early_stopping # Import log_evaluation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, clone

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
class CFG:
    train_path = "/kaggle/input/playground-series-s5e7/train.csv"
    test_path = "/kaggle/input/playground-series-s5e7/test.csv"
    sample_sub_path = "/kaggle/input/playground-series-s5e7/sample_submission.csv"
    target = "Personality"
    n_folds = 5
    seed = 42

# =========================
# Data Loading
# =========================
train = pd.read_csv(CFG.train_path)
test = pd.read_csv(CFG.test_path)
sub = pd.read_csv(CFG.sample_sub_path)

# Define the preprocess function (copied from your previous prompt for completeness)
def preprocess(df):
    """
    Performs feature engineering and basic imputation for the dataset.
    - Extracts group, deck, cabin number, and side from PassengerId and Cabin.
    - Fills missing boolean values (CryoSleep, VIP) with False.
    - Calculates total expenditure and group size.
    - Identifies solo travelers.
    - Fills remaining NaN values using forward fill.
    """
    # Assuming PassengerId is in the format 'group_id'
    if 'PassengerId' in df.columns and '_' in df['PassengerId'].iloc[0]:
        df['group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    
    # Expand Cabin into separate columns, handling potential NaNs
    if 'Cabin' in df.columns:
        df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce') # Convert CabinNum to numeric
    
    # Fill boolean columns, if they exist
    if 'CryoSleep' in df.columns:
        df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)
    if 'VIP' in df.columns:
        df['VIP'] = df['VIP'].fillna(False).astype(int)
    
    # Calculate total expenditure from service columns, filling NaNs with 0 for sum
    service_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    if all(col in df.columns for col in service_cols):
        df['Expenditure'] = df[service_cols].fillna(0).sum(axis=1)
        df['TotalSpent'] = df['Expenditure'] # Alias for clarity

    # Calculate group size based on the 'group' column, if it exists
    if 'group' in df.columns:
        df['GroupSize'] = df.groupby('group')['group'].transform('count')
        df['Solo'] = (df['GroupSize'] == 1).astype(int) # Identify solo travelers

    # Fill remaining NaNs using forward fill. This is a simple imputation strategy.
    df.fillna(method='ffill', inplace=True)
    return df

# Apply preprocessing to initial train and test sets
train = preprocess(train)
test = preprocess(test)


# Target encoding
y = train[CFG.target].copy()
le = LabelEncoder()
y_enc = le.fit_transform(y)
X = train.drop(columns=[CFG.target, "id"])
X_test = test.drop(columns=["id"])

# Ordinal encoding for categorical features
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)
cat_cols = combined.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])
X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)

# =========================
# Trainer Class
# =========================
class Trainer:
    def __init__(self, model):
        self.model = model

    def fit_predict(self, X, y, X_test, fit_args={}):
        oof_probs = np.zeros((len(X), 2))
        test_probs = np.zeros((len(X_test), 2))
        skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_train, y_val = y[tr_idx], y[val_idx]

            model = clone(self.model)
            
            # Prepare fit arguments for LightGBM specifically
            current_fit_args = fit_args.copy()
            if "LGBM" in str(type(model)):
                # If the model is an LGBMClassifier, adjust fit_args for its API
                # The original code passed 'eval_set': [(X, y_enc)] to the trainer,
                # which is the full dataset. For proper in-fold validation/early stopping,
                # eval_set should typically be [(X_val, y_val)].
                # However, to directly fix the 'verbose' error, we replace it with callbacks.
                # If you intend to use early stopping, you would add early_stopping callback here
                # and ensure eval_set uses X_val, y_val.
                current_fit_args["callbacks"] = [log_evaluation(period=0)] # Suppress verbose output
                if "verbose" in current_fit_args:
                    del current_fit_args["verbose"]
                # Ensure eval_set is correctly set for LGBM if it's in the original fit_args
                # If the original intent was to use the full dataset for eval_set, keep it.
                # Otherwise, for in-fold validation, change to:
                # current_fit_args["eval_set"] = [(X_val, y_val)]
            
            model.fit(X_train, y_train, **current_fit_args)
            oof_probs[val_idx] = model.predict_proba(X_val)
            test_probs += model.predict_proba(X_test) / CFG.n_folds

            score = accuracy_score(y_val, np.argmax(oof_probs[val_idx], axis=1))
            print(f"Fold {fold+1}: Accuracy = {score:.6f}")

            del model; gc.collect()

        oof_score = accuracy_score(y, np.argmax(oof_probs, axis=1))
        print(f"\nOverall CV Accuracy: {oof_score:.6f}")
        return oof_probs, test_probs

# =========================
# Model Definitions
# =========================
xgb_model = XGBClassifier(
    objective="binary:logistic", eval_metric="logloss", max_depth=4,
    eta=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)

lgbm_gbdt = LGBMClassifier(
    boosting_type="gbdt", learning_rate=0.016, max_depth=12, num_leaves=243,
    subsample=0.799, colsample_bytree=0.437, n_estimators=10000, device="gpu",
    reg_alpha=6.38, reg_lambda=9.39, min_child_samples=67, verbose=-1, random_state=42)

lgbm_goss = LGBMClassifier(
    boosting_type="goss", learning_rate=0.0067, max_depth=12, num_leaves=229,
    subsample=0.541, colsample_bytree=0.328, n_estimators=10000, device="gpu",
    reg_alpha=6.88, reg_lambda=4.74, min_child_samples=84, verbose=-1, random_state=42)

hgb_model = HistGradientBoostingClassifier(
    learning_rate=0.03, max_iter=500, max_depth=5, min_samples_leaf=12,
    l2_regularization=0.75, random_state=42)

# =========================
# Training Base Models
# =========================
base_models = {
    "XGB": xgb_model,
    "LGBM_GBDT": lgbm_gbdt,
    "LGBM_GOSS": lgbm_goss,
    "HGB": hgb_model
}

oof_preds = {}
test_preds = {}

for name, model in base_models.items():
    print(f"\nTraining {name}")
    trainer = Trainer(model)
    
    # Corrected fit_args for LGBM models
    if "LGBM" in name:
        # For LGBM, use 'callbacks' instead of 'verbose'.
        # log_evaluation(period=0) suppresses all logging.
        # Note: The original code used eval_set=[(X, y_enc)], which is the full training set.
        # For typical cross-validation with early stopping, eval_set should be [(X_val, y_val)]
        # within each fold of the Trainer class. This fix addresses the 'verbose' error directly.
        fit_args = {"eval_set": [(X, y_enc)], "callbacks": [log_evaluation(period=0)]}
    else:
        fit_args = {} # XGB and HGB don't need special fit_args for this setup

    oof, test_pred_model = trainer.fit_predict(X, y_enc, X_test, fit_args)
    oof_preds[name] = oof
    test_preds[name] = test_pred_model # Renamed variable to avoid conflict with global 'test'

# =========================
# Logistic Regression with Mutual Info Features
# =========================
# Re-load original data for this section to ensure correct DataFrame structure
# This prevents issues if 'train' or 'test' were inadvertently modified or converted
# to numpy arrays by previous operations not intended for this part of the pipeline.
train_for_lr = pd.read_csv(CFG.train_path)
test_for_lr = pd.read_csv(CFG.test_path)

# Apply the initial preprocessing to these fresh copies, as done for the main models
train_for_lr = preprocess(train_for_lr)
test_for_lr = preprocess(test_for_lr)


X_cat = train_for_lr.drop(columns=[CFG.target, "id"]).astype(str)
X_test_cat = test_for_lr.drop(columns=["id"]).astype(str)

# Feature interactions using 2-way combinations
from itertools import combinations
for c1, c2 in combinations(X_cat.columns, 2):
    X_cat[f"{c1}|{c2}"] = X_cat[c1] + "|" + X_cat[c2]
    X_test_cat[f"{c1}|{c2}"] = X_test_cat[c1] + "|" + X_test_cat[c2]

lr_pipeline = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression(C=0.01, max_iter=10000, random_state=0))

trainer = Trainer(lr_pipeline)
oof_lr, test_lr = trainer.fit_predict(X_cat, y_enc, X_test_cat)
oof_preds["Logistic"] = oof_lr
test_preds["Logistic"] = test_lr

# =========================
# Blending with RidgeCV
# =========================
X_stack = pd.DataFrame(np.hstack([v for v in oof_preds.values()]))
X_test_stack = pd.DataFrame(np.hstack([v for v in test_preds.values()]))

blender = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=5)
blender.fit(X_stack, y_enc)
final_probs = blender.predict(X_test_stack)
final_preds = (final_probs > 0.5).astype(int)

# =========================
# Submission
# =========================
sub[CFG.target] = le.inverse_transform(final_preds)
sub.to_csv("submission.csv", index=False)

print(sub.head())


