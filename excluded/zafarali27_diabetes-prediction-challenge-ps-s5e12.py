import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


from colorama import Fore, Style

# Print the shape of the dataframe (number of rows and columns)
print(Fore.CYAN + "train_df shape: " + Style.RESET_ALL)
print(f"{train_df.shape}\n")

# Print basic information about the dataframe (column names, data types, non-null values)
print(Fore.GREEN + "train_df info: " + Style.RESET_ALL)
print(f"{train_df.info()}\n")

# Print the count of missing (NaN) values in each column
print(Fore.YELLOW + "train_df isnull sum: " + Style.RESET_ALL)
print(f"{train_df.isnull().sum()}\n")

# Print summary statistics for numerical columns (count, mean, std, min, max, etc.)
print(Fore.MAGENTA + "train_df describe: " + Style.RESET_ALL)
print(f"{train_df.describe()}\n")



# Define the numerical & categorical
numerical_col = train_df.select_dtypes(include = ["int64","float64"]).columns
# Define the categorical
categorical_col = train_df.select_dtypes(include = "object").columns

print(f"We have features: {len(numerical_col)} numerical features {numerical_col}")
print("-"*100)
print(f"We have features: {len(categorical_col)} categorical features {categorical_col}")


from scipy.stats import chi2_contingency
chi2_test = []
for feature in categorical_col:
    if chi2_contingency(pd.crosstab(train_df['diagnosed_diabetes'], train_df[feature]))[1] < 0.05:
        chi2_test.append('Reject Null Hypothesis')
    else:
        chi2_test.append('Fail to Reject Null Hypothesis')
result = pd.DataFrame(data=[categorical_col, chi2_test]).T # Create a DataFrame to store the chi-squared test results
result.columns = ['Column', 'Hypothesis Result']
result



# After creating features
# train_df["BMI_Age"] = train_df["bmi"] * train_df["age"]
# test_df["BMI_Age"] = test_df["bmi"] * test_df["age"]
# train_df["Glucose_BMI"] = train_df["cholesterol_total"] / (train_df["bmi"] + 1)
# test_df["Glucose_BMI"] = test_df["cholesterol_total"] / (test_df["bmi"] + 1)


TARGET = "diagnosed_diabetes"
X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]

test_ids = test_df["id"]


cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


X[cat_cols] = enc.fit_transform(X[cat_cols])
test_df[cat_cols] = enc.transform(test_df[cat_cols])



KF = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

pred_lgb = np.zeros(len(test_df))
pred_cat = np.zeros(len(test_df))
pred_xgb = np.zeros(len(test_df))


# for fold, (trn_idx, val_idx) in enumerate(KF.split(X,y)):
#     print(f"\n===== FOLD {fold+1} / 5 ====")

#     X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
#     X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

#     # LightGBM
#     lgb = LGBMClassifier(
#         objective = "binary",
#         metric = "auc",
#         boosting_type = "gbdt",
#         n_estimators = 1000,
#         learning_rate = 0.01,
#         colsample_freq = 1,
#         min_child_samples = 20,
#         reg_alpha = 0.05,
#         reg_lambda = 0.1,
#         random_state = 42,
#         n_jobs = -1,
#         verbose = -1
#     )
#     lgb.fit(X_train, y_train)

#     oof_lgb[val_idx] = lgb.predict_proba(X_valid)[:, 1]
#     pred_lgb += lgb.predict_proba(X_test)[:,1] / KF.n_splits


#     # CatBoost
#     cat = CatBoostClassifier(
#         iterations = 2000,
#         learning_rate = 0.03,
#         depth = 8,
#         loss_function = "Logloss",
#         eval_metric = "AUC",
#         random_state = 42,
#         auto_class_weights = "Balanced",
#         l2_leaf_reg = 5,
#     )
#     cat.fit(X_train, y_train)
    
#     oof_cat[val_idx] = cat.predict_proba(X_valid)[:, 1]
#     pred_cat += cat.predict_proba(X_test)[:, 1] / KF,n_splits


#     xgb = XBGClassifier(
#         objective = "binary:logistic",
#         eval_metric = "auc",
#         learning_rate = 0.01,
#         max_depth = 8,
#         min_child_weight = 3,
#         colsample_bytree = 0.3,
#         subsample = 0.6,
#         reg_alpha = 0.5,
#         reg_lambda = 2.0,
#         n_estimators = 20000,
#         random_state = 42,
#         n_jobs = -1,
#         verbose = -1,
#         tree_method = "hist"
#     )

#     xgb.fit(X_train,y_train)

#     oof_xgb[val_idx] = xgb.predict_proba(X_valid)[:.1]
#     pred_xbg += xgb.predict_proba(X_test)[:, 1] / KF.n_splits
    


for fold, (trn_idx, val_idx) in enumerate(KF.split(X, y)):
    print(f"\n===== FOLD {fold+1} / 5 =====")

    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        class_weight="balanced"
    )
    lgb.fit(X_train, y_train)

    oof_lgb[val_idx] = lgb.predict_proba(X_valid)[:, 1]
    pred_lgb += lgb.predict_proba(test_df)[:, 1] / KF.n_splits

    # CatBoost
    cat = CatBoostClassifier(
    iterations=2000,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False)
    cat.fit(X_train, y_train)
    oof_cat[val_idx] = cat.predict_proba(X_valid)[:, 1]
    pred_cat += cat.predict_proba(test_df)[:, 1] / KF.n_splits

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        tree_method="hist"
    )
    xgb.fit(X_train, y_train)

    oof_xgb[val_idx] = xgb.predict_proba(X_valid)[:, 1]
    pred_xgb += xgb.predict_proba(test_df)[:, 1] / KF.n_splits



oof_blend = 0.4 * oof_lgb + 0.35 * oof_cat + 0.25 * oof_xgb
pred_blend = 0.4 * pred_lgb + 0.35 * pred_cat + 0.25 * pred_xgb


print("\nLightGBM ROC:", roc_auc_score(y, oof_lgb))
print("CatBoost ROC:", roc_auc_score(y, oof_cat))
print("XGBoost ROC:", roc_auc_score(y, oof_xgb))
print("Blended ROC:", roc_auc_score(y, oof_blend))


# Stacking

stack_train = np.vstack([oof_lgb, oof_cat, oof_xgb]).T
stack_test  = np.vstack([pred_lgb, pred_cat, pred_xgb]).T

lvl2 = LogisticRegression(max_iter=2000)
lvl2.fit(stack_train, y)

pred_final = lvl2.predict_proba(stack_test)[:, 1]

print("\nFinal Stacked ROC:",
      roc_auc_score(y, lvl2.predict_proba(stack_train)[:, 1]))


submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": pred_final
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved successfully!")

