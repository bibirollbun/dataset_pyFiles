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


import warnings
warnings.filterwarnings("ignore")

# Now run your actual code - it will work perfectly
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
# ... etc

print("Ready to train models!")
import warnings
# Aggressively suppress SQLAlchemy syntax warning
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", message=".*is not.*tuple.*")
warnings.filterwarnings("ignore", module="sqlalchemy")
import sys

# Also suppress at import time
if not sys.warnoptions:
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore::SyntaxWarning'

# Now your imports
import numpy as np
import pandas as pd
# ... rest of your code


# Install from local cache only (no internet)
!pip install sqlalchemy==1.4.50 --no-index --find-links /usr/local/lib/python3.12/dist-packages 2>/dev/null || echo "Local install failed"

# Or simply force reinstall the current version to refresh metadata
!pip install --force-reinstall sqlalchemy==1.2.19 --no-deps

!pip install --upgrade sqlalchemy
!pip install --upgrade sqlalchemy --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.net
!pip install --no-index --find-links=./packages sqlalchemy
!pip download sqlalchemy -d ./packages
!pip install --no-index --find-links=./packages sqlalchemy
!pip download sqlalchemy==2.0.25 -d ./sqlalchemy_package --only-binary=:all:
!pip install sqlalchemy-2.0.25-*.whl --force-reinstall
import pandas as pd
print(pd.__version__)

warnings.filterwarnings("ignore")

# Patch SQLAlchemy before pandas imports it
import sys
from unittest.mock import MagicMock

# Create a mock that suppresses the warning
class PatchedSQLAlchemy:
    def __getattr__(self, name):
        return MagicMock()

# Only if you don't actually need SQLAlchemy
sys.modules['sqlalchemy'] = PatchedSQLAlchemy()

# Suppress the specific SyntaxWarning from SQLAlchemy
warnings.filterwarnings("ignore", message='"is not" with.*literal', category=SyntaxWarning)

# Now import everything else
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

RANDOM_STATE = 42
N_SPLITS = 5

print("All imports successful - warning suppressed!")



train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

TARGET = "diagnosed_diabetes"

X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]
X_test = test_df.copy()

print(train_df.shape, test_df.shape)



cat_cols = X.select_dtypes(include="object").columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

cat_idx = [X.columns.get_loc(col) for col in cat_cols]



def add_features(df):
    df = df.copy()

    if "bmi" in df.columns and "age" in df.columns:
        df["bmi_age"] = df["bmi"] * df["age"]

    if "blood_pressure" in df.columns and "age" in df.columns:
        df["bp_age"] = df["blood_pressure"] * df["age"]

    if "physical_activity" in df.columns and "bmi" in df.columns:
        df["activity_bmi"] = df["physical_activity"] / (df["bmi"] + 1)

    return df

X = add_features(X)
X_test = add_features(X_test)

print("Train shape:", X.shape)
print("Test shape:", X_test.shape)




oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

cv_scores_lgb = []
cv_scores_cat = []

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)



for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nLightGBM Fold {fold+1}")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE + fold
    )

    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    val_pred = lgb_model.predict_proba(X_val)[:, 1]
    test_pred = lgb_model.predict_proba(X_test)[:, 1]

    oof_lgb[val_idx] = val_pred
    test_lgb += test_pred / N_SPLITS

    auc = roc_auc_score(y_val, val_pred)
    cv_scores_lgb.append(auc)

    print(f"AUC: {auc:.5f}")



for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nCatBoost Fold {fold+1}")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    cat_model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=200,
        verbose=False
    )

    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        cat_features=cat_idx
    )

    val_pred = cat_model.predict_proba(X_val)[:, 1]
    test_pred = cat_model.predict_proba(X_test)[:, 1]

    oof_cat[val_idx] = val_pred
    test_cat += test_pred / N_SPLITS

    auc = roc_auc_score(y_val, val_pred)
    cv_scores_cat.append(auc)

    print(f"AUC: {auc:.5f}")



import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Create a DataFrame with OOF predictions and target
oof_df = pd.DataFrame({
    'LightGBM_OOF': oof_lgb,
    'CatBoost_OOF': oof_cat,
    'target': y.values
})

# 1. Correlation coefficient
corr = oof_df[['LightGBM_OOF', 'CatBoost_OOF']].corr().iloc[0,1]
pearson_corr, p_value = pearsonr(oof_lgb, oof_cat)

print(f"Pearson correlation between LightGBM and CatBoost OOF predictions: {corr:.4f}")
print(f"P-value: {p_value:.2e}")


plt.figure(figsize=(10, 8))
sns.pairplot(
    oof_df,
    vars=['LightGBM_OOF', 'CatBoost_OOF'],
    hue='target',
    diag_kind='kde',
    plot_kws={'alpha': 0.6, 's': 30},
    diag_kws={'fill': True}
)
plt.suptitle(f'OOF Predictions Pairplot (Correlation = {corr:.4f})', y=1.02)
plt.show()


plt.figure(figsize=(8, 8))
sns.scatterplot(
    data=oof_df,
    x='LightGBM_OOF',
    y='CatBoost_OOF',
    hue='target',
    alpha=0.6,
    palette='viridis',
    s=40
)
sns.regplot(
    data=oof_df,
    x='LightGBM_OOF',
    y='CatBoost_OOF',
    scatter=False,
    color='red',
    line_kws={'linestyle': '--', 'label': f'Correlation = {corr:.4f}'}
)
plt.xlabel('LightGBM OOF Probability')
plt.ylabel('CatBoost OOF Probability')
plt.title('Scatter Plot of OOF Predictions (Colored by Target)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


## Simple ensemble performance comparison on OOF
# oof_avg = (oof_lgb + oof_cat) / 2
# auc_lgb = roc_auc_score(y, oof_lgb)
# auc_cat = roc_auc_score(y, oof_cat)
# auc_avg = roc_auc_score(y, oof_avg)

# # Stacking OOF (note: slightly optimistic since meta was trained on full OOF)
# oof_stack = meta.predict_proba(np.vstack([oof_lgb, oof_cat]).T)[:, 1]
# auc_stack = roc_auc_score(y, oof_stack)

# print("\nOOF AUC Scores:")
# print(f"LightGBM:      {auc_lgb:.5f} (CV mean: {np.mean(cv_scores_lgb):.5f})")
# print(f"CatBoost:      {auc_cat:.5f} (CV mean: {np.mean(cv_scores_cat):.5f})")
# print(f"Simple Average:{auc_avg:.5f}")
# print(f"Stacking:      {auc_stack:.5f}  ← (optimistic, trained on same OOF)")


plt.figure(figsize=(8,4))
plt.plot(cv_scores_lgb, label="LightGBM", marker="o")
plt.plot(cv_scores_cat, label="CatBoost", marker="o")
plt.axhline(np.mean(cv_scores_lgb), linestyle="--")
plt.axhline(np.mean(cv_scores_cat), linestyle="--")
plt.title("CV AUC Stability")
plt.xlabel("Fold")
plt.ylabel("ROC-AUC")


plt.legend()
plt.grid(alpha=0.3)
plt.show()


def rank_avg(preds):
    return np.mean([rankdata(p) for p in preds], axis=0)

test_rank = rank_avg([test_lgb, test_cat])
test_rank = test_rank / test_rank.max()



stack_X = np.vstack([oof_lgb, oof_cat]).T
stack_test = np.vstack([test_lgb, test_cat]).T

meta = LogisticRegression(max_iter=2000)
meta.fit(stack_X, y)

final_pred = meta.predict_proba(stack_test)[:, 1]



submission = pd.read_csv(
    "/kaggle/input/playground-series-s5e12/sample_submission.csv"
)

submission["diagnosed_diabetes"] = final_pred
submission.to_csv("submission.csv", index=False)

submission.head()


