# ✅ Install compatible versions of required libraries
!pip install -q scikit-learn==1.3.2 autogluon.tabular ray==2.10.0

# ✅ Restart the kernel/runtime after running this cell (for local runtime only)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from autogluon.tabular import TabularPredictor
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')


class CFG:
    train_path = "/kaggle/input/burnout-datathon-ieeecsmuj/train.csv"
    test_path = "/kaggle/input/burnout-datathon-ieeecsmuj/test.csv"
    sample_sub_path = "/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv"
    val_path="/kaggle/input/burnout-datathon-ieeecsmuj/val.csv"
    
    target = 'Lap_Time_Seconds'
    n_folds = 5
    seed = 1859
    time_limit = 3600 * 2  # adjust if needed


# Load data
train = pd.read_csv(CFG.train_path)
test = pd.read_csv(CFG.test_path)
val = pd.read_csv(CFG.val_path)

train = pd.concat([train, val], ignore_index=True)


train[CFG.target] = train[CFG.target].astype(float)

# Create fold column (not needed by AutoGluon but useful for tracking)
kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
for i, (_, val_idx) in enumerate(kf.split(train)):
    train.loc[val_idx, 'fold'] = i

# Initialize TabularPredictor for regression
predictor = TabularPredictor(
    label=CFG.target,
    problem_type='regression',
    eval_metric='rmse',
    verbosity=2
)

predictor.fit(
    train_data=train.drop(columns=["fold"]),
    time_limit=CFG.time_limit,
    presets='best_quality',
    ag_args_fit={
        'num_gpus': 1,
        'num_cpus': 4
    },
    # Disable stacking
    use_bag_holdout=False,     # no internal holdout set for bagging
    num_bag_folds=0,           # disables bagging
    num_stack_levels=0         # disables stacking
)


predictor.leaderboard(silent=True).style.background_gradient(subset=['score_val'], cmap='RdYlGn')


cols_to_drop = "Lap_Time_Seconds"

# Drop target column from test if present
if cols_to_drop in test.columns:
    test.drop(columns=cols_to_drop, inplace=True)

# Predict with AutoGluon predictor
pred = predictor.predict(test)

# Add predictions back to test
test["Lap_Time_Seconds"] = pred

# Save submission file
submission = test[["Lap_Time_Seconds"]]
submission.to_csv("submission_autogluon.csv", index=False)
print("✅ Submission file saved as 'submission_autogluon.csv'")

