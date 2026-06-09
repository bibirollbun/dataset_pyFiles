# ✅ Install compatible versions of required libraries
!pip install -q scikit-learn==1.3.2 autogluon.tabular ray==2.10.0

# ✅ Restart the kernel/runtime after running this cell (for local runtime only)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import pickle
import shutil
import os

warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e6/sample_submission.csv'
    original_data_path = '/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv'
    
    target = 'Personality'
    n_folds = 5
    seed = 1859
    time_limit = 3600 * 5


train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')
original = pd.read_csv(CFG.original_data_path)

skf = StratifiedKFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True)
split = skf.split(train, train[CFG.target])
for i, (_, val_index) in enumerate(split):
    train.loc[val_index, 'fold'] = i

predictor = TabularPredictor(
    log_file_path='logs.txt',
    log_to_file=True,
    problem_type='multiclass',
    eval_metric='accuracy',
    label=CFG.target,
    groups='fold',
    verbosity=2
)


predictor.fit_pseudolabel(
    train_data=train,
    pseudo_data=original,
    time_limit=CFG.time_limit,
    presets='best_quality',
    hyperparameters={
        'GBM': {'ag_args': {'name_suffix': 'LightGBM'}}
    },
    ag_args_fit={
        'excluded_model_types': ['XGBoost'],
        'num_gpus': 1,
        'num_cpus': 4
    }
)


predictor.leaderboard(silent=True).style.background_gradient(subset=['score_val'], cmap='RdYlGn')


pred = predictor.predict(test)


test["Personality"]=pred


submission = test[["Personality"]].reset_index()  # reset to bring back 'id' column
submission.to_csv("submission_autogluon.csv", index=False)
print("✅ MAP@3 submission file saved as 'submission_autogluon_map3.csv'")

