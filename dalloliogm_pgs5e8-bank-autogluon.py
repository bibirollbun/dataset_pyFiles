# %%capture
# !pip install -U autogluon


%%capture
!pip install -q xgboost==1.7.6 scikit-learn==1.3.2


%%capture
import os
import shutil
import seaborn as sns

target_dir = "/kaggle/working/"
source_dir = "/kaggle/input/autogluon-package/"

if not os.path.exists(os.path.join(target_dir, "autogluon")):
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    print("Copied autogluon installation files.")

!pip install -f --quiet --no-index --find-links='/kaggle/input/autogluon-package' 'autogluon.tabular-1.3.1-py3-none-any.whl'



# Autogluon configuration. Automatically detects if we are using an interactive notebook, and use lower defaults when debugging
import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

def is_interactive_session():
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Interactive'

is_interactive_session()

config = {
    #                  minutes     seconds       # hours
    "autogluon_time": 60       *  60        *   8,
    "autogluon_preset": "best_quality",
    #"reduce_features": 0, # Set to >0 to use only the first n features
    "tail_rows": 0 # Set to >0 to use only the last n rows in the file. Useful for debugging
    
}

if is_interactive_session():
    print("Interactive session")
    config["autogluon_time"] = 100
    #config["reduce_features"] = 200
    config["autogluon_preset"] = "medium_quality"
    config["tail_rows"] = 2000
    print(config)
else:
    print("running as job")
    print(config)


RANDOM_STATE = 666
np.random.seed(RANDOM_STATE)


hyperparameters = {
    'GBM': {},      # LightGBM
    'CAT': {},      # CatBoost
    'RF': {},       # RandomForest
    'XT': {},       # ExtraTrees
    'KNN': {},      # k-NN
    # 'NN': {},       # MXNet neural-net
    'LR': {},       # LinearModel
    'XGB': {        # XGBoost 
    },
    'TABPFN': {},
    'ENS_WEIGHTED' : {},
    "SIMPLE_ENS_WEIGHTED": {},
    'NN_TORCH': {},
    'IM_BOOSTEDRULES' : {},
    'IM_RULEFIT': {}
}

# 'RF', 'XT', 'KNN', 'GBM', 'CAT', 'XGB', 'NN_TORCH', 'LR', 'FASTAI', 'AG_TEXT_NN', 'AG_IMAGE_NN', 
# 'AG_AUTOMM', 'FT_TRANSFORMER', 'TABPFN', 'TABPFNMIX', 'FASTTEXT', 'ENS_WEIGHTED', 'SIMPLE_ENS_WEIGHTED',
# 'IM_RULEFIT', 'IM_GREEDYTREE', 'IM_FIGS', 'IM_HSTREE', 'IM_BOOSTEDRULES', 'DUMMY']


import pandas as pd
# data
path        = '/kaggle/input/playground-series-s5e8/'
train       = pd.read_csv(path + 'train.csv',             index_col = 'id')
test        = pd.read_csv(path + 'test.csv',              index_col = 'id')
submission  = pd.read_csv(path + 'sample_submission.csv', index_col = 'id')

# Reduce dataset size for debugging
if config["tail_rows"] > 0:
    train = train.head(config["tail_rows"])


train.head()


test.head()


submission.head





import numpy as np
import pandas as pd
import random

def engineer_features(df, seed=42, max_combinations=5, verbose=True):
    df = df.copy()
    random.seed(seed)
    np.random.seed(seed)

    created_features = []

    # --- 1. Domain-based feature engineering ---

    # Convert month string to number
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    if 'month' in df.columns:
        df['month_num'] = df['month'].map(month_map)
        created_features.append('month_num')

    # Flag: previously contacted
    if 'pdays' in df.columns:
        df['previously_contacted'] = (df['pdays'] != -1).astype(int)
        created_features.append('previously_contacted')

    # Bucket balance into quartiles
    if 'balance' in df.columns and df['balance'].nunique() > 4:
        df['balance_bucket'] = pd.qcut(df['balance'], q=4, labels=False, duplicates='drop')
        created_features.append('balance_bucket')

    # Interaction: job + education
    if 'job' in df.columns and 'education' in df.columns:
        df['job_edu'] = df['job'].astype(str) + "_" + df['education'].astype(str)
        created_features.append('job_edu')

    # Campaign transforms
    if 'campaign' in df.columns:
        df['log_campaign'] = np.log1p(df['campaign'])
        df['is_first_contact'] = (df['campaign'] == 1).astype(int)
        created_features += ['log_campaign', 'is_first_contact']

    # Frequency encoding
    for col in ['job', 'poutcome']:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq)
            created_features.append(f'{col}_freq')

    # --- 2. Randomized combinations ---

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns='y', errors='ignore').columns.tolist()

    # Random categorical pair interactions
    cat_pairs = random.sample(
        [(a, b) for i, a in enumerate(categorical_cols) for b in categorical_cols[i+1:]],
        k=min(max_combinations, len(categorical_cols) * (len(categorical_cols) - 1) // 2)
    )
    for a, b in cat_pairs:
        new_col = f"{a}_{b}_combo"
        df[new_col] = df[a].astype(str) + "_" + df[b].astype(str)
        created_features.append(new_col)

    # Random numeric pair ratios
    num_pairs = random.sample(
        [(a, b) for i, a in enumerate(numerical_cols) for b in numerical_cols[i+1:] if a != b],
        k=min(max_combinations, len(numerical_cols) * (len(numerical_cols) - 1) // 2)
    )
    for a, b in num_pairs:
        new_col = f"{a}_over_{b}"
        with np.errstate(divide='ignore', invalid='ignore'):
            df[new_col] = df[a] / df[b]
            df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan).fillna(0)
        created_features.append(new_col)

    # Random numeric log transforms
    for col in random.sample(numerical_cols, min(len(numerical_cols), max_combinations)):
        df[f'{col}_log'] = np.log1p(df[col])
        created_features.append(f'{col}_log')

    if verbose:
        print(f"âœ… Created {len(created_features)} engineered features.")

    return df



train_fe = engineer_features(train)
test_fe = engineer_features(test)





%%capture

predictor = TabularPredictor(
    label='y',
    problem_type='binary',
    eval_metric="roc_auc",
).fit(
    train_fe,
    hyperparameters=hyperparameters,
    time_limit=config["autogluon_time"],
    presets=config["autogluon_preset"],
    # ag_args_fit={'num_gpus': 1},
    verbosity=1,
    # keep_test_data=True
)


predictor.leaderboard()


import matplotlib.pyplot as plt

# Compute and plot feature importance
importance_df = predictor.feature_importance(data=train_fe)



top_features = importance_df.head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, x='importance', y=top_features.index)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()


%%capture
preds = predictor.predict_proba(test_fe)
preds


preds = predictor.predict_proba(test_fe)[1]
submission['y'] = preds
submission = submission.reset_index()
submission.to_csv('submission.csv', index=False)





sns.histplot(train_fe.y).set_title("Observed Y")


import seaborn as sns
sns.histplot(submission.y).set_title("Predicted Y")


predictor.fit_summary()



from sklearn.metrics import roc_auc_score

def check_overfitting(predictor, train_data, val_data=None, threshold=0.05):
    """
    Compare training and validation performance to detect overfitting.
    
    If bagging was used, computes out-of-fold (OoF) ROC AUC.
    If no val_data is provided, assumes OoF comparison only.
    """
    y_true = train_data['y']

    # Get train ROC AUC
    train_score = predictor.evaluate(train_data, silent=True)['roc_auc']

    if predictor._trainer.bagged_mode:
        # Use out-of-fold predictions for fair comparison
        oof_preds = predictor.get_oof_pred_proba(as_multiclass=False)[1]
        oof_score = roc_auc_score(y_true, oof_preds)
        score_gap = train_score - oof_score
        val_label = "Out-of-Fold"
    elif val_data is not None:
        val_score = predictor.evaluate(val_data, silent=True)['roc_auc']
        score_gap = train_score - val_score
        oof_score = val_score
        val_label = "Validation"
    else:
        print("â�— No validation data or bagging enabled. Can't assess overfitting.")
        return None

    print(f"Train ROC AUC:  {train_score:.4f}")
    print(f"{val_label} ROC AUC: {oof_score:.4f}")
    print(f"Gap:            {score_gap:.4f}")

    if score_gap > threshold:
        print("âš ï¸� Overfitting likely (gap > {:.2f})".format(threshold))
        return True
    elif score_gap < 0:
        print("âœ… Generalization is good (val > train)")
        return False
    else:
        print("ğŸ‘Œ Acceptable generalization")
        return False
#check_overfitting(predictor, train_fe)



from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Get predicted probabilities (for class 1)
y_true = train_fe['y']
y_pred = predictor.predict_proba(train_fe.drop(columns=['y']))[1]

# Step 2: Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred)

# Step 3: Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()



if not is_interactive_session():
    os.system("rm -rf /kaggle/working/*")
    print("Clean-up complete: deleted files from /kaggle/working/")
else:
    print("Not an interactive session. Skipping clean-up.")


# Save submission
submission.to_csv('submission.csv', index=False)

