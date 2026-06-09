%%time

import numpy as np, pandas as pd
from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
import numpy as np

from sklearn.model_selection import *
from tqdm import tqdm

def print_heading(title):
    print("*" * 50)
    print(f" {title} ")
    print("*" * 50)


%%time

SEED = 0
n_splits = 10

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv').drop('id', axis=1)
original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')

original = original.rename(columns={'pressure ': 'pressure', 'humidity ': 'humidity', 'cloud ': 'cloud',
                                    '         winddirection':'winddirection'})
original = original[train.columns].replace({'yes': 1, 'no': 0})
train = pd.concat([train, original], axis=0, ignore_index=True)

test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

def feature_engineering(df):
    
    df['cloud_humidity'] = df.cloud + df.humidity
    df['cloud_humidity_sunshine'] = df.cloud + df.humidity + df.sunshine
    df['cloud_sunshine'] = df.cloud * df.sunshine
    df['humidity_sunshine'] = df.humidity * df.sunshine
    df['temp_diff']=df['maxtemp']-df['mintemp']
    
    for c in ['pressure', 'maxtemp', 'temparature', 'humidity']:
        for gap in [1]:
            df[c+f"_shift{gap}"] = df[c].shift(gap)
            df[c+f"_diff{gap}"] = df[c].diff(gap)
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)


%%time

train.head()


%%time

y = train['rainfall']
X = train.drop('rainfall',axis=1)


%%time

def TRAIN_CAT(p,X_test):
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"ðŸš€ Training Fold {fold + 1}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**p,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type='CPU',
            cat_features=None,
            random_seed=SEED,
            random_strength=0,
            verbose=False
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

        train_score = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        val_score = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"âœ… Fold {fold + 1} - Train AUC: {train_score:.5f}, Val AUC: {val_score:.5f}")

    final_auc = roc_auc_score(y, oof_preds)
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    clear_output()
    print(f"ðŸ”¥ Overall Train AUC: {train_auc:.5f}")
    print(f"ðŸŽ¯ Overall OOF AUC: {final_auc:.5f}")

    return oof_preds, test_preds


%%time

params = {'n_estimators': 531, 'max_depth': 4, 'learning_rate': 0.173398811247479}

oof_preds, test_preds = TRAIN_CAT(params, test)


%%time

sample["rainfall"] = test_preds
sample.to_csv("submission.csv", index=False)
print_heading("Sub shape:")
print(sample.shape)
sample.head()

