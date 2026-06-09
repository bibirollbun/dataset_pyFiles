import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
original = pd.read_csv('/kaggle/input/original/original.csv')


original.columns


original


# labelleri duzelt
original['rainfall'] = original['rainfall'].map({'yes': 1, 'no': 0})
original.columns = original.columns.str.strip()
# gunu yila cevir
original['day'] = range(1, len(original) + 1)
# birlestir
train = pd.concat([train, original], ignore_index=True)
# id sutununu duzelt
train['id'] = train.index
train = train[:-1]


# kullanilacak feature ve targetler
remove_columns = ['id', 'rainfall']
features = [col for col in train.columns if col not in remove_columns]
target = 'rainfall'
true = train[target]
print('Features: ', features)
print('Target: ', target)


# group k fold icin yil sutunu
train['year'] = ((train.index) // 365) + 1
train.head()


groups = train['year'].sort_values(ascending=False).values
print(np.unique(groups))
n_groups = len(np.unique(groups))
gkf = GroupKFold(n_splits=n_groups)


def train_xgb_gkf():
    fold_scores = []
    oof_probs = np.zeros(len(train))
    test_probs = np.zeros(len(test))

    X = train[features]
    y = train[target]
    
    for i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        print(f"Fold {i+1}, training years: {train['year'].iloc[train_idx].unique()}, validation years: {train['year'].iloc[val_idx].unique()}")
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        model = XGBClassifier(
            max_depth=3,
            colsample_bytree=0.9,
            subsample=0.9,
            n_estimators=10000,
            learning_rate=0.01,
            eval_metric='auc',
            early_stopping_rounds=100,
            alpha=1,
            random_state=2,
            device='cuda'
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]
        print(f"Fold {i+1} Train ROC AUC: {roc_auc_score(y_train, train_probs)}")
        print(f"Fold {i+1} Val ROC AUC: {roc_auc_score(y_val, val_probs)}")

        fold_scores.append(roc_auc_score(y_val, val_probs))
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(test[features])[:, 1]
    
    return np.mean(fold_scores), roc_auc_score(true, oof_probs), test_probs / gkf.n_splits

xgb_cv_score, xgb_oof_score, xgb_test_probs = train_xgb_gkf()
print(f'XBG CV score: {xgb_cv_score}')
print(f'XGB OOF score: {xgb_oof_score}')


!pip3 install tabpfn


from tabpfn import TabPFNClassifier

def train_tabpfn_gkf():
    fold_scores = []
    oof_probs = np.zeros(len(train))
    test_probs = np.zeros(len(test))

    X = train[features]
    y = train[target]

    for i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
        print(f"Fold {i+1}, training years: {train['year'].iloc[train_idx].unique()}, validation years: {train['year'].iloc[val_idx].unique()}")
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        model = TabPFNClassifier(device='cuda')
        model.fit(X_train, y_train)
    
        train_probs = model.predict_proba(X_train)[:,1]
        val_probs = model.predict_proba(X_val)[:,1]
        print(f"Fold {i+1} Train ROC AUC: {roc_auc_score(y_train, train_probs)}")
        print(f"Fold {i+1} Val ROC AUC: {roc_auc_score(y_val, val_probs)}")
    
        fold_scores.append(roc_auc_score(y_val, val_probs))
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(test[features])[:, 1]

    return np.mean(fold_scores), roc_auc_score(true, oof_probs), test_probs / gkf.n_splits

tab_cv_score, tab_oof_score, tab_test_probs = train_tabpfn_gkf()
print(f'TabPFN CV score: {tab_cv_score}')
print(f'TabPFN OOF score: {tab_oof_score}')


from cuml.svm import SVC, LinearSVC


import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test['winddirection'].fillna(test['winddirection'].median(), inplace=True)
train['year'] = ((train.index) // 365) + 1


remove_columns = ['id', 'rainfall','year']
features = [col for col in train.columns if col not in remove_columns]
true = train[target]


m = train.rainfall.mean()
for c in features:
    n = f'{c}2'
    train[n] = train[c].map(original.groupby(c).rainfall.mean())
    train[n] = train[n].fillna(m)

for c in features:
    n = f'{c}2'
    test[n] = test[c].map(original.groupby(c).rainfall.mean())
    test[n] = test[n].fillna(m)

print(f'Train shape, {train.shape}')
print(f'Test shape, {test.shape}')


original.groupby('winddirection').rainfall.mean()


train.head()


features = [col for col in train.columns if col not in remove_columns]
print(features)
print(len(features))


groups = train['year'].sort_values(ascending=False).values
print(np.unique(groups))
n_groups = len(np.unique(groups))
gkf = GroupKFold(n_splits=n_groups)


def train_svc_gkf():
    fold_scores = []
    oof_probs = np.zeros(len(train))
    test_probs = np.zeros(len(test))
    
    X = train[features]
    y = train[target]
    for i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
        print(f"Fold {i+1}, training years: {train['year'].iloc[train_idx].unique()}, validation years: {train['year'].iloc[val_idx].unique()}")
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        # scale
        for c in features:
            m = X_train[c].mean()
            s = X_train[c].std()
            X_train[c] = (X_train[c]-m)/s
            X_val[c] = (X_val[c]-m)/s
            test[c] = (test[c]-m)/s
        
        model = SVC(C=0.1, kernel='poly', degree=1, probability=True)
        model.fit(X_train, y_train)
        
        train_probs = model.predict_proba(X_train.values)[:, 1]
        val_probs = model.predict_proba(X_val.values)[:, 1]
        
        print(f"Fold {i+1} Train ROC AUC: {roc_auc_score(y_train, train_probs)}")
        print(f"Fold {i+1} Val ROC AUC: {roc_auc_score(y_val, val_probs)}")
        
        fold_scores.append(roc_auc_score(y_val, val_probs))
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(test[features].values)[:, 1]

    return np.mean(fold_scores), roc_auc_score(true, oof_probs), test_probs / gkf.n_splits

svc_cv_score, svc_oof_score, svc_test_probs = train_svc_gkf()
print(f'SVC CV score: {svc_cv_score}')
print(f'SVC OOF score: {svc_oof_score}')
        


cv_scores_df = pd.DataFrame(columns = ['xgb', 'tabpfn', 'svc'])
cv_scores_df['xgb'] = [xgb_cv_score]
cv_scores_df['tabpfn'] = tab_cv_score
cv_scores_df['svc'] = svc_cv_score
cv_scores_df['avg_cv_score'] = (xgb_cv_score + tab_cv_score + svc_cv_score) / 3
cv_scores_df


submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


xgb_sub = submission.copy()
xgb_sub['rainfall'] = xgb_test_probs
xgb_sub.to_csv('/kaggle/working/xgb_sub.csv', index=False)


tabpfn_sub = submission.copy()
tabpfn_sub['rainfall'] = tab_test_probs
tabpfn_sub.to_csv('/kaggle/working/tabpfn_sub.csv', index=False)


svc_sub = submission.copy()
svc_sub['rainfall'] = svc_test_probs
svc_sub.to_csv('/kaggle/working/svc_sub.csv', index=False)


ensemble_sub = submission.copy()
ensemble_sub['rainfall'] = (xgb_test_probs + tab_test_probs + svc_test_probs) / 3
ensemble_sub.to_csv('/kaggle/working/ensemble_sub.csv', index=False)


xgb_tabpfn_sub = submission.copy()
xgb_tabpfn_sub['rainfall'] = (xgb_test_probs + tab_test_probs) / 2
xgb_tabpfn_sub.to_csv('/kaggle/working/xgb_tappfn_sub.csv', index=False)

