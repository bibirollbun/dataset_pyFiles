import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier as RFC
from lightgbm import LGBMClassifier as lgbm
from catboost import CatBoostClassifier as cat
from sklearn.model_selection import cross_val_score, KFold
from sklearn import metrics

import warnings 
warnings.simplefilter('ignore')

seed = 42
n_splits = 6
scorer = 'roc_auc_ovo'

# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
target = 'loan_paid_back'
train.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
test.head()


train.shape


train.info()


train[target].value_counts()


from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

scaler = StandardScaler()


X = train.copy()
X_test = test.copy()

X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

y = X.pop(target)


X_prep = X.copy()

X_prep = scaler.fit_transform(X_prep)


scaler.scale_


scaler.mean_


scaler.var_


models = {
    'rfc_clf': RFC(n_estimators=300),   # RandomForestClassifier
    
    'lgb_clf': lgbm(
        verbose=-1,
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03),  # LightGBM Classifier
    
    'cat_clf': cat(
        iterations=2750,
        verbose=0,
        depth=3,
        eval_fraction=0.3, 
        eval_metric='AUC', 
        early_stopping_rounds=100)  # CatBoost Classifier
}


score_df = pd.DataFrame()
for name, model in models.items():
    score_df[f'{name}_auc'] = cross_val_score(
        model, 
        X, y, scoring=scorer, 
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=seed+8), )

score_df.style.background_gradient(cmap='YlGn')


plt.figure(figsize=(6, 4))
ax = plt.plot(score_df['rfc_clf_auc'], color='steelblue', marker='d', 
              linestyle='--', label='Random Forest')
plt.plot(score_df['lgb_clf_auc'], color='red', marker='o', 
         linestyle='dashdot', label='LightGBM')
plt.plot(score_df['cat_clf_auc'], color='orange', marker='D', 
         linestyle='--', label='CatBoost')
plt.title(f'cv {scorer} with rf and lgbm and cat models')
plt.ylabel('scores')
plt.xlabel('fold')
plt.legend()
plt.show()


plt.figure(figsize=(9,36))

n = 6
spliter = KFold(n_splits=n, shuffle=True, random_state=seed)

test_pred_proba = pd.DataFrame()

for name, model in models.items():
    print('\n' + 'ðŸ§®' + 4*'-' + f' Using {name} model ' + 5*'-')

    for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y), 1):
        # print(15*'--' + f'Training fold {f} of {n}' + 15*'--')
        X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
        y_tr, y_va = y.iloc[tr_ind], y.iloc[va_ind]
        # Fit the model and predict_proba on validation
        clf = model.fit(X_tr, y_tr)
        preds = clf.predict_proba(X_va)[:, 1]
        # Get the acu scores
        score = metrics.roc_auc_score(y_va, preds)
        print('{}_Fold_{} ==> auc: {:.6f}'.format(name, f, score))
        # Predit proba on test data
        test_pred_proba[f'y_test_proba_{name}_fold_{f}'] = clf.predict_proba(X_test)[:, 1]

        # Plot the roc_curve of the models predictions
        plt.subplot(10, 2, f)
        tpr, fpr, _  = metrics.roc_curve(y_va, preds)
        plt.plot(tpr, fpr, label='auc with {} = {:.5f}'.format(name, score))
        plt.plot([0, 1], [0, 1], color='maroon')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title(f'roc_curves for fold_{f} on {len(preds)} candidates', 
                  color='maroon', fontsize=11, weight='bold')
    plt.tight_layout(pad=2, h_pad=3, w_pad=3)

display(test_pred_proba)


subm = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

subm[target] = test_pred_proba.iloc[:, 2:].mean(axis=1) # Average best performing Folds

subm.head()


subm.to_csv('submission.csv', index=False)

