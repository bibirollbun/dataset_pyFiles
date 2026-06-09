


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
ss = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train


X_train = train.drop(columns=['id', 'y'])
y_train = train['y']



X_train_cat = X_train.select_dtypes(exclude=['float', 'int'])
X_train_num = X_train.select_dtypes(include=['float', 'int'])


X_test = test.drop(columns=['id'])
ids = test['id']


X_test_cat = X_test.select_dtypes(exclude=['float', 'int'])
X_test_num = X_test.select_dtypes(include=['float', 'int'])




X_train.columns


for i in range(len(X_train_cat.columns)):
    print(f'{X_train_cat.columns[i]}: {X_train_cat[X_train_cat.columns[i]].unique()}')




test.isna().sum()


X_train_num.describe()


X_test_num.describe()


cat_concat = pd.concat([X_train_cat, X_test_cat], axis=0)
cat_concat_dum = pd.get_dummies(cat_concat)



cat_concat_dum


X_train_dum = cat_concat_dum[:X_train_cat.shape[0]]
X_test_dum = cat_concat_dum[X_train_cat.shape[0]:]



def data_process(df):
    df = df.copy()
    df['log_duration'] = np.log1p(df['duration'])
    df['log_age'] = np.log1p(df['age'])
    
    return df




X_train_q = pd.concat([X_train_num, X_train_dum], axis=1)
X_test_q = pd.concat([X_test_num, X_test_dum], axis=1)

X_train_f = data_process(X_train_q)
X_test_f = data_process(X_test_q)


X_train_f


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.hist(train['age'], bins=30, edgecolor='black', alpha=0.7)

plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.hist(train['duration'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Duration")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplot(2, 2, 3)
plt.hist(train['balance'], bins=90, edgecolor='black', alpha=0.7)
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.hist(train['campaign'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("campaign")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


max(train['balance'])


X_train_num["random_int"] = np.random.randint(0, 100, size=len(X_train_num))


corr_matrix = pd.concat([X_train_num, y_train, ], axis=1).corr()


sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")





pd.concat([X_train['age'], X_train['campaign'], X_train['balance'], X_train['duration']], axis=1).plot.box()


#from pandas.plotting import scatter_matrix

#scatter_matrix(pd.concat([X_train['age'], X_train['campaign'], X_train['balance'], X_train['duration']], axis=1), alpha=0.2, figsize=(6, 6), diagonal="kde");


plt.figure(figsize=(13, 6))
plt.subplot(2, 2, 1)
sns.scatterplot(
    x=X_train_f['duration'],
    y=X_train_f['balance'],
    hue=y_train,
    palette='viridis',
    alpha=0.8,
    s=100
)
plt.title('duration / balance')



plt.subplot(2, 2, 2)
sns.scatterplot(
    x=X_train_f['duration'],
    y=X_train_f['age'],
    hue=y_train,
    palette='magma',
    alpha=0.8,
    s=100
)
plt.title('duration / age')



plt.subplot(2, 2, 3)
sns.scatterplot(
    x=X_train_f['log_duration'],
    y=X_train_f['log_age'],
    hue=y_train,
    palette='crest',
    alpha=0.8,
    s=100
)
plt.title('log duration / log age')



plt.subplot(2, 2, 4)
sns.scatterplot(
    x=X_train_f['age'],
    y=X_train_f['balance'],
    hue=y_train,
    palette="flare",
    alpha=0.8,
    s=100
)
plt.title('age / balance')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression




#X_train_v, X_val, y_train_v, y_val = train_test_split(X_train_f, y_train, test_size=0.2)


#X_val


'''import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score'''


'''
import optuna

def objective(trial):
    params = {
        'objective': 'binary',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),

    }
    model_LGBM = LGBMClassifier(**params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    score = cross_val_score(model_LGBM, X=X_train_f, y=y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()

    return score
 '''   



'''study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)'''



# optuna.visualization.plot_param_importances(study) 





final_model = LGBMClassifier(
        n_estimators=20000,
        learning_rate=0.06,
        num_leaves=100,
        max_depth=10,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.79,
        reg_lambda=3.0,
        max_bin=4523,
        random_state=42,
    verbosity=-1
    )

final_model.fit(X_train_f, y_train)
pred_LGBM = final_model.predict_proba(X_test_f)


from lightgbm import plot_importance


plt.figure(figsize=(12, 8))
plot_importance(final_model, max_num_features=20, importance_type='split')
plt.title('Feature Importance (Split)')
plt.show()






#model_LGBM.fit(X_train_v, y_train_v)
#pred_LGBM = model_LGBM.predict(X_val)
#accuracy_score(y_val, pred_LGBM)
# 0.9337133333333333






ss


pd.DataFrame({'id': ids, 'y': pred_LGBM[:, 1] }).to_csv('submission.csv', index=False)

