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


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
print(train.shape, '\n')
train.head(5)


print(f'Checking class imbalance of Train:', train['y'].value_counts(normalize=True), '\n')


categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
categorical_cols


fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # 3x3 grid
axes = axes.flatten()  # flatten for easy indexing

for idx, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue="y", data=train, ax=axes[idx])
    axes[idx].set_title(f"{col} vs y")
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


def mean_encode_smooth(train, col, target, alpha=10):
    global_mean = train[target].mean()

    agg = train.groupby(col)[target].agg(['mean','count'])
    smooth = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
    mapping = smooth.to_dict()

    train_enc = train[col].map(smooth).fillna(global_mean)

    return train_enc, mapping

for col in categorical_cols:
    train[f"{col}_mean_enc"], mapping = mean_encode_smooth(train, col, 'y', alpha=10)
    test[f"{col}_mean_enc"] = test[col].map(mapping).fillna(train['y'].mean())
    # Drop original col to save memory
    train.drop(columns=[col], inplace=True)
    test.drop(columns=[col], inplace=True)


y_train = train['y'].values
train.drop(columns=['y'], inplace=True)


import lightgbm as lgb

train_data = lgb.Dataset(train, label=y_train)

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": -1,
    "verbose": -1,
    "device": "gpu",           
    "gpu_platform_id": 0,      
    "gpu_device_id": 0         
}


lgb_cv = lgb.cv(
    params,
    train_data,
    num_boost_round=1000,
    nfold=5,
    callbacks=[lgb.early_stopping(stopping_rounds=100)],
)


best_rounds = len(lgb_cv['valid auc-mean'])

lgb_model = lgb.train(params, train_data, num_boost_round=best_rounds)


# plotting the roc auc curve
from sklearn.metrics import roc_curve, auc

probas = lgb_model.predict(train)

fpr, tpr, thresholds = roc_curve(y_train, probas)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'LightGBM model (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--',
        color='black', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC of the Lightgbm model')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# Now getting test_proba
test_proba = lgb_model.predict(test)


# saving submission
sub = pd.DataFrame({'id': test['id'], 'y': test_proba})
sub.to_csv("submission.csv", index=False)
print("submission.csv saved!")
sub.head()




