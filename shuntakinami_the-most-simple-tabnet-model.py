# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install pytorch_tabnet
!pip install tabnet


from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import torch
from sklearn.preprocessing import StandardScaler,OrdinalEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.info()


target = train.y
ID = train.id
train.drop(['id','y'],inplace=True,axis=1)
test.drop('id',inplace=True,axis=1)





train.info()


num_cols = train.select_dtypes(include=['int','float']).columns
cat_cols = train.select_dtypes(include=['object']).columns
print(num_cols,cat_cols)


scaler = StandardScaler()
train[num_cols]=scaler.fit_transform(train[num_cols])
test[num_cols]=scaler.fit_transform(test[num_cols])

encoder = OrdinalEncoder()
train[cat_cols]=encoder.fit_transform(train[cat_cols])
test[cat_cols]=encoder.transform(test[cat_cols])

train


X_train,X_test,y_train,y_test=train_test_split(train,target,test_size=0.2,random_state=77)


tabnet = TabNetClassifier(
    n_d=48,
    n_a=48,
    n_steps=4,                # ステップ数は適度に
    gamma=1.2,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-3,
    momentum=0.98,            # モメンタムでより安定学習
    clip_value=1.0,           # 勾配クリッピング
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(
        lr=1e-2, 
        weight_decay=1e-4,
        eps=1e-8
    ),
    scheduler_params={"T_max": 100, "eta_min": 1e-5},
    scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
    device_name='cuda' if torch.cuda.is_available() else 'cpu',
    seed=42,
    verbose=1
)


tabnet.fit(
    X_train=np.array(X_train),
    y_train=y_train,
    eval_set=[(np.array(X_test),y_test)],
    eval_name=['test'],
    eval_metric=['accuracy'],
    max_epochs=200,
    patience=20,
    batch_size=16384,         # 大きなバッチサイズ
    virtual_batch_size=512,   # 仮想バッチサイズ
    num_workers=4,            # 並列処理
    drop_last=False,
    augmentations=None        # データ拡張は通常不要
)




test


train





sub['y'] = tabnet.predict(test.values)


sub.to_csv('submission.csv',index=False)


sub.head()




