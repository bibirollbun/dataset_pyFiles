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

tr=pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
te=pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")


y=tr['target']

x=tr.drop(['ID_code','target'],axis=1)


# ---------------------------------------------------------
# 2. 결측치 처리 (Imputation)
# ---------------------------------------------------------
print(tr.isnull().sum().sum())


# ---------------------------------------------------------
# 5. 모델링 & 앙상블 (Modeling & Ensemble)
# ---------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

scl=StandardScaler()
xs=scl.fit_transform(x)

lr_mdl=LogisticRegression(solver='liblinear',class_weight='balanced')
lr_mdl.fit(xs,y)
lr_p=lr_mdl.predict_proba(xs)[:,1]
print(f"{roc_auc_score(y,lr_p):.4f}")


xte=te.drop(['ID_code'],axis=1)

# 3.Scaling
xst=scl.transform(xte)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_p=np.zeros(xs.shape[0])
test_p=np.zeros(xst.shape[0])

for n_fold, (tr_i,val_i) in enumerate(folds.split(xs,y)):
    xtr,ytr=xs[tr_i],y[tr_i]
    xval,yval=xs[val_i],y[val_i]

    lr_mdl=LogisticRegression(solver='liblinear',class_weight='balanced')
    lr_mdl.fit(xtr,ytr)

    oof_p[val_i] = lr_mdl.predict_proba(xval)[:,1]

    test_p += lr_mdl.predict_proba(xst)[:,1] / folds.n_splits

    fold_auc = roc_auc_score(yval,oof_p[val_i])
    print(f"Fold {n_fold+1} AUC: {fold_auc:.5f}")

print(f"\nFinal CV AUC: {roc_auc_score(y, oof_p):.5f}")

# submission 파일 생성
submission = pd.DataFrame({
    "ID_code": te["ID_code"],
    "target": test_p
})
submission.to_csv("submission.csv", index=False)


import pandas as pd

# 리더보드 데이터 로드
lb = pd.read_csv("/kaggle/input/leaderboard/leaderboard.csv") # 파일명 확인 필요


# 내 점수(0.86xxx)가 어느 정도 위치인지 확인
my_score = 0.8615
my_rank = lb[lb['Score'] >= my_score].shape[0]

print(f"내 점수({my_score}) 위로 {my_rank}명의 참가자가 있습니다.")
print(f"전체 참가자 수: {len(lb)}")
print(f"상위 {my_rank / len(lb) * 100 :.2f}% 입니다.")

