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


tr=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
te=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# ---------------------------------------------------------
# Magic Feature: Frequency Encoding (빈도수 인코딩)
# ---------------------------------------------------------

# 원본 데이터 복사 (안전을 위해)
tr_magic = tr.copy()
te_magic = te.copy()

# 학습 데이터와 테스트 데이터를 합쳐서 세야 정확합니다! (중요)
dt = pd.concat([tr_magic,te_magic], axis=0)

features = [col for col in tr.columns if col.startswith('var')]

# 3. 임시 저장소 만들기 (리스트)
train_new_cols = []
test_new_cols = []

# 4. Loop: 리스트에 담기만 함 (DataFrame 건드리지 않음)
for col in features:
    count_map = dt[col].value_counts()
    
    # map 결과를 Series로 만들고, 이름(name)을 지정해서 리스트에 추가
    train_new_cols.append(tr_magic[col].map(count_map).rename(f'{col}_count'))
    test_new_cols.append(te_magic[col].map(count_map).rename(f'{col}_count'))

# 5. 한 번에 합치기 (concat) - 여기서 경고 해결!
tr_magic = pd.concat([tr_magic] + train_new_cols, axis=1)
te_magic = pd.concat([te_magic] + test_new_cols, axis=1)

print(tr_magic.shape)


tr_magic.head()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

y=tr_magic['target']
x=tr_magic.drop(['target','ID_code'],axis=1)
xte=te_magic.drop(['ID_code'],axis=1)


ss=StandardScaler()
xs=ss.fit_transform(x)
xtes=ss.transform(xte)


# Model Learning
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lr=np.zeros(len(x))
te_p_lr = np.zeros(len(xte))

for tr_i,val_i in folds.split(x,y):
    xtr,ytr=xs[tr_i],y.iloc[tr_i]
    xval,yval=xs[val_i],y.iloc[val_i]

    clf = LogisticRegression(solver='liblinear',class_weight='balanced', C=1.0, penalty='l2')
    clf.fit(xtr,ytr)

    oof_lr[val_i] = clf.predicta_proba(xval)[:,1]
    te_p_lr += clf.predicta_proba(xtes)[:,1]/folds.n_splits

sub_lr = pd.DataFrame({"ID_code": te["ID_code"], "target":te_p_lr})
sub_lr.to_csv('submission.csv',index=False)


import pandas as pd

# 리더보드 데이터 로드
lb = pd.read_csv("/kaggle/input/leaderboard/leaderboard.csv") # 파일명 확인 필요


# 내 점수(0.86xxx)가 어느 정도 위치인지 확인
my_score = 0.8661
my_rank = lb[lb['Score'] >= my_score].shape[0]

print(f"내 점수({my_score}) 위로 {my_rank}명의 참가자가 있습니다.")
print(f"전체 참가자 수: {len(lb)}")
print(f"상위 {my_rank / len(lb) * 100 :.2f}% 입니다.")

