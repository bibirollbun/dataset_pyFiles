import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
X = train.drop(['id', 'accident_risk'], axis=1)
y = train['accident_risk']
X_test = test.drop(['id'], axis=1)

# Encode categorical
for col in X.select_dtypes(['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# Feature engineering
for col in X.columns[:6]:
    X[f'{col}_sq'] = X[col]**2
    X_test[f'{col}_sq'] = X_test[col]**2

# Models  
models = {
    'X':XGBRegressor(n_estimators=1500,lr=0.02,max_depth=7,subsample=0.8,random_state=42,tree_method='hist',n_jobs=-1),
    'L':LGBMRegressor(n_estimators=1500,learning_rate=0.02,max_depth=8,subsample=0.8,random_state=42,verbose=-1,n_jobs=-1),
    'C':CatBoostRegressor(iterations=1500,learning_rate=0.02,depth=7,random_seed=42,verbose=0),
    'E':ExtraTreesRegressor(n_estimators=400,max_depth=16,random_state=42,n_jobs=-1)
}

kf = KFold(5,shuffle=True,random_state=42)
oof,tst,scr = {},{},{}

for n,m in models.items():
    print(f'\n{n}:')
    o,t = np.zeros(len(X)),np.zeros(len(X_test))
    for f,(tri,vli) in enumerate(kf.split(X),1):
        m.fit(X.iloc[tri],y.iloc[tri])
        o[vli] = m.predict(X.iloc[vli])
        t += m.predict(X_test)/5
        print(f'F{f}:{np.sqrt(mean_squared_error(y.iloc[vli],o[vli])):.6f}')
    r = np.sqrt(mean_squared_error(y,o))
    print(f'OOF:{r:.6f}')
    oof[n],tst[n],scr[n] = o,t,r

# Ensemble
ti = sum(1/s for s in scr.values())
w = {n:(1/s)/ti for n,s in scr.items()}
print('\n'+'='*50)
for n in w:
    print(f'{n}:w={w[n]:.4f},RMSE={scr[n]:.6f}')

ens_o = sum(oof[n]*w[n] for n in models)
ens_t = sum(tst[n]*w[n] for n in models)
avg_o = np.mean(list(oof.values()),axis=0)
avg_t = np.mean(list(tst.values()),axis=0)

ens_r = np.sqrt(mean_squared_error(y,ens_o))
avg_r = np.sqrt(mean_squared_error(y,avg_o))

print(f'\nWgtEns:{ens_r:.6f}')
print(f'AvgEns:{avg_r:.6f}')

final = ens_t if ens_r<avg_r else avg_t
sub = pd.DataFrame({'id':test['id'],'accident_risk':np.clip(final,0,1)})
print(f'\n{sub.head()}')
sub.to_csv('submission.csv',index=False)
print('\nSaved!') 


