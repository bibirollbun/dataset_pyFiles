%reset -f

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

warnings.simplefilter('ignore')

train = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv'
                   ).set_index('ID')
test = pd.read_csv('../input/equity-post-HCT-survival-predictions/test.csv'
                   ).set_index('ID')

Xt = train.drop(columns=['efs', 'efs_time'])
Xf = Xt.select_dtypes('float')
Xc = Xt.select_dtypes('object').astype('category')
X = pd.concat([Xf, Xc], axis=1)

Xt_test = test
Xf_test = Xt_test.select_dtypes('float')
Xc_test = Xt_test.select_dtypes('object').astype('category')
for col in Xc_test.columns:
    Xc_test[col] = Xc_test[col].cat.set_categories(X[col].cat.categories)
X_test = pd.concat([Xf_test, Xc_test], axis=1)

yt = train[['efs', 'efs_time']]
yt.loc[:, 'score'] = np.log(yt['efs_time'])
yt.loc[yt['efs'] == 0, 'score'] = RobustScaler().fit_transform(yt.loc[yt['efs'] == 0, ['score']])
yt.loc[yt['efs'] == 1, 'score'] = RobustScaler().fit_transform(yt.loc[yt['efs'] == 1, ['score']])
yt.loc[yt['efs'] == 0, 'score'] = yt.loc[yt['efs'] == 0, 'score'] + (4 * yt.loc[yt['efs'] == 1, 'score'].std())
yt.loc[:, 'score']  = RobustScaler().fit_transform(yt.loc[:, ['score']])
y = yt['score']

fit_d = lgb.Dataset(X, y)
params = {'verbosity': -1}
m = lgb.train(params, fit_d)
submission = pd.DataFrame(-m.predict(X_test), index=X_test.index, columns=['prediction'])
submission.to_csv('submission.csv')

