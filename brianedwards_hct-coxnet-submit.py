# %reset -f

# import warnings
# import numpy as np
# import pandas as pd
# from sklearn import set_config
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sksurv.preprocessing import OneHotEncoder
# from sksurv.linear_model import CoxnetSurvivalAnalysis

# from sklearn.model_selection import GridSearchCV, KFold

# warnings.simplefilter('ignore')
# set_config(display='text')

# train_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv')
# train = train_csv.set_index('ID')
# test_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv')
# test = train_csv.set_index('ID')

# X = train.drop(columns=['efs', 'efs_time'])
# Xf = X.select_dtypes('float')
# Xf = Xf.fillna(Xf.median())
# Xc = X.select_dtypes('object').astype('category')
# for col in Xc.columns:
#     Xc[col] = Xc[col].cat.add_categories('unknown')
# Xc = Xc.fillna('unknown')
# Xc = OneHotEncoder().fit_transform(Xc)
# X = pd.concat([Xf, Xc], axis=1)

# y = np.array(list(zip(train['efs'].astype('bool'), train['efs_time'])),
#              dtype=[('efs', 'bool'),
#                     ('efs_time', train['efs_time'].dtype)])

# coxnet_pipe = make_pipeline(StandardScaler(),
#                             CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100))
# coxnet_pipe.fit(X, y)
# estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

# gcv = GridSearchCV(
#     make_pipeline(StandardScaler(),
#                   CoxnetSurvivalAnalysis(l1_ratio=0.9)),
#     param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
#     cv=KFold(n_splits=5, shuffle=True, random_state=0),
#     error_score=0.5,
#     n_jobs=-1,
# ).fit(X, y)

# print(gcv.best_params_)


%reset -f

import io
import csv
import warnings
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxnetSurvivalAnalysis

warnings.simplefilter('ignore')
set_config(display='text')

train_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv')
train = train_csv.set_index('ID')
test_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/test.csv')
test = test_csv.set_index('ID')
d = pd.read_csv('../input/equity-post-HCT-survival-predictions/data_dictionary.csv')
d = d.set_index('variable')

def transform(X):
    Xf = X.select_dtypes('float')
    Xf = Xf.fillna(Xf.median())
    Xc = X.select_dtypes('object').astype('category')
    # for col in Xc.columns:
    #     Xc[col] = Xc[col].cat.add_categories('unknown')
    # Xc = Xc.fillna('unknown')
    for col in Xc.columns:
        values_raw = d.loc[col, 'values']
        values = (values_raw
                  .replace(' nan ', " 'unknown' ")
                  .replace(' nan]', " 'unknown' ")
                  .replace('\n', '')
                  [1:-1])
        string_io = io.StringIO()
        string_io.write(values)
        string_io.seek(0)
        csv_reader = csv.reader(string_io, delimiter=' ', quotechar="'")
        Xc[col] = Xc[col].cat.set_categories(set(next(csv_reader) + ['unknown']))
        Xc[col] = Xc[col].fillna('unknown')
    Xc = OneHotEncoder().fit_transform(Xc)
    return pd.concat([Xf, Xc], axis=1)

fit_X = transform(train.drop(columns=['efs', 'efs_time']))
pred_X = transform(test)

fit_y = np.array(list(zip(train['efs'].astype('bool'), train['efs_time'])),
                 dtype=[('efs', 'bool'),
                        ('efs_time', train['efs_time'].dtype)])

m = make_pipeline(StandardScaler(),
                  CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[0.002290152640140058]))

m.fit(fit_X, fit_y)

submission = pd.DataFrame(m.predict(pred_X),
                          index=pred_X.index,
                          columns=['prediction'])

submission.to_csv('submission.csv')




