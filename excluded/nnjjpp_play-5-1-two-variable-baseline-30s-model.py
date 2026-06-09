import pandas as pd 
import numpy as np


SEED = 777


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', index_col='id')
comp = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

target = 'num_sold'


from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
linear_reg = TransformedTargetRegressor(SGDRegressor(random_state = SEED),
                                        func=np.log, inverse_func=np.exp)

pipe = Pipeline([('encode', OneHotEncoder(sparse_output=False,
                                          handle_unknown='ignore')),
                 ('interact', PolynomialFeatures(degree=2,
                                                 interaction_only=True)),
                 ('mod', linear_reg)]).set_output(transform='pandas')


nm_ix = np.bitwise_not(train[target].isna())
X = train.loc[nm_ix,:].drop(target, axis=1)[['country', 'product']]
y = train.loc[nm_ix, target]

year = train.loc[nm_ix,:]['date'].map(lambda x: int(x[:4]))


from sklearn.model_selection import cross_validate, GroupKFold
mape_cv = cross_validate(estimator = pipe, X=X, y=y,
               cv = GroupKFold(n_splits=7),
               groups = year, 
               scoring = 'neg_mean_absolute_percentage_error')['test_score'].mean()
print(f'CV Estimate of MAPE is {-mape_cv:.3f}')


pipe.fit(X,y)


sub['num_sold'] = pipe.predict(comp[['country', 'product']])
sub.to_csv('submission.csv', index=False)

