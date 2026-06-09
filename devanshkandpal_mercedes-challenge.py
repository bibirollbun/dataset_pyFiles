!unzip /kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip


!unzip /kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip


import pandas as pd
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn


from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
df.head()
X = df.drop(columns=['ID', 'y'])
y = df['y']

for col in X.columns:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)



def get_cv(X, y, model, verbose=True):
    fit_params={"verbose": False} if verbose else {}
    scores = cross_val_score(model, X, y, cv=5, fit_params=fit_params)
    return scores.mean()

def get_r2_score(X, y_test, model):
    y_pred = model.predict(X)
    ss_res = ((y_test - y_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# these are the manual hp tuned parameters
catboost_model = CatBoostRegressor(silent=True, iterations=170, cat_features=X.columns.tolist(), random_seed=69, depth=4, l2_leaf_reg=3.5, grow_policy='Lossguide', random_strength=0.3)
catboost_model.fit(X, y, verbose=False)
# print(get_cv(X, y, catboost_model))


imp = pd.Series(catboost_model.feature_importances_)
imp.index = X.columns
imp.sort_values(ascending=False, inplace=True)

# use top_n as a hyperparameter
top_n = 130
X_pruned = X.loc[:,imp[:top_n].index]
pruned_model = CatBoostRegressor(iterations=170, cat_features=X_pruned.columns.tolist(), random_seed=69, depth=4, l2_leaf_reg=3.5, grow_policy='Lossguide', random_strength=0.3)
pruned_model.fit(X_pruned, y, verbose=False)

# print('Baseline without pruning:')
# print(n, get_cv(X, y, catboost_model))

# for n in [300, 350]:
#     X_pruned = X.loc[:,imp[:n].index]
#     pruned_model = CatBoostRegressor(iterations=170, cat_features=X_pruned.columns.tolist(), random_seed=69, depth=4, l2_leaf_reg=3.5, grow_policy='Lossguide', random_strength=0.3)
#     print(n, get_cv(X_pruned, y, pruned_model))

# 150 features: 0.57035
# 130: 0.570956


# testing for hp tuning

# print(get_r2_score(y_test, y_pred))
# print(get_cv(X, y, catboost_model))

# for it in [160, 180, 190]:
#     catboost_model = CatBoostRegressor(iterations=it, cat_features=X.columns.tolist(), random_seed=69, depth=4, l2_leaf_reg=3.5, grow_policy='Lossguide', random_strength=0.3)
#     catboost_model.fit(X, y, verbose=False)
#     score = get_cv(X, y, catboost_model)
#     print(f'iter: {it}, score: {score}')



from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

enc = TargetEncoder()
X_enc = enc.fit_transform(X, y)


# HP tuning
# for alpha in range(1, 20):
#     alpha /= 10
#     lasso_model = Lasso(alpha=alpha, random_state=69)
#     print('alpha: ', alpha, get_cv(X_enc, y, lasso_model, verbose=False))

# final model: alpha=0.2, CV=0.5777756
lasso_model = Lasso(alpha=0.2, random_state=69)
lasso_model.fit(X_enc, y)


enc_catboost_model = CatBoostRegressor(silent=True, iterations=400, random_seed=69, depth=6, l2_leaf_reg=10.0, grow_policy='SymmetricTree', random_strength=0.717)
enc_catboost_model.fit(X_enc, y, verbose=False)

# manual hpt
# depth = 6
# iter = 400
# for x in range(350, 450, 10):
#     enc_catboost_model = CatBoostRegressor(iterations=x, random_seed=69, depth=6, l2_leaf_reg=10.0, grow_policy='SymmetricTree', random_strength=0.717)
#     print(x, get_cv(X_enc, y, enc_catboost_model))


# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer

# opt = BayesSearchCV(
#      CatBoostRegressor(random_seed=69, silent=True),
#      {
#          # 'iterations': Integer(50, 1000),
#          'l2_leaf_reg': Real(0.01, 10, prior='log-uniform'),
#          'random_strength': Real(0.01, 10, prior='log-uniform'),
#          'grow_policy': Categorical(['SymmetricTree', 'Depthwise', 'Lossguide']),
#      },
#      n_iter=32,
#      random_state=0,
#  )
# # opt.fit(X_enc, y)
# # opt.best_params_
# # found params:
# # ('grow_policy', 'SymmetricTree'),
# # ('l2_leaf_reg', 10.0),
# # ('random_strength', 0.7172563068730811)])


# TIME TO STACKKK
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline

enc_catboost_pipeline = Pipeline(
    steps=[
        ('encoder', enc),
        ('catboost', enc_catboost_model)
    ]
)

lasso_pipeline = Pipeline(
    steps=[
        ('encoder', enc),
        ('lasso', lasso_model)
    ]
)

estimators = [
    ('catboost', catboost_model),
    ('enc_catboost', enc_catboost_pipeline),
    ('lasso', lasso_pipeline)
]

reg = StackingRegressor(estimators=estimators, cv='prefit', final_estimator=Lasso(random_state=69, alpha=0.2))
reg.fit(X, y)
# print(get_cv(X, y, reg, verbose=False))



# # tuning the stack
# X_enc_train = enc.transform(X_train)
# X_enc_test = enc.transform(X_test)

# t_enc_catboost_model = CatBoostRegressor(silent=True, iterations=400, random_seed=69, depth=6, l2_leaf_reg=10.0, grow_policy='SymmetricTree', random_strength=0.717)
# t_enc_catboost_model.fit(X_enc_train, y_train)
# print('enc_catboost', get_r2_score(X_enc_test, y_test, t_enc_catboost_model))

# t_lasso_model = Lasso(alpha=0.2, random_state=69)
# t_lasso_model.fit(X_enc_train, y_train)
# print('lasso', get_r2_score(X_enc_test, y_test, t_lasso_model))

# t_catboost_model = CatBoostRegressor(silent=True, iterations=170, cat_features=X.columns.tolist(), random_seed=69, depth=4, l2_leaf_reg=3.5, grow_policy='Lossguide', random_strength=0.3)
# t_catboost_model.fit(X_train, y_train)
# print('catboost', get_r2_score(X_test, y_test, t_catboost_model))


# t_enc_catboost_pipeline = Pipeline(
#     steps=[
#         ('encoder', enc),
#         ('catboost', t_enc_catboost_model)
#     ]
# )

# t_lasso_pipeline = Pipeline(
#     steps=[
#         ('encoder', enc),
#         ('lasso', t_lasso_model)
#     ]
# )

# t_estimators = [
#     ('catboost', t_catboost_model),
#     ('enc_catboost', t_enc_catboost_pipeline),
#     ('lasso', t_lasso_pipeline)
# ]

# t_reg = StackingRegressor(estimators=estimators, cv='prefit', final_estimator=Lasso(random_state=69, alpha=0.2))
# t_reg.fit(X_train, y_train)
# print(get_r2_score(X_test, y_test, t_reg))


# submission code
df_sub = pd.read_csv('test.csv')
id_col = df_sub['ID']
X_sub = df_sub.drop(columns=['ID'])

X_pruned_sub = X_sub.loc[:,imp[:top_n].index]
X_enc_sub = enc.transform(X_sub)

for col in X_sub.columns:
    X_sub[col] = X_sub[col].astype('category')

sub = reg.predict(X_sub)
sub = pd.DataFrame(sub)
sub.columns = ['y']
sub.index = id_col
sub.index.name = 'ID'
sub.to_csv('submission.csv')



















