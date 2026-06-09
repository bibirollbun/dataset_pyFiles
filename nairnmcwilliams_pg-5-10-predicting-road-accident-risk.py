# Set run_submission to true if running for a submission to the competition.
# Otherwise, we reduce the number of iterations on some of the slower processes by an order of magnitude.
run_debug = False
run_submission = True

seed = 1000 if run_submission else 0


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


dir_name = '/kaggle/input/playground-series-s5e10'

df_sample = pd.read_csv(f'{dir_name}/sample_submission.csv', index_col='id')

df_train = pd.read_csv(f'{dir_name}/train.csv', index_col='id')
df_test = pd.read_csv(f'{dir_name}/test.csv', index_col='id')


target_feature = 'accident_risk'

df_train.drop(target_feature, axis=1).isnull().sum() + df_test.isnull().sum()


df_train.head(10)


binary_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']

# Combine the training and test datasets so that we can one-hot encode the MTRANS feature in both sets at the same time.
df_full = pd.concat([df_train, df_test])

print('Categorical data in full dataset')
for feature in categorical_features:
    print(f"Categories in feature {feature} = {[i for i in df_full[feature].unique()]}") 

# Perform the one-hot encoding, assigning each category a binary 0/1 value.
df_full = pd.get_dummies(df_full, columns=binary_features + categorical_features, drop_first=True, dtype=int)
df_full["night_and_fog"] = (df_full["lighting_night"] & df_full["weather_foggy"]).astype(int)

df_full.head(10)


# Separate the training and test sets using the indices of each.
df_train = df_full.loc[df_train.index]
df_test = df_full.loc[df_test.index]
df_test = df_test.drop(target_feature, axis=1)


from sklearn.model_selection import train_test_split

X = df_train.copy()
y = df_train[target_feature].copy()
X = X.drop(target_feature, axis=1)

X_test = df_test.copy()

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=seed)


def plot_correlation_matrix(X, y):
    X_y = X.join(y, on='id')
    
    _, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(X_y.corr(),
                cmap='RdBu',
                annot=True,
                fmt=".2f",
                ax=ax)
    plt.show()

plot_correlation_matrix(X, y)


from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(root_mean_squared_error)

def create_base_model():
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'n_estimators': 1000 if run_submission else 100,
        'random_state': seed,
        'early_stopping_rounds': 50
    }
    
    return XGBRegressor(**params)

def score_model(X, y, model=None, print_score=True, return_model=False):
    Xt, Xv, yt, yv = train_test_split(X, y, random_state=seed)
    _model = create_base_model() if model is None else model
    
    cv = 5
    cv_score = cross_val_score(_model, Xt, yt, scoring=rmse_scorer, cv=cv, fit_params={
        'eval_set': [(Xv, yv)],
        'verbose': run_debug
    })
    
    if run_debug:
        print(f'Cross validation scores from latest run = {cv_score}.')
    
    _score = cv_score.mean()
    
    if print_score:
        print(f'Baseline CV score with {type(_model)} model = {_score:.5f} ± {cv_score.std()/np.sqrt(cv):.5f}.')
    
    if return_model:
        return _score, _model
    return _score


base_score, base_model = score_model(X, y, return_model=True)


from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [1000] if run_submission else [100],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [0.5, 1, 2],
    'reg_alpha': [0, 0.1, 0.5],
}

xgb_model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=seed,
    tree_method='hist',
    n_jobs=-1
)

search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_grid,
    n_iter=20 if run_submission else 3,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=1
)

search.fit(X_train, y_train, verbose=run_debug)

print("Best CV RMSE:", -search.best_score_)
print("Best params:", search.best_params_)

score, model = score_model(X, y, model=search.best_estimator_, return_model=True)


if score < base_score:
    best_score = score
    best_model = model
else:
    best_score = base_score
    best_model = base_model


from sklearn.inspection import permutation_importance
import pandas as pd

params = best_model.get_params()
params['early_stopping_rounds'] = None
best_model = XGBRegressor(**params)

best_model.fit(X_train, y_train)
perm_imp = permutation_importance(best_model, X_val, y_val, n_repeats=5, scoring='neg_root_mean_squared_error', random_state=seed)
perm_imp = pd.Series(perm_imp.importances_mean, index=X_val.columns).sort_values(ascending=False)
perm_imp


reduced_features = perm_imp.loc[perm_imp.abs() > 0.001].index
X_rc = X[reduced_features]
X_rc_train = X_train[reduced_features]
X_rc_val = X_val[reduced_features]

search.fit(X_rc_train, y_train, verbose=run_debug)

print("Best params:", search.best_params_)

score, model = score_model(X_rc, y, model=search.best_estimator_, return_model=True)


if score < best_score:
    best_score = score
    best_model = model
    X = X[reduced_features]
    X_test = X_test[reduced_features]


params = best_model.get_params()
params['early_stopping_rounds'] = 50
params['n_estimators'] = 5000

model = XGBRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=run_debug)

params['early_stopping_rounds'] = None
params['n_estimators'] = model.best_iteration

y_preds = []
for s in [seed, seed + 10000, seed + 20000, seed + 30000]:
    params['random_state'] = s
    model = XGBRegressor(**params)
    model.fit(X, y)
    y_pred = model.predict(X_test)
    y_preds.append(y_pred)

y_pred = np.clip(np.mean(y_preds, axis=0), 0.0, 1.0)
y_pred = pd.DataFrame(y_pred, index=X_test.index, columns=[target_feature])

y_pred


y_pred.to_csv('submission.csv')

