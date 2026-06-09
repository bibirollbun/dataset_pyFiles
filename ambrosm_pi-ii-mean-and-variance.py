print("Installing scikit_learn with TargetEncoder...")
!pip install -q --no-deps scikit_learn==1.4.2



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xgboost
import lightgbm
import catboost
import pickle
from functools import partial

from sklearn.base import RegressorMixin, BaseEstimator, clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, QuantileRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error, mean_squared_log_error, root_mean_squared_error
from sklearn.dummy import DummyRegressor

def mwis_loss(y_true, y_pred):
    assert y_pred.shape[1] == 2
    lower = y_pred[:,0]
    upper = y_pred[:,1]
    alpha = 0.1
    mwis = (2 / alpha) * (mean_pinball_loss(y_true, lower, alpha=alpha/2)
                          + mean_pinball_loss(y_true, upper, alpha=1-alpha/2))
    return mwis

def fast_mwis_score(y_true, lower, upper, alpha):
    """
    This is an implementation of the Winkler Interval Score
    (https://otexts.com/fpp3/distaccuracy.html#winkler-score).
    The mean over all of the individual Winkler Interval scores (MWIS) is returned,
    along with the coverage and the percentages below and above.
    
    alpha is the non-coverage rate, i.e., set alpha to
    0.1 for a coverage of 90 %.
    
    Return values:
    - mwis: mean Winkler Interval Score
    - below, coverage, above: percentage of samples below / within / above the interval
    """

    assert y_true.ndim == 1, "y_true: pandas Series or 1D array expected"
    assert lower.ndim == 1, "lower: pandas Series or 1D array expected"
    assert upper.ndim == 1, "upper: pandas Series or 1D array expected"
    assert isinstance(alpha, float), "alpha: float expected"
    assert (lower <= upper).all(), ("lower must be <= upper",
                                    lower[lower > upper],
                                    upper[lower > upper])

    total_interval_width = upper.sum() - lower.sum()
    error_above = (y_true - upper)[y_true > upper].sum()
    error_below = (lower - y_true)[y_true < lower].sum()
    total_error = error_above + error_below
    mwis = (total_interval_width + total_error * 2 / alpha) / len(y_true)
    below = (y_true < lower).mean()
    above = (upper < y_true).mean()
    coverage = ((lower <= y_true) & (y_true <= upper)).mean()
    return mwis, below, coverage, above
    


# Configuration
COMPUTE_TEST_PRED = True
n_repeats = 15

# Containers for predictions
oof = {name: {} for name in ['mean', 'var', 'abs', 'lower', 'upper']}
test_pred = {name: {} for name in ['mean', 'var', 'abs', 'lower', 'upper']}
preds = [oof, test_pred] if COMPUTE_TEST_PRED else [oof]


train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv',
                    parse_dates=['sale_date'],
                    index_col='id') # 200000 rows × 46 columns
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv',
                    parse_dates=['sale_date'],
                    index_col='id') # 200000 rows × 45 columns

# Feature engineering
for df in [train, test]:
    df['date_float'] = (df.sale_date - pd.Timestamp('1998-06-01 00:00:00'))  / np.timedelta64(365,'D')
    df['age'] = df.sale_date.dt.year - df.year_built
    df['age_reno'] = df.sale_date.dt.year - np.maximum(df.year_built, df.year_reno)
    df['val'] = df.imp_val + df.land_val
    # df['sale_year'] = df.sale_date.dt.year
    df['sale_year'] = df['date_float'].astype(int) # our year starts in June
    df['sale_month'] = df.sale_date.dt.month
    df['sale_nbr'] = df.sale_nbr.clip(0, 2) # missing values remain missing

# Set the type for categorical features
cat_features = list(train.select_dtypes('object').columns) + ['area']
print(f"Categorical features: {cat_features}")
for col in cat_features:
    train.fillna({col: 'NaN'}, inplace=True)
    test.fillna({col: 'NaN'}, inplace=True)
    dtype = pd.CategoricalDtype(list(set(train[col]).union(set(test[col]))))
    train[col] = train[col].astype(dtype)
    test[col] = test[col].astype(dtype)

# features = list(test.columns)
num_features = list(test.select_dtypes('int').columns) + list(test.select_dtypes('float').columns)
num_features = [f for f in num_features if f not in ['sale_year']]

te_features = cat_features
# te_features = ['sale_warning', 'city', 'zoning', 'subdivision', 'submarket', 'area']
column_transformer = ColumnTransformer(
    [('te', TargetEncoder(target_type='continuous', random_state=1), te_features)],
    remainder='passthrough'
).set_output(transform='pandas')



# Transform the target
train['log_price'] = np.log(train.sale_price)

# Find trend and seasonality
# Fit a linear model for log_price to the one-hot encoded year and month of the sale
ohe = OneHotEncoder(drop='first', sparse_output=False)
X = ohe.fit_transform(train[['sale_year', 'sale_month']])
X_te = ohe.transform(test[['sale_year', 'sale_month']])
trend_model = LinearRegression()
print(f"R2: {cross_val_score(trend_model, X, train.log_price).mean():.3f}") # r2 is 0.416
trend_model.fit(X, train.log_price)
# print(trend_model.coef_.round(3), trend_model.intercept_.round(3))

# Detrend and deseasonalize the target
train['trend'] = trend_model.predict(X)
test['trend'] = trend_model.predict(X_te)

def detrend(df, price):
    """Compute the transformed target for price"""
    return np.log(price) - df['trend'] # transformed target

def retrend(df, y_pred_detrended):
    """Compute the predictions for the original target from the predicted detrended target"""
    return np.exp(y_pred_detrended + df['trend'])
    
train['detrended_price'] = train.log_price - train['trend'] # transformed target

# # Add a detrended form of the val feature
# for df in [train, test]:
#     df['val_detrended'] = df.val / np.exp(df.trend)
#     # df['max_detrended_price'] = np.log(3000000) - train['trend'] # transformed maximum price
# num_features = list(set(num_features).union({'val_detrended'}))#.union({'max_detrended_price'}))


# Plot, resampling the prices by year
rule = 'YE'
plt.figure(figsize=(15, 4))
temp = train.copy()
temp['ls'] = temp.detrended_price
series = temp.resample(on='sale_date', rule=rule, label='left').detrended_price.mean()
plt.plot(series.index, series, label='mean')
q05 = temp.resample(on='sale_date', rule=rule, label='left').detrended_price.quantile(0.05)
q95 = temp.resample(on='sale_date', rule=rule, label='left').detrended_price.quantile(0.95)
series_std = temp.resample(on='sale_date', rule=rule, label='left').detrended_price.std()
plt.fill_between(series.index, q05, q95, alpha=0.6, label='90 %') # fill area between quantiles
# plt.fill_between(series.index, series-1.67*series_std, series+1.67*series_std, alpha=0.6, label='90 % if lognormal')
# plt.plot(temp.resample(on='sale_date', rule=rule, label='left').max_detrended_price.mean(), ':', color='k', label='max price (3M$)')
# plt.plot(temp.resample(on='sale_date', rule=rule, label='left').detrended_price.quantile(0.98), color='gray', label='98 %')
# plt.plot(temp.resample(on='sale_date', rule=rule, label='left').detrended_price.quantile(0.02), color='gray', label='2 %')
plt.legend(loc='upper left')
plt.title('Yearly detrended log sale_price')
plt.xlabel('year')
plt.ylabel('detrended log sale_price')
plt.show()


def cross_validate(model, 
                   rv_name, y,
                   label=None,
                   n_repeats=n_repeats,
                   X_dev=train[num_features + cat_features],
                   X_te=test[num_features + cat_features]):
    """Cross-validate the model, retrain and compute test predictions.
    
    Parameters:
    - model: the model to cross-validate
    - label: a string label for the model
    - rv_name: the name of the predicted quantity ('mean', 'var', 'lower', 'upper')
    - y: the true target values
    - n_repeats: the count of repetitions for the ensemble
    - X_dev: dataset for training and validation
    - X_te: dataset for testing
    
    Output in global variables
    - oof[rv_name][label]: out-of-fold predictions
    - test_pred[rv_name][label]: test predictions
    """
    start_time = datetime.datetime.now()

    m = model
    if isinstance(m, Pipeline): m = m[-1]
    if isinstance(m, TransformedTargetRegressor): m = m.regressor
    if label is None:
        label = type(m).__name__
    cb_with_uncertainty = isinstance(m, catboost.CatBoostRegressor) and m.get_param('loss_function') == 'RMSEWithUncertainty'
    scoring_function = (partial(mean_pinball_loss, alpha=0.05) if rv_name == 'lower'
                        else partial(mean_pinball_loss, alpha=0.95) if rv_name == 'upper'
                        else root_mean_squared_error)
    scoring_name = 'Pinball loss' if rv_name in ['lower', 'upper'] else 'RMSE'
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99999)
    oof_preds = np.full((len(X_dev), 2), np.nan) if cb_with_uncertainty else np.full((len(X_dev),), np.nan)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X_dev, pd.qcut(train.sale_price, 10, labels=False))):
        if isinstance(X_dev, pd.DataFrame):
            X_tr = X_dev.iloc[idx_tr]
            X_va = X_dev.iloc[idx_va]
        else:
            X_tr = X_dev[idx_tr]
            X_va = X_dev[idx_va]
        y_tr = y.iloc[idx_tr]
        y_va = y.iloc[idx_va]
        
        def fit_predict_repeatedly(model, X_tr, y_tr, X_va):
            """Average n_repeats runs of the model with different seeds."""
            pred_va = np.zeros((len(X_va), 2), dtype=float) if cb_with_uncertainty else np.zeros(len(X_va), dtype=float)
            for i in range(n_repeats):
                model1 = clone(model)
                m = model1
                if isinstance(m, Pipeline): m = m[-1]
                if isinstance(m, TransformedTargetRegressor): m = m.regressor
                if not isinstance(m, DummyRegressor) and not isinstance(m, BayesianRidge):
                    m.set_params(random_state=i)
                model1.fit(X_tr, y_tr)
                if isinstance(m, Ridge)or isinstance(m, BayesianRidge):
                    print('                              Ensemble weights:', m.coef_.round(3), np.round(m.intercept_, 3))
                pred_va += model1.predict(X_va)
            pred_va /= n_repeats
            return pred_va
        
        pred_va = fit_predict_repeatedly(model, X_tr, y_tr, X_va)
        score = scoring_function(y_va, pred_va[:,0] if cb_with_uncertainty else pred_va)
        print(f"# Fold {fold:2} {score:6.4f}")
        oof_preds[idx_va] = pred_va

    score = scoring_function(y, oof_preds[:,0] if cb_with_uncertainty else oof_preds)
    repeats = f" {n_repeats=}" if n_repeats > 1 else ''
    elapsed_time = datetime.datetime.now() - start_time
    print(f"# Overall {scoring_name} {score:6.4f}"
          f" {label} {rv_name} {repeats}   {int(np.round(elapsed_time.total_seconds() / 60))} min")
    if cb_with_uncertainty:
        oof['mean'][label] = oof_preds[:,0]
        oof['var'][label] = oof_preds[:,1]
    else:
        oof[rv_name][label] = oof_preds

    if COMPUTE_TEST_PRED:
        test_preds = fit_predict_repeatedly(model, X_tr, y_tr, X_te)
        if cb_with_uncertainty:
            test_pred['mean'][label] = test_preds[:,0]
            test_pred['var'][label] = test_preds[:,1]
        else:
            test_pred[rv_name][label] = test_preds


# Prepare for predicting the mean
cross_validate_mean = partial(cross_validate, rv_name='mean', y=train.detrended_price)


hgb_params = {'learning_rate': 0.10685277042497433, 'max_features': 0.9642865268110019, 'l2_regularization': 94.0842832340528, 'max_depth': 8, 'min_samples_leaf': 12, 'max_leaf_nodes': 133, 'max_iter': 1000, 'early_stopping': False} # 0.01770868232484414 detrended_price
model = make_pipeline(
    ColumnTransformer([('te', TargetEncoder(target_type='continuous', random_state=11), cat_features)],
                      remainder='passthrough').set_output(transform='pandas'),
    HistGradientBoostingRegressor(**hgb_params))
cross_validate_mean(model)



xgb_params = {'learning_rate': 0.114198727291001, 'min_child_weight': 81, 'colsample_bytree': 0.8527952811639532, 'reg_lambda': 78.62906182739489, 'max_depth': 7, 'n_estimators': 1500} # 0.017950306044295913 v16
model = make_pipeline(
    column_transformer,
    xgboost.XGBRegressor(**xgb_params, enable_categorical=True)
)
cross_validate_mean(model)
# Overall RMSE 0.1330 XGBRegressor mean    2 min


xgb_params = {'learning_rate': 0.09662987098560075, 'min_child_weight': 99, 'colsample_bytree': 0.8909273926182449, 'reg_lambda': 86.40095583865487, 'max_depth': 7, 'n_estimators': 3000} # 0.017537369391218246 detrended_price v25
model = make_pipeline(
    column_transformer,
    xgboost.XGBRegressor(**xgb_params)
)
cross_validate_mean(model, label='XGB3000')
# Overall RMSE 0.1329 XGB3000 mean    4 min


lgbm_params = {'learning_rate': 0.13375728851189173, 'num_leaves': 282, 'min_child_samples': 51, 'colsample_bytree': 0.8148949621016217, 'reg_lambda': 96.17522289797597, 'n_estimators': 800, 'force_row_wise': True, 'verbose': -1} # 0.017501700009063228 detrended_price v26
model = make_pipeline(
    column_transformer,
    lightgbm.LGBMRegressor(**lgbm_params)
)
cross_validate_mean(model)
# Overall RMSE 0.1333 LGBMRegressor mean    3 min


cb_params = {'learning_rate': 0.16979216957121673, 'colsample_bylevel': 0.8658567122166555, 'subsample': 0.9931050800114454, 'reg_lambda': 21.330305237029084, 'depth': 8, 'iterations': 2000, 'loss_function': 'RMSE', 'task_type': 'CPU', 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'verbose': False} # 0.0178251537446623
model = make_pipeline(
    column_transformer,
    catboost.CatBoostRegressor(**cb_params)
)
cross_validate_mean(model)
# Fold  0 0.1329
# Fold  1 0.1323
# Fold  2 0.1344
# Fold  3 0.1323
# Fold  4 0.1324
# Overall RMSE 0.1328 CatBoostRegressor mean    7 min


# Ensemble the predicted means
potential_members = sorted([name for name in oof['mean'].keys() if 'Ridge' not in name])
print(potential_members)
X_dev = np.column_stack([oof['mean'][name] for name in potential_members])
if COMPUTE_TEST_PRED:
    X_te = np.column_stack([test_pred['mean'][name] for name in potential_members])
else:
    X_te = None
model = BayesianRidge(tol=1e-10)
cross_validate_mean(model, n_repeats=1, X_dev=X_dev, X_te=X_te)

with open("mean_prediction.pickle", "wb") as f:
    pickle.dump(retrend(train, oof['mean']['BayesianRidge']), f)

# Overall RMSE 0.1306 BayesianRidge mean    0 min

plt.scatter(oof['mean']['BayesianRidge'], train.detrended_price, s=1)
plt.gca().set_aspect('equal')
plt.xlabel('detrended log price pred')
plt.ylabel('detrended log price true')
plt.show()

for label in ['mean']:
    print(f"oof       {label}: {retrend(train, oof[label]['BayesianRidge']).min():.0f}..{retrend(train, oof[label]['BayesianRidge']).max():.0f}")
for label in ['mean']:
    print(f"test_pred {label}: {retrend(test, test_pred[label]['BayesianRidge']).min():.0f}..{retrend(test, test_pred[label]['BayesianRidge']).max():.0f}")

for X, p in [(train, oof), (test, test_pred)]:
    p['mean']['BayesianRidge'] = detrend(X, retrend(X, p['mean']['BayesianRidge']).clip(train.sale_price.min(), train.sale_price.max()))



# Prepare for predicting the variances
USE_ABS = True
scale_name = 'abs' if USE_ABS else 'var'

oof_residual = train.detrended_price - oof['mean']['BayesianRidge']
X_dev = train[num_features + cat_features].copy()
X_dev['mean'] = oof['mean']['BayesianRidge']

if COMPUTE_TEST_PRED:
    X_te = test[num_features + cat_features].copy()
    X_te['mean'] = test_pred['mean']['BayesianRidge']

cross_validate_var = partial(cross_validate,
                             rv_name=scale_name,
                             X_dev=X_dev,
                             X_te=X_te,
                             y=np.abs(oof_residual) if USE_ABS else np.square(oof_residual))

model = DummyRegressor()
cross_validate_var(model)
# Overall RMSE 0.0710 DummyRegressor var    0 min
# Overall RMSE 0.0975 DummyRegressor abs    0 min


xgb_params = {'learning_rate': 0.06446244773503793, 'min_child_weight': 958, 'colsample_bytree': 0.8779839849735629, 'reg_lambda': 85.88542657781878, 'max_depth': 5, 'n_estimators': 1500} # 0.006675845480713263
model = make_pipeline(
    column_transformer,
    xgboost.XGBRegressor(**xgb_params, enable_categorical=True)
) # mse
cross_validate_var(model, label='XGB1')
# Fold  0 0.0618
# Fold  1 0.0680
# Fold  2 0.0700
# Fold  3 0.0763
# Fold  4 0.0623
# Overall RMSE 0.0679 XGB1 var    1 min
# Fold  0 0.0858
# Fold  1 0.0863
# Fold  2 0.0886
# Fold  3 0.0868
# Fold  4 0.0864
# Overall RMSE 0.0868 XGB1 abs    1 min


xgb_params = {'learning_rate': 0.08558907376076766, 'min_child_weight': 826, 'colsample_bytree': 0.8788889346686256, 'reg_lambda': 71.08012969986461, 'max_depth': 5, 'n_estimators': 1500, 'objective': 'reg:gamma'} # 0.005427833769866527
model = make_pipeline(
    column_transformer,
    xgboost.XGBRegressor(**xgb_params)
) # gamma, predicts only positive variances
cross_validate_var(model, label='XGB2')
# Overall RMSE 0.0677 XGB2 var    2 min
# Overall RMSE 0.0869 XGB2 abs    2 min


xgb_params = {'objective': 'reg:gamma', 'learning_rate': 0.06446244773503793, 'min_child_weight': 958, 'colsample_bytree': 0.8779839849735629, 'reg_lambda': 85.88542657781878, 'max_depth': 5, 'n_estimators': 1500} # 0.006675845480713263
model = make_pipeline(
    column_transformer,
    xgboost.XGBRegressor(**xgb_params)
) # gamma, predicts only positive variances
cross_validate_var(model, label='XGB3')
# Overall RMSE 0.0677 XGB3 var    2 min
# Overall RMSE 0.0869 XGB3 abs    2 min


xgb_params = {'learning_rate': 0.02115489626917683, 'min_child_weight': 291, 'colsample_bytree': 0.9892823035534885, 'reg_lambda': 64.18541691878656, 'max_depth': 7, 'n_estimators': 1500} # 0.00801164049646578 abs_residual
model = make_pipeline(
    column_transformer,
    xgboost.XGBRegressor(**xgb_params, enable_categorical=True)
)
cross_validate_var(model, label='XGB4')
# Overall RMSE 0.0679 XGB4 var    2 min
# Overall RMSE 0.0866 XGB4 abs    2 min


lgbm_params = {'learning_rate': 0.11098486232288532, 'num_leaves': 123, 'min_child_samples': 159, 'colsample_bytree': 0.9024448852759349, 'reg_lambda': 20.52349933755892, 'min_gain_to_split': 0.01370024305156118, 'n_estimators': 1500, 'force_row_wise': True, 'verbose': -1} # 0.008023354201353965 abs_residual
model = make_pipeline(
    column_transformer,
    lightgbm.LGBMRegressor(**lgbm_params)
)
cross_validate_var(model)
# Overall RMSE 0.0678 LGBMRegressor var    1 min
# Overall RMSE 0.0866 LGBMRegressor abs    1 min


cb_params = {'learning_rate': 0.07804095663227534, 'colsample_bylevel': 0.9532352011300971, 'subsample': 0.9965705687571, 'reg_lambda': 55.020721908976356, 'depth': 7, 'iterations': 2000, 'loss_function': 'RMSE', 'task_type': 'CPU', 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'verbose': False}
model = make_pipeline(
    column_transformer,
    # catboost.CatBoostRegressor(**cb_params, cat_features=['remainder__join_status'])
    catboost.CatBoostRegressor(**cb_params)
)
cross_validate_var(model)
# Overall RMSE 0.0680 CatBoostRegressor var    5 min
# Overall RMSE 0.0867 CatBoostRegressor abs    5 min


# Ensemble the predicted variances
potential_members = sorted([name for name in oof[scale_name].keys() if 'Ridge' not in name and 'Dummy' not in name])
print(potential_members)
X_dev = np.column_stack([oof[scale_name][name] for name in potential_members])
if COMPUTE_TEST_PRED:
    X_te = np.column_stack([test_pred[scale_name][name] for name in potential_members])
model_var = BayesianRidge(tol=1e-10)
cross_validate_var(model_var, n_repeats=1, X_dev=X_dev, X_te=X_te)
# Overall RMSE 0.0864 BayesianRidge abs    0 min


# Show the minimum predictions
for key in sorted(oof[scale_name].keys()):
    print(key, oof[scale_name][key].min())


# Estimate the uncertainty parameter
for pred in preds:
    print('Minimum and maximum variance:', pred[scale_name]['BayesianRidge'].min(), pred[scale_name]['BayesianRidge'].max())
    if USE_ABS:
        pred['std'] = pred['abs']['BayesianRidge'].clip(0, None)
    else:
        pred['std'] = np.sqrt(pred['var']['BayesianRidge'].clip(0, None))
    print('Minimum and maximum standard deviation:', pred['std'].min(), pred['std'].max())

plt.scatter(oof['mean']['BayesianRidge'], oof['std'], s=1, c='m', alpha=0.2)
# plt.scatter(test_pred['mean']['BayesianRidge'], test_pred['std'], s=1, c='c', alpha=0.2)
plt.xlabel('predicted mean')
plt.ylabel('predicted uncertainty')
plt.show()



%%time
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class CompleteModel(BaseEstimator, RegressorMixin):
    def __init__(self, quantile, random_state=1):
        assert quantile == 0.05 or quantile == 0.95
        self.quantile = quantile
        self.random_state = random_state
        
    def fit(self, X, y):

        def objective(trial):
            w_mean = trial.suggest_float('w_mean', 0.95, 1.05)
            w_std = trial.suggest_float('w_std', -2.5, -1.2) if self.quantile < 0.5 else trial.suggest_float('w_std', 1.2, 2.5)
            w_intercept = trial.suggest_float('w_intercept', -0.03, 0.03)
            y_pred = X['mean'] * w_mean + X['std'] * w_std + w_intercept
            return mean_pinball_loss(y, y_pred, alpha=self.quantile)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=400, timeout=4*3600)
        self.w_mean_ = study.best_params['w_mean']
        self.w_std_ = study.best_params['w_std']
        self.w_intercept_ = study.best_params['w_intercept']
        print(f'                                                       Weights:'
              f' {self.w_mean_:.3f} {self.w_std_:6.3f} {self.w_intercept_:8.5f}')
        return self
        
    def predict(self, X):
        y = X['mean'] * self.w_mean_ + X['std'] * self.w_std_ + self.w_intercept_
        return y

X_dev = train[['age']].copy()
X_dev['mean'] = oof['mean']['BayesianRidge']
X_dev['std'] = oof['std']

if COMPUTE_TEST_PRED:
    X_te = test[['age']].copy()
    X_te['mean'] = test_pred['mean']['BayesianRidge']
    X_te['std'] = test_pred['std']


for rv_name, quantile in [('lower', 0.05), ('upper', 0.95)]:
    model = CompleteModel(quantile=quantile)
    cross_validate(model, rv_name, train.detrended_price,
                       n_repeats=1,
                       X_dev=X_dev,
                       X_te=X_te)
    print()
    print()

# var: 0.95..1.05, +-1.2..+-1.6, -0.03..0.03
# abs: 1..1.05, +-1.9..+-2.5, 0..0.01


model_names = ['CompleteModel']
result_list = []
for model_name in model_names:
    alpha = 0.1
    mwis, below, coverage, above = fast_mwis_score(train.sale_price, 
                                                   retrend(train, oof['lower'][model_name]).clip(train.sale_price.min(), train.sale_price.max()), 
                                                   retrend(train, oof['upper'][model_name]).clip(train.sale_price.min(), train.sale_price.max()),
                                                   alpha)
    pin_lower = mean_pinball_loss(train.sale_price, retrend(train, oof['lower'][model_name]).clip(train.sale_price.min(), train.sale_price.max()), alpha=0.05)
    pin_upper = mean_pinball_loss(train.sale_price, retrend(train, oof['upper'][model_name]).clip(train.sale_price.min(), train.sale_price.max()), alpha=0.95)
    print(f"# Overall MWIS {mwis:6.0f}={20*pin_lower:6.0f}+{20*pin_upper:6.0f}"
          f"   [{below:.1%} {coverage:.1%} {above:.1%}]   {model_name} {n_repeats=} {scale_name}")
    result_list.append([mwis, pin_lower, pin_upper, coverage])
# Overall MWIS 289628=132515+157113   [5.0% 90.0% 5.0%]   CompleteModel n_repeats=1 abs


# Histogram of lower and upper bounds
plt.figure(figsize=(12, 2))
plt.hist(oof['lower']['CompleteModel'], bins=np.linspace(-2, 2, 201), label='lower', alpha=0.5)
plt.hist(oof['upper']['CompleteModel'], bins=np.linspace(-2, 2, 201), label='upper', alpha=0.5)
plt.title('Histogram of lower and upper interval bounds (detrended)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 2))
plt.hist(retrend(train, oof['lower']['CompleteModel']), bins=np.linspace(0, 3e6, 201), label='lower', alpha=0.5)
plt.hist(retrend(train, oof['upper']['CompleteModel']), bins=np.linspace(0, 3e6, 201), label='upper', alpha=0.5)
plt.title('Histogram of lower and upper interval bounds')
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(retrend(train, oof['lower']['CompleteModel']).clip(train.sale_price.min(), train.sale_price.max()),
            retrend(train, oof['upper']['CompleteModel']).clip(train.sale_price.min(), train.sale_price.max()),
            s=1,
            c='g',
            alpha=0.4
           )
plt.plot([0, 2.5e6], [0, 2.5e6], color='gray')
plt.gca().set_aspect('equal')
plt.xlabel('lower')
plt.ylabel('upper')
plt.title('Lower and upper interval bounds')
plt.show()

for label in ['lower', 'upper']:
    print(f"oof       {label}: {retrend(train, oof[label]['CompleteModel']).min():.0f}..{retrend(train, oof[label]['CompleteModel']).max():.0f}")
for label in ['lower', 'upper']:
    print(f"test_pred {label}: {retrend(test, test_pred[label]['CompleteModel']).min():.0f}..{retrend(test, test_pred[label]['CompleteModel']).max():.0f}")
    


if COMPUTE_TEST_PRED:
    submission = pd.DataFrame({'pi_lower': retrend(test, test_pred['lower']['CompleteModel']).clip(train.sale_price.min(), train.sale_price.max()),
                               'pi_upper': retrend(test, test_pred['upper']['CompleteModel']).clip(train.sale_price.min(), train.sale_price.max())},
                              index=test.index)
    submission.to_csv(f'submission.csv')
    display(submission)





