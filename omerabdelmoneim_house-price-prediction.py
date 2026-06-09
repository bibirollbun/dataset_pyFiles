import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, PolynomialFeatures
import os
from typing import Literal, Optional
pd.set_option('display.max_columns', None)


train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv',parse_dates =['sale_date'])
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv',parse_dates =['sale_date'])
sample_submission = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv')


X = train.drop(columns=['sale_price'])
y = train['sale_price']
min_date = train['sale_date'].min()


def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """Compute the Winkler Interval Score for prediction intervals.

    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.

    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)


def fix_predictions(low,high):
    swap_index = low > high
    temp = np.copy(low)
    low[swap_index] = high[swap_index]
    high[swap_index] = temp[swap_index]
    low = np.clip(low,0,low)
    return low, high


def predict(estimator,X,alpha):
    if hasattr(estimator, 'predict_interval'):
        preds, low ,high = estimator.predict_interval(X,alpha)

    elif "Mapie" in estimator.__class__.__name__:
        # MAPIE estimator with built-in interval support
        preds, intervals = estimator.predict(X, return_pred_int=True, alpha=alpha)
        low, high = intervals[:, 0], intervals[:, 1]
    
    else:
        preds = estimator.predict(X)
        low = preds * alpha
        high = preds * (2 - alpha)
    
    low,high = fix_predictions(low, high)
    return preds, low,high


def make_submission(estimator,X_train=X,y_train=y,X_test=test,alpha=0.1):
    print("Fitting estimator: ")
    estimator.fit(X_train,y_train)
    print("Predicting: ")
    _, low, high = predict(estimator,X_test,alpha)
    submission = pd.DataFrame({
        'id':X_test['id'],
        'pi_lower':low,
        'pi_upper':high
                 })
    print("Saving submission: ")
    submission.to_csv('submission.csv', index=False)
    return submission


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

estimator_last_score = {}
test_after_train_percantage = (test['sale_date'] > train['sale_date'].max()).mean()
date_cutoff = train['sale_date'].quantile(1 - test_after_train_percantage)

def change_type(value):
    if value < 0:
        return "ğŸ“ˆ improved"
    elif value > 0:
        return "ğŸ“‰ degraded"
    else:
        return "â�¸ï¸� no change"
        
def evaluate_estimator(estimator, X=X, y=y, n_splits=4,shuffle=True,random_state=0, estimator_name=None, print_result=True,alpha=0.1):
    random_state = random_state if shuffle else None
    is_holdout = X['sale_date'] >= date_cutoff
    X_holdout = X[is_holdout]
    y_holdout = y[is_holdout]
    X = X[~is_holdout]
    y = y[~is_holdout]
    kfold = KFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state)
    folds = [(train_idx, val_idx) for train_idx, val_idx in kfold.split(X, y)]
    fold_winkler_scores = []
    fold_rmse_scores = []
    all_scores = []
    mean_coverage = 0
    mean_interval_width = 0
    for train_idx, val_idx in tqdm(folds,desc="cross validate"):
        X_train, X_val = X.iloc[train_idx,:], X.iloc[val_idx,:]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        estimator_ = clone(estimator)
        estimator_.fit(X_train, y_train)
        preds, low, high = predict(estimator_,X_val,alpha)
        mean_interval_width += np.mean(high - low)
        fold_scores, coverage = winkler_score(y_val,low,high, return_coverage=True)
        fold_winkler_scores.append(np.mean(fold_scores))
        rmse = mean_squared_error(y_val, preds, squared=False)
        fold_rmse_scores.append(rmse)
        mean_coverage += coverage
        all_scores.append(fold_scores)
        
    mean_score = np.mean(fold_winkler_scores)
    median_score = np.median(fold_winkler_scores)
    mean_rmse = np.mean(fold_rmse_scores)
    mean_coverage /= n_splits
    mean_interval_width /= n_splits
    winkler_std = np.std(fold_winkler_scores)
    if print_result:
        print(f"\n=== Evaluation Results [{estimator_name}] ===")
        print(f"Mean Winkler: {mean_score:.4f}")
        print(f"Median Winkler: {median_score:.4f}")
        print(f"Mean RMSE: {mean_rmse:.4f}")
        print(f"Mean Coverage: {mean_coverage:.4f}")
        print(f"Winkler Std: {winkler_std:.4f}")
        print(f"Winkler Scores: {[f'{score:.4f}' for score in fold_winkler_scores]}")
        print(f"Mean Interval Width: {mean_interval_width} ({(mean_interval_width * 100 / y.mean()):.2f}%)")
        print(f"\nEvaluating on Holdout (Post-{date_cutoff.date()})")
            
    estimator_final = clone(estimator)
    estimator_final.fit(X, y)
    preds_holdout, low, high = predict(estimator_final, X_holdout, alpha)
    holdout_scores, holdout_coverage = winkler_score(y_holdout, low, high, return_coverage=True)
    mean_holdout_score = np.mean(holdout_scores)
    rmse_holdout = mean_squared_error(y_holdout, preds_holdout, squared=False)
    final_score = (mean_holdout_score * len(y_holdout) + mean_score * len(y)) / (len(y_holdout) + len(y))
    final_rmse = (rmse_holdout * len(y_holdout) + mean_rmse * len(y)) / (len(y_holdout) + len(y))
    if print_result:
        print(f"\n=== Holdout (Post-{date_cutoff.date()}) Evaluation ===")
        print(f"Holdout Winkler: {mean_holdout_score:.4f}")
        print(f"Holdout RMSE: {rmse_holdout:.4f}")
        print(f"Coverage: {holdout_coverage:.4f}")
        print(f"Holdout Size: {len(y_holdout)}")
        print(f"Final Score: {final_score:.4f}")

    if print_result and estimator_name:
        if estimator_name in estimator_last_score:
            last = estimator_last_score[estimator_name]
            mean_diff = mean_score - last['mean_winkler']
            median_diff = median_score - last['median_winkler']
            holdout_diff = mean_holdout_score - last['holdout_winkler']
            final_diff = final_score - last['final_winkler']
            rmse_diff = mean_rmse - last['mean_rmse']
            rmse_holdout_diff = rmse_holdout - last['holdout_rmse']
            rmse_final_diff = final_rmse - last['final_rmse']

            print("\n=== Performance Change ===")
            print(f"Mean Winkler Î”: {mean_diff:+.4f} ({change_type(mean_diff)})")
            print(f"Median Winkler Î”: {median_diff:+.4f} ({change_type(median_diff)})")
            print(f"Mean RMSE Î”: {rmse_diff:+.4f} ({change_type(rmse_diff)})")
            
            print(f"Holdout Winkler Î”: {holdout_diff:+.4f} ({change_type(holdout_diff)})")
            print(f"Holdout RMSE Î”: {rmse_holdout_diff:+.4f} ({change_type(rmse_holdout_diff)})")
            
            print(f"Final Winkler Î”: {final_diff:+.4f} ({change_type(final_diff)})")
            print(f"Final RMSE Î”: {rmse_final_diff:+.4f} ({change_type(rmse_final_diff)})")
            if all(x < 0 for x in [mean_diff,median_diff,rmse_diff,holdout_diff,
                                    rmse_holdout_diff,final_diff]):
                print("Improved over all")
        else:
            print("\n(First evaluation of this estimator)")
            
        estimator_last_score[estimator_name] = {
            'mean_winkler': mean_score,
            'median_winkler': median_score,
            'holdout_winkler': mean_holdout_score,
            'final_winkler': final_score,
            'mean_rmse': mean_rmse,
            'holdout_rmse': rmse_holdout,
            'final_rmse': final_rmse
            }
    return {
        "fold_winkler_scores": fold_winkler_scores,
        "holdout_winkler": mean_holdout_score,
        "final_winkler": final_score,
        "fold_rmse_scores": fold_rmse_scores,
        "holdout_rmse": rmse_holdout,
        "final_rmse": final_rmse,
        "mean_winkler": mean_score,
        "median_winkler": median_score,
        "mean_rmse": mean_rmse,
        "mean_coverage": mean_coverage,
        "winkler_std": winkler_std,
        "holdout_coverage": holdout_coverage,
        "holdout_size": len(y_holdout),
        "cutoff_date": date_cutoff.date()
    }


from scipy.stats import t 
from sklearn.base import clone, BaseEstimator, RegressorMixin
class QuantilePredictor(BaseEstimator,RegressorMixin):
    
    def __init__(self,estimator,clip = True, 
                 residual_mode:Literal['absolute', 'relative'] = 'absolute'):
        super().__init__()
        self.estimator = estimator
        self.clip = clip
        self.residual_mode = residual_mode
        
    def fit(self,X,y):
        self.estimator_ = clone(self.estimator).fit(X,y)
        preds = self.estimator_.predict(X)
        resid = y - preds
        if self.residual_mode == 'relative':
            resid = resid / preds
        self.std_ = np.std(resid)
        self.n_ = len(y)
        self.dof_ = max(self.n_ - 1, 1)
        self.min_ = np.min(y)
        self.max_ = np.max(y)
        return self
        
    def predict(self,X):
        return self.estimator_.predict(X)

    def predict_interval(self, X, alpha):
        preds = self.predict(X)
        # Two-tailed t-value (for equal-tailed confidence interval)
        t_val = t.ppf(1 - alpha / 2, df=self.dof_)  # e.g., for 90% CI, alpha=0.1, use 0.95
    
        # Symmetric interval around predictions
        margin = t_val * self.std_
        
        if self.residual_mode == 'relative':
            margin = margin * preds
            
        lower = preds - margin
        upper = preds + margin
        if self.clip:
            lower = np.clip(lower, self.min_, self.max_)
            upper = np.clip(upper, self.min_, self.max_)
    
        return preds, lower, upper


from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy as np

class DetrendedRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        trend_estimator=None,
        main_estimator=None,
        residual_mode: Literal['absolute', 'relative'] = 'absolute',
        freq: Optional[str] = None,
        date_col: Optional[str] = 'sale_date'
    ):
        self.trend_estimator = trend_estimator
        self.main_estimator = main_estimator
        self.residual_mode = residual_mode
        self.freq = freq
        self.date_col = date_col

    def _maybe_aggregate(self, X, y):
        """Aggregate by frequency if freq and date_col are set"""
        if self.freq is None or self.date_col is None:
            return X, y

        df = X.copy()
        df['_target'] = y
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        grouped = df.groupby(pd.Grouper(key=self.date_col, freq=self.freq)).agg('median',numeric_only=True)
        grouped = grouped.dropna(subset=['_target'])
        grouped = grouped.reset_index()
        
        y_agg = grouped['_target']
        X_agg = grouped.drop(columns=['_target'])

        return X_agg, y_agg

    def fit(self, X, y):
        if self.trend_estimator is not None:
            # Aggregate before fitting trend model (optional)
            X_agg, y_agg = self._maybe_aggregate(X, y)
            self.trend_estimator_ = clone(self.trend_estimator)
            self.trend_estimator_.fit(X_agg, y_agg)

            # Predict trend on original X
            trend_pred = self.trend_estimator_.predict(X)

            if self.residual_mode == 'relative':
                with np.errstate(divide='ignore', invalid='ignore'):
                    residuals = (y - trend_pred) / trend_pred
                    residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                residuals = y - trend_pred
        else:
            residuals = y
            self.trend_estimator_ = None

        # Fit main estimator on residuals
        self.main_estimator_ = clone(self.main_estimator)
        self.main_estimator_.fit(X, residuals)

        return self

    def predict(self, X):
        trend_pred = 0
        if self.trend_estimator_ is not None:
            trend_pred = self.trend_estimator_.predict(X)

        residual_pred = self.main_estimator_.predict(X)

        if self.trend_estimator_ is not None and self.residual_mode == 'relative':
            return trend_pred + residual_pred * trend_pred
        else:
            return trend_pred + residual_pred


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class OrderedClusterTransformerDF(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, cluster_columns=None, n_init='auto', random_state=0, cluster_col_name='cluster_ordered'):
        self.n_clusters = n_clusters
        self.cluster_columns = cluster_columns
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_col_name = cluster_col_name
        
    def fit(self, X: pd.DataFrame, y):
        if self.cluster_columns is None:
            raise ValueError("cluster_columns parameter must be set to list of columns to cluster on")
        
        X_cluster = X[self.cluster_columns]
        
        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              n_init=self.n_init,
                              random_state=self.random_state)
        clusters = self.kmeans_.fit_predict(X_cluster.values)
        
        cluster_means = []
        for c in range(self.n_clusters):
            mean_y = y[clusters == c].mean()
            cluster_means.append((c, mean_y))
        
        cluster_means_sorted = sorted(cluster_means, key=lambda x: x[1])
        self.label_map_ = {old_label: new_label for new_label, (old_label, _) in enumerate(cluster_means_sorted)}
        
        return self
    
    def transform(self, X: pd.DataFrame):
        if self.cluster_columns is None:
            raise ValueError("cluster_columns parameter must be set to list of columns to cluster on")
        
        X_cluster = X[self.cluster_columns]
        # Use the fitted kmeans model to predict clusters on new data
        clusters = self.kmeans_.predict(X_cluster.values)
        
        ordered_labels = np.array([self.label_map_[c] for c in clusters])
        
        X_new = X.copy()
        X_new[self.cluster_col_name] = ordered_labels
        return X_new



# columns that should be encoded using One Hot Encoding
one_hot_columns = ['sale_warning', 'join_status','city','submarket']
# columns containing many 0s
sparse_columns = [
    'year_reno', 'wfnt', 'golf', 'greenbelt', 'noise_traffic',
    'view_rainier', 'view_olympics', 'view_cascades', 'view_territorial',
    'view_skyline', 'view_sound', 'view_lakewash', 'view_lakesamm',
    'view_otherwater', 'view_other'
]
# encode catogrical features
encode_cat_step = ('encoder',ColumnTransformer([('cat',OneHotEncoder(sparse_output=False, handle_unknown='ignore'),one_hot_columns)],
                                               remainder='passthrough'))
# selects numeric, boolean and one hot encoded columns only.
select_columns_step = ('select_columns',
                       FunctionTransformer(lambda df: df.drop(columns=['id']).select_dtypes(['number', 'bool']).join(df[one_hot_columns])))
# adds days since first sale 
add_days_since_start_step = ('add_days_since_start',
                            FunctionTransformer(lambda df: df.assign(days_since_start = (df['sale_date'] - min_date).dt.days)))
# derives new feature from year_built and sale_date 
add_years_since_build_step = ('add_years_since_build',
                             FunctionTransformer(
                                 lambda df: df.assign(
                                     years_since_build = df['sale_date'].dt.year - df['year_built']
                                 )
                             ))
# add year
add_year_step = ('add_year',
                FunctionTransformer(
                    lambda df: df.assign(
                        year = df['sale_date'].dt.year
                    )
                ))
# add cluster based only on latitude and longitude. n_clusters=8 is based on highest silhouette score.
add_location_cluster_step = ("add_location_cluster",
                             OrderedClusterTransformerDF(
                                 n_clusters=8, 
                                 cluster_columns=['latitude','longitude']
                             ))
# Adds a binary column indicating whether a house has a basement. 
# Assumes houses without a basement have 0 sqft_fbsmt.
add_basement_step = ('add_basement',
                    FunctionTransformer(
                        lambda df: df.assign(
                            basement = df['sqft_fbsmt'] != 0
                        )
                    ))
# get a step that adds season columns
def get_season_step(days = 1):
    return (f'add_season_{days}',
           FunctionTransformer(
               lambda df: df.assign(
                    **{
                       f"sin_{days}":np.sin(2 * np.pi * df['sale_date'].dt.dayofyear  / days),
                       f"cos_{days}":np.cos(2 * np.pi * df['sale_date'].dt.dayofyear  / days)
                   }
               )
           ))
# step that select given columns and drop all other columns
def get_column_selecter_step(columns):
    return ("select_columns", FunctionTransformer(lambda x:x[columns]))
    
# step that drop given columns
def get_column_drop_step(columns):
    return ("drop_columns",FunctionTransformer(lambda x:x.drop(columns=columns)))

# step to add cluster based on sparse columns.
# could be used to reduce dimensionality by dropping sparse columns after adding cluster.
def get_sparse_cluster_step(n_clusters=3):
    return ("add_cluster",OrderedClusterTransformerDF(n_clusters=n_clusters,cluster_columns=sparse_columns))


from sklearn.linear_model import LinearRegression
trend_model = Pipeline([
    add_days_since_start_step,
    get_column_selecter_step(['days_since_start']),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("reg", LinearRegression(n_jobs=-1))
])
evaluate_estimator(trend_model, estimator_name='trend_model',alpha = 0.1)


def get_interval_estimator(
    estimator,
    trend_estimator=trend_model,
    clip=True,
    quantile_residual_mode: Literal['absolute', 'relative'] = 'absolute',
    trend_residual_mode: Literal['absolute', 'relative'] = 'absolute',
    freq=None
):
    """
    Wraps a base estimator into an interval predictor with optional trend removal and residual-based quantile calibration.

    Parameters:
    ----------
    estimator : regressor
        The main model to use for prediction.

    trend_estimator : regressor or None, default=trend_model
        An optional model to capture the trend. If provided, a DetrendedRegressor is used to fit the residuals
        instead of the original target values.

    clip : bool, default=True
        If True, prediction intervals are clipped to be non-negative.

    quantile_residual_mode : {'absolute', 'relative'}, default='absolute'
        Mode for computing the prediction interval margins:
        - 'absolute': adds a constant width to all predictions.
        - 'relative': adds a constant proportion of the prediction value.

    trend_residual_mode : {'absolute', 'relative'}, default='absolute'
        Mode for computing residuals in the trend model if used.

    freq : str or None, default=None
        Frequency of the time series, used when fitting trend models like time-based regressors.

    Returns:
    -------
    QuantilePredictor
        A model that can be used to predict point estimates and intervals.
    """
    estimator_ = estimator if trend_estimator is None else DetrendedRegressor(
        trend_estimator=trend_estimator,
        main_estimator=estimator,
        residual_mode=trend_residual_mode,
        freq=freq
    )
    return QuantilePredictor(estimator_, clip=clip, residual_mode=quantile_residual_mode)



from xgboost import XGBRegressor
xgb = Pipeline([add_days_since_start_step,
                add_basement_step,
                select_columns_step,
                 encode_cat_step,
                ('xgb',XGBRegressor(n_jobs=-1,random_state=0))])
quantile_xgb = get_interval_estimator(xgb, 
                                      quantile_residual_mode = 'relative',
                                      trend_residual_mode = 'absolute',
                                     freq='ME')
evaluate_estimator(quantile_xgb, estimator_name='quantile_xgb',alpha = 0.1)


from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
lin = Pipeline([add_days_since_start_step,
                add_year_step,
                add_basement_step,
                add_location_cluster_step,
                select_columns_step,
                encode_cat_step,
                ('impute',SimpleImputer()),
                ('lin',LinearRegression(n_jobs=-1))])
quantile_lin = get_interval_estimator(lin,
                                      clip=False,
                                      trend_residual_mode='relative')
evaluate_estimator(quantile_lin, estimator_name='quantile_lin',alpha = 0.1)


from sklearn.ensemble import HistGradientBoostingRegressor

hist =Pipeline([add_days_since_start_step,
                add_location_cluster_step,
                select_columns_step,
                encode_cat_step,
                ('hist', HistGradientBoostingRegressor(random_state=0))])

quantile_hist = get_interval_estimator(hist,
                                       trend_estimator=None,
                                       quantile_residual_mode = 'relative')
evaluate_estimator(quantile_hist, estimator_name='quantile_hist', alpha=0.1)



from catboost import CatBoostRegressor
X_fixed = X.copy()
cat_cols = X_fixed.select_dtypes(['object']).columns
cat_feature_indices = [X_fixed.columns.get_loc(col) for col in cat_cols]
X_fixed[cat_cols] = X_fixed[cat_cols].astype(str).fillna("nan")
catboost = CatBoostRegressor(verbose=0, 
                             cat_features=cat_feature_indices)
quantile_catboost = get_interval_estimator(catboost, 
                                           quantile_residual_mode = 'relative',
                                           trend_residual_mode='relative')
evaluate_estimator(quantile_catboost,X_fixed,y, estimator_name='quantile_catboost',alpha = 0.1)


test_fixed = test.copy()
cat_cols = test_fixed.select_dtypes(['object']).columns
cat_feature_indices = [test_fixed.columns.get_loc(col) for col in cat_cols]
test_fixed[cat_cols] = test_fixed[cat_cols].astype(str).fillna("nan")
submission = make_submission(quantile_catboost,X_train=X_fixed,X_test=test_fixed)
submission.head(15)

