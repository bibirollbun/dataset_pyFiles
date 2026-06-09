from IPython.core.display import HTML

# Define custom CSS directly in Python variable
custom_css = """
<style>
  :root {
    --header1_color: #204709;
    --header2_color: #42841F;
    --header3_color: #6EAF4B;
    --keyword_color: #cc241d; /* import */
    --string_color: #79740e;
    --number_color: #b16286;
    --def_color: #689d6a; /* class name */
    --property_color: #458588; /* python properties */
    --builtin_color: #689d6a;
    --comment_color: #9f9f9f;
    --comment_color_2: #458588; /* equals sign */
    --operator_color: #a221f2;
    --font_color: #3c3836; /* general font */
    --variable2_color: #b16286; /*self keyworda */
    --box_color: #fffdee; /* Remove opacity */
  }

  /* Add the following style for headers with background color */
  h1,
  .h1 {
    font-family: "Trebuchet MS", sans-serif;
    font-size: 2em !important;
    letter-spacing: 1px;
    color: var(--header1_color);
    border-bottom: 3px solid var(--header1_color);
    background-color: #000080;
    padding: 0.5em;
    color: #ffff00 !important;
  }

  h2,
  .h2 {
    font-family: "Trebuchet MS";
    font-size: 1.7em !important;
    color: var(--header2_color);
    background-color: #000080;
    padding: 0.5em;
    color: #ffff00 !important;
  }

  h3,
  .h3 {
    font-family: "Trebuchet MS";
    font-size: 1.4em !important;
    color: var(--header3_color);
    background-color: #000080;
    padding: 0.5em;
    color: #ffff00 !important;
  }

  /* Rest of your existing styles... */

  body[data-jp-theme-light="true"] .jp-Notebook .CodeMirror.cm-s-jupyter {
    background-color: var(--box_color) !important;
  }

  div.input_area {
    background-color: var(--box_color) !important;
  }
</style>
"""

# Apply custom CSS
HTML(custom_css)


%%capture
!pip install bluecast


from category_encoders import (
    GLMMEncoder,
    LeaveOneOutEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    WOEEncoder,
)

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer, LabelEncoder, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, RepeatedKFold

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import re
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from xgboost import plot_tree

import warnings
warnings.filterwarnings("ignore")


from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.conformal_prediction.evaluation import prediction_interval_coverage
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.general_utils.general_utils import save_to_production, load_for_production

from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest


train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")

submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")


target = 'sale_price'
print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


train


test


train.info()


submission


submission.columns


train['sale_date_year'] = pd.to_datetime(train['sale_date']).dt.year
train['sale_date_month'] = pd.to_datetime(train['sale_date']).dt.month
train['sale_date_week_of_year'] = pd.to_datetime(train['sale_date']).apply(lambda x: x.isocalendar()[1])
train['sale_date_day_of_week'] = pd.to_datetime(train['sale_date']).dt.dayofweek
train['sale_date_day'] = pd.to_datetime(train['sale_date']).dt.day
train['sale_date_hour'] = pd.to_datetime(train['sale_date']).dt.hour

test['sale_date_year'] = pd.to_datetime(test['sale_date']).dt.year
test['sale_date_month'] = pd.to_datetime(test['sale_date']).dt.month
test['sale_date_week_of_year'] = pd.to_datetime(test['sale_date']).apply(lambda x: x.isocalendar()[1])
test['sale_date_day_of_week'] = pd.to_datetime(test['sale_date']).dt.dayofweek
test['sale_date_day'] = pd.to_datetime(test['sale_date']).dt.day
test['sale_date_hour'] = pd.to_datetime(test['sale_date']).dt.hour

#train = train.drop(columns=['date'])
#test = test.drop(columns=['date'])


train.columns


def add_shift(df):
    shifts_feats = ['sale_nbr']
    
    for col in shifts_feats:
        # shift for last hors
        for shift_length in range(1, 4):
            df[f'{col}_shifted_{shift_length}'] = df[col].shift(shift_length)
            df[f'{col}_running_mean_{shift_length}'] = df[col].rolling(shift_length).mean()
            
        # daily shift
        for shift_length in range(1, 7):
            df[f'{col}_shifted_{shift_length}_days'] = df.groupby('sale_date_day')[col].transform(lambda x: x.shift(shift_length))
            
        # days_of_week shift and rolling mean per _days_of_week of year (i.e. Wednesday to Wednesday)
        for shift_length in [1, 2, 3, 4, 5, 52, 53]:
            df[f'{col}_shifted_{shift_length}_days_of_week'] = df.groupby('sale_date_day_of_week')[col].transform(lambda x: x.shift(shift_length))
            #df[f'{col}_rolling_mean_{shift_length}_weeks'] = df.groupby('week_of_year')[col].rolling(shift_length).mean().reset_index(0, drop=True)
            df[f'{col}_rolling_mean_{shift_length}__days_of_week'] = df.groupby('sale_date_day_of_week')[col].transform(lambda x: x.rolling(shift_length, 1).mean())
            
        # weekly shift and rolling mean per week of year
        for shift_length in range(1, 2):
            df[f'{col}_shifted_{shift_length}_weeks'] = df.groupby('sale_date_week_of_year')[col].transform(lambda x: x.shift(shift_length))
            #df[f'{col}_rolling_mean_{shift_length}_weeks'] = df.groupby('week_of_year')[col].rolling(shift_length).mean().reset_index(0, drop=True)
            df[f'{col}_rolling_mean_{shift_length}_weeks'] = df.groupby('sale_date_week_of_year')[col].transform(lambda x: x.rolling(shift_length, 1).mean())
        
        # monthly shift and rolling mean per month of year
        for shift_length in range(1, 25):
            df[f'{col}_shifted_{shift_length}_months'] = df.groupby('sale_date_month')[col].transform(lambda x: x.shift(shift_length))
            #df[f'{col}_rolling_mean_{shift_length}_months'] = df.groupby('date_month')[col].rolling(shift_length).mean().reset_index(0, drop=True)
            df[f'{col}_rolling_mean_{shift_length}_months'] = df.groupby('sale_date_month')[col].transform(lambda x: x.rolling(shift_length, 1).mean())
    return df


train["source"] = 0
test["source"] = 1
all_data = pd.concat([train, test])
all_data = all_data.sort_values(by=["sale_date"], ascending=[True])


# all_data = add_shift(all_data)


# cyclic transformation
for feature in ["sale_date_hour", "sale_date_day_of_week", "sale_date_week_of_year", "sale_date_month"]:
    min_f = all_data[feature].min() 
    max_f = all_data[feature].max()
    
    rel_diff = (all_data[feature] - min_f) / (max_f - min_f)
    all_data[f'sin_{feature}'] = np.sin(2 * np.pi * rel_diff)
    all_data[f'cos_{feature}'] = np.cos(2 * np.pi * rel_diff)


# add shifts of target column: keep ony when correlation is high up to a certain limit of new columns
def add_target_shifts(all_data, range_min, range_max, corr_thres):
    cols_added = 0
    for lookback in range(range_min, range_max):# [5360, 35040]: # len test, # len 1 year
        shifted_target = all_data[target].shift(lookback)
        corr, _ = pearsonr(shifted_target.head(len(train.index)).fillna(0), all_data.head(len(train.index))[target])
        if corr > corr_thres or corr < corr_thres * -1:
            print(lookback, corr)
            all_data[f"{target}_shifted_{lookback}_days"] = shifted_target
            cols_added += 1
            if cols_added >= 10: # we don't need more dimensions
                return all_data
    return all_data


#all_data = add_target_shifts(all_data, 1, 730, 0.6)


train = all_data.loc[all_data["source"] == 0]
test = all_data.loc[all_data["source"] == 1]

train = train.drop("source", axis=1)
test = test.drop(["source", target], axis=1)


train.info()


print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


nb_train_rows = len(train.index)

train, eval_df = train_test_split(
    train,
    test_size=0.20,
    random_state=300,   # controls the shuffle
    shuffle=True
)

train = train.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)


from bluecast.eda.analyse import (
    bi_variate_plots,
    correlation_heatmap,
    correlation_to_target,
    plot_pca,
    plot_theil_u_heatmap,
    plot_tsne,
    univariate_plots,
    check_unique_values,
    plot_null_percentage,
    mutual_info_to_target
)

from bluecast.preprocessing.feature_types import FeatureTypeDetector
from bluecast.monitoring.data_monitoring import DataDrift


ignore_cols = []

feat_type_detector = FeatureTypeDetector()
train_data = feat_type_detector.fit_transform_feature_types(train.drop(ignore_cols, axis=1))
train_data = train_data.sample(10000, random_state=78)

len(feat_type_detector.num_columns)


train_data_disc = train_data.copy()

est = KBinsDiscretizer(
        n_bins=10, encode='ordinal', strategy='uniform', subsample=None
)
est.fit(train_data_disc[target].values.reshape(-1, 1))
train_data_disc[target] = est.transform(train_data_disc[target].values.reshape(-1, 1))
train_data_disc[target].value_counts()


feat_type_detector.num_columns.remove(target)

data_drift_checker = DataDrift()
# statistical data drift checks for numerical features
data_drift_checker.kolmogorov_smirnov_test(train.loc[:, feat_type_detector.num_columns], test.loc[:, feat_type_detector.num_columns], threshold=0.05)
# show flags
print(data_drift_checker.kolmogorov_smirnov_flags)


# statistical data drift checks for categorical features
data_drift_checker.population_stability_index(train.loc[:, feat_type_detector.cat_columns], test.loc[:, feat_type_detector.cat_columns])
# show flags
print(data_drift_checker.population_stability_index_flags)
# show psi values
print(data_drift_checker.population_stability_index_values)


train = train.drop("id", axis=1)
eval_df = eval_df.drop("id", axis=1)
test = test.drop("id", axis=1)


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


from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    PredictedClasses,  # just for linting checks
    PredictedProbas,  # just for linting checks
)
from catboost import CatBoostRegressor, Pool
import optuna
from optuna.integration import CatBoostPruningCallback

RST = 3

class CustomModel(BaseClassMlModel):
    def __init__(self):
        self.model = None
        
    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        
        eval_dataset = Pool(x_test, y_test, cat_features=[])
        
        #quantile_levels = [0.1, 0.5, 0.90]
        #quantile_str = str(quantile_levels).replace('[','').replace(']','')

        def objective(trial):
            # this part is taken from: https://www.kaggle.com/code/syerramilli/catboost-multi-quantile-regression
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 2000, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 5, 10, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 1e6, log=True),
                'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.1, 1),
                'subsample': trial.suggest_float("subsample", 0.3, 1),
                "sampling_frequency": trial.suggest_categorical("sampling_frequency", ["PerTree", "PerTreeLevel"]),
            }
            model = CatBoostRegressor(
                loss_function="RMSEWithUncertainty", # f'MultiQuantile:alpha={quantile_str}',
                thread_count= 4,
                cat_features=[],
                bootstrap_type="Bernoulli",
                random_seed=RST,
                **param
            )
            #pruning_callback = CatBoostPruningCallback(trial, "learn")
            
            # train model
            try:
                model.fit(x_train, y_train, verbose=0)
            except Exception:
                return 999999

            # get predictions
            preds = model.predict(x_test)

            #quantile_levels = [0.10, 0.5, 0.90]
            
            #preds = pd.DataFrame(
            #    preds,
            #    columns=quantile_levels,
            #    index=x_test.index
            #).reset_index(drop=False)

            # this part comes from: https://www.kaggle.com/code/paddykb/pi-ii-hpp-catboost-rmsewithuncertainty
            mean_preds, var_preds = preds[:,0], preds[:,1]
            # return prediction intervals
            y_min, y_max = y_train.min(), y_train.max()
            pi = np.zeros((len(x_test.index), 2))
            # 1.67 is a fiddle factor to approximate a 90% coverage.
            pi[:, 0] = np.round((mean_preds - np.sqrt(var_preds) * 1.67 ).clip(y_min, y_max))
            pi[:, 1] = np.round((mean_preds + np.sqrt(var_preds) * 1.67 ).clip(y_min, y_max))

            winkler_sc, coverage = winkler_score(
                y_test, 
                pi[:, 0], 
                pi[:, 1], 
                return_coverage=True
            )
            print(f"Winkler={winkler_sc:.4f}, Coverage={coverage:.4f}")

            return winkler_sc# ** (1 - coverage) # penalize bad coverage
        
        sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=1000
            )
        study = optuna.create_study(
            direction="minimize", sampler=sampler, study_name=f"catboost"
        )
        study.optimize(
            objective,
            n_trials=200,
            timeout=60 * 60 * 9,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        best_parameters = study.best_trial.params
        self.model = CatBoostRegressor(
                loss_function="RMSEWithUncertainty", # f'MultiQuantile:alpha={quantile_str}',
                thread_count= 4,
                cat_features=[],
                random_seed=RST,
                bootstrap_type ="Bernoulli",
                **best_parameters
            ).fit(
            x_train,
            y_train,
            eval_set=eval_dataset,
            use_best_model=True,
            early_stopping_rounds=20,
            plot=False,
            verbose=0,
        )
        

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        # predict Catboost classifier
        preds = self.model.predict(df)

        mean_preds, var_preds = preds[:,0], preds[:,1]
        # return prediction intervals
        y_min, y_max = train[target].min(), train[target].max()
        pi = np.zeros((len(df.index), 2))
        # 1.67 is a fiddle factor to approximate a 90% coverage.
        pi[:, 0] = np.round((mean_preds - np.sqrt(var_preds) * 1.67 ).clip(y_min, y_max))
        pi[:, 1] = np.round((mean_preds + np.sqrt(var_preds) * 1.67 ).clip(y_min, y_max))

        preds_df = pd.DataFrame(
            {
                "0.10": pi[:, 0],
                "0.90": pi[:, 1]
            }
        )

        print(preds_df)
        
        return preds.mean(axis=1)


from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig, XgboostTuneParamsRegressionConfig

# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.global_random_state = 600
train_config.calculate_shap_values = False # takes too long without GPU

catboost_model = CustomModel()


automl = BlueCastRegression(
        class_problem="regression", # also multiclass is possible
        #stratifier=skf,
        conf_training=train_config,
        #conf_xgboost=xgboost_param_config,
        ml_model=catboost_model,
        #custom_feature_selector=custom_feature_selector,
        )


automl.fit(train.copy(), target_col=target)


automl.calibrate(eval_df.drop(target, axis=1), eval_df[target])


preds_eval = automl.predict_interval(eval_df.drop(target, axis=1), alphas=[0.10])


prediction_interval_coverage(eval_df[target], preds_eval, alphas=[0.10])


preds = automl.predict_interval(test.copy(), alphas=[0.10])
preds


quantiles = ['0.1', '0.9']

preds.columns = preds.columns.str.rstrip('_low')
preds.columns = preds.columns.str.rstrip('_high')

preds = preds.loc[:, quantiles]
preds = preds.reindex(preds.columns.to_list()[::], axis=1)
preds


quantile_levels = ["pi_lower", "pi_upper"]

submission = submission[["id"]]
for quantile_numeric, quantile in zip(quantile_levels, quantiles):
    submission[quantile_numeric] = preds[quantile]

submission.to_csv('submission_catboost_via_bluecast_conformal_prediction_estimation.csv', index=False)
submission


# save pipeline including tracker
#save_to_production(automl, "/kaggle/working/", "bluecast_cv_pipeline")

# in production or for further experiments this can be loaded again
#automl_loaded = load_for_production("/kaggle/working/", "bluecast_cv_pipeline")




