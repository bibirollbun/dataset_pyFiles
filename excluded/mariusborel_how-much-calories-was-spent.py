import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import r2_score, mean_squared_log_error, mean_squared_error, roc_auc_score, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold, TimeSeriesSplit as TSS)

import sklearn
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, plot_importance, cv
from sklearn.compose import make_column_transformer
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

import shap

import optuna

import warnings
warnings.filterwarnings('ignore')

# verify the versions of my tools
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'sklearn version: {sklearn.__version__}')
# print(f'optuna version : {optuna.__version__}')


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test_raw = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
orig_raw = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv').drop(columns=['User_ID'])
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

target = 'Calories'

orig_raw.columns = train_raw.columns
train_raw.head(3)


# Should we engeneer the features?
create_new_features = True

# Define function for features engeneering
class Feature_Eng(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()
        if create_new_features:
            df['Max_heart_rate'] = 207 - 0.7*df['Age']
            df['%_heart_rate'] = df['Heart_Rate']/df['Max_heart_rate']
            df['heart_rate_/_weight'] = df['Heart_Rate']/df['Weight']
            df['Heart_Rate_x_Duration'] = df['Heart_Rate']*df['Duration']
            df['bmi'] = df['Weight']/(0.01*df['Height'])**2
            df['log_weight_x_height'] = np.log(df['Weight']*df['Height'])
            df['body_temp_x_weight'] = df['Body_Temp']*np.log(df['Weight'])
            df['Body_Temp - 37'] = df['Body_Temp'] - 37 # assuming 37 as body temp at rest
            df['heat'] = df['Weight']*df['Body_Temp - 37']
            df['work'] = df['heat']*df['Duration']/60
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=[1, 2, 3, 4, 5]).astype('int')
            df['Age_pred_max_heart_rate'] = pd.cut(df['Age'], 
                                                   bins=[10, 30, 35, 40, 45, 50, 55, 60, 65, 70, 90], 
                                                   labels=[200, 190, 185, 180, 175, 170, 165, 160, 155, 150]
                                                  ).astype('int')
            df['diff_heart_rate'] = df['Max_heart_rate'] - df['Age_pred_max_heart_rate']
            df['Sex'] = df['Sex'] == 'female'
            df['Temp_Weight_ratio'] = df['Body_Temp']/df['Weight']
            df['HeartRate_BodyTemp_ratio'] = df['Heart_Rate']/df['Body_Temp']
        else:
            df['Sex'] = df['Sex'] == 'female'
        
        return df


prep_X = train_raw.copy()
prep_X = Feature_Eng().fit_transform(prep_X)
prep_X.head(3)


plt.figure(figsize=(10, 9))
prep_corr = prep_X.corr().abs()
sns.heatmap(prep_corr, annot=True, fmt='.2f', cmap='YlGn', cbar=False, vmin=0.8);


class DropHighCorrFeatures(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df, threshold=0.97):
        # Calculate the correlation matrix
        corr_matrix = df.corr().abs()
        # Select the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Initialize a set to keep track of features to drop
        to_drop = set()
        
        # Iterate over the columns and find features with correlation greater than the threshold
        for column in upper.columns:
            if any(upper[column] > threshold):
                # Add the column to the set of features to drop
                to_drop.add(column)
                # Remove the correlated columns from further consideration
                correlated_columns = upper.index[upper[column] > threshold].tolist()
                to_drop.update(correlated_columns)
        
        # Drop the features from the dataset
        df.drop(to_drop, axis=1, inplace=True)
    
        return df


X_prep = prep_X.copy()
y_prep = X_prep.pop(target)


model_shap = XGBRegressor()
model_shap.fit(X_prep, y_prep)

explainer_xgb = shap.TreeExplainer(model_shap)
shap_values_xgb = explainer_xgb.shap_values(X_prep)

shap_values_mean = np.mean(shap_values_xgb, axis=0)

shap.summary_plot(shap_values_xgb, X_prep, cmap='YlGn')


prep_X = train_raw.copy()
prep_X.pop(target)

prep_treator = make_pipeline(Feature_Eng(), DropHighCorrFeatures())
prep_X = prep_treator.fit_transform(prep_X)

num_feats = prep_X.select_dtypes(include='number').columns.tolist()

prep_X.head()


def data_target_prep(df):
    X = df.copy()
    y = X.pop(target)
    y_log = np.log1p(y)

    return X, y, y_log


X, y, y_log = data_target_prep(train_raw)

X_or, y_or, y_log_or = data_target_prep(orig_raw)


X_tr, X_va, y_log_tr, y_log_va = train_test_split(X, y_log, test_size=0.1, random_state=36)

[d.shape for d in [X_tr, X_va, y_log_tr, y_log_va]]


model = make_pipeline(Feature_Eng(), 
                      DropHighCorrFeatures(), 
                      XGBRegressor(n_estimators=2000))
model.fit(X_tr, y_log_tr)


y_va_hat = model.predict(X_va)
va_rmse = np.sqrt(mean_squared_error(y_log_va, y_va_hat))
va_r2 = model.score(X_va, y_log_va)
y_or_hat = model.predict(X_or)
or_r2 = model.score(X_or, y_log_or)
or_rmse = np.sqrt(mean_squared_error(y_log_or, y_or_hat))

plt.figure(figsize=(9,4))
plt.subplot(121)
sns.scatterplot(x=y_log_va, y=y_va_hat, color='green')
plt.xlabel('True')
plt.ylabel('preds')
plt.title('va_rmse: {:.4} || va_r2: {:.4}'.format(va_rmse, va_r2))
plt.subplot(122)
sns.scatterplot(x=y_log_or, y=y_or_hat, color='gold')
plt.xlabel('True')
plt.title('or_rmse: {:.4} || or_r2: {:.4}'.format(or_rmse, or_r2))
plt.tight_layout()


spliter = KFold(n_splits=6, shuffle=True, random_state=15)

reg_pipe = make_pipeline(Feature_Eng(), 
                      DropHighCorrFeatures(), 
                      XGBRegressor(n_estimators=2000))

plt.figure(figsize=(14,8))
for f, (tr_ind, va_ind) in enumerate(spliter.split(X, y_log), start=1):
    X_tr, X_va = X.loc[tr_ind], X.loc[va_ind]
    y_tr, y_va = y_log.loc[tr_ind], y_log.loc[va_ind]
    # fit and predict for the fold
    reg_pipe.fit(X_tr, y_tr)
    y_va_hat = reg_pipe.predict(X_va)
    y_va_hat = np.clip(y_va_hat, orig_raw[target].min(), orig_raw[target].max())
    # score the prediction for the fold
    rmsle_score = np.sqrt(mean_squared_error(y_va, y_va_hat))
    r2 = r2_score(y_va, y_va_hat)
    # plot a scatter plot to compare predicted vs true values
    plt.subplot(2,3,f)
    sns.scatterplot(x=y_va, y=y_va_hat, palette='Set1')
    plt.title("Fold_{} || RMSLE score: {:.4f} || r2: {:.4}".format(f, rmsle_score, r2), color='yellowgreen')
    plt.xlabel('true_values')
    plt.ylabel('predicted')
plt.tight_layout()


3 # Fit the model
final_model = make_pipeline(Feature_Eng(), 
                      DropHighCorrFeatures(), 
                      XGBRegressor(n_estimators=2000)).fit(X, y_log)
# predict on test data
pred_log = np.clip(final_model.predict(test_raw),
                orig_raw[target].min(),
                orig_raw[target].max()
               )
# Convert the preds from log
preds = np.expm1(pred_log)
# Assign the preds to the submission data
sample_submission[target] = preds

display(sample_submission.head(10))

# Save the submission file
sample_submission.to_csv('submission.csv', index=False)
print('Your predictions are ready for submission!')

