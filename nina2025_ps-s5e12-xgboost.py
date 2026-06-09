import numpy as np, pandas as pd

import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

import warnings; warnings.filterwarnings('ignore')


train  = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test   = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig   = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


orig['id'] = orig.index

orig = orig[train.columns.to_list()]

print(f'Train shape: {train.shape}')
print(f'Test shape:  {test.shape}' )
print(f'Orig shape:  {orig.shape}' )

train = train.merge(orig, how='outer')
train.shape


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder that supports multiple aggregation functions,
    internal cross-validation for leakage prevention, and smoothing.

    Parameters
    ----------
    cols_to_encode : list of str
        List of column names to be target encoded.

    aggs : list of str, default=['mean']
        List of aggregation functions to apply. Any function accepted by
        pandas' `.agg()` method is supported, such as:
        'mean', 'std', 'var', 'min', 'max', 'skew', 'nunique',
        'count', 'sum', 'median'.
        Smoothing is applied only to the 'mean' aggregation.

    cv : int, default=5
        Number of folds for cross-validation in fit_transform.

    smooth : float or 'auto', default='auto'
        The smoothing parameter `m`. A larger value puts more weight on the
        global mean. If 'auto', an empirical Bayes estimate is used.

    drop_original : bool, default=False
        If True, the original columns to be encoded are dropped.
    """
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        """
        Learn mappings from the entire dataset.
        These mappings are used for the transform method on validation/test data.
        """
        temp_df = X.copy()
        temp_df['target'] = y

        # Learn global statistics for each aggregation
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        # Learn category-specific mappings
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping

        return self

    def transform(self, X):
        """
        Apply learned mappings to the data.
        Unseen categories are filled with global statistics.
        """
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)

        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit and transform the data using internal cross-validation to prevent leakage.
        """
        # First, fit on the entire dataset to get global mappings for transform method
        self.fit(X, y)

        # Initialize an empty DataFrame to store encoded features
        encoded_features = pd.DataFrame(index=X.index)

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                # --- Calculate mappings only on the training part of the fold ---
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'

                    # Calculate global stat for this fold
                    fold_global_stat = y_train.agg(agg_func)

                    # Calculate category stats for this fold
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    # --- Apply smoothing only for 'mean' aggregation ---
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()

                        m = self.smooth
                        if self.smooth == 'auto':
                            # Empirical Bayes smoothing
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0  # No smoothing if no variance between groups
                        # Apply smoothing formula
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)

                    # Store encoded values for the validation fold
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        # Merge with original DataFrame
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)

        return X_transformed


# bmi score categories
def bmi_classification(score):
  if score >= 30:
    return 'Obese'
  elif score >= 25:
    return 'Overwieght'
  elif score >= 18.5:
    return 'Normal'
  else:
    return 'Underweight'

for d in [train, test, orig]:
  d['bmi_category'] = d['bmi'].apply(bmi_classification)


new_features,features = [], test.columns.to_list()

for col in features:
    # Mean
    tmp      = orig.groupby(col)['diagnosed_diabetes'].mean()
    new_name = 'orig_' + str(col)
    tmp.name = new_name
    train    = train.merge(tmp, how='left', on=col)
    test     = test .merge(tmp, how='left', on=col)
    new_features.append(new_name)
    # Count
    tmp_cnt_name = 'orig_cnt_' + str(col)
    tmp_cnt  = orig .groupby(col).size().reset_index(name=tmp_cnt_name)
    train    = train.merge(tmp_cnt, how='left', on=col)
    test     = test .merge(tmp_cnt, how='left', on=col)
    new_features.append(tmp_cnt_name)

new_base = features + new_features

print(f'New features: {len(new_base)}')


# look at the categorical columns for their unique values

dbase,objs = [],train.select_dtypes(include='object').columns.to_list()

for obj in objs:
    values = train[obj].unique()
    dbase.append({'column':obj, 'unique values':values, 'their number':len(values)})

df = pd.DataFrame(dbase)

df


# convert the object datatype columns into category dtypes

objs = train.select_dtypes(include='object').columns.to_list()

for obj in objs:
    train[obj] = train[obj].astype('category')
    test [obj] = test [obj].astype('category')

train.info()


# label encoding really helped the score improve
# focus only on the object columns for labe encoding
objects = train.select_dtypes('category').columns

for obj in objects:
    le = LabelEncoder()
    le.fit(pd.concat([train[obj],test[obj]], axis=0).astype(str))
    train[obj] = le.transform(train[obj].astype(str))
    test [obj] = le.transform(test [obj].astype(str))


X = train.drop(columns=['diagnosed_diabetes'])
y = train['diagnosed_diabetes']


int_cols = X.select_dtypes(include=['int','float']).columns.to_list()
folds    = 5
skf      = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
oof      = np.zeros(len(X))

xgb_test_preds = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(skf.split(X, y),1):
    print('\n','cross-validation: fold',i,'\n')
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    TE = TargetEncoder(cols_to_encode=int_cols, cv=5, smooth='auto',aggs=['mean'], drop_original=False)
    X_train = TE.fit_transform(X_train, y_train)
    X_test  = TE    .transform(X_test)
    test    = TE    .transform(test)

    neg, pos = y_train.value_counts()
    scale_pos_weight = neg / pos
    
    model = xgb.XGBClassifier(
        n_estimators     = 5_000,
        max_depth        = 7,
        learning_rate    = 0.015,
        subsample        = 0.85,
        colsample_bytree = 0.74,
        min_child_weight = 3,
        gamma            = 1,              # requires more gain to split
        reg_alpha        = 0.3,            # L1 regularization
        reg_lambda       = 3,              # L2 regularization
        scale_pos_weight = scale_pos_weight,
        random_state     = 42,
        device           ='cuda',
        eval_metric      ='auc',
        early_stopping_rounds=300
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=300
    )
    
    oof[test_index] = model.predict_proba(X_test)[:,1]
    xgb_test_preds += model.predict_proba(  test)[:,1] / folds


xgb.plot_importance(model, max_num_features=30)

auc = roc_auc_score(y, oof)

print("\n\nValidation AUC:", auc, '\n') # fold.1 = 0.74662 - 0.78259


submission = pd.DataFrame({'id': sample['id'],'diagnosed_diabetes':xgb_test_preds})
submission.to_csv('submission.csv', index=False)
submission

