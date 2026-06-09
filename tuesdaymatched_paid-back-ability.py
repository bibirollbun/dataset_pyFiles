import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import optuna
import itertools


df = pd.read_csv(f'/kaggle/input/playground-series-s5e11/train.csv')
df_test = pd.read_csv(f'/kaggle/input/playground-series-s5e11/test.csv')


sns.countplot(df, x = 'loan_paid_back')
plt.xlabel('Will this customers pay their loan')
plt.ylabel('How many?')
plt.title('Customers count for payment abilities' )
plt.xticks([0, 1], ['Not Paid', 'Paid'], rotation=20)
plt.tight_layout()
plt.show()


category = df.select_dtypes('object').columns
fig, axes = plt.subplots(3, 2, figsize=(30, 20))
axes = axes.flatten()

for ax, i in zip(axes, category):
    counts = df[i].value_counts()
    ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white'}
    )
    ax.set_title(f'mean of accident risk for: {i}')
    ax.set_xlabel(i)
    ax.set_ylabel('Customer can pay their loan')
    for i in ax.get_xticklabels():
        i.set_rotation(45)
plt.tight_layout()
plt.show()


df['loan_per_income'] = df['loan_amount'] / df['annual_income']
df['income_per_debt'] = df['annual_income'] / (df['loan_amount'] + 1)
df['grade'] = df['grade_subgrade'].str[0]
df['subgrade'] = df['grade_subgrade'].str[1].astype(int)

grade_map = {g: i for i, g in enumerate(list('ABCDEFG'), 1)}
df['grade_num'] = df['grade'].map(grade_map)
df['risk_score'] = (df['grade_num'] - 1) * 5 + df['subgrade']
df['credit_minus_risk'] = df['credit_score'] - df['risk_score']
df.drop(columns = ['grade_subgrade','grade','subgrade'], inplace = True)


df_test['loan_per_income'] = df_test['loan_amount'] / df_test['annual_income']
df_test['income_per_debt'] = df_test['annual_income'] / (df_test['loan_amount'] + 1)
df_test['grade'] = df_test['grade_subgrade'].str[0]
df_test['subgrade'] = df_test['grade_subgrade'].str[1].astype(int)

grade_test = {g: i for i, g in enumerate(list('ABCDEFG'), 1)}
df_test['grade_num'] = df_test['grade'].map(grade_test)
df_test['risk_score'] = (df_test['grade_num'] - 1) * 5 + df_test['subgrade']
df_test['credit_minus_risk'] = df_test['credit_score'] - df_test['risk_score']
df_test.drop(columns = ['grade_subgrade','grade','subgrade'], inplace = True)


from sklearn.base import BaseEstimator, TransformerMixin

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


CAT_FEATURE = df.select_dtypes(include = ['object']).columns
X = df.drop(columns= ['id','loan_paid_back'])
y = df['loan_paid_back']
X_test = df_test.drop(columns = ['id'])



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        
        te = TargetEncoder(cols_to_encode=CAT_FEATURE ,aggs=['mean','std'], 
                       cv=5, smooth=1.0, drop_original=True)
        X_train = te.fit_transform(X_train, y_train) 
        X_val = te.transform(X_val)
X_test = te.transform(X_test)


X_new = pd.concat([X_train, X_val])
y_new = pd.concat([y_train, y_val])


df_new = X_new.copy()
df_new['id'] = df['id']
df_new['loan_paid_back'] = y_new.values
df_new.to_csv('Train_encode_new.csv', index=False)


df_test_new = X_test.copy()
df_test_new['id'] = df_test['id']
df_test_new.to_csv('Test_encode_new.csv', index=False)

