# for more information : https://www.kaggle.com/competitions/playground-series-s5e10/discussion/612990

!pip install xgboost -U -q # should be xgboost 3.1.1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

TARGET = 'loan_paid_back'
N_SPLITS = 11
SEED = 42


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig_df = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

print(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}, Orig Shape: {orig_df.shape}")


train_df['is_train'] = 1
test_df['is_train'] = 0

combined_df = pd.concat([train_df, test_df.assign(loan_paid_back=0)], ignore_index=True)

print(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}, Orig Shape: {orig_df.shape}")
print(f"Combined Shape: {combined_df.shape}")


CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
BASE_FEATURES = [col for col in train_df.columns if col not in ['id', TARGET, 'is_train']]


print(train_df[TARGET].value_counts(normalize=True))


print(orig_df[TARGET].value_counts(normalize=True))


print(combined_df[TARGET].value_counts(normalize=True))


ORIG_FEATURES = []


for col in BASE_FEATURES:

    agg_df = orig_df.groupby(col)[TARGET].agg(['mean', 'size']).reset_index()
    
    new_mean_col_name = f"orig_mean_{col}"
    new_count_col_name = f"orig_count_{col}"
    agg_df.rename(columns={'mean': new_mean_col_name, 'size': new_count_col_name}, inplace=True)
    
    combined_df = combined_df.merge(agg_df, on=col, how='left')
    
    ORIG_FEATURES.extend([new_mean_col_name, new_count_col_name])

combined_df.annual_income.astype(int)
print(f"{len(ORIG_FEATURES)} features created from original data.")


bins = [-1, 2000, 3000, 100000]
labels = ['rare_emp_status', 'medium_freq_emp_status', 'high_freq_emp_status']
new_freq_feature = 'employment_status_freq_cat'

combined_df[new_freq_feature] = pd.cut(
    combined_df['orig_count_employment_status'],
    bins=bins,
    labels=labels
).astype('category')

print(f"new feature '{new_freq_feature}' created.")
print(combined_df[new_freq_feature].value_counts())


ALL_FEATURES = BASE_FEATURES + ORIG_FEATURES + [new_freq_feature]
CATS.append(new_freq_feature)


combined_df["annual_income"] = (combined_df["annual_income"]//100)*100
combined_df["loan_amount"] = (combined_df["loan_amount"]//100)*100

combined_df['credit_score_cat'] = pd.qcut(combined_df['credit_score'], q=15, labels=False, duplicates='drop')
ALL_FEATURES.append('credit_score_cat')

combined_df['dti_cat'] = pd.qcut(combined_df['debt_to_income_ratio'], q=15, labels=False, duplicates='drop')
ALL_FEATURES.append('dti_cat')


CATS = []
NUMS = []
for c in ALL_FEATURES:
    t = "CAT"
    if combined_df[c].dtype=='object':
        CATS.append(c)
    else:
        NUMS.append(c)
        t = "NUM"
    n = combined_df[c].nunique()
    na = combined_df[c].isna().sum()
    print(f"[{t}] {c} has {n} unique and {na} NA")
print("CATS:", CATS )
print("NUMS:", NUMS )


for col in CATS:
    combined_df[col] = combined_df[col].astype('category')
    
train_final = combined_df[combined_df['is_train'] == 1].drop(columns=['is_train'])
test_final = combined_df[combined_df['is_train'] == 0].drop(columns=['is_train', TARGET])

X = train_final[ALL_FEATURES]
y = train_final[TARGET]
X_test = test_final[ALL_FEATURES]

print(f"Features {len(ALL_FEATURES)}")


from sklearn.model_selection import StratifiedKFold, KFold
skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_estimators': 10_000,
    'learning_rate': 0.01,
    'early_stopping_rounds': 1000,
    'random_state': SEED,
    'n_jobs': -1,
    'device': 'cuda', # 'cpu'
    'enable_categorical': True
    # 'scale_pos_weight': 1
}


# taken from: https://www.kaggle.com/code/masayakawamata/s5e11-te-xgb-interaction-features
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder with support for multiple aggregation functions,
    cross-validation leakage prevention, and smoothing.
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
        temp_df = X.copy()
        temp_df['target'] = y

        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)


        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        
        return self

    def transform(self, X):
        X_transformed = X.copy()

        for col in self.cols_to_encode:
            if pd.api.types.is_categorical_dtype(X_transformed[col]):
                X_transformed[col] = X_transformed[col].astype(str)

        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X_transformed[col].astype(str).map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
        
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx].copy()

            for col in self.cols_to_encode:
                if pd.api.types.is_categorical_dtype(X_train[col]):
                    X_train[col] = X_train[col].astype(str)
                    X_val[col] = X_val[col].astype(str)

            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()

                        m = self.smooth
                        if self.smooth == 'auto':
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            m = avg_variance_within / variance_between if variance_between > 0 else 0

                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)

                    encoded_values_filled = encoded_values.fillna(fold_global_stat).astype(float)
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values_filled

        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
            
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed



oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importances = pd.DataFrame(index=ALL_FEATURES)

COLS_TO_ENCODE = [
    'grade_subgrade', 
    'gender', 
    'marital_status', 
    'education_level', 
    'loan_purpose'
]

CATS_FOR_XGB = [col for col in CATS if col not in COLS_TO_ENCODE]

gc.collect()


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- Fold {fold}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    

    X_test_fold = X_test.copy()


    TE = TargetEncoder(cols_to_encode=COLS_TO_ENCODE, cv=5  , smooth='auto', aggs=['mean'], drop_original=True)
    

    X_train = TE.fit_transform(X_train, y_train)
    

    X_val = TE.transform(X_val)
    X_test_fold = TE.transform(X_test_fold)


    for col in CATS_FOR_XGB:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')
        X_test_fold[col] = X_test_fold[col].astype('category')



    model = XGBClassifier(**params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=1000)


    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'Fold {fold} AUC: {fold_score:.4f}')
    

    test_preds += model.predict_proba(X_test_fold)[:, 1] / N_SPLITS


overall_auc = roc_auc_score(y, oof_preds)
print(f'Overall OOF AUC: {overall_auc:.4f}')


feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
    'feature': X_train.columns, 
    'importance': feature_importances
})

importance_df = importance_df.sort_values('importance', ascending=False)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 20))
sns.barplot(x='importance', 
            y='feature', 
            data=importance_df.head(20)) 
plt.title(f'Feature Importance ({N_SPLITS} FOLDS model)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()

plt.show()


# # refit 
# final_params = params.copy()
# final_params['n_estimators'] = model.best_iteration 
# del final_params['early_stopping_rounds']

# final_model = XGBClassifier(**final_params)
# final_model.fit(X, y, verbose=False)

# plt.rcParams["figure.dpi"] = 160      
# fig, ax = plt.subplots(figsize=(15, 20))

# xgb.plot_importance(
#     final_model,
#     max_num_features=20,
#     importance_type="gain",
#     ax=ax,
#     show_values=False,                
#     grid=False
# )

# ax.set_title("XGB Feature Importances", fontsize=18)
# ax.tick_params(axis="both", labelsize=12)
# fig.tight_layout()

# plt.show()

# # del final_model
# # gc.collect()


pd.DataFrame({'id': train_df.id, TARGET: oof_preds}).to_csv(f'oof_xgb+te_{N_SPLITS}FOLDS_cv_{overall_auc:.5}.csv', index=False)
pd.DataFrame({'id': test_df.id, TARGET: test_preds}).to_csv(f'oof_xgb+te_{N_SPLITS}FOLDS_cv_{overall_auc:.5}_submission.csv', index=False)




