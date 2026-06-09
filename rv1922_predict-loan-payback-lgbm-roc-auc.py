import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

TARGET = 'loan_paid_back'

# 2. Feature Definitions
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
NUMS = [col for col in BASE if col not in CATS]

# 3. Feature Engineering: Interactions
INTER = []
TE_BASE = [col for col in BASE if col not in ['annual_income', 'loan_amount']]
for col1, col2 in combinations(TE_BASE, 2):
    new_col_name = f'{col1}_{col2}'
    INTER.append(new_col_name)
    for df in [train, test, orig]:
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)

# 4. Feature Engineering: Rounding
ROUND = []
rounding_levels = {'1s': 0, '10s': -1}
for col in ['annual_income', 'loan_amount']:
    for suffix, level in rounding_levels.items():
        new_col_name = f'{col}_ROUND_{suffix}'
        ROUND.append(new_col_name)
        for df in [train, test, orig]:
            df[new_col_name] = df[col].round(level).astype(int)

# 5. Feature Engineering: Original Data Stats
ORIG = []
for col in BASE:
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

FEATURES = BASE + ORIG + INTER + ROUND
X = train[FEATURES]
y = train[TARGET]

# 6. Target Encoder Class
class TargetEncoder(BaseEstimator, TransformerMixin):
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
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series).fillna(self.global_stats_[agg_func])
        
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
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
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0
                        
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
            
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

# 7. Model Parameters (Updated with Pruning)
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'colsample_bytree': 0.3,
    'subsample': 0.55,
    'learning_rate': 0.01,
    'device': 'cuda',  # Ensure you are on a GPU runtime, else change to 'cpu'
    'random_state': 42,
    
    'gamma': 1.5,
    
    'min_child_weight': 5
}

# 8. Training Loop
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- Fold {fold}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test[FEATURES].copy()

    # Apply Target Encoding
    TE = TargetEncoder(cols_to_encode=INTER, cv=5, smooth=1.0, aggs=['mean'], drop_original=True)
    X_train = TE.fit_transform(X_train, y_train)
    X_val = TE.transform(X_val)
    X_test = TE.transform(X_test)

    TE2 = TargetEncoder(cols_to_encode=ROUND, cv=5, smooth=1.0, aggs=['mean'], drop_original=False)
    X_train = TE2.fit_transform(X_train, y_train)
    X_val = TE2.transform(X_val)
    X_test = TE2.transform(X_test)

    # Convert to Categorical for DMatrix (XGBoost handles categories efficiently)
    X_train[CATS] = X_train[CATS].astype('category')
    X_val[CATS] = X_val[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    # Train
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dval, 'val')],
        early_stopping_rounds=200,
        verbose_eval=1000
    )

    val_preds = model.predict(dval)
    oof_preds[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'Fold {fold} AUC: {fold_score:.4f}')
    
    fpr, tpr, _ = roc_curve(y_val, val_preds)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {fold_score:.4f})')

    test_preds += model.predict(dtest) / N_SPLITS

# 9. Finalize Plot and Scores
overall_auc = roc_auc_score(y, oof_preds)
print(f'====================')
print(f'Overall OOF AUC: {overall_auc:.4f}')
print(f'====================')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (Overall AUC = {overall_auc:.4f})')
plt.legend(loc="lower right")
plt.show()

submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_preds
})
submission.to_csv('submission_XGBoost_Pruned.csv', index=False)

