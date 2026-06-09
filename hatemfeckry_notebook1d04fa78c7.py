# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


y =  train["y"]
train.drop(columns = ["y"], inplace=True)


df = pd.concat(
    [
        train,
        test
    ]
)
df.head()



month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df['month_numeric'] = df['month'].map(month_mapping).astype(int)


df['quarter'] = ((df['month_numeric'] - 1) // 3) + 1

df['is_year_end'] = df['month_numeric'].isin([11, 12]).astype(int)

df.drop(columns = ["month"], inplace = True)



df['age_balance_ratio'] = df['balance'] / (df['age'] + 1)
df['age_job_interaction'] = df['age'] * df['job'].astype('category').cat.codes


df['campaign_per_previous'] = df['campaign'] / (df['previous'] + 1)
df['total_contacts'] = df['campaign'] + df['previous']


df['has_loan_or_default'] = ((df['loan'] == 'yes') | (df['default'] == 'yes')).astype(int)
df['financial_stress'] = ((df['balance'] < 0) & (df['housing'] == 'yes')).astype(int)


df["month_numeric"].unique()


df['date_str'] = '2024-' + df['month_numeric'].astype(str) + '-' + df['day'].astype(str)
df['date'] = pd.to_datetime(df['date_str'], errors='coerce')


df.info()


age_group_means = df.groupby('age')['balance'].transform('mean')
df['balance_to_age_group_ratio'] = df['balance'] / (age_group_means + 1) # +1 to avoid division by zero


df['balance_rank_in_job'] = df.groupby('job')['balance'].rank(pct=True)

df['contact_recency'] = 1 / (df['pdays'] + 2)
df.loc[df['pdays'] == -1, 'contact_recency'] = 0 

df['day_of_week'] = df['date'].dt.dayofweek

df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)

conditions = [
    (df['age'] < 30) & (df['marital'] == 'single'),
    (df['age'] >= 30) & (df['age'] < 55) & (df['marital'] == 'married'),
    (df['age'] >= 55)
]
choices = ['Young_Single', 'Married_Professional', 'Senior']
df['life_stage'] = np.select(conditions, choices, default='Other')

# Clean up temporary columns
df = df.drop(columns=['date_str', 'date'])

print(df[['balance_to_age_group_ratio', 'day_of_week', 'life_stage']].head())


df.info()


to_be_multiplied = ["duration", "balance", "age", "quarter"]
for feature in to_be_multiplied:
    for target in to_be_multiplied:
        if feature != target:
            df[feature+"_"+target] = df[feature] * df[target]


from sklearn.preprocessing import OneHotEncoder
one_enc = OneHotEncoder(handle_unknown='ignore', sparse_output = False)


test = df[750000:]
df = df[0:750000]


test.drop(columns =["id"], inplace=True)
df.drop(columns = ["id"], inplace = True)


high_cardinality_features = ['month_numeric', 'day']
print(df[high_cardinality_features].shape, y.shape)


from category_encoders import TargetEncoder
high_cardinality_features = ['month_numeric', 'day']

# Initialize the encoder
encoder = TargetEncoder(cols=high_cardinality_features)

# Fit the encoder on the training data only
encoder.fit(df[high_cardinality_features], y)

# Transform both training and test data
train_encoded = encoder.transform(df[high_cardinality_features])
test_encoded = encoder.transform(test[high_cardinality_features])

# Add the encoded features to your datasets
for feature in high_cardinality_features:
    df[f'{feature}_encoded'] = train_encoded[feature]
    test[f'{feature}_encoded'] = test_encoded[feature]

# Drop the original categorical columns
df = df.drop(high_cardinality_features, axis=1)
test = test.drop(high_cardinality_features, axis=1)


df.info()




one_columns = [ 
    "marital", "housing", "loan", "poutcome", "default", "contact", 'job', 'education', 'life_stage',
    "quarter"
]
one_results = one_enc.fit_transform(df[one_columns])
column_names = one_enc.get_feature_names_out(one_columns)
one_df = pd.DataFrame(one_results, columns =column_names, index = df.index)




to_be_scaled = df.drop(columns = one_columns)
to_be_scaled.info()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_array = scaler.fit_transform(to_be_scaled)
scaled_df = pd.DataFrame(scaled_array, 
                         index=to_be_scaled.index, 
                         columns=to_be_scaled.columns)


df = pd.concat([
    to_be_scaled,
    one_df
],
    axis=1
)


# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()

X = df


# model.fit(X, y)


# test.drop(columns = ["day"], inplace = True)





test_one_columns = one_columns
test_one_results = one_enc.transform(test[one_columns])
test_column_names = one_enc.get_feature_names_out(one_columns)
test_one_df = pd.DataFrame(test_one_results, columns =column_names, index = test.index)


test_to_be_scaled = test.drop(columns = test_one_columns)



test_scaled_array = scaler.transform(test_to_be_scaled)
test_scaled_df = pd.DataFrame(
    test_scaled_array,
    index = test_to_be_scaled.index,
    columns = test_to_be_scaled.columns
)


test = pd.concat(
    [
        test_to_be_scaled,
        test_one_df
    ],
    axis = 1
)


sub_sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sub_sample.head()


id_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")["id"]


# import xgboost as xgb
# second_model = xgb.XGBClassifier(random_state=42, n_estimators=200)


# second_model.fit(X, y)


# predictions = second_model.predict(test)
# results = pd.DataFrame()
# results["id"] = id_df
# results["y"] = predictions
# results.head()
# results.to_csv("submission.csv", index=False)


# from sklearn.ensemble import RandomForestClassifier
# third_model = RandomForestClassifier(n_estimators=100, random_state=42)
# third_model.fit(X, y)


# predictions = third_model.predict(test)
# results = pd.DataFrame()
# results["id"] = id_df
# results["y"] = predictions
# results.head()
# results.to_csv("submission.csv", index=False)


# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import roc_auc_score

# param_dist = {
#     'n_estimators': [50, 100, 200, 300, 400, 500],
#     'max_depth': [3, 4, 5, 6, 7, 8],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
#     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'gamma': [0, 0.1, 0.2, 0.3, 0.4],
#     'reg_alpha': [0, 0.1, 0.5, 1],
#     'reg_lambda': [0.5, 1, 1.5, 2]
# }

# random_search = RandomizedSearchCV(
#     estimator=second_model,
#     param_distributions=param_dist,
#     n_iter=100, 
#     scoring='roc_auc',
#     cv=3,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1 
# )

# random_search.fit(X, y)


 # print(random_search.best_params_)


best_params = {'subsample': 0.8, 'reg_lambda': 2, 'reg_alpha': 1, 'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 0.7}


# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# estimators = [
#     ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#     ('xgb', xgb.XGBClassifier(**best_params,random_state=42))
# ]

# stacker = StackingClassifier(
#     estimators=estimators,
#     final_estimator=LogisticRegression(penalty='elasticnet', 
#                                        solver='saga', 
#                                        l1_ratio=0.5,
#                                        random_state=42,
#                                        max_iter=1000),
#     cv=5,
#     n_jobs=-1
# )

# stacker.fit(X, y)




# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import RidgeClassifier
# from sklearn.model_selection import cross_val_score
# import numpy as np


# scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

# xgb_model = xgb.XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42)


# xgb_model.fit(X, y)


X.isna().sum().sum()


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

def create_meta_features(X_train, y_train, X_test, base_models, n_splits=5):
    """
    Create meta-features using k-fold cross-validation
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize arrays for meta-features
    train_meta = np.zeros((X_train.shape[0], len(base_models)))
    test_meta = np.zeros((X_test.shape[0], len(base_models)))
    test_meta_folds = np.zeros((n_splits, X_test.shape[0], len(base_models)))
    
    # Train each base model and create meta-features
    for i, model in enumerate(base_models):
        print(f"Training base model {i+1}/{len(base_models)}")
        
        # For storing test predictions from each fold
        fold_test_preds = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model on training fold
            model.fit(X_tr, y_tr)
            
            # Predict on validation fold (meta-features for training)
            val_preds = model.predict_proba(X_val)[:, 1]
            train_meta[val_idx, i] = val_preds
            
            # Predict on test data (we'll average these later)
            test_preds = model.predict_proba(X_test)[:, 1]
            test_meta_folds[fold, :, i] = test_preds
        
        # Average test predictions across folds
        test_meta[:, i] = test_meta_folds[:, :, i].mean(axis=0)
    
    return train_meta, test_meta

# Usage example:
base_models = [
    HistGradientBoostingClassifier(
        random_state=42, 
        class_weight='balanced'  # For handling class imbalance
    ),
    HistGradientBoostingClassifier(
        random_state=123, 
        class_weight='balanced'  # For handling class imbalance
    ),
    XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42),
    XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=123),
    LGBMClassifier(scale_pos_weight=scale_pos_weight, n_estimators=400, random_state=42),
    LGBMClassifier(scale_pos_weight=scale_pos_weight, n_estimators=400, random_state=123),
    
]

# Create meta-features
train_meta, test_meta = create_meta_features(X, y, test, base_models)

print(train_meta.shape)

# Train meta-learner on meta-features
meta_learner = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
meta_learner.fit(train_meta, y)

# Make predictions on test meta-features
test_predictions = meta_learner.predict_proba(test_meta)[:, 1]


# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
# import numpy as np
# from sklearn.calibration import CalibratedClassifierCV
# import lightgbm as lgbm


# scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

# # 1. Define diverse base models
# estimators = [
#     ('xgb-1', xgb.XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42)),
#     ('xgb-2', xgb.XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=24)),
#     ('lgbm', lgbm.LGBMClassifier(n_estimators=400, class_weight = {0: 1, 1:scale_pos_weight}, random_state = 42))
# ]

# # --- 2. Define Meta-Learner ---
# meta_learner = LogisticRegressionCV(cv=5, scoring='roc_auc', random_state=42)


# stacking_clf = StackingClassifier(
#     estimators=estimators,
#     final_estimator=meta_learner,
#     cv = 3, # Inner CV for generating base model predictions
#     stack_method='predict_proba',
#     n_jobs=-1,
#     passthrough=False 
# )

# stacking_clf.fit(X, y)


# predictions = xgb_model.predict(test)
results = pd.DataFrame()
results["id"] = id_df
results["y"] = test_predictions
results.head()
results.to_csv("submission.csv", index=False)

