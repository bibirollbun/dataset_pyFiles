import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

import xgboost as xgb
import lightgbm as lgb

sns.set(style="whitegrid", context="talk")


train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

train.columns


display(train.head())


display(train.describe(include="all").transpose())


missing = train.isnull().mean() * 100
print("Missing values (%):\n", missing[missing > 0].sort_values(ascending=False))


plt.figure(figsize=(10,6))
sns.histplot(train['efs_time'], bins=30, kde=True)
plt.title("Distribution of efs_time")
plt.xlabel("Time")
plt.ylabel("Count")


plt.figure(figsize=(8,4))
sns.countplot(x="race_group", data=train)
plt.title("Distribution of Race Groups")
plt.xticks(rotation=45)
plt.show()


target_cols = ['efs', 'efs_time']
features = train.drop(columns=target_cols)
target = train[target_cols].copy()

target['log_efs_time'] = np.log(target['efs_time'] + 1e-8)


X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()


print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)


num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])



X_train_pre = preprocessor.fit_transform(X_train)
X_val_pre = preprocessor.transform(X_val)


num_features = numerical_cols

cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features = cat_encoder.get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([num_features, cat_features])


X_train_pre_df = pd.DataFrame(X_train_pre, columns=all_feature_names, index=X_train.index)
X_val_pre_df = pd.DataFrame(X_val_pre, columns=all_feature_names, index=X_val.index)


xgb_estimator = ('xgb', xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    device='cuda',
    verbosity=1,
    random_state=42
))

lgb_estimator = ('lgb', lgb.LGBMRegressor(
    objective='regression',
    device_type='gpu',
    gpu_platform_id=0,
    gpu_device_id=0,
    random_state=42
))

ridge_estimator = ('ridge', Ridge())

base_estimators = [xgb_estimator, lgb_estimator, ridge_estimator]


stacked_regressor = StackingRegressor(
    estimators=base_estimators,
    final_estimator=Ridge(),
    cv=5,
    n_jobs=-1
)


ensemble_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stack', stacked_regressor)
])


param_grid = {
    'stack__xgb__n_estimators': [100, 200],
    'stack__xgb__max_depth': [3, 5, 7],
    'stack__xgb__learning_rate': [0.01, 0.1, 0.2],
    'stack__lgb__n_estimators': [100, 200],
    'stack__lgb__max_depth': [3, 5, 7],
    'stack__lgb__learning_rate': [0.01, 0.1, 0.2],
    'stack__final_estimator__alpha': [0.1, 1.0, 10.0]
}


n_iter = 50

grid = RandomizedSearchCV(
    estimator=ensemble_pipeline,
    param_distributions=param_grid,
    n_iter=n_iter,
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=10,
    random_state=42
)


grid.fit(X_train, y_train['log_efs_time'])


print("Best parameters:", grid.best_params_)
print("Best CV MSE:", -grid.best_score_)


best_model = grid.best_estimator_


def concordance_index_custom(times, events, predictions):
    """
    Compute the concordance index.
    times: array-like of survival times.
    events: array-like of event indicators (1 if event occurred, 0 if censored).
    predictions: array-like of risk scores (higher risk corresponds to shorter survival).
    """
    n = len(times)
    num_concordant = 0.0
    num_tied = 0.0
    num_pairs = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            if times[i] < times[j] and events[i] == 1:
                num_pairs += 1
                if predictions[i] > predictions[j]:
                    num_concordant += 1
                elif predictions[i] == predictions[j]:
                    num_tied += 1
            elif times[j] < times[i] and events[j] == 1:
                num_pairs += 1
                if predictions[j] > predictions[i]:
                    num_concordant += 1
                elif predictions[i] == predictions[j]:
                    num_tied += 1
    if num_pairs == 0:
        return np.nan
    return (num_concordant + 0.5 * num_tied) / num_pairs


pred_log_times_val = best_model.predict(X_val)
risk_scores_val = -pred_log_times_val


global_cindex = concordance_index_custom(y_val['efs_time'].values, y_val['efs'].values, risk_scores_val)
print("Global Concordance Index:", global_cindex)


race_groups = X_val['race_group'].unique()
group_cindices = []
print("\nC-index by Race Group:")
for group in race_groups:
    idx = (X_val['race_group'] == group)
    if idx.sum() > 0:
        group_times = y_val.loc[idx, 'efs_time'].values
        group_events = y_val.loc[idx, 'efs'].values
        group_risks = risk_scores_val[idx]
        c_idx = concordance_index_custom(group_times, group_events, group_risks)
        group_cindices.append(c_idx)
        print(f"Race: {group}, C-index: {c_idx:.3f}")


stratified_cindex = np.mean(group_cindices) - np.std(group_cindices)
print("\nStratified C-index (mean - std):", stratified_cindex)


X_test_pre = preprocessor.transform(test)


pred_log_times_test = best_model.predict(test)
risk_scores_test = -pred_log_times_test


submission = pd.DataFrame({
    'ID': test['ID'],
    'prediction': risk_scores_test
})


submission.head()


test.shape


submission.to_csv("submission.csv", index=False)

