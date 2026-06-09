import pandas as pd


df_train_m = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df_train = df_train_m.copy()


print(df_train.shape)
print(df_test.shape)


df_train.head()


df_train.isnull().sum()


df_train_numeric = df_train.select_dtypes(include='number')


df_train_numeric.corr()['diagnosed_diabetes'].sort_values(ascending=False)


df_train['gender'].value_counts()


# gender	ethnicity	education_level	income_level	smoking_status	employment_status

def gender(x):
  if(x == 'Female'):
    return 0
  if(x == 'Male'):
    return 1
  else:
    return 2




df_train['gender'] = df_train['gender'].apply(gender)
df_test['gender'] = df_test['gender'].apply(gender)


df_train['ethnicity'].value_counts()


from sklearn.preprocessing import LabelEncoder


from numpy import dtype
le_ethnicity = LabelEncoder()
df_train['ethnicity_encoded'] = le_ethnicity.fit_transform(df_train['ethnicity'])
df_test['ethnicity_encoded'] = le_ethnicity.transform(df_test['ethnicity'])


df_train = pd.get_dummies(df_train, columns=['education_level'],dtype = int)
df_test = pd.get_dummies(df_test, columns=['education_level'],dtype=int)


df_train['income_level'].value_counts()


df_train['employment_status'].value_counts()


df_train = pd.get_dummies(df_train, columns=['employment_status'],dtype=int)
df_test = pd.get_dummies(df_test, columns=['employment_status'],dtype=int)


df_train.dtypes


df_train.describe()


df_train.head()


from sklearn.preprocessing import OrdinalEncoder

income_order = [
    ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
]

oe_income = OrdinalEncoder(categories=income_order)

df_train['income_encoded'] = oe_income.fit_transform(
    df_train[['income_level']]
)

df_test['income_encoded'] = oe_income.transform(
    df_test[['income_level']]
)



df_train['smoking_status'].value_counts()


df_train = pd.get_dummies(df_train, columns=['smoking_status'],dtype=int)
df_test = pd.get_dummies(df_test, columns=['smoking_status'],dtype=int)


df_train['ethnicity'].value_counts()


df_train = pd.get_dummies(df_train, columns=['ethnicity'],dtype=int)
df_test = pd.get_dummies(df_test, columns=['ethnicity'],dtype=int)


df_train.head()


df_train_numeric = df_train.select_dtypes(include='number')


corr_target = df_train_numeric.corr()['diagnosed_diabetes'].sort_values(ascending=False)
print(corr_target)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot
plt.figure(figsize=(10,6))
sns.barplot(
    x=corr_target.values,
    y=corr_target.index,
    palette='coolwarm'
)
plt.title('Correlation of Numeric Features with Diagnosed Diabetes')
plt.xlabel('Correlation coefficient')
plt.ylabel('Features')
plt.show()



import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df_fe = df_train_m.copy()


df_fe['bmi_age'] = df_fe['bmi'] * df_fe['age']
df_fe['waist_to_hip_age'] = df_fe['waist_to_hip_ratio'] / df_fe['age']
df_fe['bp_ratio'] = df_fe['systolic_bp'] / df_fe['diastolic_bp']

df_fe['family_bmi'] = df_fe['family_history_diabetes'] * df_fe['bmi']
df_fe['age_physical_activity'] = df_fe['age'] * df_fe['physical_activity_minutes_per_week']

age_bins = [0, 30, 45, 60, 120]
age_labels = ['<30', '30-45', '45-60', '60+']
df_fe['age_bin'] = pd.cut(df_fe['age'], bins=age_bins, labels=age_labels)

bmi_bins = [0, 18.5, 24.9, 29.9, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df_fe['bmi_cat'] = pd.cut(df_fe['bmi'], bins=bmi_bins, labels=bmi_labels)

df_fe['cholesterol_risk_score'] = df_fe['cholesterol_total'] + df_fe['ldl_cholesterol'] - df_fe['hdl_cholesterol']
df_fe['cardio_risk_score'] = df_fe['hypertension_history'] + df_fe['cardiovascular_history']


income_order = [['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']]
oe_income = OrdinalEncoder(categories=income_order)
df_fe['income_encoded'] = oe_income.fit_transform(df_fe[['income_level']])

one_hot_features = ['gender', 'smoking_status', 'education_level', 'employment_status']
df_fe = pd.get_dummies(df_fe, columns=one_hot_features, drop_first=False,dtype=int)

ethnicity_target_mean = df_fe.groupby('ethnicity')['diagnosed_diabetes'].mean()
df_fe['ethnicity_encoded'] = df_fe['ethnicity'].map(ethnicity_target_mean)

df_fe = df_fe.drop(columns=['ethnicity', 'age', 'bmi', 'waist_to_hip_ratio',
                            'systolic_bp', 'diastolic_bp', 'cholesterol_total',
                            'ldl_cholesterol', 'hdl_cholesterol',
                            'hypertension_history', 'cardiovascular_history'])



df_fe.head()


df_fe = pd.get_dummies(df_fe, columns=['age_bin', 'bmi_cat'], drop_first=False,dtype=int)


df_fe_num = df_fe.select_dtypes(include='number')


corr_target = df_fe_num.corr()['diagnosed_diabetes'].sort_values(ascending=False)
print(corr_target)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot
plt.figure(figsize=(10,6))
sns.barplot(
    x=corr_target.values,
    y=corr_target.index,
    palette='coolwarm'
)
plt.title('Correlation of Numeric Features with Diagnosed Diabetes')
plt.xlabel('Correlation coefficient')
plt.ylabel('Features')
plt.show()



df_fe.dtypes


df_fe['bmi_over25'] = (df_train['bmi'] > 25).astype(int)  # 1 if overweight/obese, 0 otherwise


import pandas as pd
from sklearn.model_selection import train_test_split

df_train_ready = df_fe.copy()

# categorical_cols = ['age_bin', 'bmi_cat']
# df_train_ready = pd.get_dummies(df_train_ready, columns=categorical_cols, drop_first=False)

df_train_ready = df_train_ready.drop(columns=['id', 'income_level'])


X = df_train_ready.drop(columns=['diagnosed_diabetes'])
y = df_train_ready['diagnosed_diabetes']





X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target distribution:\n", y_train.value_counts(normalize=True))


# Replace characters [, ], <, >, spaces with _
X_train.columns = X_train.columns.str.replace(r'[\[\]<> ]', '_', regex=True)
X_test.columns = X_test.columns.str.replace(r'[\[\]<> ]', '_', regex=True)


import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# xgb_clf = xgb.XGBClassifier( #----AUC: 0.7206378823772285
#     objective='binary:logistic',
#     eval_metric='auc',
#     use_label_encoder=False,
#     scale_pos_weight=scale_pos_weight,
#     random_state=42
# )

xgb_clf = xgb.XGBClassifier( #---- AUC: 0.7236812312155625
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1
)

xgb_clf.fit(X_train, y_train)

# Predictions
y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:,1]

# Evaluate
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))



!pip install catboost


from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()

cat_clf = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=10,
    bagging_temperature=0.8,
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
    class_weights=[1, scale_pos_weight],
    early_stopping_rounds=50
)

cat_clf.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

y_proba_cat = cat_clf.predict_proba(X_test)[:,1]
print("Optimized CatBoost AUC:", roc_auc_score(y_test, y_proba_cat))



import lightgbm as lgb
from sklearn.metrics import roc_auc_score

X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
categorical_cols = [c for c in X_train_lgb.columns if 'bin_' in c or 'bmi_cat' in c or 'age_bin' in c or 'smoking_status' in c or 'gender' in c or 'education_level' in c or 'employment_status' in c]
for col in categorical_cols:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')



scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 40,
    'max_depth': 7,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbosity': -1
}



import lightgbm as lgb
from sklearn.metrics import roc_auc_score
train_data = lgb.Dataset(X_train_lgb, label=y_train, categorical_feature=categorical_cols)
test_data = lgb.Dataset(X_test_lgb, label=y_test, reference=train_data, categorical_feature=categorical_cols)

lgb_model=lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[train_data, test_data],
    valid_names=['train','valid'],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(stopping_rounds=50),lgb.log_evaluation(100)]
)

y_proba_lgb=lgb_model.predict(X_test_lgb)
print("LightGBM Optimized AUC:",roc_auc_score(y_test, y_proba_lgb))



best_iter = lgb_model.best_iteration
print("Best iteration:", best_iter)



scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 40,
    'max_depth': 7,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbosity': -1
}

train_data = lgb.Dataset(
    X_train_lgb,
    label=y_train,
    categorical_feature=categorical_cols
)

valid_data = lgb.Dataset(
    X_test_lgb,
    label=y_test,
    reference=train_data,
    categorical_feature=categorical_cols
)

lgb_model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[valid_data],
    valid_names=['valid'],
    num_boost_round=2000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(100)
    ]
)

print("Validation AUC:",
      roc_auc_score(y_test, lgb_model.predict(X_test_lgb)))



best_iter = lgb_model.best_iteration
print("Best iteration:", best_iter)



scale_pos_weight_full = (y == 0).sum() / (y == 1).sum()

lgb_params['scale_pos_weight'] = scale_pos_weight_full



full_train_data = lgb.Dataset(
    X,
    label=y
)




final_model = lgb.train(
    params=lgb_params,
    train_set=full_train_data,
    num_boost_round=best_iter
)




df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df_test.head()


import pandas as pd

df_fe = df_test.copy()

df_fe['bmi_age'] = df_fe['bmi'] * df_fe['age']
df_fe['waist_to_hip_age'] = df_fe['waist_to_hip_ratio'] / df_fe['age']
df_fe['bp_ratio'] = df_fe['systolic_bp'] / df_fe['diastolic_bp']

df_fe['family_bmi'] = df_fe['family_history_diabetes'] * df_fe['bmi']
df_fe['age_physical_activity'] = df_fe['age'] * df_fe['physical_activity_minutes_per_week']

df_fe['age_bin'] = pd.cut(
    df_fe['age'],
    bins=[0, 30, 45, 60, 120],
    labels=['<30', '30-45', '45-60', '60+']
)

df_fe['bmi_cat'] = pd.cut(
    df_fe['bmi'],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)
df_fe['cholesterol_risk_score'] = (
    df_fe['cholesterol_total']
    + df_fe['ldl_cholesterol']
    - df_fe['hdl_cholesterol']
)

df_fe['cardio_risk_score'] = (
    df_fe['hypertension_history']
    + df_fe['cardiovascular_history']
)
df_fe['income_encoded'] = oe_income.transform(
    df_fe[['income_level']]
)
one_hot_features = [
    'gender',
    'smoking_status',
    'education_level',
    'employment_status'
]

df_fe = pd.get_dummies(
    df_fe,
    columns=one_hot_features,
    drop_first=False,
    dtype=int
)

#(USING TRAIN MAPPING)
df_fe['ethnicity_encoded'] = (
    df_fe['ethnicity']
    .map(ethnicity_target_mean)
    .fillna(ethnicity_target_mean.mean())
)
df_fe = df_fe.drop(columns=[
    'ethnicity',
    'age',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'cholesterol_total',
    'ldl_cholesterol',
    'hdl_cholesterol',
    'hypertension_history',
    'cardiovascular_history',
    'income_level'
])



X_test_final = df_fe.reindex(
    columns=X.columns,
    fill_value=0
)



test_pred_proba = final_model.predict(X_test_final)


# submission = pd.DataFrame({
#     'id': df_test['id'],
#     'diagnosed_diabetes': test_pred_proba
# })

# submission.to_csv('submission.csv', index=False)


