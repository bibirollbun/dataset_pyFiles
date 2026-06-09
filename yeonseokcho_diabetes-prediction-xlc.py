import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=Warning)


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
print(sample_submission.shape)
sample_submission.head()


test= pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
print(test.shape)
test.head()


train= pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
print(train.shape)
train.head()


train.info()


train.describe(include="all").T


train['diagnosed_diabetes'].value_counts()
# 1.0 diagnosed with diabetes, 0.0 not diagnosed with diabetes


df = train.copy()
df['diagnosed_diabetes'] = df['diagnosed_diabetes'].astype('category')
df['diagnosed_diabetes'].value_counts()
df['diagnosed_diabetes'] = df['diagnosed_diabetes'].map({1.0: 'Diagnosed', 0.0: 'Not_Diagnosed'})
df['diagnosed_diabetes'].value_counts()


# Categorical df with target variable
df_cat_target = df.select_dtypes(include=['object', 'category']) 
print(df_cat_target.shape)
df_cat_target.head()


# Counts and proportions of diabetes status by Categorical variable

def plot_dist(df, col):

    #ct = pd.crosstab(df[col], df['diagnosed_diabetes'], normalize='index')
    #sorted_order = ct.sort_values(by='Diagnosed', ascending=True).index

    plt.figure(figsize=(10, 3))

    # Countplot 
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x=col, hue='diagnosed_diabetes')

    plt.title(f'{col} (Count)')
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Proportion Plot
    #df[col] = pd.Categorical(df[col], categories=sorted_order, ordered=True)

    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=col, hue='diagnosed_diabetes', multiple='fill')

    plt.title(f'{col} (Proportion)')
    plt.ylabel('Proportion')
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()


plot_dist(df_cat_target, 'gender')


plot_dist(df_cat_target, 'ethnicity')


plot_dist(df_cat_target, 'education_level')


plot_dist(df_cat_target, 'income_level')


plot_dist(df_cat_target, 'smoking_status')


plot_dist(df_cat_target, 'employment_status')


# Numeric features with target variable
df_num = df.select_dtypes(include=['int', 'float'])
df_num_target = pd.concat([df_num, df['diagnosed_diabetes']], axis=1)
print(df_num_target.shape)
df_num_target.head()


# Density distributions and proportions of diabetes status by Numerical variable
def plot_num_dist(df, col, bins=50):

    plt.figure(figsize=(10, 3))

    # Distribution plot
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df, x=col, hue='diagnosed_diabetes', fill=True)
    
    plt.title(f'{col} Distribution (Density)')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Proportion plot
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=col, hue='diagnosed_diabetes', multiple='fill', bins=bins)
    
    plt.title(f'{col} Proportion by Target')
    plt.xlabel(col)
    plt.ylabel('Proportion')
    
    # range 1~99%
    plt.xlim(df[col].quantile(0.01), df[col].quantile(0.99))
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


plot_num_dist(df_num_target, 'age')


plot_num_dist(df_num_target, 'alcohol_consumption_per_week')


plot_num_dist(df_num_target, 'physical_activity_minutes_per_week')


plot_num_dist(df_num_target, 'diet_score')


plot_num_dist(df_num_target, 'sleep_hours_per_day')


plot_num_dist(df_num_target, 'screen_time_hours_per_day')


plot_num_dist(df_num_target, 'bmi')


plot_num_dist(df_num_target, 'waist_to_hip_ratio')


plot_num_dist(df_num_target, 'systolic_bp')


plot_num_dist(df_num_target, 'diastolic_bp')


plot_num_dist(df_num_target, 'heart_rate')


plot_num_dist(df_num_target, 'cholesterol_total')


# "good" cholesterol
plot_num_dist(df_num_target, 'hdl_cholesterol') 


# "bad" cholesterol
plot_num_dist(df_num_target, 'ldl_cholesterol')


plot_num_dist(df_num_target, 'triglycerides')


plot_num_dist(df_num_target, 'family_history_diabetes')


plot_num_dist(df_num_target, 'hypertension_history')


plot_num_dist(df_num_target, 'cardiovascular_history')


print(df.shape)
df.head()


# Select numeric columns to process (excluding ID, Target, and categorical variables)
numeric_cols = df_num_target.columns

exclude_cols = ['id', 'diagnosed_diabetes', 'hypertension_history', 
                'cardiovascular_history', 'family_history_diabetes']
cols_to_clip = [c for c in numeric_cols if c not in exclude_cols]

# Processing outliers (Clipping 1% ~ 99%)
for col in cols_to_clip:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    
# Clip values outside this range to the boundary values (maintaining data count)
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print(df.shape)
df.head()


# Generation features / Train

# Part 1 - Lipid Metabolism & Insulin Resistance Related Features
# 1. Triglycerides / HDL Ratio (TG_HDL_Ratio)
df['TG_HDL_Ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)

# 2. LDL / HDL Ratio (LDL_HDL_Ratio)
df['LDL_HDL_Ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)

# 3. Non-HDL Cholesterol (Non_HDL)
df['Non_HDL'] = df['cholesterol_total'] - df['hdl_cholesterol']

# Part 2 - Hemodynamic Stress & Cardiovascular Load
# 4. Pulse Pressure
df['Pulse_Pressure'] = df['systolic_bp'] - df['diastolic_bp']

# 5. Mean Arterial Pressure (MAP)
df['MAP'] = df['diastolic_bp'] + (df['Pulse_Pressure'] / 3)

# 6. Rate Pressure Product (RPP)
df['Rate_Pressure_Product'] = df['heart_rate'] * df['systolic_bp']

# 7. Cardio-Metabolic Stress Index
df['Cardio_Stress_Index'] = df['systolic_bp'] * df['cholesterol_total']

# Part 3 - Lifestyle & Anthropometry Indices
# 8. Sleep Deviation
df['Sleep_Deviation'] = abs(df['sleep_hours_per_day'] - 7.5)

# 9. Physical Activity to Screen Time Ratio (Activity_Screen_Ratio)
df['Activity_Screen_Ratio'] = df['physical_activity_minutes_per_week'] / (df['screen_time_hours_per_day'] * 60 + 1)

# 10. Obesity Index
df['Obesity_Index'] = df['bmi'] * df['waist_to_hip_ratio']

# 11. Accelerated Aging Index (BMI_x_Age)
df['BMI_x_Age'] = df['bmi'] * df['age']

# Part 4 - Socioeconomic Status (SES)
edu_map = {'No formal': 0, 'Primary': 1, 'Highschool': 2, 'Graduate': 3, 'Postgraduate': 4}
inc_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}

# 12. Education Score (Edu_Score)
df['Edu_Score'] = df['education_level'].map(edu_map).fillna(1)

# 13. Income Score (Inc_Score)
df['Inc_Score'] = df['income_level'].map(inc_map).fillna(1)

# 14. Socioeconomic Status Score (SES_Score)
df['SES_Score'] = df['Edu_Score'] + df['Inc_Score']

# 15. Retired & Overweight Risk Group (Retired_Overweight)
df['Retired_Overweight'] = ((df['employment_status'] == 'Retired') & (df['bmi'] >= 25)).astype(int)

# Part 5 - Metabolic Syndrome Risk Score
# 16. Initialize Metabolic Syndrome Score
df['Metabolic_Risk_Score'] = 0

# Condition 1: Obesity (BMI >= 30)
df['Metabolic_Risk_Score'] += (df['bmi'] >= 30).astype(int)

# Condition 2: Hypertension (Systolic >= 130 OR Diastolic >= 85)
df['Metabolic_Risk_Score'] += ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)).astype(int)

# Condition 3: High Triglycerides (>= 150)
df['Metabolic_Risk_Score'] += (df['triglycerides'] >= 150).astype(int)

# Condition 4: Low HDL (< 45)
df['Metabolic_Risk_Score'] += (df['hdl_cholesterol'] < 45).astype(int)

# Part 6 - Categorical Interactions
# 17. Combine Gender & Smoking Status 
df['Gender_Smoking'] = df['gender'].astype(str) + "_" + df['smoking_status'].astype(str)

print(df.shape)
df.head()


df["Metabolic_Risk_Score"].value_counts()


# Select numeric columns to process (excluding ID, Target, and categorical variables)
test_num = test.select_dtypes(include=['int', 'float']) 

test_num_cols = test_num.columns

exclude_cols = ['id', 'hypertension_history', 
                'cardiovascular_history', 'family_history_diabetes']
cols_to_clip = [c for c in test_num_cols if c not in exclude_cols]

# Processing outliers (Clipping 1% ~ 99%)
for col in cols_to_clip:
    lower_bound = test[col].quantile(0.01)
    upper_bound = test[col].quantile(0.99)
    
# Clip values outside this range to the boundary values (maintaining data count)
    test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)

print(test.shape)
test.head()


# Generation features test

# Part 1 - Lipid Metabolism & Insulin Resistance Related Features
# 1. Triglycerides / HDL Ratio (TG_HDL_Ratio)
test['TG_HDL_Ratio'] = test['triglycerides'] / (test['hdl_cholesterol'] + 1)

# 2. LDL / HDL Ratio (LDL_HDL_Ratio)
test['LDL_HDL_Ratio'] = test['ldl_cholesterol'] / (test['hdl_cholesterol'] + 1)

# 3. Non-HDL Cholesterol (Non_HDL)
test['Non_HDL'] = test['cholesterol_total'] - test['hdl_cholesterol']

# Part 2 - Hemodynamic Stress & Cardiovascular Load
# 4. Pulse Pressure
test['Pulse_Pressure'] = test['systolic_bp'] - test['diastolic_bp']

# 5. Mean Arterial Pressure (MAP)
test['MAP'] = test['diastolic_bp'] + (test['Pulse_Pressure'] / 3)

# 6. Rate Pressure Product (RPP)
test['Rate_Pressure_Product'] = test['heart_rate'] * test['systolic_bp']

# 7. Cardio-Metabolic Stress Index
test['Cardio_Stress_Index'] = test['systolic_bp'] * test['cholesterol_total']

# Part 3 - Lifestyle & Anthropometry Indices
# 8. Sleep Deviation
test['Sleep_Deviation'] = abs(test['sleep_hours_per_day'] - 7.5)

# 9. Physical Activity to Screen Time Ratio (Activity_Screen_Ratio)
test['Activity_Screen_Ratio'] = test['physical_activity_minutes_per_week'] / (test['screen_time_hours_per_day'] * 60 + 1)

# 10. Obesity Index
test['Obesity_Index'] = test['bmi'] * test['waist_to_hip_ratio']

# 11. Accelerated Aging Index (BMI_x_Age)
test['BMI_x_Age'] = test['bmi'] * test['age']

# Part 4 - Socioeconomic Status (SES)
edu_map = {'No formal': 0, 'Primary': 1, 'Highschool': 2, 'Graduate': 3, 'Postgraduate': 4}
inc_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}

# 12. Education Score (Edu_Score)
test['Edu_Score'] = test['education_level'].map(edu_map).fillna(1)

# 13. Income Score (Inc_Score)
test['Inc_Score'] = test['income_level'].map(inc_map).fillna(1)

# 14. Socioeconomic Status Score (SES_Score)
test['SES_Score'] = test['Edu_Score'] + test['Inc_Score']

# 15. Retired & Overweight Risk Group (Retired_Overweight)
test['Retired_Overweight'] = ((test['employment_status'] == 'Retired') & (test['bmi'] >= 25)).astype(int)

# Part 5 - Metabolic Syndrome Risk Score
# 16. Initialize Metabolic Syndrome Score
test['Metabolic_Risk_Score'] = 0

# Condition 1: Obesity (BMI >= 30)
test['Metabolic_Risk_Score'] += (test['bmi'] >= 30).astype(int)

# Condition 2: Hypertension (Systolic >= 130 OR Diastolic >= 85)
test['Metabolic_Risk_Score'] += ((test['systolic_bp'] >= 130) | (test['diastolic_bp'] >= 85)).astype(int)

# Condition 3: High Triglycerides (>= 150)
test['Metabolic_Risk_Score'] += (test['triglycerides'] >= 150).astype(int)

# Condition 4: Low HDL (< 45)
test['Metabolic_Risk_Score'] += (test['hdl_cholesterol'] < 45).astype(int)

# Part 6 - Categorical Interactions
# 17. Combine Gender & Smoking Status 
test['Gender_Smoking'] = test['gender'].astype(str) + "_" + test['smoking_status'].astype(str)

print(test.shape)
test.head()


feature_num = df.select_dtypes(include=['int', 'float']) 
test_num = test.select_dtypes(include=['int', 'float']) 
print(feature_num.shape, test_num.shape)
feature_num.head()


# Categorical feature
feature_cat = df.select_dtypes(include=['object', 'category']) 
feature_cat = feature_cat.drop(['diagnosed_diabetes'], axis=1)
print(feature_cat.shape)
feature_cat.head()


# Categorical test
test_cat = test.select_dtypes(include=['object', 'category']) 
print(test_cat.shape)
test_cat.head()


# for XGBoost
from sklearn.preprocessing import LabelEncoder

# Create copies to preserve the original datasets
feature_cat_le = feature_cat.copy()
test_cat_le = test_cat.copy()
cat_cols = feature_cat.columns.tolist()

# 2. Apply Label Encoding
for c in cat_cols:
    le = LabelEncoder()
    # Fit on combined Train and Test data (prevents errors with unseen categories in the Test set)
    all_data = pd.concat([feature_cat[c], test_cat[c]], axis=0).astype(str)
    le.fit(all_data)
    
    # Transform each dataset
    feature_cat_le[c] = le.transform(feature_cat[c].astype(str))
    test_cat_le[c] = le.transform(test_cat[c].astype(str))

# 3. Combine with numerical features (Final data for XGBoost)
feature_le = pd.concat([feature_cat_le, feature_num], axis=1)
test_le = pd.concat([test_cat_le, test_num], axis=1)

# 4. Drop unnecessary ID column
feature_le = feature_le.drop(['id'], axis=1, errors='ignore')
test_le = test_le.drop(['id'], axis=1, errors='ignore')

print(feature_le.shape, test_le.shape) # for XGB
feature_le.head()


# for LGBM/CatBoost

cat_cols = feature_cat.columns.tolist()

# 2. Convert to 'category' type (Key step)
for c in cat_cols:
    feature_cat[c] = feature_cat[c].astype('category')
    test_cat[c] = test_cat[c].astype('category')

# 3. Combine with numerical features (Final data for LGBM/CatBoost)
feature_final = pd.concat([feature_cat, feature_num], axis=1)
test_final = pd.concat([test_cat, test_num], axis=1)

# 4. Drop unnecessary ID column
feature_final = feature_final.drop(['id'], axis=1, errors='ignore')
test_final = test_final.drop(['id'], axis=1, errors='ignore')

print(feature_final.shape, test_final.shape) # for LGB/Catboost
feature_final.head()


# target
target = train[['diagnosed_diabetes']]
print(target.shape)
target.head()


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# target
y = target['diagnosed_diabetes']

# 1. Data for XGBoost (Label Encoded data)
X_train_le, X_val_le, y_train, y_val = train_test_split(
    feature_le, y, test_size=0.2, random_state=2512, stratify=y
)

# 2. Data for LightGBM & CatBoost (non_treated))
X_train_final, X_val_final, _, _ = train_test_split(
    feature_final, y, test_size=0.2, random_state=2512, stratify=y
)

print(X_train_le.shape, X_val_le.shape, y_train.shape, y_val.shape) #for XGB
print(X_train_final.shape, X_val_final.shape, y_train.shape, y_val.shape) #for LGB, Catboost


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# XGB model
xgb_model = XGBClassifier(
    n_estimators=2000, 
    learning_rate=0.05, 
    max_depth=6,
    random_state=2512, 
    n_jobs=-1, 
    enable_categorical=False,
    early_stopping_rounds=50, 
    eval_metric=['logloss','auc'] 
)

# learning
xgb_model.fit(
    X_train_le, y_train,
    eval_set=[(X_train_le, y_train), (X_val_le, y_val)],
    verbose=0
)

# evaluation
train_proba = xgb_model.predict_proba(X_train_le)[:, 1]
val_proba = xgb_model.predict_proba(X_val_le)[:, 1]

train_auc_xgb = roc_auc_score(y_train, train_proba)
val_auc_xgb = roc_auc_score(y_val, val_proba)

print(f"XGBoost Train AUC: {train_auc_xgb:.8f}")
print(f"XGBoost Val AUC:   {val_auc_xgb:.8f}")
print(f"Gap (Train - Val): {train_auc_xgb - val_auc_xgb:.8f}")

# XGBoost Train AUC: 0.76373122
# XGBoost Val AUC:   0.72538095
# Gap (Train - Val): 0.03835027


from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# Model
lgbm_model = LGBMClassifier(
    n_estimators=2000, 
    learning_rate=0.05, 
    max_depth=6,
    random_state=2512, 
    n_jobs=-1, 
    metric='auc', 
    verbosity=-1)

# learning
lgbm_model.fit(
    X_train_final, y_train,
    eval_set=[(X_train_final, y_train), (X_val_final, y_val)],
    eval_metric=['binary_logloss', 'auc'],
    callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)])

# evaluation
train_proba_lgbm = lgbm_model.predict_proba(X_train_final)[:, 1]
val_proba_lgbm = lgbm_model.predict_proba(X_val_final)[:, 1]

train_auc_lgbm = roc_auc_score(y_train, train_proba_lgbm)
val_auc_lgbm = roc_auc_score(y_val, val_proba_lgbm)

print(f"LightGBM Train AUC: {train_auc_lgbm:.8f}")
print(f"LightGBM Val AUC:   {val_auc_lgbm:.8f}")
print(f"Gap (Train - Val):  {train_auc_lgbm - val_auc_lgbm:.8f}")

# LightGBM Train AUC: 0.75408046
# LightGBM Val AUC:   0.72661167
# Gap (Train - Val):  0.02746879


from catboost import CatBoostClassifier

# catboost model
cat_model = CatBoostClassifier(
    n_estimators=2000, 
    learning_rate=0.05, 
    depth=6,
    random_state=2512, 
    eval_metric='AUC', 
    early_stopping_rounds=50)

# learning
cat_model.fit(
    X_train_final, y_train,
    cat_features=cat_cols, 
    eval_set=[(X_train_final, y_train), (X_val_final, y_val)],
    use_best_model=True, 
    verbose=0)

# evaluation
train_proba_cat = cat_model.predict_proba(X_train_final)[:, 1]
val_proba_cat = cat_model.predict_proba(X_val_final)[:, 1]

train_auc_cat = roc_auc_score(y_train, train_proba_cat)
val_auc_cat = roc_auc_score(y_val, val_proba_cat)

print(f"CatBoost Train AUC: {train_auc_cat:.8f}")
print(f"CatBoost Val AUC:   {val_auc_cat:.8f}")
print(f"Gap (Train - Val):  {train_auc_cat - val_auc_cat:.8f}")

# CatBoost Train AUC: 0.74249578
# CatBoost Val AUC:   0.72645667
# Gap (Train - Val):  0.01603911


# Visualizing Training History

def plot_history(model_name, train_loss, val_loss, train_auc, val_auc):
    epochs = range(len(train_loss))
    
    plt.figure(figsize=(10, 2))
    
    plt.plot(epochs, train_loss, label='loss', color='tab:blue')
    plt.plot(epochs, train_auc, label='auc', color='tab:orange')
    plt.plot(epochs, val_loss, label='val_loss', color='tab:green')
    plt.plot(epochs, val_auc, label='val_auc', color='tab:red')
    
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Score / Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.2, 1.0) 
    plt.show()


# Visualizing Training History

# 1. XGBoost
results_xgb = xgb_model.evals_result()
plot_history(
    'XGBoost',
    results_xgb['validation_0']['logloss'],
    results_xgb['validation_1']['logloss'],
    results_xgb['validation_0']['auc'],
    results_xgb['validation_1']['auc']
)

# 2. LightGBM
results_lgbm = lgbm_model.evals_result_
plot_history(
    'LightGBM',
    results_lgbm['training']['binary_logloss'],
    results_lgbm['valid_1']['binary_logloss'],
    results_lgbm['training']['auc'],
    results_lgbm['valid_1']['auc']
)

# 3. CatBoost
results_cat = cat_model.get_evals_result()

plot_history(
    'CatBoost',
    results_cat['validation_0']['Logloss'], 
    results_cat['validation_1']['Logloss'], 
    results_cat['validation_0']['AUC'],     
    results_cat['validation_1']['AUC']
)


from sklearn.inspection import permutation_importance

importances = xgb_model.feature_importances_

perm_importance_df_xgb = pd.DataFrame({
    'feature': feature_final.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

perm_importance_df_xgb.head(15)


# Extract feature importance using LightGBM's built-in function

from sklearn.inspection import permutation_importance

importance_type = 'gain' 

importances = lgbm_model.booster_.feature_importance(importance_type=importance_type)
perm_importance_df = pd.DataFrame({
    'feature': feature_final.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

perm_importance_df.head(15)


plt.figure(figsize=(12, 6))
sns.barplot(
    x='feature', 
    y='importance',  
    data=perm_importance_df # 
)

plt.title('LightGBM Feature Importance (Type: Gain)', fontsize=12)
plt.xlabel('Feature', fontsize=10)
plt.ylabel('Importance Score (Gain)', fontsize=10) 

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


TOP_N = 8  # select top 25 features

selected_features = perm_importance_df.head(TOP_N)['feature'].tolist()

print(f"Selected {len(selected_features)} features:")
print(selected_features)

common_features_le = [f for f in selected_features if f in X_train_le.columns]
X_train_le_selected = X_train_le[common_features_le]
X_val_le_selected = X_val_le[common_features_le]

common_features_final = [f for f in selected_features if f in X_train_final.columns]
X_train_final_selected = X_train_final[common_features_final]
X_val_final_selected = X_val_final[common_features_final]

print(f"Original Shape (LGBM): {X_train_final.shape}")
print(f"Selected Shape (LGBM): {X_train_final_selected.shape}")

print(f"XGBoost : {X_train_le_selected.shape}, {X_val_le_selected.shape}")
print(f"LGBM/Cat: {X_train_final_selected.shape}, {X_val_final_selected.shape}")
print(f"Target  : {y_train.shape}, {y_val.shape}")


# XGB Model with selected features
xgb_model_sel = XGBClassifier(
    n_estimators=2000, 
    learning_rate=0.05, 
    max_depth=6,
    random_state=2512, 
    n_jobs=-1, 
    enable_categorical=False,
    early_stopping_rounds=50, 
    eval_metric=['logloss','auc'] 
)

# Learning 
xgb_model_sel.fit(
    X_train_le_selected, y_train,
    eval_set=[(X_train_le_selected, y_train), (X_val_le_selected, y_val)],
    verbose=0
)

# Evaluation
train_sel_proba = xgb_model_sel.predict_proba(X_train_le_selected)[:, 1]
val_sel_proba = xgb_model_sel.predict_proba(X_val_le_selected)[:, 1]

train_sel_auc_xgb = roc_auc_score(y_train, train_sel_proba)
val_sel_auc_xgb = roc_auc_score(y_val, val_sel_proba)

print(f"XGBoost Train_sel AUC: {train_sel_auc_xgb:.8f}")
print(f"XGBoost Val_sel AUC:   {val_sel_auc_xgb:.8f}")
print(f"Gap (Train_sel - Val_sel): {train_sel_auc_xgb - val_sel_auc_xgb:.8f}")

# XGBoost Train_sel AUC: 0.74832485
# XGBoost Val_sel AUC:   0.72518465
# Gap (Train_sel - Val_sel): 0.02314020


# LightGBM Model with selected features

lgbm_model_sel = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    random_state=2512,
    n_jobs=-1,
    metric='auc',
    verbosity=-1
)

# Learning
lgbm_model_sel.fit(
    X_train_final_selected, y_train,
    eval_set=[(X_train_final_selected, y_train), (X_val_final_selected, y_val)],
    eval_metric=['binary_logloss', 'auc'],
    callbacks=[
        early_stopping(stopping_rounds=50, verbose=False),
        log_evaluation(period=0)
    ]
)

# Evaluation
train_sel_proba_lgbm = lgbm_model_sel.predict_proba(X_train_final_selected)[:, 1]
val_sel_proba_lgbm = lgbm_model_sel.predict_proba(X_val_final_selected)[:, 1]

train_sel_auc_lgbm = roc_auc_score(y_train, train_sel_proba_lgbm)
val_sel_auc_lgbm = roc_auc_score(y_val, val_sel_proba_lgbm)

print(f"LightGBM Train_sel AUC: {train_sel_auc_lgbm:.8f}")
print(f"LightGBM Val_sel AUC:   {val_sel_auc_lgbm:.8f}")
print(f"Gap (Train_sel - Val_sel): {train_sel_auc_lgbm - val_sel_auc_lgbm:.8f}")

# LightGBM Train_sel AUC: 0.74344505
# LightGBM Val_sel AUC:   0.72624189
# Gap (Train_sel - Val_sel): 0.01720316


# CatBoost Model with selected features

cat_cols_sel = [c for c in cat_cols if c in X_train_final_selected.columns]

cat_model_sel = CatBoostClassifier(
    n_estimators=2000, 
    learning_rate=0.05, 
    depth=6,
    random_state=2512, 
    eval_metric='AUC', 
    early_stopping_rounds=50
)

cat_model_sel.fit(
    X_train_final_selected, y_train,
    cat_features=cat_cols_sel,  
    eval_set=[(X_train_final_selected, y_train), (X_val_final_selected, y_val)],
    use_best_model=True, 
    verbose=0
)

train_sel_proba_cat = cat_model_sel.predict_proba(X_train_final_selected)[:, 1]
val_sel_proba_cat = cat_model_sel.predict_proba(X_val_final_selected)[:, 1]
train_sel_auc_cat = roc_auc_score(y_train, train_sel_proba_cat)
val_sel_auc_cat = roc_auc_score(y_val, val_sel_proba_cat)

print(f"CatBoost Train_sel AUC: {train_sel_auc_cat:.8f}")
print(f"CatBoost Val_sel AUC:   {val_sel_auc_cat:.8f}")
print(f"Gap (Train_sel - Val_sel): {train_sel_auc_cat - val_sel_auc_cat:.8f}")

# CatBoost Train_sel AUC: 0.73779441
# CatBoost Val_sel AUC:   0.72628309
# Gap (Train_sel - Val_sel): 0.01151132


# Visualizing Training History

# 1. XGBoost
results_xgb = xgb_model_sel.evals_result()
plot_history(
    'XGBoost',
    results_xgb['validation_0']['logloss'],
    results_xgb['validation_1']['logloss'],
    results_xgb['validation_0']['auc'],
    results_xgb['validation_1']['auc']
)

# 2. LightGBM
results_lgbm = lgbm_model_sel.evals_result_
plot_history(
    'LightGBM',
    results_lgbm['training']['binary_logloss'],
    results_lgbm['valid_1']['binary_logloss'],
    results_lgbm['training']['auc'],
    results_lgbm['valid_1']['auc']
)

# 3. CatBoost
results_cat = cat_model_sel.get_evals_result()

plot_history(
    'CatBoost',
    results_cat['validation_0']['Logloss'], 
    results_cat['validation_1']['Logloss'], 
    results_cat['validation_0']['AUC'],     
    results_cat['validation_1']['AUC']
)


# XGBoost 
pred_sel_xgb = xgb_model_sel.predict_proba(X_val_le_selected)[:, 1]
train_sel_proba_xgb = xgb_model_sel.predict_proba(X_train_le_selected)[:, 1]

# LightGBM 
pred_sel_lgbm = lgbm_model_sel.predict_proba(X_val_final_selected)[:, 1]
train_sel_proba_lgbm = lgbm_model_sel.predict_proba(X_train_final_selected)[:, 1]

# CatBoost 
pred_sel_cat = cat_model_sel.predict_proba(X_val_final_selected)[:, 1]
train_sel_proba_cat = cat_model_sel.predict_proba(X_train_final_selected)[:, 1]

# weight optimization by Grid Search
auc_best_weighted = 0
weights_best = {}
step = 0.01

# Finding best ensemble weights
# LightGBM weight loop (0 ~ 1.0)
for weight_lgb in np.arange(0, 1.0 + step, step):
    
    # XGBoost weight loop (0 ~ (1.0 - LGBM weight))
    for weight_xgb in np.arange(0, 1.0 - weight_lgb + step, step):
        
        # CatBoost weight 
        weight_cat = 1.0 - weight_lgb - weight_xgb
        
        if weight_cat < -1e-9:
            continue
            
        # Voting ensemble 
        preds_ensemble_weighted = (weight_lgb * pred_sel_lgbm + 
                                   weight_xgb * pred_sel_xgb + 
                                   weight_cat * pred_sel_cat)
        
        # AUC Check
        auc_current = roc_auc_score(y_val, preds_ensemble_weighted)
        
        # Best Update
        if auc_current > auc_best_weighted:
            auc_best_weighted = auc_current
            weights_best = {
                'LightGBM': weight_lgb,
                'XGBoost': weight_xgb,
                'CatBoost': weight_cat
            }

print(f"Best Ensemble Val AUC: {auc_best_weighted:.8f}")
print("Best Weights:")
for model_name, weight in weights_best.items():
    print(f" - {model_name}: {weight:.4f}")

final_train_pred = (weights_best['LightGBM'] * train_sel_proba_lgbm + 
                    weights_best['XGBoost'] * train_sel_proba_xgb + 
                    weights_best['CatBoost'] * train_sel_proba_cat)

final_val_pred = (weights_best['LightGBM'] * pred_sel_lgbm + 
                  weights_best['XGBoost'] * pred_sel_xgb + 
                  weights_best['CatBoost'] * pred_sel_cat)

final_train_auc = roc_auc_score(y_train, final_train_pred)
final_val_auc = roc_auc_score(y_val, final_val_pred)

print(f"Final Ensemble Train AUC: {final_train_auc:.8f}")
print(f"Final Ensemble Val AUC:   {final_val_auc:.8f}")
print(f"Gap (Train - Val):        {final_train_auc - final_val_auc:.8f}")

# Final Ensemble Train AUC: 0.74153243
# Final Ensemble Val AUC:   0.72667495
# Gap (Train - Val):        0.01485748


test_le_selected = test_le[X_train_le_selected.columns]
test_final_selected = test_final[X_train_final_selected.columns]
test_le_selected.shape, test_final_selected.shape


pred_test_xgb = xgb_model_sel.predict_proba(test_le_selected)[:, 1]
pred_test_lgbm = lgbm_model_sel.predict_proba(test_final_selected)[:, 1]
pred_test_cat = cat_model_sel.predict_proba(test_final_selected)[:, 1]

test_pred = (0 * pred_test_lgbm + 
             0 * pred_test_xgb + 
             1 * pred_test_cat)
test_pred

# catboost model 


submission = pd.DataFrame({'id': test.id, 'diagnosed_diabetes': test_pred})
print(submission.shape)
submission.head()


submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()







