import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
                   
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
        
import warnings
warnings.filterwarnings("ignore")


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
print(sample_submission.shape)
sample_submission.head()


test= pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
print(test.shape)
test.head()


train= pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
print(train.shape)
train.head()


train.info()


# NaN Counts in each columns
nan_count = train.isna().sum()

# NaN ratios in each columns
nan_percent = (nan_count / len(train)) * 100

print(nan_percent)


# NaN Counts in each columns
nan_count = test.isna().sum()

# NaN ratios in each columns
nan_percent = (nan_count / len(test)) * 100

print(nan_percent)


# Class Ratios

plt.figure(figsize=(12,3))
colors = sns.color_palette('pastel')

plt.subplot(1, 3, 1)    
train['Personality'].value_counts().plot.pie(counterclock=False, autopct='%.0f%%', colors = colors)
plt.title('Personality') # non_balanced target data

plt.subplot(1, 3, 2)    
train['Stage_fear'].value_counts().plot.pie(counterclock=False, autopct='%.0f%%', colors = colors)
plt.title('Stage_fear')

plt.subplot(1, 3, 3)    
train['Drained_after_socializing'].value_counts().plot.pie(counterclock=False, autopct='%.0f%%', colors = colors)
plt.title('Drained_after_socializing')

plt.show() 


print(train['Stage_fear'].unique())
train.Stage_fear.value_counts(dropna=False)
# -> np.nan


print(train['Drained_after_socializing'].unique())
train.Drained_after_socializing.value_counts(dropna=False)
# -> np.nan


# numeric data

train_num = train.select_dtypes(include=['float', 'int'])
train_num = train_num.drop(['id'], axis=1)
print(train_num.shape)
train_num.head()


train_num_target = pd.concat([train_num, train[['Personality']]], axis=1)

print(train_num_target.shape)
train_num_target.head()


# Distribution Plot
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
sns.violinplot(data=train_num_target, x='Personality', y='Time_spent_Alone', inner='quartile')
plt.title('Time_spent_Alone')
plt.xlabel('Personality')
plt.ylabel('Time_spent_Alone')

plt.subplot(2, 3, 2)
sns.violinplot(data=train_num_target, x='Personality', y='Social_event_attendance', inner='quartile')
plt.title('Social_event_attendance')
plt.xlabel('Personality')
plt.ylabel('Social_event_attendance')

plt.subplot(2, 3, 3)
sns.violinplot(data=train_num_target, x='Personality', y='Going_outside', inner='quartile')
plt.title('Going_outside')
plt.xlabel('Personality')
plt.ylabel('Going_outside')

plt.subplot(2, 3, 4)
sns.violinplot(data=train_num_target, x='Personality', y='Friends_circle_size', inner='quartile')
plt.title('Friends_circle_size')
plt.xlabel('Personality')
plt.ylabel('Friends_circle_size')

plt.subplot(2, 3, 5)
sns.violinplot(data=train_num_target, x='Personality', y='Post_frequency', inner='quartile')
plt.title('Post_frequency')
plt.xlabel('Personality')
plt.ylabel('Post_frequency')

plt.tight_layout()
plt.show()


# Categoric data
train_cat = train.select_dtypes(include=['object']).columns
train_cat = train[train_cat]
print(train_cat.shape)
train_cat.head()


# Categoric data
test_cat = test.select_dtypes(include=['object']).columns
test_cat = test[test_cat]
print(test_cat.shape)
test_cat.head()



from sklearn.preprocessing import LabelEncoder

# 1. data copy
train_cat_filled = train_cat.copy()

# 2. convert all kinds missing values to np.nan
train_cat_filled['Stage_fear_enc'] = train_cat_filled['Stage_fear'].astype(str)
train_cat_filled['Drained_after_socializing_enc'] = train_cat_filled['Drained_after_socializing'].astype(str)
print(train_cat_filled['Stage_fear_enc'].unique())

train_cat_filled['Stage_fear_enc'].replace('nan', np.nan, inplace=True)
train_cat_filled['Drained_after_socializing_enc'].replace('nan', np.nan, inplace=True)
print(train_cat_filled['Stage_fear_enc'].unique())


# 1. test data copy
test_cat_filled = test_cat.copy()

# 2. convert all kinds missing values to np.nan
test_cat_filled['Stage_fear_enc'] = test_cat_filled['Stage_fear'].astype(str)
test_cat_filled['Drained_after_socializing_enc'] = test_cat_filled['Drained_after_socializing'].astype(str)
print(test_cat_filled['Stage_fear_enc'].unique())

test_cat_filled['Stage_fear_enc'].replace('nan', np.nan, inplace=True)
test_cat_filled['Drained_after_socializing_enc'].replace('nan', np.nan, inplace=True)
print(test_cat_filled['Stage_fear_enc'].unique())


# 3. Convert to numerical labels with LabelEncoder
le_stage = LabelEncoder()
le_drained = LabelEncoder()

# np.nan -> missing -> labelencoding
train_cat_filled['Stage_fear_enc_temp'] = train_cat_filled['Stage_fear_enc'].fillna('missing')
train_cat_filled['Drained_after_socializing_enc_temp'] = train_cat_filled['Drained_after_socializing_enc'].fillna('missing')

test_cat_filled['Stage_fear_enc_temp'] = test_cat_filled['Stage_fear_enc'].fillna('missing')
test_cat_filled['Drained_after_socializing_enc_temp'] = test_cat_filled['Drained_after_socializing_enc'].fillna('missing')

train_cat_filled['Stage_fear_enc'] = le_stage.fit_transform(train_cat_filled['Stage_fear_enc_temp'])
test_cat_filled['Stage_fear_enc'] = le_stage.transform(test_cat_filled['Stage_fear_enc_temp'])

train_cat_filled['Drained_after_socializing_enc'] = le_drained.fit_transform(train_cat_filled['Drained_after_socializing_enc_temp'])
test_cat_filled['Drained_after_socializing_enc'] = le_drained.fit_transform(test_cat_filled['Drained_after_socializing_enc_temp'])

train_cat_filled.head()


test_cat_filled.head()


# 4. restore np.nan values & drop temp columns
train_cat_filled.loc[train_cat_filled['Stage_fear_enc_temp'] == 'missing', 'Stage_fear_enc'] = np.nan
train_cat_filled.loc[train_cat_filled['Drained_after_socializing_enc_temp'] == 'missing', 'Drained_after_socializing_enc'] = np.nan

test_cat_filled.loc[test_cat_filled['Stage_fear_enc_temp'] == 'missing', 'Stage_fear_enc'] = np.nan
test_cat_filled.loc[test_cat_filled['Drained_after_socializing_enc_temp'] == 'missing', 'Drained_after_socializing_enc'] = np.nan

train_cat_filled = train_cat_filled.drop(['Stage_fear_enc_temp', 'Drained_after_socializing_enc_temp'], axis=1)
test_cat_filled = test_cat_filled.drop(['Stage_fear_enc_temp', 'Drained_after_socializing_enc_temp'], axis=1)

print(train_cat_filled.shape, test_cat_filled.shape)
train_cat_filled.head()


test_cat_filled.head()


# 5. apply KNNImputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3) #5 -> 1

imputed_train_array = imputer.fit_transform(train_cat_filled[['Stage_fear_enc', 'Drained_after_socializing_enc']])
imputed_test_array = imputer.transform(test_cat_filled[['Stage_fear_enc', 'Drained_after_socializing_enc']])

# convert to int
train_cat_filled['Stage_fear_enc'] = imputed_train_array[:, 0].round().astype(int)
train_cat_filled['Drained_after_socializing_enc'] = imputed_train_array[:, 1].round().astype(int)

test_cat_filled['Stage_fear_enc'] = imputed_test_array[:, 0].round().astype(int)
test_cat_filled['Drained_after_socializing_enc'] = imputed_test_array[:, 1].round().astype(int)

print(train_cat_filled.shape, test_cat_filled.shape)
train_cat_filled.head()


test_cat_filled.head()


# 5. drop unnecessary columns 
train_cat_filled.drop(columns=['Stage_fear', 'Drained_after_socializing'], inplace=True)
test_cat_filled.drop(columns=['Stage_fear', 'Drained_after_socializing'], inplace=True)

print(train_cat_filled.isna().sum().sum(), test_cat_filled.isna().sum().sum())
print(train_cat_filled.shape, test_cat_filled.shape)
train_cat_filled.head()


test_cat_filled.head()


# numeric data
train_num = train.select_dtypes(include=['float', 'int']).columns
train_num = train[train_num]
train_num = train_num.drop(['id'], axis=1)

test_num = test.select_dtypes(include=['float', 'int']).columns
test_num = test[test_num]
test_num = test_num.drop(['id'], axis=1)

print(train_num.shape, test_num.shape)
train_num.head()


train_num.isna().sum()


# 1. Converted outliers to NaN 
Q1 = train_num.quantile(0.25)  
Q3 = train_num.quantile(0.75)  
IQR = Q3 - Q1

lower_bound = Q1 - 2.0 * IQR
upper_bound = Q3 + 2.0 * IQR

for col in train_num.columns:
    outliers_idx = (train_num[col] < lower_bound[col]) | (train_num[col] > upper_bound[col])
    train_num.loc[outliers_idx, col] = np.nan

train_num.isna().sum()


test_num.isna().sum()


for col in train_num.columns:
    outliers_idx = (test_num[col] < lower_bound[col]) | (test_num[col] > upper_bound[col])
    test_num.loc[outliers_idx, col] = np.nan

test_num.isna().sum()


# 2. imputed missing values using KNN
train_num_filled = imputer.fit_transform(train_num)
test_num_filled = imputer.transform(test_num)

train_num_filled = pd.DataFrame(train_num_filled, columns=train_num.columns, index=train_num.index)
test_num_filled = pd.DataFrame(test_num_filled, columns=test_num.columns, index=test_num.index)

print(train_num_filled.isna().sum().sum(), test_num_filled.isna().sum().sum())
train_num_filled.head()


test_num_filled.head()


# 3. Robust Scaling
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(train_num_filled)  

train_num_scaled = scaler.transform(train_num_filled)
train_num_scaled = pd.DataFrame(train_num_scaled, columns=train_num_filled.columns, index=train_num_filled.index)

test_num_scaled = scaler.transform(test_num_filled)
test_num_scaled = pd.DataFrame(test_num_scaled, columns=test_num_filled.columns, index=test_num_filled.index)

print(train_num_scaled.shape, test_num_scaled.shape)
train_num_scaled.head()


train_scaled = pd.concat([train_cat_filled, train_num_scaled], axis=1)
print(train_scaled.shape)
train_scaled.head()


test_scaled = pd.concat([test_cat_filled, test_num_scaled], axis=1)
print(test_scaled.shape)
test_scaled.head()


# correlation with Personality
train_corr = train_scaled.copy()
train_corr['Personality'] = train_corr['Personality'].replace(['Introvert', 'Extrovert'], [0, 1])
corrmat = train_corr.corr()
print(corrmat.iloc[0])


# correlation heatmap

plt.figure(figsize=(6, 6))
k=8
cols = corrmat.nlargest(k, 'Personality').index 
cols_abs_sorted = corrmat['Personality'].abs().nlargest(k).index # absolute corr values
cm = np.corrcoef(train_corr[cols_abs_sorted].values.T)
sns.set(font_scale=1.)
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.3f', annot_kws={'size': 10}, 
                 yticklabels=cols_abs_sorted.values, xticklabels=cols_abs_sorted.values, 
                 cmap="rainbow", cbar_kws={"shrink": 0.8})
hm.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.show()


# generate new features
train_scaled_enl = train_scaled.copy()

# 1. Social media activity score
train_scaled_enl['Social_Activity_Score'] = (train_scaled_enl['Post_frequency'] * 
                                             train_scaled_enl['Social_event_attendance'])

# 2. Stage_fear_enc 'Yes' plus Drained_after_socializing_enc 'Yes'
train_scaled_enl['Fear_and_Drained'] = (train_scaled_enl['Stage_fear_enc'] + 
                                        train_scaled_enl['Drained_after_socializing_enc'])

# 3. Product of social event attendance and going outside 
train_scaled_enl['Event_Outside_Product'] = (train_scaled_enl['Social_event_attendance'] * 
                                             train_scaled_enl['Going_outside'])

# 4. Whether the person has many friends 
#train_scaled_enl['Many_Friends'] = (train_scaled_enl['Friends_circle_size'] > 
#                                    train_scaled_enl['Friends_circle_size'].median()).astype(int)

# 5. Whether the person spends a lot of time alone
train_scaled_enl['Much_Alone'] = (train_scaled_enl['Time_spent_Alone'] > 
                                  train_scaled_enl['Time_spent_Alone'].median()).astype(int)

# 6. Whether the person is an active poster on social media 
#train_scaled_enl['Active_Poster'] = (train_scaled_enl['Post_frequency'] > 
#                                    train_scaled_enl['Post_frequency'].median()).astype(int)

# 7. Extroversion score 
train_scaled_enl['Extroversion_Score'] = (train_scaled_enl['Social_event_attendance'] + 
                                          train_scaled_enl['Going_outside'] + 
                                          train_scaled_enl['Friends_circle_size'] + 
                                          train_scaled_enl['Post_frequency'] - 
                                          train_scaled_enl['Time_spent_Alone'] - 
                                          5 * train_scaled_enl['Stage_fear_enc'] - 
                                          5 * train_scaled_enl['Drained_after_socializing_enc'])

print(train_scaled_enl.shape)
train_scaled_enl.head()


# generate new features
test_scaled_enl = test_scaled.copy()

# 1. Social media activity score
test_scaled_enl['Social_Activity_Score'] = (test_scaled_enl['Post_frequency'] * 
                                             test_scaled_enl['Social_event_attendance'])

# 2. Stage_fear_enc 'Yes' plus Drained_after_socializing_enc 'Yes'
test_scaled_enl['Fear_and_Drained'] = (test_scaled_enl['Stage_fear_enc'] + 
                                        test_scaled_enl['Drained_after_socializing_enc'])

# 3. Product of social event attendance and going outside 
test_scaled_enl['Event_Outside_Product'] = (test_scaled_enl['Social_event_attendance'] * 
                                             test_scaled_enl['Going_outside'])

# 4. Whether the person has many friends
#test_scaled_enl['Many_Friends'] = (test_scaled_enl['Friends_circle_size'] > 
#                                    test_scaled_enl['Friends_circle_size'].median()).astype(int)

# 5. Whether the person spends a lot of time alone
test_scaled_enl['Much_Alone'] = (test_scaled_enl['Time_spent_Alone'] > 
                                  test_scaled_enl['Time_spent_Alone'].median()).astype(int)

# 6. Whether the person is an active poster on social media
#test_scaled_enl['Active_Poster'] = (test_scaled_enl['Post_frequency'] > 
#                                    test_scaled_enl['Post_frequency'].median()).astype(int)

# 7. Extroversion score
test_scaled_enl['Extroversion_Score'] = (test_scaled_enl['Social_event_attendance'] + 
                                          test_scaled_enl['Going_outside'] + 
                                          test_scaled_enl['Friends_circle_size'] + 
                                          test_scaled_enl['Post_frequency'] - 
                                          test_scaled_enl['Time_spent_Alone'] - 
                                          5 * test_scaled_enl['Stage_fear_enc'] - 
                                          5 * test_scaled_enl['Drained_after_socializing_enc'])

print(test_scaled_enl.shape)
test_scaled_enl.head()


train_scaled_enl.describe(include='all').T
# Fear & Drained std 0.87


# correlation with Personality
train_scaled_enl_corr = train_scaled_enl.copy()
train_scaled_enl_corr['Personality'] = train_scaled_enl_corr['Personality'].replace(['Introvert', 'Extrovert'], [0, 1])
corrmat = train_scaled_enl_corr.corr()

corr_sorted = corrmat.iloc[0].reindex(corrmat.iloc[0].abs().sort_values(ascending=False).index)
print(corr_sorted)


# correlation heatmap

plt.figure(figsize=(12, 12))
k=15
cols = corrmat.nlargest(k, 'Personality').index 
cols_abs_sorted = corrmat['Personality'].abs().nlargest(k).index # absolute corr values
cm = np.corrcoef(train_scaled_enl_corr[cols_abs_sorted].values.T)
sns.set(font_scale=1.)
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.3f', annot_kws={'size': 10}, 
                 yticklabels=cols_abs_sorted.values, xticklabels=cols_abs_sorted.values, 
                 cmap="rainbow", cbar_kws={"shrink": 0.8})
hm.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.show()


print(train_scaled_enl.shape)
train_scaled_enl.head()


# target and Features
target = train_corr['Personality']
features = train_scaled_enl.drop(['Personality'], axis=1)

print(features.shape, target.shape)
features.head()


# train and validation data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(features, target, stratify=target, random_state=42)

X_train.shape, X_val.shape, y_train.shape, y_val.shape 


y_train.value_counts()


scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
scale_pos_weight


# xgb_model
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

y_train_pred = xgb_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train Accuracy:", train_accuracy)

y_val_pred = xgb_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Val Accuracy:",val_accuracy)

print("Train-Val Gap:", train_accuracy - val_accuracy)
print(classification_report(y_val, y_val_pred, digits=4))
print(confusion_matrix(y_val, y_val_pred))


# xgb_model Visualization
evals_result = xgb_model.evals_result()
train_logloss = evals_result['validation_0']['logloss']
val_logloss = evals_result['validation_1']['logloss']
rounds = range(1, len(train_logloss) + 1)

plt.figure(figsize=(8, 3))
plt.title('"xgb_model" Learning Curve')
plt.xlabel('Boosting Round')
plt.ylabel('Logloss')
sns.lineplot(x=rounds, y=train_logloss, label='Train Logloss', color='blue')
sns.lineplot(x=rounds, y=val_logloss, label='Validation Logloss', color='orange')
plt.legend()
plt.grid(True)
plt.show()
# Default n_estimators is 100, but since overfitting increases after 7, tuning this hyperparameter is essential


# hyperparameter optimization for xgb_model
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

xgb_param_distributions = {
    'n_estimators': randint(5, 50),          
    'max_depth': randint(2, 8),              
    'learning_rate': uniform(0.01, 0.2),     
    'subsample': uniform(0.7, 0.3),         
    'colsample_bytree': uniform(0.7, 0.3),   
    'gamma': uniform(0, 5)                   
}

# xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_distributions,
    n_iter=50,          
    cv=5,                
    scoring='accuracy',   
    n_jobs=-1,           
    random_state=42,
    verbose=0
)

# hyperparameter tuning
xgb_random_search.fit(X_train, y_train)

# Best parameters & Score
print("XGB Best parameters:", xgb_random_search.best_params_)
print("XGB Best cross-validation score:", xgb_random_search.best_score_)


# xgb_best model
xgb_best_params = xgb_random_search.best_params_

# modeling with best hyperparameter
xgb_best = XGBClassifier(
    eval_metric='logloss', 
    random_state=42, 
    use_label_encoder=False,
    **xgb_best_params
)

xgb_best.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

y_train_pred = xgb_best.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train Accuracy:", train_accuracy)

y_val_pred = xgb_best.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Val Accuracy:", val_accuracy)

print("Train-Val Gap:", train_accuracy - val_accuracy)
print(classification_report(y_val, y_val_pred, digits=4))
print(confusion_matrix(y_val, y_val_pred))

# Better generalization performance, higher validation accuracy, less overfitting
"""
Train Accuracy: 0.9708486288058734
Val Accuracy: 0.9645864824012093
Train-Val Gap: 0.0062621464046641595

Train Accuracy: 0.9712085222774058
Val Accuracy: 0.965450226732887
Train-Val Gap: 0.005758295544518788

Train Accuracy: 0.970776650111567
Val Accuracy: 0.9652342906499676
Train-Val Gap: 0.005542359461599422
"""


# xgb_best model visualization
evals_result = xgb_best.evals_result()
train_logloss = evals_result['validation_0']['logloss']
val_logloss = evals_result['validation_1']['logloss']
rounds = range(1, len(train_logloss) + 1)

plt.figure(figsize=(8, 3))
plt.title('"xgb_best" Learning Curve')
plt.xlabel('Boosting Round')
plt.ylabel('Logloss')
sns.lineplot(x=rounds, y=train_logloss, label='Train Logloss', color='blue')
sns.lineplot(x=rounds, y=val_logloss, label='Validation Logloss', color='orange')
plt.legend()
plt.grid(True)
plt.show()


# lgb_model
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(eval_metric='logloss', random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])

y_train_pred = lgb_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train Accuracy:", train_accuracy)

y_val_pred = lgb_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Val Accuracy:", val_accuracy)

print("Train-Val Gap:", train_accuracy - val_accuracy)
print(classification_report(y_val, y_val_pred, digits=4))
print(confusion_matrix(y_val, y_val_pred))


# lgb_model visualization
evals_result = lgb_model.evals_result_ ##
train_logloss = evals_result['training']['binary_logloss']
val_logloss = evals_result['valid_1']['binary_logloss']
rounds = range(1, len(train_logloss) + 1)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 3))
plt.title('"lgb_model" Learning Curve')
plt.xlabel('Boosting Round')
plt.ylabel('Logloss')
sns.lineplot(x=rounds, y=train_logloss, label='Train Logloss', color='blue')
sns.lineplot(x=rounds, y=val_logloss, label='Validation Logloss', color='orange')
plt.legend()
plt.grid(True)
plt.show()


# hyperparameter tuning for lgb_model
lgb_param_distributions = {
    'n_estimators': randint(100, 1001),          
    'max_depth': randint(3, 9),                  
    'learning_rate': uniform(0.01, 0.19),         
    'subsample': uniform(0.7, 0.3),             
    'colsample_bytree': uniform(0.7, 0.3),       
    'reg_alpha': uniform(0, 1),                 
    'reg_lambda': uniform(0, 1)                 
}

# lgb_model = lgb.LGBMClassifier(eval_metric='logloss', random_state=42, verbose=-1)

lgb_random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=lgb_param_distributions,
    n_iter=50,            
    cv=5,                 
    scoring='accuracy',   
    n_jobs=-1,            
    random_state=42,
    verbose=0
)

lgb_random_search.fit(X_train, y_train)

print("LGB Best parameters:", lgb_random_search.best_params_)
print("LGB Best cross-validation score:", lgb_random_search.best_score_)


# lgb_best model
lgb_best_params = lgb_random_search.best_params_
lgb_best = lgb.LGBMClassifier(
    eval_metric='logloss', 
    random_state=42, 
    **lgb_best_params, 
    verbose=0, 
    verbosity=-1
) 

lgb_best.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])

y_train_pred = lgb_best.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train Accuracy:", train_accuracy)

y_val_pred = lgb_best.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Val Accuracy:", val_accuracy)

print("Train-Val Gap:", train_accuracy - val_accuracy)
print(classification_report(y_val, y_val_pred, digits=4))
print(confusion_matrix(y_val, y_val_pred))

# Better generalization performance, higher validation accuracy, less overfitting
"""
Train Accuracy: 0.9721442453033902
Val Accuracy: 0.9650183545670482
Train-Val Gap: 0.0071258907363419555

Train Accuracy: 0.973295904412294
Val Accuracy: 0.9652342906499676
Train-Val Gap: 0.008061613762326392

Train Accuracy: 0.9717123731375513
Val Accuracy: 0.9652342906499676
Train-Val Gap: 0.006478082487583747

"""


# lgb_best visualization
evals_result = lgb_best.evals_result_

train_logloss = evals_result['training']['binary_logloss']
val_logloss = evals_result['valid_1']['binary_logloss']
rounds = range(1, len(train_logloss) + 1)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 3))
plt.title('"lgb_best" Learning Curve')
plt.xlabel('Boosting Round')
plt.ylabel('Logloss')
sns.lineplot(x=rounds, y=train_logloss, label='Train Logloss', color='blue')
sns.lineplot(x=rounds, y=val_logloss, label='Validation Logloss', color='orange')
plt.legend()
plt.grid(True)
plt.show()


from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(objective='Logloss', random_seed=42, verbose=False)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

# accuracy with train data
y_train_pred = cat_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train Accuracy:", train_accuracy)

# accuracy with validation data
y_val_pred = cat_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Val Accuracy:", val_accuracy)

print("Train-Val Gap:", train_accuracy - val_accuracy)
print(classification_report(y_val, y_val_pred, digits=4))
print(confusion_matrix(y_val, y_val_pred))


# cat_model visualization
evals_result = cat_model.evals_result_

train_logloss = evals_result['learn']['Logloss']
val_logloss = evals_result['validation']['Logloss']
rounds = range(1, len(train_logloss) + 1)

plt.figure(figsize=(8, 3))
plt.title('"cat_model" Learning Curve')
plt.xlabel('Boosting Round')
plt.ylabel('Logloss')
sns.lineplot(x=rounds, y=train_logloss, label='Train Logloss', color='blue')
sns.lineplot(x=rounds, y=val_logloss, label='Validation Logloss', color='orange')
plt.legend()
plt.grid(True)
plt.show()


# hyperparameter tuning for cat_model
cat_param_distributions = {
    'iterations': randint(100, 1001),           
    'depth': randint(3, 9),                    
    'learning_rate': uniform(0.01, 0.19),       
    'l2_leaf_reg': uniform(1, 9),              
    'bagging_temperature': uniform(0, 1),      
    'border_count': randint(32, 256),         
}

# cat_model = CatBoostClassifier(objective='Logloss', random_seed=42, verbose=False)

cat_random_search = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=cat_param_distributions,
    n_iter=50,           
    cv=5,                 
    scoring='accuracy',   
    n_jobs=-1,           
    random_state=42,
    verbose=0
)

cat_random_search.fit(X_train, y_train)

print("Best parameters:", cat_random_search.best_params_)
print("Best cross-validation score:", cat_random_search.best_score_)


# cat_best model
cat_best_params = cat_random_search.best_params_
cat_best = CatBoostClassifier(objective='Logloss', random_seed=42, verbose=False, **cat_best_params)
cat_best.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

y_train_pred = cat_best.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train Accuracy:", train_accuracy)

y_val_pred = cat_best.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

print("Train_Val Gap:", train_accuracy - val_accuracy)
print(classification_report(y_val, y_val_pred, digits=4))
print(confusion_matrix(y_val, y_val_pred))

"""
Train Accuracy: 0.9704167566400346
Validation Accuracy: 0.9648024184841287
Train_Val Gap: 0.00561433815590584

Train Accuracy: 0.970488735334341
Validation Accuracy: 0.9650183545670482
Train_Val Gap: 0.005470380767292782

Train Accuracy: 0.9709206075001799
Validation Accuracy: 0.9650183545670482
Train_Val Gap: 0.005902252933131735

"""


# cat_best model visualization
evals_result = cat_best.evals_result_

train_logloss = evals_result['learn']['Logloss']
val_logloss = evals_result['validation']['Logloss']
rounds = range(1, len(train_logloss) + 1)

plt.figure(figsize=(8, 3))
plt.title('"cat_best" Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Logloss')
sns.lineplot(x=rounds, y=train_logloss, label='Train Logloss', color='blue')
sns.lineplot(x=rounds, y=val_logloss, label='Validation Logloss', color='orange')
plt.legend()
plt.grid(True)
plt.show()


# 1. Voting Classifier
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[('xgb', xgb_best), ('lgb', lgb_best), ('cat', cat_best)],
    voting='soft',
    n_jobs=-1
)

voting_clf.fit(X_train, y_train)

# Evaluation of train
train_pred_proba = voting_clf.predict_proba(X_train)[:, 1]
train_pred = (train_pred_proba > 0.5).astype(int)

train_accuracy = accuracy_score(y_train, train_pred)
print("Train Accuracy:", train_accuracy)

# Evaluation of val
val_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
val_pred = (val_pred_proba > 0.5).astype(int)

val_accuracy = accuracy_score(y_val, val_pred)
print("Validation Accuracy:", val_accuracy)

print("Train_Val Gap:", train_accuracy - val_accuracy)
print(classification_report(y_val, val_pred, digits=4))
print(confusion_matrix(y_val, val_pred))

"""
Train Accuracy: 0.9723601813863096
Validation Accuracy: 0.965450226732887
Train_Val Gap: 0.00690995465342259

Train Accuracy: 0.9709206075001799
Validation Accuracy: 0.965450226732887
Train_Val Gap: 0.005470380767292893

"""


# 2. model stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


base_models = []
base_models.append(('xgb',
                     XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False, **xgb_best_params)))
base_models.append(('lgb',
                     lgb.LGBMClassifier(eval_metric='logloss', random_state=42, **lgb_best_params, verbose=0, verbosity=-1)))
base_models.append(('cat',
                     CatBoostClassifier(objective='Logloss', random_seed=42, verbose=False, **cat_best_params)))

meta_model = LogisticRegression(max_iter=1000, random_state=42)
model_stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, n_jobs=-1, passthrough=False)
model_stack.fit(X_train, y_train)

# Train set 
train_pred_proba = model_stack.predict_proba(X_train)[:, 1]
train_pred = (train_pred_proba > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, train_pred)
print("Train Accuracy:", train_accuracy)

# Validation set
val_pred_proba = model_stack.predict_proba(X_val)[:, 1]
val_pred = (val_pred_proba > 0.5).astype(int)
val_accuracy = accuracy_score(y_val, val_pred)
print("Validation Accuracy:", val_accuracy)
print("Train_Val Gap:", train_accuracy - val_accuracy)

print("\nClassification Report (Validation):")
print(classification_report(y_val, val_pred, digits=4))
print("Confusion Matrix (Validation):")
print(confusion_matrix(y_val, val_pred))

"""
Train Accuracy: 0.9719283092204707
Validation Accuracy: 0.9652342906499676
Train_Val Gap: 0.006694018570503113

Train Accuracy: 0.9710645648887929
Validation Accuracy: 0.9650183545670482
Train_Val Gap: 0.006046210321744683
"""



# H2O Initialization and Data Conversion

import h2o
from h2o.automl import H2OAutoML

h2o.init()

# convert pandas DataFrame to H2OFrame
train = pd.concat([X_train, y_train], axis=1)
val = pd.concat([X_val, y_val], axis=1)

train_h2o = h2o.H2OFrame(train)
val_h2o = h2o.H2OFrame(val)

# target and features columns
target_col = y_train.name  
feature_cols = X_train.columns.tolist()

# target conversion to factor
train_h2o[target_col] = train_h2o[target_col].asfactor()
val_h2o[target_col] = val_h2o[target_col].asfactor()


# H2O AutoML Training

aml = H2OAutoML(
    max_runtime_secs=600, # Maximum training time (seconds)
    seed=42,
    balance_classes=True # Automatically balance class distribution
)
aml.train(x=feature_cols, y=target_col, training_frame=train_h2o)


# Prediction & Evaluation
train_preds = aml.leader.predict(train_h2o)
train_pred_labels = train_preds.as_data_frame()['predict']

val_preds = aml.leader.predict(val_h2o)
val_pred_labels = val_preds.as_data_frame()['predict']

train_preds = aml.leader.predict(train_h2o)
train_pred_labels = train_preds.as_data_frame()['predict']

train_accuracy = accuracy_score(y_train, train_pred_labels)
val_accuracy = accuracy_score(y_val, val_pred_labels)

print(train_accuracy)
print(val_accuracy)
print(train_accuracy - val_accuracy)
print(classification_report(y_val, val_pred_labels, digits=4))
print(confusion_matrix(y_val, val_pred_labels))





# 

test_voting_proba = voting_clf.predict_proba(test_scaled_enl)[:, 1]
test_voting_pred = (test_voting_proba > 0.5).astype(int)
test_voting_pred


submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_voting_pred
})
submission.head()


submission['Personality'] = submission['Personality'].replace({0: 'Introvert', 1: 'Extrovert'})
submission.head()


submission.to_csv('submission.csv', index=False)



print(len(test['id']), len(test_voting_pred))


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()

