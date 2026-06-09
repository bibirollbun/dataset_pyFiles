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
warnings.filterwarnings("ignore", category=Warning)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
print(sample_submission.shape)
sample_submission.head(2)


# Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
print(test.shape)
test.head(2)

# 730/365 = 2 years time series data?


test = test.rename(columns={'temparature': 'temperature'})
test.head(1)


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
print(train.shape)
train.tail(2)

# 2190/365 = 6 years time series data?


train = train.rename(columns={'temparature': 'temperature'})


train.rainfall.value_counts()
# unbalanced data


train.info()


# Correlation
train_temp = train.drop(['id', 'day'], axis=1)
corrmat = train_temp.corr()
corrmat


plt.figure(figsize=(7, 7))
k=11
cols = corrmat.nlargest(k, 'rainfall')['rainfall'].index
cm = np.corrcoef(train_temp[cols].values.T)
sns.set(font_scale=0.8)
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, 
                 yticklabels=cols.values, xticklabels=cols.values,cmap="Blues")
hm.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.show()


def diagnostic_plots(df, variable):
    plt.figure(figsize=(12, 2))

    plt.subplot(1, 3, 1)
    
    rain_values = df['rainfall'].unique()
    skews = {}
    for rain_value in rain_values:
        skews[rain_value] = df[df['rainfall'] == rain_value][variable].skew()
    
    sns.histplot(data=df, x=variable, hue="rainfall", kde=True, stat="density", bins=30)
    
    for rain_value, skew in skews.items():
        plt.text(0.76, 0.90 - rain_values.tolist().index(rain_value) * 0.1, 
                 f'Rain {rain_value} Skew: {skew:.4f}', 
                 transform=plt.gca().transAxes, horizontalalignment='right')
        
    plt.title('Histogram')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x="rainfall", y=variable)  
    plt.title('Boxplot')

    plt.subplot(1, 3, 3)
    sns.barplot(data=df, x="rainfall", y=variable)
    plt.title('Barplot')

    plt.show()

variables = ['cloud', 'humidity', 'windspeed', 'dewpoint', 'winddirection', 
             'mintemp', 'temperature', 'pressure', 'maxtemp', 'sunshine']

for variable in variables:
    diagnostic_plots(train_temp, variable)


# enhanced train data
train_en = train.copy()
print(train_en.shape)
train_en.head(1)


# Features generation for train data
# 1. Temperature change in a day
train_en['temp_range'] = train_en['maxtemp'] - train_en['mintemp']
train_en['max_temp_diff'] = train_en['maxtemp'] - train_en['temperature']
train_en['temp_min_diff'] = train_en['temperature'] - train_en['mintemp']

# 2. Humidity
train_en['humidity_deficit'] = train_en['humidity'] - train_en['dewpoint']
train_en['dewpoint_depression'] = train_en['temperature'] - train_en['dewpoint']
train_en['humidity_dewpoint'] = train_en['humidity'] / (train_en['dewpoint'] + 0.1) 

# 3. Wind Direction Sin/Cos
train_en['winddirection_sin'] = np.sin(train_en['winddirection'] * np.pi / 180)
train_en['winddirection_cos'] = np.cos(train_en['winddirection'] * np.pi / 180)

# 4. Wind Vector Components
train_en['u_wind'] = -train_en['windspeed'] * train_en['winddirection_sin'] #East-West component
train_en['v_wind'] = -train_en['windspeed'] * train_en['winddirection_cos'] #North-South component

# 5. cloud to sunshine ratio
train_en['cloud_sunshine'] = train_en['cloud'] / (train_en['sunshine'] + 0.1) 

# 6. Temperature-Humidity Index
train_en['THI'] = train_en['temperature'] - (0.55 * (1 - train_en['humidity']/100) * (train_en['temperature'] - 14.5))
train_en['perceived_temp'] = (0.8 * train_en['temperature'] + 
                              (train_en['humidity'] / 100) * (train_en['temperature'] - 14.3) + 46.4)
# 7. wind & humidity interaction
train_en['wind_humidity_factor'] = train_en['windspeed'] * (train_en['humidity'] / 100)**2
train_en['wind_humidity_factor_lag'] = train_en['windspeed'].shift(1).fillna(0) * train_en['humidity'].shift(1).fillna(0)
train_en['wind_humidity_pressure_factor'] = train_en['windspeed'] * (train_en['humidity'] / 100) * train_en['pressure']

# 8. Pressure change
train_en['pressure_diff'] = train_en['pressure'].diff().fillna(0)
train_en['pressure_trend_3d'] = train_en['pressure'].diff(3).fillna(0)
train_en['pressure_rolling_mean_3d'] = train_en['pressure'].rolling(window=3).mean().fillna(train_en['pressure'].mean())
train_en['pressure_rolling_mean_7d'] = train_en['pressure'].rolling(window=7).mean().fillna(train_en['pressure'].mean())
train_en['pressure_rolling_mean_14d'] = train_en['pressure'].rolling(window=14).mean().fillna(train_en['pressure'].mean())

# 9. temperature change
train_en['temperature_diff'] = train_en['temperature'].diff().fillna(0)
train_en['temperature_diff_3d'] = train_en['temperature'].diff(3).fillna(0)
train_en['temperature_rolling_mean_3d'] = train_en['temperature'].rolling(window=3).mean().fillna(train_en['temperature'].mean())
train_en['temperature_rolling_mean_7d'] = train_en['temperature'].rolling(window=7).mean().fillna(train_en['temperature'].mean())
train_en['temperature_rolling_mean_14d'] = train_en['temperature'].rolling(window=14).mean().fillna(train_en['temperature'].mean())

# 10. humidity change
train_en['humidity_diff'] = train_en['humidity'].diff().fillna(0)
train_en['humidity_diff_3d'] = train_en['humidity'].diff(3).fillna(0)
train_en['humidity_rolling_mean_3d'] = train_en['humidity'].rolling(window=3).mean().fillna(train_en['humidity'].mean())
train_en['humidity_rolling_mean_7d'] = train_en['humidity'].rolling(window=7).mean().fillna(train_en['humidity'].mean())
train_en['humidity_rolling_mean_14d'] = train_en['humidity'].rolling(window=14).mean().fillna(train_en['humidity'].mean())

# 11. cloud change
train_en['cloud_diff'] = train_en['cloud'].diff().fillna(0)
train_en['cloud_diff_3d'] = train_en['cloud'].diff(3).fillna(0)
train_en['cloud_rolling_mean_3d'] = train_en['cloud'].rolling(window=3).mean().fillna(train_en['cloud'].mean())
train_en['cloud_rolling_mean_7d'] = train_en['cloud'].rolling(window=7).mean().fillna(train_en['cloud'].mean())
train_en['cloud_rolling_mean_14d'] = train_en['cloud'].rolling(window=14).mean().fillna(train_en['cloud'].mean())

# 12. windspeed change
train_en['windspeed_diff'] = train_en['windspeed'].diff().fillna(0)
train_en['windspeed_diff_3d'] = train_en['windspeed'].diff(3).fillna(0)
train_en['windspeed_rolling_mean_3d'] = train_en['windspeed'].rolling(window=3).mean().fillna(train_en['windspeed'].mean())
train_en['windspeed_rolling_mean_7d'] = train_en['windspeed'].rolling(window=7).mean().fillna(train_en['windspeed'].mean())
train_en['windspeed_rolling_mean_14d'] = train_en['windspeed'].rolling(window=14).mean().fillna(train_en['windspeed'].mean())

# 13. pressure acceleration
train_en['pressure_acceleration'] = train_en['pressure_diff'].diff().fillna(0)

# 14. month
train_en['month'] = (((train_en['day'] - 1) // 30) + 1).clip(upper=12)

# 15. cyclical nature of days in a year
train_en['day_of_year_sin'] = np.sin(2 * np.pi * train_en['day'] / 365)
train_en['day_of_year_cos'] = np.cos(2 * np.pi * train_en['day'] / 365)

train_en = train_en.drop(["id", "day"], axis=1)    
print(train_en.shape)
train_en.head(1) 


# enhanced test data
test_en = test.copy()
print(test_en.shape)
test_en.head(1)


# feature generation for test data
# 1. Temperature change in a day
test_en['temp_range'] = test_en['maxtemp'] - test_en['mintemp']
test_en['max_temp_diff'] = test_en['maxtemp'] - test_en['temperature']
test_en['temp_min_diff'] = test_en['temperature'] - test_en['mintemp']

# 2. Humidity
test_en['humidity_deficit'] = test_en['humidity'] - test_en['dewpoint']
test_en['dewpoint_depression'] = test_en['temperature'] - test_en['dewpoint']
test_en['humidity_dewpoint'] = test_en['humidity'] / (test_en['dewpoint'] + 0.1) 

# 3. Wind Direction Sin/Cos
test_en['winddirection_sin'] = np.sin(test_en['winddirection'] * np.pi / 180)
test_en['winddirection_cos'] = np.cos(test_en['winddirection'] * np.pi / 180)

# 4. Wind Vector Components
test_en['u_wind'] = -test_en['windspeed'] * test_en['winddirection_sin'] #East-West component
test_en['v_wind'] = -test_en['windspeed'] * test_en['winddirection_cos'] #North-South component

# 5. cloud to sunshine ratio
test_en['cloud_sunshine'] = test_en['cloud'] / (test_en['sunshine'] + 0.1) 

# 6. Temperature-Humidity Index
test_en['THI'] = test_en['temperature'] - (0.55 * (1 - test_en['humidity']/100) * (test_en['temperature'] - 14.5))
test_en['perceived_temp'] = (0.8 * test_en['temperature'] + 
                              (test_en['humidity'] / 100) * (test_en['temperature'] - 14.3) + 46.4)
# 7. wind & humidity interaction
test_en['wind_humidity_factor'] = test_en['windspeed'] * (test_en['humidity'] / 100)**2
test_en['wind_humidity_factor_lag'] = test_en['windspeed'].shift(1).fillna(0) * test_en['humidity'].shift(1).fillna(0)
test_en['wind_humidity_pressure_factor'] = test_en['windspeed'] * (test_en['humidity'] / 100) * test_en['pressure']

# 8. Pressure change
test_en['pressure_diff'] = test_en['pressure'].diff().fillna(0)
test_en['pressure_trend_3d'] = test_en['pressure'].diff(3).fillna(0)
test_en['pressure_rolling_mean_3d'] = test_en['pressure'].rolling(window=3).mean().fillna(test_en['pressure'].mean())
test_en['pressure_rolling_mean_7d'] = test_en['pressure'].rolling(window=7).mean().fillna(test_en['pressure'].mean())
test_en['pressure_rolling_mean_14d'] = test_en['pressure'].rolling(window=14).mean().fillna(test_en['pressure'].mean())

# 9. temperature change
test_en['temperature_diff'] = test_en['temperature'].diff().fillna(0)
test_en['temperature_diff_3d'] = test_en['temperature'].diff(3).fillna(0)
test_en['temperature_rolling_mean_3d'] = test_en['temperature'].rolling(window=3).mean().fillna(test_en['temperature'].mean())
test_en['temperature_rolling_mean_7d'] = test_en['temperature'].rolling(window=7).mean().fillna(test_en['temperature'].mean())
test_en['temperature_rolling_mean_14d'] = test_en['temperature'].rolling(window=14).mean().fillna(test_en['temperature'].mean())

# 10. humidity change
test_en['humidity_diff'] = test_en['humidity'].diff().fillna(0)
test_en['humidity_diff_3d'] = test_en['humidity'].diff(3).fillna(0)
test_en['humidity_rolling_mean_3d'] = test_en['humidity'].rolling(window=3).mean().fillna(test_en['humidity'].mean())
test_en['humidity_rolling_mean_7d'] = test_en['humidity'].rolling(window=7).mean().fillna(test_en['humidity'].mean())
test_en['humidity_rolling_mean_14d'] = test_en['humidity'].rolling(window=14).mean().fillna(test_en['humidity'].mean())

# 11. cloud change
test_en['cloud_diff'] = test_en['cloud'].diff().fillna(0)
test_en['cloud_diff_3d'] = test_en['cloud'].diff(3).fillna(0)
test_en['cloud_rolling_mean_3d'] = test_en['cloud'].rolling(window=3).mean().fillna(test_en['cloud'].mean())
test_en['cloud_rolling_mean_7d'] = test_en['cloud'].rolling(window=7).mean().fillna(test_en['cloud'].mean())
test_en['cloud_rolling_mean_14d'] = test_en['cloud'].rolling(window=14).mean().fillna(test_en['cloud'].mean())

# 12. windspeed change
test_en['windspeed_diff'] = test_en['windspeed'].diff().fillna(0)
test_en['windspeed_diff_3d'] = test_en['windspeed'].diff(3).fillna(0)
test_en['windspeed_rolling_mean_3d'] = test_en['windspeed'].rolling(window=3).mean().fillna(test_en['windspeed'].mean())
test_en['windspeed_rolling_mean_7d'] = test_en['windspeed'].rolling(window=7).mean().fillna(test_en['windspeed'].mean())
test_en['windspeed_rolling_mean_14d'] = test_en['windspeed'].rolling(window=14).mean().fillna(test_en['windspeed'].mean())

# 13. pressure acceleration
test_en['pressure_acceleration'] = test_en['pressure_diff'].diff().fillna(0)

# 14. month
test_en['month'] = (((test_en['day'] - 1) // 30) + 1).clip(upper=12)

# 15. cyclical nature of days in a year
test_en['day_of_year_sin'] = np.sin(2 * np.pi * test_en['day'] / 365)
test_en['day_of_year_cos'] = np.cos(2 * np.pi * test_en['day'] / 365)

test_en_features = test_en.drop(["id", "day"], axis=1)    
print(test_en_features.shape)
test_en_features.head(1) 


train_en_features = train_en.drop(['rainfall'], axis=1)
train_en_features.shape


# Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_en_features)

train_en_st = scaler.transform(train_en_features)
train_en_st = pd.DataFrame(train_en_st, columns=train_en_features.columns)

test_en_st = scaler.transform(test_en_features)
test_en_st = pd.DataFrame(test_en_st, columns=test_en_features.columns)

print(train_en_st.shape, test_en_st.shape)
train_en_st.head(1)


# Correlation
train_en_st_temp = pd.concat([train_en_st, train[['rainfall']]], axis=1)
corrmat = train_en_st_temp.corr()
plt.figure(figsize=(20, 20))
k=56
cols = corrmat.nlargest(k, 'rainfall')['rainfall'].index
cm = np.corrcoef(train_en_st_temp[cols].values.T)
sns.set(font_scale=0.8)
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, 
                 yticklabels=cols.values, xticklabels=cols.values,cmap="Blues")
hm.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.show()


# Removing high correlated features
corr_rate_threshold = 0.77
cor_matrix = train_en_st.corr().abs()

# remove mirror and diagonal values
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# Drop columns with higher correlation than rate_corr_threshold
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= corr_rate_threshold)]
print(to_drop)


# Features selection to mitigate multicollinearity
train_features_sel = train_en_st.drop(train_en_st[to_drop], axis=1)
print(train_features_sel.shape)
train_features_sel.head(2)


train_features_sel.columns


# Features selection to mitigate multicollinearity
test_features_sel = test_en_st.drop(test_en_st[to_drop], axis=1) 
print(test_features_sel.shape)
test_features_sel.head(2)


# Correlation
train_features_sel_temp = pd.concat([train_features_sel, train[['rainfall']]], axis=1)
corrmat = train_features_sel_temp.corr()
plt.figure(figsize=(20, 20))
k=28
cols = corrmat.nlargest(k, 'rainfall')['rainfall'].index
cm = np.corrcoef(train_features_sel_temp[cols].values.T)
sns.set(font_scale=1.0)
hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values,cmap="Blues")
hm.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.show()


target = train['rainfall']
train_features_sel.shape, target.shape, test_features_sel.shape


print(target.value_counts(), target.value_counts(normalize=True))


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_features_sel, target, random_state = 1, stratify=target)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# XGBClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

weights = np.where(y_train == 1, 1, 1650/540)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

roc_auc_scores_train = []
roc_auc_scores_val = []

for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    weights_fold = weights[train_index]

    model_xgb = XGBClassifier(random_state=1, n_jobs=-1, 
                              max_depth=5,  
                              learning_rate=0.1, 
                              subsample=0.8,  
                              colsample_bytree=0.8,  
                              reg_alpha=0.1, 
                              reg_lambda=0.1)
    model_xgb.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)
    
    xgb_pred_train = model_xgb.predict_proba(X_train_fold)[:, 1]
    roc_auc_scores_train.append(roc_auc_score(y_train_fold, xgb_pred_train))
    
    xgb_pred_val = model_xgb.predict_proba(X_val_fold)[:, 1]
    roc_auc_scores_val.append(roc_auc_score(y_val_fold, xgb_pred_val))

avg_roc_auc_train = np.mean(roc_auc_scores_train)
avg_roc_auc_val = np.mean(roc_auc_scores_val)

print(f"Training ROC AUC: {avg_roc_auc_train:.8f}") 
print(f"Validation ROC AUC: {avg_roc_auc_val:.8f}")
Val_to_Train_Ratio = avg_roc_auc_train / avg_roc_auc_val
print(f"Val to Train Ratio: {Val_to_Train_Ratio: .4f}")
"""
Training ROC AUC: 0.99999750
Validation ROC AUC: 0.87714119
Val to Train Ratio:  1.1401
"""


# importance features
feature_importance = pd.DataFrame({'features': train_features_sel.columns, 'importance': model_xgb.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
feature_importance


# Top N-Features Selection
top_n_features = feature_importance['features'].head(20).tolist()
important_features = train_features_sel[top_n_features]
print(important_features.shape)
important_features.head(1)


important_features.columns


test_features = test_features_sel[important_features.columns]
test_features.shape


from sklearn.model_selection import train_test_split
X_train_sel, X_val_sel, y_train_sel, y_val_sel = train_test_split(important_features, target, random_state = 1, stratify=target)
X_train_sel.shape, X_val_sel.shape, y_train_sel.shape, y_val_sel.shape


# XGBClassifier to select Top N-Features

weights = np.where(y_train_sel == 1, 1, 1650/540)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

roc_auc_scores_train = []
roc_auc_scores_val = []

for train_index, val_index in skf.split(X_train_sel, y_train_sel):
    X_train_fold, X_val_fold = X_train_sel.iloc[train_index], X_train_sel.iloc[val_index]
    y_train_fold, y_val_fold = y_train_sel.iloc[train_index], y_train_sel.iloc[val_index]
    weights_fold = weights[train_index]

    model_xgb = XGBClassifier(random_state=1, n_jobs=-1, 
                              max_depth=5,  
                              learning_rate=0.1,  
                              subsample=0.8,  
                              colsample_bytree=0.8,  
                              reg_alpha=0.1, 
                              reg_lambda=0.1 
                             )
    model_xgb.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)
    
    xgb_pred_train = model_xgb.predict_proba(X_train_fold)[:, 1]
    roc_auc_scores_train.append(roc_auc_score(y_train_fold, xgb_pred_train))
    
    xgb_pred_val = model_xgb.predict_proba(X_val_fold)[:, 1]
    roc_auc_scores_val.append(roc_auc_score(y_val_fold, xgb_pred_val))

avg_roc_auc_train = np.mean(roc_auc_scores_train)
avg_roc_auc_val = np.mean(roc_auc_scores_val)

print(f"Training ROC AUC: {avg_roc_auc_train:.8f}") 
print(f"Validation ROC AUC: {avg_roc_auc_val:.8f}")
Val_to_Train_Ratio = avg_roc_auc_train / avg_roc_auc_val
print(f"Val to Train Ratio: {Val_to_Train_Ratio: .4f}")
"""
Training ROC AUC: 0.99999564
Validation ROC AUC: 0.87621058
Val to Train Ratio:  1.1413
"""


# hyperparameter tuned XGBClassifier
weights = np.where(y_train_sel == 1, 1, 1650/540)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

roc_auc_scores_train = []
roc_auc_scores_val = []

for train_index, val_index in skf.split(X_train_sel, y_train_sel):
    X_train_fold, X_val_fold = X_train_sel.iloc[train_index], X_train_sel.iloc[val_index]
    y_train_fold, y_val_fold = y_train_sel.iloc[train_index], y_train_sel.iloc[val_index]
    weights_fold = weights[train_index]

    model_xgb = XGBClassifier(random_state=1, n_jobs=-1, 
                              max_depth=2, min_child_weight=12, gamma=0.30005718740867243, 
                              n_estimators=223, learning_rate=0.018430904947696832, 
                              subsample=0.6273875931979261, colsample_bytree=0.6923385947687978, 
                              reg_alpha=0.14898013363049112, reg_lambda=4)
    model_xgb.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)

    xgb_pred_train = model_xgb.predict_proba(X_train_fold)[:, 1]
    roc_auc_scores_train.append(roc_auc_score(y_train_fold, xgb_pred_train))

    xgb_pred_val = model_xgb.predict_proba(X_val_fold)[:, 1]
    roc_auc_scores_val.append(roc_auc_score(y_val_fold, xgb_pred_val))

avg_roc_auc_train = np.mean(roc_auc_scores_train)
avg_roc_auc_val = np.mean(roc_auc_scores_val)

print(f"Training ROC AUC: {avg_roc_auc_train:.8f}")
print(f"Validation ROC AUC: {avg_roc_auc_val:.8f}")
Val_to_Train_Ratio = avg_roc_auc_train / avg_roc_auc_val
print(f"Val to Train Ratio: {Val_to_Train_Ratio:.4f}")
"""
Training ROC AUC: 0.91707148
Validation ROC AUC: 0.89130722
Val to Train Ratio: 1.0289
"""


# hyperparameter tuned LGBMClassifier
from lightgbm import LGBMClassifier

weights = np.where(y_train_sel == 1, 1, 1650/540)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

roc_auc_scores_train = []
roc_auc_scores_val = []

for train_index, val_index in skf.split(X_train_sel, y_train_sel):
    X_train_fold, X_val_fold = X_train_sel.iloc[train_index], X_train_sel.iloc[val_index]
    y_train_fold, y_val_fold = y_train_sel.iloc[train_index], y_train_sel.iloc[val_index]
    weights_fold = weights[train_index]

    model_lgb = LGBMClassifier(random_state=1, n_jobs=-1, verbose=-1, 
                               num_leaves=67, max_depth=2, min_child_samples=114, 
                               n_estimators=535, learning_rate=0.010030037966920426, 
                               subsample=0.8, colsample_bytree=0.6001143748173449, 
                               reg_alpha=0.01, reg_lambda=1.7337794540855653, 
                               min_child_weight=1.90) 
    model_lgb.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)
    
    lgb_pred_train = model_lgb.predict_proba(X_train_fold)[:, 1]
    roc_auc_scores_train.append(roc_auc_score(y_train_fold, lgb_pred_train))

    lgb_pred_val = model_lgb.predict_proba(X_val_fold)[:, 1]
    roc_auc_scores_val.append(roc_auc_score(y_val_fold, lgb_pred_val))

avg_roc_auc_train = np.mean(roc_auc_scores_train)
avg_roc_auc_val = np.mean(roc_auc_scores_val)

print(f"Training ROC AUC: {avg_roc_auc_train:.8f}")
print(f"Validation ROC AUC: {avg_roc_auc_val:.8f}")
Val_to_Train_Ratio = avg_roc_auc_train / avg_roc_auc_val
print(f"Val to Train Ratio: {Val_to_Train_Ratio:.4f}")

"""
Training ROC AUC: 0.92068072
Validation ROC AUC: 0.89042842
Val to Train Ratio: 1.0340
"""


# 3. hyperparameter tuned CatBoostClassifier
from catboost import CatBoostClassifier

weights = np.where(y_train_sel == 1, 1, 1650/540)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

roc_auc_scores = []

for train_index, val_index in skf.split(X_train_sel, y_train_sel):
    X_train_fold, X_val_fold = X_train_sel.iloc[train_index], X_train_sel.iloc[val_index]
    y_train_fold, y_val_fold = y_train_sel.iloc[train_index], y_train_sel.iloc[val_index]
    weights_fold = weights[train_index]

    model_cat = CatBoostClassifier(random_state=1, verbose=0, 
                                   depth=3, border_count=134, 
                                   iterations=309, learning_rate=0.013354284130669362, 
                                   leaf_estimation_iterations=3)
    model_cat.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)

    cat_pred_train = model_cat.predict_proba(X_train_fold)[:, 1]
    roc_auc_scores_train.append(roc_auc_score(y_train_fold, cat_pred_train))

    cat_pred_val = model_cat.predict_proba(X_val_fold)[:, 1]
    roc_auc_scores_val.append(roc_auc_score(y_val_fold, cat_pred_val))

avg_roc_auc_train = np.mean(roc_auc_scores_train)
avg_roc_auc_val = np.mean(roc_auc_scores_val)

print(f"Training ROC AUC: {avg_roc_auc_train:.8f}")
print(f"Validation ROC AUC: {avg_roc_auc_val:.8f}")
Val_to_Train_Ratio = avg_roc_auc_train / avg_roc_auc_val
print(f"Val to Train Ratio: {Val_to_Train_Ratio:.4f}")
"""
Training ROC AUC: 0.91858413
Validation ROC AUC: 0.89099168
Val to Train Ratio: 1.0310
"""


# 1. Voting Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report 

model_voting = VotingClassifier(estimators=[('xgb', model_xgb), ('lgb', model_lgb), ('cat', model_cat)],
                                voting='soft', n_jobs=-1)

model_voting.fit(X_train_sel, y_train_sel)

# Evaluation of train
voting_pred_train = model_voting.predict_proba(X_train_sel)[:, 1]
voting_pred_label_train = np.where(voting_pred_train > 0.5, 1, 0)
roc_auc_score_train = roc_auc_score(y_train_sel, voting_pred_train)

print("\nVoting Classifier Training Evaluation:")
print(confusion_matrix(y_train_sel, voting_pred_label_train))
print(classification_report(y_train_sel, voting_pred_label_train))
print(f"Training ROC AUC: {roc_auc_score_train}")

# Evaluation of val
voting_pred_val = model_voting.predict_proba(X_val_sel)[:, 1]
voting_pred_label_val = np.where(voting_pred_val > 0.5, 1, 0)
roc_auc_score_val = roc_auc_score(y_val_sel, voting_pred_val)

print("\nVoting Classifier Validation Evaluation:")
print(confusion_matrix(y_val_sel, voting_pred_label_val))
print(classification_report(y_val_sel, voting_pred_label_val))
print(f"Validation ROC AUC: {roc_auc_score_val}")

Val_to_Train_Ratio = roc_auc_score_train / roc_auc_score_val
print(f"Val to Train Ratio: {Val_to_Train_Ratio:.4f}")


# 2. model stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

base_models = []
base_models.append(('xgb',
                     XGBClassifier(random_state=1, n_jobs=-1,
                                   max_depth=2, min_child_weight=12, gamma=0.30005718740867243,
                                   n_estimators=223, learning_rate=0.018430904947696832,
                                   subsample=0.6273875931979261, colsample_bytree=0.6923385947687978,
                                   reg_alpha=0.14898013363049112, reg_lambda=4)))
base_models.append(('lgb',
                     LGBMClassifier(random_state=1, n_jobs=-1, verbose=-1,
                                    num_leaves=67, max_depth=2, min_child_samples=114,
                                    n_estimators=535, learning_rate=0.010030037966920426,
                                    subsample=0.8, colsample_bytree=0.6001143748173449,
                                    reg_alpha=0.01, reg_lambda=1.7337794540855653,
                                    min_child_weight=1.90)))
base_models.append(('cat',
                     CatBoostClassifier(random_state=1, verbose=0,
                                        depth=3, border_count=134,
                                        iterations=309, learning_rate=0.013354284130669362,
                                        leaf_estimation_iterations=3)))

meta_model = LogisticRegression()
model_stack = StackingClassifier(estimators=base_models, final_estimator=meta_model)
model_stack.fit(X_train_sel, y_train_sel)

# Evaluation of train
stack_pred_train = model_stack.predict_proba(X_train_sel)[:, 1]
stack_pred_label_train = np.where(stack_pred_train > 0.5, 1, 0)
roc_auc_score_train = roc_auc_score(y_train_sel, stack_pred_train)

print("\nStacking Classifier Training Evaluation:")
print(confusion_matrix(y_train_sel, stack_pred_label_train))
print(classification_report(y_train_sel, stack_pred_label_train))
print(f"Training ROC AUC: {roc_auc_score_train}")

# Evaluation of val
stack_pred_val = model_stack.predict_proba(X_val_sel)[:, 1]
stack_pred_label_val = np.where(stack_pred_val > 0.5, 1, 0)
roc_auc_score_val = roc_auc_score(y_val_sel, stack_pred_val)

print("\nStacking Classifier Validation Evaluation:")
print(confusion_matrix(y_val_sel, stack_pred_label_val))
print(classification_report(y_val_sel, stack_pred_label_val))
print(f"Validation ROC AUC: {roc_auc_score_val}")

Val_to_Train_Ratio = roc_auc_score_train / roc_auc_score_val
print(f"Val to Train Ratio: {Val_to_Train_Ratio:.4f}")


y_val_pred = pd.DataFrame(data= np.c_[voting_pred_label_val, stack_pred_label_val, y_val], 
                          columns=['voting', 'stack', 'y_val'])

voting_accuracy = (y_val_pred['voting'] == y_val_pred['y_val']).mean()
stack_accuracy = (y_val_pred['stack'] == y_val_pred['y_val']).mean()

accuracy_df = pd.DataFrame({
    'voting': [voting_accuracy],
    'stack': [stack_accuracy]
})

y_val_pred.loc['accuracy'] = [voting_accuracy, stack_accuracy,'-']

y_val_pred.T


test_features


test_pred = model_voting.predict_proba(test_features)[:, 1]
test_pred[:5]


submission = pd.DataFrame({'id': test.id, 'rainfall': test_pred})
print(submission.shape)
submission.head()


submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
print(submission.shape)
submission.head()

