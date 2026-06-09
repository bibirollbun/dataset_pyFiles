import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


sample = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
print(sample.shape)
sample.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
print(test.shape)
test.head()


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
print(train.shape)
train.head()


train.info()


train.describe(include='all')


train['y'].value_counts()


# numeric data
train_num_target = train.select_dtypes(include=['float', 'int'])
test_num = test.select_dtypes(include=['float', 'int'])

train_num_target = train_num_target.drop(['id'], axis=1)
test_num = test_num.drop(['id'], axis=1)

print(train_num_target.shape, test_num.shape)
train_num_target.head(1)


# diagnostic plots
def diagnostic_plots_num(df, variable, target, axes): 
    unique_values = df[target].unique()

    category_1 = df[df[target] == unique_values[0]][variable]
    category_2 = df[df[target] == unique_values[1]][variable]
    
    sns.kdeplot(data=category_1, ax=axes[0], color='blue', label=str(unique_values[0]))
    sns.kdeplot(data=category_2, ax=axes[0], color='orange', label=str(unique_values[1]))
    axes[0].set_title(f'KDE Plot (Skew: {df[variable].skew():.3f})')
    axes[0].legend()

    sns.boxplot(ax=axes[1], y=variable, x=target, data=df)
    axes[1].set_title('Boxplot')

variables = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
fig, axes = plt.subplots(len(variables), 2, figsize=(15, 22)) 

for i, variable in enumerate(variables):
    diagnostic_plots_num(train_num_target, variable, 'y', axes[i])

plt.tight_layout()
plt.show()


train_num = train_num_target.drop(['y'], axis=1)
target = train_num_target['y']

train_num.shape, target.shape, test_num.shape


train_num_features = train_num.copy()

# over 60
train_num_features['is_senior'] = (train_num_features['age'] >= 60).astype(int)

# balance
train_num_features['positive_balance'] = (train_num_features['balance'] > 0).astype(int)
    
# log transformation 
train_num_features['log_balance'] = np.log1p(train_num_features['balance'] - train_num_features['balance'].min())

# contact number in campaign & previous
train_num_features['total_contacts'] = train_num_features['campaign'] + train_num_features['previous']
    
# first contact in campaign
train_num_features['is_first_contact'] = (train_num_features['campaign'] == 1).astype(int)

# contacted in past
train_num_features['previously_contacted'] = (train_num_features['pdays'] != -1).astype(int)

print(train_num_features.shape)
train_num_features.head()


test_num_features = test_num.copy()

# over 60
test_num_features['is_senior'] = (test_num_features['age'] >= 60).astype(int)

# balance
test_num_features['positive_balance'] = (test_num_features['balance'] > 0).astype(int)
    
# log transformation
test_num_features['log_balance'] = np.log1p(test_num_features['balance'] - test_num_features['balance'].min())

# contact number in campaign & previous
test_num_features['total_contacts'] = test_num_features['campaign'] + test_num_features['previous']
    
# first contact in campaign
test_num_features['is_first_contact'] = (test_num_features['campaign'] == 1).astype(int)

# contacted in past
test_num_features['previously_contacted'] = (test_num_features['pdays'] != -1).astype(int)

print(test_num_features.shape)
test_num_features.head()


print(train_num.shape, target.shape, test_num.shape)
print(train_num_features.shape, target.shape, test_num_features.shape)





# categorical data
train_cat = train.select_dtypes(include=['object'])
test_cat = test.select_dtypes(include=['object'])

print(train_cat.shape, test_cat.shape)
train_cat.head(1)


train_cat_target = pd.concat([train_cat, train['y']], axis=1)

print(train_cat_target.shape)
train_cat_target.head(1)


# Diagnostic Plots
def diagnostic_plots_cat(data, col, hue=None, rotation=15):
    order = data[col].value_counts().index
    sns.countplot(x=col, hue=hue, data=data, order=order)
    plt.title(f"countplot of {col}")
    plt.xticks(rotation=rotation, ha='right')
    if hue is not None:
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.figure(figsize=(15, 18))

plt.subplot(5, 2, 1)
diagnostic_plots_cat(train_cat_target, 'job', hue='y')

plt.subplot(5, 2, 2)
diagnostic_plots_cat(train_cat_target, 'marital', hue='y')

plt.subplot(5, 2, 3)
diagnostic_plots_cat(train_cat_target, 'education', hue='y')

plt.subplot(5, 2, 4)
diagnostic_plots_cat(train_cat_target, 'default', hue='y')

plt.subplot(5, 2, 5)
diagnostic_plots_cat(train_cat_target, 'housing', hue='y')

plt.subplot(5, 2, 6)
diagnostic_plots_cat(train_cat_target, 'loan', hue='y')

plt.subplot(5, 2, 7)
diagnostic_plots_cat(train_cat_target, 'contact', hue='y')

plt.subplot(5, 2, 8)
diagnostic_plots_cat(train_cat_target, 'month', hue='y') 	

plt.subplot(5, 2, 9)
diagnostic_plots_cat(train_cat_target, 'poutcome', hue='y') 

plt.tight_layout()
plt.show()


train_cat_features = train_cat.copy()

# age in life stage
train_cat_features['age_group'] = pd.cut(train_num_features['age'], 
                                bins=[0, 29, 60, 100], 
                                labels=['Young', 'Middle-aged', 'Senior'])

# high success month
high_success_months = ['mar', 'sep', 'oct', 'dec']
train_cat_features['high_success_month'] = train_cat_features['month'].isin(high_success_months).astype(int)

# high success job
high_success_jobs = ['student', 'retired']
train_cat_features['high_success_job'] = train_cat_features['job'].isin(high_success_jobs).astype(int)
    
# number of loans?
train_cat_features['num_loans'] = (train_cat_features['housing'] == 'yes').astype(int) + (train_cat_features['loan'] == 'yes').astype(int)

# have a loans?
train_cat_features['any_loan'] = (train_cat_features['num_loans'] > 0).astype(int)

# success in past + Phone
train_cat_features['poutcome_contact'] = train_cat_features['poutcome'] + "_" + train_cat_features['contact']

print(train_cat_features.shape)
train_cat_features.head()


train_cat_features['num_loans'].value_counts()


test_cat_features = test_cat.copy()

# age in life stage
test_cat_features['age_group'] = pd.cut(test_num_features['age'], 
                                bins=[0, 29, 60, 100], 
                                labels=['Young', 'Middle-aged', 'Senior'])

# high success month
high_success_months = ['mar', 'sep', 'oct', 'dec']
test_cat_features['high_success_month'] = test_cat_features['month'].isin(high_success_months).astype(int)

# high success job
high_success_jobs = ['student', 'retired']
test_cat_features['high_success_job'] = test_cat_features['job'].isin(high_success_jobs).astype(int)
    
# number of loans?
test_cat_features['num_loans'] = (test_cat_features['housing'] == 'yes').astype(int) + (test_cat_features['loan'] == 'yes').astype(int)

# have a loans?
test_cat_features['any_loan'] = (test_cat_features['num_loans'] > 0).astype(int)

# success in past + Phone
test_cat_features['poutcome_contact'] = test_cat_features['poutcome'] + "_" + test_cat_features['contact']

print(test_cat_features.shape)
test_cat_features.head()


print(train_cat.shape, target.shape, test_cat.shape)
print(train_cat_features.shape, target.shape, test_cat_features.shape)


 


from sklearn.preprocessing import LabelEncoder

encoders = {}
train_cat_features_encoded = train_cat_features.copy()

for col in train_cat_features_encoded.columns:
    le = LabelEncoder()
    train_cat_features_encoded[col] = le.fit_transform(train_cat_features_encoded[col])
    encoders[col] = le  

test_cat_features_encoded = test_cat_features.copy()

for col in test_cat_features_encoded.columns:
    if col in encoders:
        le = encoders[col]
        test_cat_features_encoded[col] = test_cat_features_encoded[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    else:
        test_cat_features_encoded[col] = test_cat_features_encoded[col]

print(train_cat_features_encoded.shape, test_cat_features_encoded.shape)
train_cat_features_encoded.head(1)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_num_features_scaled = scaler.fit_transform(train_num_features)
test_num_features_scaled = scaler.transform(test_num_features)

train_num_features_scaled = pd.DataFrame(train_num_features_scaled, columns=train_num_features.columns)
test_num_features_scaled = pd.DataFrame(test_num_features_scaled, columns=test_num_features.columns)

print(train_num_features_scaled.shape, test_num_features_scaled.shape)
train_num_features_scaled.head(1)


train_feature = pd.concat([train_cat_features_encoded, train_num_features_scaled], axis=1)
test_feature = pd.concat([test_cat_features_encoded, test_num_features_scaled], axis=1)

print(train_feature.shape, target.shape, test_feature.shape)
train_feature.head().T


import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
import random

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

X = train_feature.values
y = target.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

input_dim = X_train.shape[1]

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train)

class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

model_dnn = Sequential([

    layers.Dense(256, input_shape=(input_dim,), kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),

    layers.Dense(128,  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),

    layers.Dense(64,  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)  # learning_rate default = 0.001 # overfit @ 0.0001 

model_dnn.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['AUC']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_dnn = model_dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=1024,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=2
)

test_X = test_feature.values
preds_dnn = model_dnn.predict(test_X)

pred_labels_dnn = (preds_dnn >= 0.5).astype(int)
pred_labels_dnn[:20].flatten()


print(history_dnn.history.keys())


#  Visualize Training History
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history_dnn.history['loss'], label='Train Loss')
plt.plot(history_dnn.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_dnn.history['AUC'], label='Train AUC')         
plt.plot(history_dnn.history['val_AUC'], label='Validation AUC')  
plt.title('AUC vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


from sklearn.metrics import roc_auc_score, roc_curve

val_preds_dnn = model_dnn.predict(X_val).ravel()  # shape (samples,)
val_true = y_val

auc_score = roc_auc_score(val_true, val_preds_dnn)
print(f'model_dnn validation roc auc: {auc_score:.4f}')

fpr, tpr, thresholds = roc_curve(val_true, val_preds_dnn)

plt.figure(figsize=(4, 3))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


from sklearn.metrics import classification_report

val_pred_labels_dnn = (val_preds_dnn >= 0.7).astype(int)

print(classification_report(y_val, val_pred_labels_dnn, target_names=['No (0)', 'Yes (1)']))


import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"Scale Pos Weight for XGBoost: {scale_pos_weight:.6f}")

model_xgb = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric=['logloss', 'auc'],  
    use_label_encoder=False,
    random_state=42,
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=10,
    verbose=0
)

val_preds_proba_xgb = model_xgb.predict_proba(X_val)[:, 1]
val_auc_xgb = roc_auc_score(y_val, val_preds_proba_xgb)
print(f"model_xgb validation roc auc: {val_auc_xgb:.8f}")

pred_labels_xgb = (val_preds_proba_xgb >= 0.5).astype(int)
pred_labels_xgb[:20].flatten()

# model_xgb validation roc auc: 0.96779550
# array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


#  Visualize Training History

evals_result_xgb = model_xgb.evals_result()

# Extract metrics from evals_result_xgb
train_loss_xgb = evals_result_xgb['validation_0']['logloss'] if 'logloss' in evals_result_xgb['validation_0'] else None
val_loss_xgb = evals_result_xgb['validation_1']['logloss'] if 'logloss' in evals_result_xgb['validation_1'] else None

train_auc_xgb = evals_result_xgb['validation_0']['auc'] if 'auc' in evals_result_xgb['validation_0'] else None
val_auc_xgb = evals_result_xgb['validation_1']['auc'] if 'auc' in evals_result_xgb['validation_1'] else None

plt.figure(figsize=(10, 4))

# Loss Plot
plt.subplot(1, 2, 1)

if train_loss_xgb and val_loss_xgb:
    plt.plot(train_loss_xgb, label='Train Loss')
    plt.plot(val_loss_xgb, label='Validation Loss')
plt.title('Loss vs Epochs (XGBoost)') 
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)

# AUC Plot
plt.subplot(1, 2, 2)

if train_auc_xgb and val_auc_xgb:
    plt.plot(train_auc_xgb, label='Train AUC')
    plt.plot(val_auc_xgb, label='Validation AUC')
plt.title('AUC vs Epochs (XGBoost)') 
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout() 
plt.show()


# ROC Curve 

auc_score = roc_auc_score(y_val, val_preds_proba_xgb)
print(f'Validation ROC AUC: {auc_score:.8f}')

fpr, tpr, _ = roc_curve(y_val, val_preds_proba_xgb)
plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.grid(True)
plt.show()

# Validation ROC AUC: 0.96707851


# classification report
val_preds_labels_xgb = (val_preds_proba_xgb >= 0.5).astype(int)
print(classification_report(y_val, val_preds_labels_xgb, target_names=['No (0)', 'Yes (1)']))





# Hyperparameter Tuning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# param_dist_xgb = {
#     'max_depth': randint(3, 11),   
#     'min_child_weight': randint(1, 11),         
              
#     'subsample': uniform(0.7, 0.3),            
#     'colsample_bytree': uniform(0.6, 0.4),        
              
#     'learning_rate': loguniform(0.01, 0.3),       
#     'n_estimators': randint(100, 1000),        
             
#     'gamma': uniform(0, 0.5),                   
#     'reg_alpha': loguniform(1e-5, 10.0),         
#     'reg_lambda': loguniform(1e-5, 10.0),        
# }

# base_model_xgb = xgb.XGBClassifier(
#     scale_pos_weight=scale_pos_weight, 
#     random_state=42,
#     use_label_encoder=False,  
#     eval_metric='logloss'     
# )

# random_search_xgb = RandomizedSearchCV(
#     estimator=base_model_xgb,
#     param_distributions=param_dist_xgb,
#     n_iter=40,            
#     scoring='roc_auc',
#     cv=3,
#     verbose=0,
#     n_jobs=-1,
#     random_state=42
# )

# random_search_xgb.fit(X_train, y_train)

# print("Best CV ROC AUC:", random_search_xgb.best_score_)
# print("Best Parameters of xgb:", random_search_xgb.best_params_)


# model_xgb_final with Best Parameters
best_params = {
    'max_depth': 7,    
    'min_child_weight': 8,         
    'subsample': 0.9909729556485982,             
    'colsample_bytree': 0.608233797718321,        
    'learning_rate': 0.05958389350068958,        
    'n_estimators': 847,      
    'gamma': 0.4828160165372797,                  
    'reg_alpha': 6.220025976819156,        
    'reg_lambda': 4.9352962094020985,      
}

all_params = {
    **best_params,
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'use_label_encoder': False,
    'random_state': 42,
    'tree_method': 'hist',
    'scale_pos_weight': scale_pos_weight
}

model_xgb_final = xgb.XGBClassifier(**all_params)

# Prepare evaluation set for monitoring performance
eval_set = [(X_train, y_train), (X_val, y_val)]

model_xgb_final.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)


val_preds_proba_xgb_final = model_xgb_final.predict_proba(X_val)[:, 1]
val_auc_xgb_final = roc_auc_score(y_val, val_preds_proba_xgb_final)
print(f"model_xgb_final validation roc auc: {val_auc_xgb_final:.8f}")

pred_labels_xgb_final = (val_preds_proba_xgb_final >= 0.5).astype(int)
pred_labels_xgb_final[:20].flatten()

# model_xgb_final validation roc auc: 0.96785735
# array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


# Visualize Training History

evals_result_xgb_final = model_xgb_final.evals_result()

train_loss_xgb_final = evals_result_xgb_final['validation_0']['logloss'] if 'logloss' in evals_result_xgb_final['validation_0'] else None
val_loss_xgb_final = evals_result_xgb_final['validation_1']['logloss'] if 'logloss' in evals_result_xgb_final['validation_1'] else None

train_auc_xgb_final = evals_result_xgb_final['validation_0']['auc'] if 'auc' in evals_result_xgb_final['validation_0'] else None
val_auc_xgb_final = evals_result_xgb_final['validation_1']['auc'] if 'auc' in evals_result_xgb_final['validation_1'] else None

plt.figure(figsize=(10, 4))

# Loss Plot
plt.subplot(1, 2, 1)
if train_loss_xgb_final and val_loss_xgb_final:
    plt.plot(train_loss_xgb_final, label='Train Loss')
    plt.plot(val_loss_xgb_final, label='Validation Loss')
plt.title('Loss vs Epochs (XGBoost Final)')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)

# AUC Plot
plt.subplot(1, 2, 2)
if train_auc_xgb_final and val_auc_xgb_final:
    plt.plot(train_auc_xgb_final, label='Train AUC')
    plt.plot(val_auc_xgb_final, label='Validation AUC')
plt.title('AUC vs Epochs (XGBoost Final)')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ROC Curve 
auc_score = roc_auc_score(y_val, val_preds_proba_xgb_final)
print(f'Validation ROC AUC: {auc_score:.8f}')

fpr, tpr, _ = roc_curve(y_val, val_preds_proba_xgb_final)
plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.grid(True)
plt.show()

# Validation ROC AUC: 0.96785735 (<- 0.96707851)


# classification report
val_preds_labels_xgb_final = (val_preds_proba_xgb_final >= 0.5).astype(int)
print(classification_report(y_val, val_preds_labels_xgb_final, target_names=['No (0)', 'Yes (1)']))


import lightgbm as lgb

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_lgb = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight, 
    objective='binary',
    boosting_type='gbdt',
    metric=['binary_logloss', 'auc'],  
    learning_rate=0.1,
    n_estimators=100,
    max_depth=10,
    num_leaves=63,
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8
)

model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)]
)

val_preds_proba_lgb = model_lgb.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds_proba_lgb)
print(f"Validation ROC AUC (LightGBM): {val_auc:.4f}")

pred_labels_lgb = (val_preds_proba_lgb >= 0.5).astype(int)
pred_labels_lgb[:20].flatten()

# Validation ROC AUC (LightGBM): 0.9549
# array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


# Visualize Training History
evals_result_lgb = model_lgb.evals_result_

train_loss_lgb = evals_result_lgb['training']['binary_logloss'] if 'binary_logloss' in evals_result_lgb.get('training', {}) else None
val_loss_lgb = evals_result_lgb['valid_1']['binary_logloss'] if 'binary_logloss' in evals_result_lgb.get('valid_1', {}) else None

train_auc_lgb = evals_result_lgb['training']['auc'] if 'auc' in evals_result_lgb.get('training', {}) else None
val_auc_lgb = evals_result_lgb['valid_1']['auc'] if 'auc' in evals_result_lgb.get('valid_1', {}) else None

plt.figure(figsize=(10, 4))

# Loss Plot
plt.subplot(1, 2, 1)

if train_loss_lgb and val_loss_lgb:
    epochs_lgb = range(len(train_loss_lgb))
    plt.plot(epochs_lgb, train_loss_lgb, label='Train Loss')
    plt.plot(epochs_lgb, val_loss_lgb, label='Validation Loss')
plt.title('Loss vs Epochs (LightGBM)')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)

# AUC Plot
plt.subplot(1, 2, 2)

if train_auc_lgb and val_auc_lgb:
    epochs_lgb = range(len(train_auc_lgb))
    plt.plot(epochs_lgb, train_auc_lgb, label='Train AUC')
    plt.plot(epochs_lgb, val_auc_lgb, label='Validation AUC')
plt.title('AUC vs Epochs (LightGBM)')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ROC Curve
best_val_auc_lgb = val_auc_lgb[-1] if isinstance(val_auc_lgb, (list, tuple)) else val_auc_lgb

auc_score_lgb = roc_auc_score(y_val, val_preds_proba_lgb)
print(f'Validation ROC AUC: {best_val_auc_lgb:.8f}')

fpr_lgb, tpr_lgb, _ = roc_curve(y_val, val_preds_proba_lgb)

plt.figure(figsize=(5, 4))

plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {auc_score_lgb:.4f})')
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LightGBM')
plt.legend() 
plt.grid(True)
plt.show()

# Validation ROC AUC: 0.95829677


# classification_report
val_preds_labels_lgb = (val_preds_proba_lgb >= 0.5).astype(int)
print(classification_report(y_val, val_preds_labels_lgb, target_names=['No (0)', 'Yes (1)']))





# hyperparameter tuning

# param_dist_lgb = {
#         'max_depth': randint(3, 15), 

#         'num_leaves': randint(20, 1000),   

#         'min_child_samples': randint(5, 100), 

#         'subsample': uniform(0.6, 0.4),  
#         'colsample_bytree': uniform(0.6, 0.4), 

#         'learning_rate': loguniform(0.01, 0.3),  
   
#         'n_estimators': randint(100, 1000),        
               
#         'reg_alpha': loguniform(1e-5, 10.0),        
#         'reg_lambda': loguniform(1e-5, 10.0),                
# }

# base_model_lgb = lgb.LGBMClassifier(
#     scale_pos_weight=scale_pos_weight, 
#     objective='binary',
#     boosting_type='gbdt',
#     random_state=42,
#     metric='auc', 
#     verbosity =-1
# )

# random_search_lgb = RandomizedSearchCV(
#     estimator=base_model_lgb,
#     param_distributions=param_dist_lgb,
#     n_iter=40,
#     scoring='roc_auc',
#     cv=3,
#     verbose=0,
#     n_jobs=-1,
#     random_state=42
# )

# random_search_lgb.fit(X_train, y_train)

# print("Best CV ROC AUC (LightGBM):", f"{random_search_lgb.best_score_:.4f}")
# print("Best Parameters (LightGBM):", random_search_lgb.best_params_)


# model_lgb_final with Best Parameters

best_params_lgb = {
    'max_depth' : 14, 
    'num_leaves' : 290,    
    'min_child_samples': 62,
    'subsample': 0.8099025726528951,  
    'colsample_bytree': 0.7216968971838151, 
    'learning_rate': 0.102493222, 
    'n_estimators': 158,      
    'reg_alpha': 6.220025976819156,      
    'reg_lambda': 0.7085721663941598,
}

# all parameters
all_params_lgb = {
    **best_params_lgb,
    'objective': 'binary', 
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'random_state': 42,
    'scale_pos_weight': scale_pos_weight 
}

# final lgb model
model_lgb_final = lgb.LGBMClassifier(**all_params_lgb)

eval_set = [(X_train, y_train), (X_val, y_val)]

model_lgb_final.fit(
    X_train, y_train,
    eval_set=eval_set,
    callbacks=[
        lgb.early_stopping(stopping_rounds=10, verbose=False)
    ]
)

# evaluation
val_preds_proba_lgb_final = model_lgb_final.predict_proba(X_val)[:, 1]
val_auc_lgb_final = roc_auc_score(y_val, val_preds_proba_lgb_final)
print(f"model_lgb_final validation roc auc: {val_auc_lgb_final:.8f}")

pred_labels_lgb_final = (val_preds_proba_lgb_final >= 0.5).astype(int)
print(pred_labels_lgb_final[:20].flatten())


# Visualize Training History
evals_result_lgb_final = model_lgb_final.evals_result_

train_loss_lgb_final = evals_result_lgb_final['training']['binary_logloss'] if 'binary_logloss' in evals_result_lgb_final.get('training', {}) else None
val_loss_lgb_final = evals_result_lgb_final['valid_1']['binary_logloss'] if 'binary_logloss' in evals_result_lgb_final.get('valid_1', {}) else None

train_auc_lgb_final = evals_result_lgb_final['training']['auc'] if 'auc' in evals_result_lgb_final.get('training', {}) else None
val_auc_lgb_final = evals_result_lgb_final['valid_1']['auc'] if 'auc' in evals_result_lgb_final.get('valid_1', {}) else None

plt.figure(figsize=(10, 4))

# Loss Plot
plt.subplot(1, 2, 1)
if train_loss_lgb_final and val_loss_lgb_final:
    epochs_lgb_final = range(len(train_loss_lgb_final))
    plt.plot(epochs_lgb_final, train_loss_lgb_final, label='Train Loss')
    plt.plot(epochs_lgb_final, val_loss_lgb_final, label='Validation Loss')
plt.title('Loss vs Epochs (LightGBM Final)')
plt.xlabel('Epoch') 
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)

# AUC Plot
plt.subplot(1, 2, 2)
if train_auc_lgb_final and val_auc_lgb_final:
    epochs_lgb_final = range(len(train_auc_lgb_final))
    plt.plot(epochs_lgb_final, train_auc_lgb_final, label='Train AUC')
    plt.plot(epochs_lgb_final, val_auc_lgb_final, label='Validation AUC')
plt.title('AUC vs Epochs (LightGBM Final)')
plt.xlabel('Epoch') 
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ROC Curve 
auc_score = roc_auc_score(y_val, val_preds_proba_lgb_final)
print(f'Validation ROC AUC: {auc_score:.8f}')

fpr, tpr, _ = roc_curve(y_val, val_preds_proba_lgb_final)
plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LGBoost')
plt.grid(True)
plt.show()

# Validation ROC AUC: 0.95999888 (<- 0.95829677)


# classification report
val_preds_labels_lgb_final = (val_preds_proba_lgb_final >= 0.5).astype(int)
print(classification_report(y_val, val_preds_labels_lgb_final, target_names=['No (0)', 'Yes (1)']))


# optimal weight ratio for models
preds_xgb_val = model_xgb_final.predict_proba(X_val)[:, 1]
preds_lgb_val = model_lgb_final.predict_proba(X_val)[:, 1]
preds_dnn_val = model_dnn.predict(X_val).flatten()

best_auc = 0
best_weights = {}

for w1 in np.arange(0.5, 1.0, 0.01):
    for w2 in np.arange(0.01, 0.5, 0.01):
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue

        weights = {'xgb': w1, 'lgb': w2, 'dnn': w3}

        ensemble_preds = (
            preds_xgb_val * weights['xgb'] +
            preds_lgb_val * weights['lgb'] +
            preds_dnn_val * weights['dnn']
        )
        
        current_auc = roc_auc_score(y_val, ensemble_preds)

        if current_auc > best_auc:
            best_auc = current_auc
            best_weights = weights

print(f"Best Validation AUC: {best_auc:.8f}")
print(f"Best Weights: {best_weights}")


# evaluation & AUC curcve
final_ensemble_preds_proba = (
    preds_xgb_val * best_weights['xgb'] +
    preds_lgb_val * best_weights['lgb'] +
    preds_dnn_val * best_weights['dnn']
)

final_ensemble_auc_score = roc_auc_score(y_val, final_ensemble_preds_proba)
print(f"Final Ensemble Model Validation ROC AUC with Best Weights: {final_ensemble_auc_score:.8f}")

fpr, tpr, thresholds = roc_curve(y_val, final_ensemble_preds_proba)

# ROC Curve
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f'Ensemble (AUC = {final_ensemble_auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')

plt.title('ROC Curve - Final Ensemble Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show() 

# Ensemble Model 0.96783616 Vs Tuned XGB 0.96785735


test_feature.head()


test_proba = model_xgb_final.predict_proba(test_feature)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)
test_pred[:20].flatten()


submission = pd.DataFrame({
    'id': test['id'],
    'y': test_pred
})
submission.head(10)


submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head(10)

