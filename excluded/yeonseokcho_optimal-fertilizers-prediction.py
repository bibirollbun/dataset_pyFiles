import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
print(sample_submission.shape)
sample_submission.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
print(test.shape)
test.head()


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
print(train.shape)
train.head()


train.info()


train.describe(include='all')


train['Fertilizer Name'].value_counts()
# 7 types of fertilizers


train_en = train.copy() 
test_en = test.copy()

train_en.shape, test_en.shape


# 

print(train_en.shape)
train_en.head()


#

print(test_en.shape)
test_en.head()


train_df = train_en.drop(columns=['id'])
test_df = test_en.drop(columns=['id'])

train_df.shape, test_df.shape


# Categorical variable encoding
from sklearn.preprocessing import LabelEncoder

# Encoding Soil Type, Crop Type, and Fertilizer Name (target)
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

train_df['Soil Type'] = le_soil.fit_transform(train_df['Soil Type'])
train_df['Crop Type'] = le_crop.fit_transform(train_df['Crop Type'])
train_df['Fertilizer Name'] = le_fert.fit_transform(train_df['Fertilizer Name'])

test_df['Soil Type'] = le_soil.fit_transform(test_df['Soil Type'])
test_df['Crop Type'] = le_crop.fit_transform(test_df['Crop Type'])

print(train_df.shape, test_df.shape)
train_df.head()


features = train_df.drop(columns=['Fertilizer Name'])
test_features = test_df.copy()

target = train_df['Fertilizer Name']

features.shape, target.shape, test_features.shape


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

X_train.shape, X_val.shape, y_train.shape, y_val.shape


# 1. XGBoost 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# used optimized hyperparameters
model_xgb = XGBClassifier(random_state=42, 
                          max_depth=7, min_child_weight=5, 
                          subsample=0.9598528437324805, colsample_bytree=0.7174250836504598, 
                          learning_rate=0.13982006857683707, n_estimators=271,
                          eval_metric='mlogloss', early_stopping_rounds=50) 

model_xgb.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

# 1. Get predicted probabilities for each class
y_val_xgb_proba = model_xgb.predict_proba(X_val)  # (n_samples, n_classes)

# 2. Extract the indices of the top 3 classes with the highest probabilities
y_val_xgb_pred_top3_idx = np.argsort(y_val_xgb_proba, axis=1)[:, -3:][:, ::-1]

# 3. Convert indices to fertilizer names (strings)
y_val_xgb_pred_top3_label = le_fert.inverse_transform(y_val_xgb_pred_top3_idx.ravel()).reshape(y_val_xgb_pred_top3_idx.shape)
y_val_label = le_fert.inverse_transform(y_val)

# 4. Compare predictions with actual values
xgb_pred_vs_real = np.column_stack((y_val_xgb_pred_top3_label, y_val_label))
print(xgb_pred_vs_real[:20]) 


# prediction accuracy
xgb_correct = [row[-1] in row[:3] for row in xgb_pred_vs_real]
xgb_accuracy = sum(xgb_correct) / len(xgb_correct)

print("Top-3 XGB Accuracy:", xgb_accuracy)

# Top-3 XGB Accuracy: 0.5233333333333333


# evaluation plot
plt.figure(figsize=(8, 3))
plt.title('Learning Curve - XGBoost')
plt.xlabel('Boosting Round')
plt.ylabel('MLogLoss')

sns.lineplot(data=model_xgb.evals_result()['validation_0']['mlogloss'], 
             label='Train', color='blue')

sns.lineplot(data=model_xgb.evals_result()['validation_1']['mlogloss'], 
             label='Validation', color='orange')

plt.ylim(1.8, 2.0 if max(model_xgb.evals_result()['validation_1']['mlogloss']) > 2 else 2.0)
plt.axvline(model_xgb.best_iteration, color='red', linestyle='--')  
plt.legend()
plt.show()


# 2. LightGBM
import lightgbm as lgb
from sklearn.metrics import accuracy_score

model_lgb = lgb.LGBMClassifier(random_state=42, verbose=-1, 
                               max_depth=7, 
                               num_leaves=55, 
                               min_child_samples=47, 
                               subsample=0.9852142919, colsample_bytree=0.8123620356542087, 
                               learning_rate=0.10408361292114134, n_estimators=274, 
                               eval_metric='mlogloss', early_stopping_rounds=50) 

model_lgb.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_val, y_val)])

y_val_lgb_proba = model_lgb.predict_proba(X_val)

y_val_lgb_pred_top3_idx = np.argsort(y_val_lgb_proba, axis=1)[:, -3:][:, ::-1]

y_val_lgb_pred_top3_label = le_fert.inverse_transform(y_val_lgb_pred_top3_idx.ravel()).reshape(y_val_lgb_pred_top3_idx.shape)
y_val_label = le_fert.inverse_transform(y_val)

lgb_pred_vs_real = np.column_stack((y_val_lgb_pred_top3_label, y_val_label))
print(lgb_pred_vs_real[:20]) 


# prediction accuracy
lgb_correct = [row[-1] in row[:3] for row in lgb_pred_vs_real]

lgb_accuracy = sum(lgb_correct) / len(lgb_correct)
print("Top-3 LGB Accuracy:", lgb_accuracy)

# Top-3 LGB Accuracy: 0.51836


# print(model_lgb.evals_result_)


# evaluation plot
plt.figure(figsize=(8, 3))
plt.title('Learning Curve - LGBoost')
plt.xlabel('Boosting Round')
plt.ylabel('multi_logloss')

sns.lineplot(data=model_lgb.evals_result_['training']['multi_logloss'], 
             label='Train', color='blue')

sns.lineplot(data=model_lgb.evals_result_['valid_1']['multi_logloss'], 
             label='Validation', color='orange')

plt.ylim(1.8, 2.0 if max(model_lgb.evals_result_['valid_1']['multi_logloss']) > 2 else 2.0)
plt.axvline(model_lgb.best_iteration_, color='red', linestyle='--')  
plt.legend()
plt.show()


# 3. CatBoostClassifier
from catboost import CatBoostClassifier
model_cat = CatBoostClassifier(random_state=42, verbose=0, 
                               iterations=1160, 
                               bagging_temperature=0.8745401188473625
                              ) 

model_cat.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_val, y_val)])

y_val_cat_proba = model_cat.predict_proba(X_val)  

y_val_cat_pred_top3_idx = np.argsort(y_val_cat_proba, axis=1)[:, -3:][:, ::-1]

y_val_cat_pred_top3_label = le_fert.inverse_transform(y_val_cat_pred_top3_idx.ravel()).reshape(y_val_cat_pred_top3_idx.shape)
y_val_label = le_fert.inverse_transform(y_val)

cat_pred_vs_real = np.column_stack((y_val_cat_pred_top3_label, y_val_label))
print(cat_pred_vs_real[:20]) 


# prediction accuracy

cat_correct = [row[-1] in row[:3] for row in cat_pred_vs_real]

cat_accuracy = sum(cat_correct) / len(cat_correct)
print("Top-3 CAT Accuracy:", cat_accuracy)

# Top-3 CAT Accuracy: 0.5118933333333333


# print(model_cat.evals_result_)
# print(model_cat.evals_result_['learn']['MultiClass'])


# evaluation plot

plt.figure(figsize=(8, 3))
plt.title('Learning Curve - CatBoost')
plt.xlabel('Boosting Round')
plt.ylabel('MultiLogloss')

sns.lineplot(
    data=model_cat.evals_result_['learn']['MultiClass'],
    label='Train', color='blue'
)
sns.lineplot(
    data=model_cat.evals_result_['validation_1']['MultiClass'],
    label='Validation_1', color='orange'
) # train and validation curve are overlapped

plt.ylim(min(model_cat.evals_result_['learn']['MultiClass']), 
         max(model_cat.evals_result_['learn']['MultiClass']))

plt.axvline(model_cat.get_best_iteration(), color='red', linestyle='--')

plt.legend()
plt.show() 


# Ensemble Model - VotingClassifier
from sklearn.ensemble import VotingClassifier

model_xgb = XGBClassifier(random_state=42, 
                          max_depth=7, min_child_weight=5, 
                          subsample=0.9598528437324805, colsample_bytree=0.7174250836504598, 
                          learning_rate=0.13982006857683707, n_estimators=271) 

model_lgb = lgb.LGBMClassifier(random_state=42, verbose=-1,
                               max_depth=7, 
                               num_leaves=55, 
                               min_child_samples=47, 
                               subsample=0.9852142919, colsample_bytree=0.8123620356542087, 
                               learning_rate=0.10408361292114134, n_estimators=274)

model_cat = CatBoostClassifier(random_state=42, verbose=0, 
                               iterations=1160, 
                               bagging_temperature=0.8745401188473625)

Voting_models = [
    ('XGBoost', model_xgb), 
    ('LGBM', model_lgb), 
    ('CatBoost', model_cat)
]

voting_model = VotingClassifier(estimators=Voting_models, voting='soft')
voting_model.fit(X_train, y_train)

y_val_vot_proba = voting_model.predict_proba(X_val)

y_val_vot_pred_top3_idx = np.argsort(y_val_vot_proba, axis=1)[:, -3:][:, ::-1]

y_val_vot_pred_top3_label = le_fert.inverse_transform(y_val_vot_pred_top3_idx.ravel()).reshape(y_val_vot_pred_top3_idx.shape)
y_val_label = le_fert.inverse_transform(y_val)

vot_pred_vs_real = np.column_stack((y_val_vot_pred_top3_label, y_val_label))
print(vot_pred_vs_real[:20])


# prediction accuracy

vot_correct = [row[-1] in row[:3] for row in vot_pred_vs_real]
vot_accuracy = sum(vot_correct) / len(vot_correct)

print("Top-3 Voting Accuracy:", vot_accuracy)

# Top-3 Voting Accuracy: 0.5235933333333334
# Top-3 Voting Accuracy: 0.5234266666666667


#  Ensemble Model - StackingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

base_models = []
base_models.append(('xgb', XGBClassifier(random_state=42, 
                                         max_depth=7, min_child_weight=5, 
                                         subsample=0.9598528437324805, colsample_bytree=0.7174250836504598, 
                                         learning_rate=0.13982006857683707, n_estimators=271)))

base_models.append(('lgb', lgb.LGBMClassifier(random_state=42, verbose=-1, 
                                             max_depth=7, 
                                             num_leaves=55, 
                                             min_child_samples=47, 
                                             subsample=0.9852142919, colsample_bytree=0.8123620356542087, 
                                             learning_rate=0.10408361292114134, n_estimators=274)))

base_models.append(('cat', CatBoostClassifier(random_state=42, verbose=0, 
                                              iterations=1160, 
                                              bagging_temperature=0.8745401188473625)))

meta_model = LogisticRegression()
stack_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
stack_model.fit(X_train, y_train)

y_val_sta_proba = stack_model.predict_proba(X_val)

y_val_sta_pred_top3_idx = np.argsort(y_val_sta_proba, axis=1)[:, -3:][:, ::-1]

y_val_sta_pred_top3_label = le_fert.inverse_transform(y_val_sta_pred_top3_idx.ravel()).reshape(y_val_sta_pred_top3_idx.shape)
y_val_label = le_fert.inverse_transform(y_val)

sta_pred_vs_real = np.column_stack((y_val_sta_pred_top3_label, y_val_label))
print(sta_pred_vs_real[:20])


# prediction accuracy

sta_correct = [row[-1] in row[:3] for row in sta_pred_vs_real]
sta_accuracy = sum(sta_correct) / len(sta_correct)

print("Top-3 Stacking Accuracy:", sta_accuracy)

# Top-3 Stacking Accuracy: 0.52484


test_proba = stack_model.predict_proba(test_features)

test_pred_top3_idx = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
test_pred_top3_label = le_fert.inverse_transform(test_pred_top3_idx.ravel()).reshape(test_pred_top3_idx.shape)

submission = pd.DataFrame({
    'id': test['id'], 
    'Fertilizer Name': [' '.join(row) for row in test_pred_top3_label]
})

submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')

print(submission.shape)
submission.head()

