import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')


train_df.head()


train_df.shape


train_df.info()


train_df.isnull().sum()


test_df.head()


test_df.shape


test_df.info()


test_df.isnull().sum()


ax = sns.countplot(x=train_df['NObeyesdad'], palette='muted', order=train_df['NObeyesdad'].value_counts().index)

plt.title('Obesity Levels')
plt.xticks(rotation=45)
plt.xlabel('Obesity')
plt.ylabel('Number of People')
ax.bar_label(ax.containers[0])
plt.show()


train_df.describe()


train_df.corr(numeric_only=True)


cmap = sns.diverging_palette(220, 20, as_cmap=True, s=60, l=90)
sns.heatmap(train_df.corr(numeric_only=True), annot=True, linewidths=0.5, fmt=".2f", cmap=cmap);


numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
train_df[numerical_cols].hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x=train_df['Height'], y=train_df['Weight'], hue=train_df['NObeyesdad'], alpha=0.6, palette='muted')
plt.title('Height vs Weight - Obesity Level')
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.countplot(data=train_df, x='family_history_with_overweight', hue='NObeyesdad', ax=axes[0, 0], palette='muted')
axes[0, 0].set_title('Family History - Obesity')

sns.countplot(data=train_df, x='FAVC', hue='NObeyesdad', ax=axes[0, 1], palette='muted')
axes[0, 1].set_title('FAVC - Obesity')

sns.countplot(data=train_df, x='CAEC', hue='NObeyesdad', ax=axes[1, 0], palette='muted')
axes[1, 0].set_title('CAEC - Obesity')

sns.countplot(data=train_df, x='MTRANS', hue='NObeyesdad', ax=axes[1, 1], palette='muted')
axes[1, 1].set_title('MTRANS - Obesity')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.countplot(data=train_df, x='Gender', hue='NObeyesdad', ax=axes[0, 0], palette='muted')
axes[0, 0].set_title('Gender - Obesity')

sns.countplot(data=train_df, x='SMOKE', hue='NObeyesdad', ax=axes[0, 1], palette='muted')
axes[0, 1].set_title('Smoke - Obesity')

sns.countplot(data=train_df, x='SCC', hue='NObeyesdad', ax=axes[1, 0], palette='muted')
axes[1, 0].set_title('SCC - Obesity')

sns.countplot(data=train_df, x='CALC', hue='NObeyesdad', ax=axes[1, 1], palette='muted')
axes[1, 1].set_title('CALC - Obesity')

plt.tight_layout()
plt.show()


train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['NObeyesdad'] = np.nan 
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)


#BMI 
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
df['BMI_2'] = df['BMI'] ** 2
#Age
df['Age'] = df['Age'].round().astype(int)
df['IsYoung'] = df['Age'].apply(lambda x: x < 25).astype(int)
df['IsAging'] = df['Age'].apply(lambda x: 25 <= x < 40).astype(int)
df['IsOld'] = df['Age'].apply(lambda x: 40 <= x <= 61).astype(int)


df['Family_BMI_Interaction'] = df['family_history_with_overweight'].astype(str) + "_" + (df['BMI'] > 25).astype(str)
df['Gender_Age_Interaction'] = df['Gender'].astype(str) + "_" + (df['Age'] > 30).astype(str)


categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS',
                   'Family_BMI_Interaction', 'Gender_Age_Interaction']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


integer_cols = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in integer_cols:
    df[col] = df[col].round().astype(int)


train = df[df['is_train'] == 1].drop(['is_train', 'id'], axis=1)
test = df[df['is_train'] == 0].drop(['is_train', 'id', 'NObeyesdad'], axis=1)


x = train.drop('NObeyesdad', axis=1)
y = train['NObeyesdad']
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


target_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
target_mapping


inv_target_mapping = {v: k for k, v in target_mapping.items()}


X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


models = {"LightGBM": lgb.LGBMClassifier(random_state=42, verbosity=-1),
          "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
          "CatBoost": CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False),
          "RandomForest": RandomForestClassifier(random_state=42, n_estimators=200)}
results = []
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    elapsed_time = time.time() - start_time
    results.append({"Model": name,"Accuracy": acc,"Time (s)": elapsed_time})   
    print(f"{name} -> Accuracy: %{acc*100:.2f}")
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)


results_df


from sklearn.ensemble import VotingClassifier
lgbm_params = {"objective": "multiclass","metric": "multi_logloss","verbosity": -1,"boosting_type": "gbdt",
               "random_state": 42,"num_class": 7,"learning_rate": 0.03,"n_estimators": 1500,"num_leaves": 40,
               "max_depth": -1,"min_child_samples": 20,"subsample": 0.8,"colsample_bytree": 0.8}

xgb_params = {"objective": "multi:softmax","eval_metric": "mlogloss","random_state": 42,"num_class": 7,
              "learning_rate": 0.03,"n_estimators": 1200,"max_depth": 6,"subsample": 0.8,"colsample_bytree": 0.8,
              "reg_alpha": 0.1,"reg_lambda": 1.5}

clf_lgbm = lgb.LGBMClassifier(**lgbm_params)
clf_xgb = xgb.XGBClassifier(**xgb_params)
clf_cat = CatBoostClassifier(random_state=42, verbose=0, iterations=1500, learning_rate=0.03, depth=6)

voting_clf = VotingClassifier(estimators=[('lgbm', clf_lgbm),('xgb', clf_xgb),('cat', clf_cat)],
                              voting='soft',weights=[1, 1, 1])

voting_clf.fit(X_train, y_train)
val_pred = voting_clf.predict(X_val)
acc = accuracy_score(y_val, val_pred)
print(f"Accuracy: %{acc * 100:.2f}")


predictions = voting_clf.predict(test)
predictions_labels = target_encoder.inverse_transform(predictions)


submission = pd.DataFrame({'id': test_df['id'],'NObeyesdad': predictions_labels})
submission.to_csv('submission.csv', index=False)


from sklearn.model_selection import StratifiedKFold
lgbm_params = {"objective": "multiclass","metric": "multi_logloss","verbosity": -1,"boosting_type": "gbdt",
               "num_class": 7,"learning_rate": 0.02,"n_estimators": 1000,"random_state": 42}
xgb_params = {"objective": "multi:softprob","eval_metric": "mlogloss","num_class": 7,"learning_rate": 0.03,
              "n_estimators": 800,"max_depth": 6,"random_state": 42}
cat_params = {"loss_function": "MultiClass","verbose": 0,"learning_rate": 0.03,"iterations": 1000,"depth": 6,
              "random_state": 42}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_probs_lgbm = np.zeros((len(test), 7))
test_probs_xgb = np.zeros((len(test), 7))
test_probs_cat = np.zeros((len(test), 7))

oof_preds = [] 
oof_targets = []

for fold, (train_index, val_index) in enumerate(skf.split(x, y)):
    X_train_fold, X_val_fold = x.iloc[train_index], x.iloc[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    #LightGBM
    clf_lgbm = lgb.LGBMClassifier(**lgbm_params)
    clf_lgbm.fit(X_train_fold, y_train_fold)
    test_probs_lgbm += clf_lgbm.predict_proba(test) / 5
    #XGBoost
    clf_xgb = xgb.XGBClassifier(**xgb_params)
    clf_xgb.fit(X_train_fold, y_train_fold)
    test_probs_xgb += clf_xgb.predict_proba(test) / 5
    #CatBoost
    clf_cat = CatBoostClassifier(**cat_params)
    clf_cat.fit(X_train_fold, y_train_fold)
    test_probs_cat += clf_cat.predict_proba(test) / 5
    #Validation
    val_pred = clf_lgbm.predict(X_val_fold)
    score = accuracy_score(y_val_fold, val_pred)
    print(f"Fold {fold+1} Val Accuracy (LGBM): %{score*100:.2f}")


# ENSEMBLE
final_test_probs = (test_probs_lgbm * 0.35) + (test_probs_xgb * 0.45) + (test_probs_cat * 0.20)

final_preds_indices = np.argmax(final_test_probs, axis=1)
final_labels = target_encoder.inverse_transform(final_preds_indices)


submission = pd.DataFrame({'id': test_df['id'],'NObeyesdad': final_labels})
submission.to_csv('submission_kfold.csv', index=False)


import joblib 
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS',
                    'Family_BMI_Interaction', 'Gender_Age_Interaction']
encoders = {} 
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le 

model = {'model': voting_clf,'encoders': encoders,'target_encoder': target_encoder}

joblib.dump(model, 'model.pkl')

