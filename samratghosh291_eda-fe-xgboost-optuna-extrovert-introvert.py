import pandas as pd
import seaborn as sns
import numpy as np
import optuna
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder 
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss,f1_score, roc_curve, auc, roc_auc_score
import kagglehub
from optuna.visualization import plot_optimization_history,plot_parallel_coordinate,plot_slice,plot_contour,plot_param_importances
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.filterwarnings('ignore')


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

# Download latest version
kagglehub.dataset_download("rakeshkapilavai/extrovert-vs-introvert-behavior-data")


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
org_train_1 = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
org_train_2=pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')


print(f" Train Shape: {train.shape}")
print(f" Test Shape: {test.shape}")
print(f" Original Dataset 1 Shape: {org_train_1.shape}")
print(f" Original Dataset 2 Shape: {org_train_2.shape}")


#drop 'id' column in train and test
train.drop(columns=['id'],inplace=True)
test_id=test['id']
test.drop(columns=['id'],inplace=True)
print(f"Id Drop âœ…")

#merge train dataset with original datasets
train = pd.concat([train, org_train_1,org_train_2], ignore_index=True)
print(f"Merge Dataset doneâœ…")

# Returns a boolean Series: True = duplicate row (after first occurrence)
duplicates = train.duplicated()

# Count total duplicate rows
duplicate_count = duplicates.sum()
print(f"Number of duplicate rows being dropped: {duplicate_count}")

# Remove duplicates (keep first occurrence)
train = train.drop_duplicates(keep='first')



print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")


train.info()


train.head()


train.isnull().sum()/train.shape[0]


test.isnull().sum()/len(test)


train.nunique()


test.nunique()


plt.figure(figsize=(10,6))
sns.heatmap(train.isna().transpose(),
            cmap="magma",
            cbar_kws={'label': 'Missing Data'})


plt.figure(figsize=(10,6))
sns.heatmap(test.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})


train['Stage_fear'].value_counts()/train['Stage_fear'].value_counts().sum()


test['Stage_fear'].value_counts()/test['Stage_fear'].value_counts().sum()


#Replace the NaN value with mode("No")
train['Stage_fear']=train['Stage_fear'].fillna(train['Stage_fear'].mode()[0])
test['Stage_fear']=test['Stage_fear'].fillna(test['Stage_fear'].mode()[0])


train['Drained_after_socializing'].value_counts()/train['Drained_after_socializing'].value_counts().sum()


test['Drained_after_socializing'].value_counts()/test['Drained_after_socializing'].value_counts().sum()


#Replace the NaN value with mode("No")
train['Drained_after_socializing']=train['Drained_after_socializing'].fillna(train['Stage_fear'].mode()[0])
test['Drained_after_socializing']=test['Drained_after_socializing'].fillna(test['Stage_fear'].mode()[0])


train= pd.get_dummies(train, columns=['Stage_fear','Drained_after_socializing'])
test= pd.get_dummies(test, columns=['Stage_fear','Drained_after_socializing'])


#Replace NaN with mean value
boolean_cols= train.select_dtypes(include=['bool'])


for col in boolean_cols:
    train[col]=train[col].map({True:1,False:0})
    test[col]=test[col].map({True:1,False:0})


numerical_cols=['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']


for cols in numerical_cols:
    train[cols]=train[cols].fillna(train[cols].mean())
    test[cols]=test[cols].fillna(test[cols].mean())


scaler = MinMaxScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.fit_transform(test[numerical_cols])


le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])


train


train


test


X=train.drop(columns='Personality')
y=train['Personality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


plt.figure(figsize=(8, 6)) # Adjust figure size as needed
sns.heatmap(train.corr(), annot=True, cmap='magma', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'verbosity': 0,
    'eta': 0.01,
    'seed': 42
}

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 4,
    'num_leaves': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'verbosity': -1,
    'random_state': 42,
    'force_col_wise': True
}

cat_params = {
    'iterations': 1000,
    'depth': 4,
    'learning_rate': 0.01,
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'verbose': 0,
    'random_seed': 42,
    'early_stopping_rounds': 50
}


SPLITS = 10
REPEATS = 2
skf = RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=42)

# ----------------- Data Preparation ------------------

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

y_pred_xgb = np.zeros(len(X_test))
y_pred_lgb = np.zeros(len(X_test))
y_pred_cat = np.zeros(len(X_test))


# ----------------- Cross Validation ------------------

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    ### XGBoost ###
    dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
    dval_xgb = xgb.DMatrix(X_val, label=y_val)
    dtest_xgb = xgb.DMatrix(X_test)

    model_xgb = xgb.train(
        xgb_params,
        dtrain_xgb,
        num_boost_round=1000,
        evals=[(dval_xgb, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    pred_xgb_val = model_xgb.predict(dval_xgb)
    pred_xgb_test = model_xgb.predict(dtest_xgb)

    ### LightGBM ###
    model_lgb = LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
    )

    pred_lgb_val = model_lgb.predict_proba(X_val)[:, 1]
    pred_lgb_test = model_lgb.predict_proba(X_test)[:, 1]

    ### CatBoost ###
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    pred_cat_val = model_cat.predict_proba(X_val)[:, 1]
    pred_cat_test = model_cat.predict_proba(X_test)[:, 1]

    ### Aggregate ###
    oof_xgb[val_idx] += pred_xgb_val / REPEATS
    oof_lgb[val_idx] += pred_lgb_val / REPEATS
    oof_cat[val_idx] += pred_cat_val / REPEATS

    y_pred_xgb += pred_xgb_test / (REPEATS * SPLITS)
    y_pred_lgb += pred_lgb_test / (REPEATS * SPLITS)
    y_pred_cat += pred_cat_test / (REPEATS * SPLITS)

# ----------------- Final Predictions ------------------

oof_preds = (oof_xgb + oof_lgb + oof_cat) / 3
y_pred = (y_pred_xgb + y_pred_lgb + y_pred_cat) / 3



# ----------------- Evaluation ------------------

y_pred_class = (oof_preds > 0.5).astype(int)

print("Accuracy:", accuracy_score(y, y_pred_class))
print("CV Logloss:", log_loss(y, oof_preds))
print("F1 Score:", f1_score(y, y_pred_class))
print("\nClassification Report:")
print(classification_report(y, y_pred_class))

cm = confusion_matrix(y, y_pred_class)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model_xgb, max_num_features=7, ax=ax)
ax.set_title('Feature Importance')
plt.tight_layout()
plt.show()


# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y, oof_preds)  # Use predicted probabilities
roc_auc = roc_auc_score(y, oof_preds)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random guessing')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# Find best threshold Youdenâ€™s J statistic
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Youdenâ€™s J statistic Threshold:", optimal_threshold)


# Train final models on full data
final_xgb = xgb.train(
    xgb_params,
    xgb.DMatrix(X, label=y),
    num_boost_round=1000
)

final_lgb = LGBMClassifier(**lgb_params)
final_lgb.fit(X, y)

final_cat = CatBoostClassifier(**cat_params)
final_cat.fit(X, y)


# Define a function to make predictions on unseen data
def predict_unseen(X_new):
    pred_xgb = final_xgb.predict(xgb.DMatrix(X_new))
    pred_lgb = final_lgb.predict_proba(X_new)[:, 1]
    pred_cat = final_cat.predict_proba(X_new)[:, 1]
    final_pred = (pred_xgb + pred_lgb + pred_cat) / 3
    return final_pred


pred=predict_unseen(test)
print(pred)

for i in range(pred.shape[0]):
    if pred[i]<0.5:
        pred[i]=0
    else:
        pred[i]=1

print(pred)


label_map = {0: 'Extrovert', 1: 'Introvert'}
predicted_labels = [label_map[val] for val in pred]


predicted_labels


result_df = pd.DataFrame({
    'id': test_id,
    'Personality': predicted_labels
})

# Step 5: Save to CSV
result_df.to_csv('submission.csv', index=False)

print("âœ… CSV saved as 'personality_predictions.csv'")

