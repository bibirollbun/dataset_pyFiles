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


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
import xgboost as xgb 
import lightgbm as lgb
import catboost as cb
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv') 
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')



train.head()


train.info()


train.describe()


train["Personality"].value_counts()


sns.countplot(x="Personality",data=train)


plt.figure(figsize=(15,8))
sns.histplot(x="Friends_circle_size",data=train,kde=True)


plt.figure(figsize=(15,8))
plt.hist(train["Friends_circle_size"], bins=30, edgecolor='black', alpha=0.7)


train['Friends_circle_size'].value_counts()


plt.figure(figsize=(15,8))
sns.countplot(x="Friends_circle_size" ,data=train)
plt.show()


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

for i in num_cols:
    plt.figure(figsize=(15,8))
    plt.title(f"Hist plot of {i}")
    sns.histplot(x=i,data=train,kde=True)
    train[i].value_counts()
    plt.title(f"Count plot of {i}")
    sns.countplot(x=i ,data=train)
    plt.show()


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']



train[num_cols] = train[num_cols].fillna(train[num_cols].mean()) 
test[num_cols] = test[num_cols].fillna(test[num_cols].mean()) 
train[cat_cols] = train[cat_cols].fillna(train[cat_cols].mode().iloc[0])
test[cat_cols] = test[cat_cols].fillna(test[cat_cols].mode().iloc[0])


def check(x):
    if x=="Yes":
        return 1
    else: 
        return 0


train['Stage_fear']=train['Stage_fear'].apply(check)


test['Stage_fear']=test['Stage_fear'].apply(check)
train['Drained_after_socializing']=train['Drained_after_socializing'].apply(check)
test['Drained_after_socializing']=test['Drained_after_socializing'].apply(check)


train.sample(5)


test.sample(5)


target_le = LabelEncoder()
train['Personality'] = target_le.fit_transform(train['Personality'])



X=train.drop(columns=['Personality']) 
y = train['Personality'] 
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)



# Model Evaluation Function
def evaluate_model(model, model_name):
    print(f"\nTraining {model_name}")
    fold_accuracies = []
    test_preds = np.zeros((test.shape[0], len(np.unique(y))))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
        fold_accuracies.append(acc)

        test_preds += model.predict_proba(test) / skf.n_splits

    mean_acc = np.mean(fold_accuracies)
    print(f"Average Accuracy for {model_name}: {mean_acc:.4f}")
    return mean_acc, np.argmax(test_preds, axis=1)
    


# Models to try
models = {
    "XGBoost": xgb.XGBClassifier(),
    "LightGBM": lgb.LGBMClassifier(),
    "CatBoost": cb.CatBoostClassifier(verbose=0),
    "RandomForest": RandomForestClassifier()
}


# Run all models
results = {}
final_submissions = {}

for name, model in models.items():
    acc, preds = evaluate_model(model, name)
    results[name] = acc
    final_submissions[name] = preds

# Best model selection
best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model} with Accuracy: {results[best_model]:.4f}")



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =======================
# 8. BASIC XGBOOST (CPU)
# =======================
model= xgb.XGBClassifier(
    n_estimators=1000,  
    max_depth=5,                   
    subsample=0.8,
    learning_rate=0.01,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# =======================
# 9. VALIDATION METRICS
# =======================
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_acc:.4f}\n")



fold_accuracies = []
test_preds = np.zeros((test.shape[0], len(np.unique(y))))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
     

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
        fold_accuracies.append(acc)

        test_preds += model.predict_proba(test) / skf.n_splits

fold_accuracies


test.head()


# ================
# 10. PREDICTION
# ================
test_preds_numeric = model.predict(test)  # model = your trained model
test_preds = target_le.inverse_transform(test_preds_numeric)  # target_le = your fitted LabelEncoder

# =======================
# 11. SUBMISSION
# =======================
submission = pd.DataFrame({
    'id': test['id'],                   # make sure the column is 'id' not 'id_col'
    'Personality': test_preds           # original column name from dataset
})

submission.to_csv('submission.csv', index=False)
submission.head()


submission 




