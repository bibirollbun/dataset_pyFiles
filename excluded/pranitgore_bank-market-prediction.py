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


filepath = '/kaggle/input/playground-series-s5e8/train.csv'
data = pd.read_csv(filepath, index_col = 'id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col = 'id')
data.head()


data.columns


data.isna().sum()


list(set(data.dtypes.to_list()))


data_num = data.select_dtypes(include = ['int64'])
print(data_num.describe())


import seaborn as sns
import matplotlib.pyplot as plt

#plt.figure(figsize = (16,10))
#sns.barplot(data=data, y='y', x='age')

age_category = [0,10,20,30,40,50,60,70,80,90,100]
data['age_category'] = pd.cut(data['age'],bins=age_category, right=False)
plt.figure(figsize=(12,6))
sns.barplot(data=data, x='age_category', y='y')
plt.show()



correlation_matrix = data_num.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#plt.title('Correlation Heatmap')
plt.show()
plt.savefig('Correlation for Bank Market')


data_p = data_num[['pdays', 'previous']]
plt.figure(figsize=(12,6))
sns.scatterplot(data=data_p)



filepath = '/kaggle/input/playground-series-s5e8/train.csv'
data = pd.read_csv(filepath, index_col='id')


marital_dummies = pd.get_dummies(data['marital'], prefix='marital', drop_first=False)

print(marital_dummies.head())
data = pd.concat([data, marital_dummies], axis=1)
data = data.drop('marital', axis=1)


job_dummies = pd.get_dummies(data['job'], prefix='job', drop_first=False)

print(job_dummies.head())
data = pd.concat([data, job_dummies], axis=1)
data = data.drop('job', axis=1)


education_dummies = pd.get_dummies(data['education'], prefix='education', drop_first=False)

print(education_dummies.head())
data = pd.concat([data, education_dummies], axis=1)
data = data.drop('education', axis=1)


contact_dummies = pd.get_dummies(data['contact'], prefix='contact', drop_first=False)

print(contact_dummies.head())
data = pd.concat([data, contact_dummies], axis=1)
data = data.drop('contact', axis=1)


month_dummies = pd.get_dummies(data['month'], prefix='month', drop_first=False)

print(month_dummies.head())
data = pd.concat([data, month_dummies], axis=1)
data = data.drop('month', axis=1)


poutcome_dummies = pd.get_dummies(data['poutcome'], prefix='poutcome', drop_first=False)

print(poutcome_dummies.head())
data = pd.concat([data, poutcome_dummies], axis=1)
data = data.drop('poutcome', axis=1)


data.head(10)


data['pdays'] = data['pdays'].replace(-1, 0)
data.head(10)


#data.dtypes
# default, housing, loan
data['default'] = data['default'].map({'yes': 1, 'no': 0})
data['housing'] = data['housing'].map({'yes': 1, 'no': 0})
data['loan'] = data['loan'].map({'yes': 1, 'no': 0})


import numpy as np

def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset [~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(data)
print("{} examples in training, {} examples in testing".format(len(train_ds_pd), len(valid_ds_pd)))


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
y = data.y
features = [
    # Numerical Columns
    'age',
    'balance',
    'day',
    'duration',
    'campaign',
    'pdays',
    'previous',
    'job_unemployed', 'job_services', 'job_management', 'job_blue-collar',
    'job_self-employed', 'job_technician', 'job_entrepreneur',
    'job_housemaid', 'job_retired', 'job_admin.', 'job_student',
    'marital_married', 'marital_single', 'marital_divorced',
    'education_tertiary', 'education_secondary', 'education_primary',
    'default',
    'housing',
    'loan',
    'contact_cellular', 'contact_telephone', 'contact_unknown',
    'month_jan', 'month_feb', 'month_mar', 'month_apr', 'month_may',
    'month_jun', 'month_jul', 'month_aug', 'month_sep', 'month_oct',
    'month_nov', 'month_dec',
    'poutcome_failure', 'poutcome_other', 'poutcome_success', 'poutcome_unknown'
]
X = data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


list(set(data.dtypes.to_list()))


# Specify Model
model = DecisionTreeClassifier(random_state=1)
# Fit Model
model.fit(train_X, train_y)


val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))


from sklearn.metrics import accuracy_score

# Make predictions
val_predictions = model.predict(val_X)

# Calculate Accuracy
val_accuracy = accuracy_score(val_predictions, val_y)
print("Validation Accuracy: {:.2f}".format(val_accuracy))


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


model = DecisionTreeClassifier(max_leaf_nodes=500, random_state=1)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_accuracy = accuracy_score(val_y, val_predictions)

print("Validation Accuracy for max_leaf_nodes=500: {:.2f}".format(val_accuracy))


model = DecisionTreeClassifier(max_leaf_nodes=50, random_state=1)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_accuracy = accuracy_score(val_y, val_predictions)

print("Validation Accuracy for max_leaf_nodes=50: {:.2f}".format(val_accuracy))


model = DecisionTreeClassifier(max_leaf_nodes=100, random_state=1)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_accuracy = accuracy_score(val_y, val_predictions)

print("Validation Accuracy for max_leaf_nodes=100: {:.2f}".format(val_accuracy))


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=1)

rf_model.fit(train_X, train_y)


rf_predictions = rf_model.predict(val_X)
rf_accuracy = accuracy_score(val_y, rf_predictions)
print("Validation Accuracy: {:.2f}".format(rf_accuracy))


rf_model = RandomForestClassifier(max_leaf_nodes=500, random_state=1)
rf_model.fit(train_X, train_y)
rf_predictions = rf_model.predict(val_X)
rf_accuracy = accuracy_score(val_y, rf_predictions)

print("Validation Accuracy for max_leaf_nodes=500: {:.2f}".format(rf_accuracy))


rf_model = RandomForestClassifier(max_leaf_nodes=100, random_state=1)
rf_model.fit(train_X, train_y)
rf_predictions = rf_model.predict(val_X)
rf_accuracy = accuracy_score(val_y, rf_predictions)

print("Validation Accuracy for max_leaf_nodes=100: {:.2f}".format(rf_accuracy))


rf_model = RandomForestClassifier(max_leaf_nodes=50, random_state=1)
rf_model.fit(train_X, train_y)
rf_predictions = rf_model.predict(val_X)
rf_accuracy = accuracy_score(val_y, rf_predictions)

print("Validation Accuracy for max_leaf_nodes=50: {:.2f}".format(rf_accuracy))


import xgboost as xgb
from sklearn.metrics import accuracy_score

xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=1, use_label_encoder=False, eval_metric='logloss')


xgb_model.fit(train_X, train_y)
xgb_predictions = xgb_model.predict(val_X)

# Calculate Accuracy
xgb_accuracy = accuracy_score(val_y, xgb_predictions)
print("XGBoost Validation Accuracy: {:.2f}".format(xgb_accuracy))


xgb_model = xgb.XGBClassifier(max_leaf_nodes=50, random_state=1)
xgb_model.fit(train_X, train_y)
xgb_predictions = xgb_model.predict(val_X)
xgb_accuracy = accuracy_score(val_y, xgb_predictions)

print("Validation Accuracy for max_leaf_nodes=50: {:.2f}".format(xgb_accuracy))


xgb_model = xgb.XGBClassifier(max_leaf_nodes=100, random_state=1)
xgb_model.fit(train_X, train_y)
xgb_predictions = xgb_model.predict(val_X)
xgb_accuracy = accuracy_score(val_y, xgb_predictions)

print("Validation Accuracy for max_leaf_nodes=100: {:.2f}".format(xgb_accuracy))


xgb_model = xgb.XGBClassifier(max_leaf_nodes=500, random_state=1)
xgb_model.fit(train_X, train_y)
xgb_predictions = xgb_model.predict(val_X)
xgb_accuracy = accuracy_score(val_y, xgb_predictions)

print("Validation Accuracy for max_leaf_nodes=500: {:.2f}".format(xgb_accuracy))


bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)



data.columns


num_cols = ['age', 'balance', 'campaign', 'day', 'duration', 'pdays', 'previous']
binary_cols = data.drop(columns=num_cols)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])


X = data.drop('y', axis=1)
y = data.y


lgbm_best_params ={'n_estimators': 1582,
                     'max_depth': 15,
                     'learning_rate': 0.04436352313699452,
                     'num_leaves': 77,
                     'min_child_samples': 81,
                     'subsample': 0.8677563315146003,
                     'colsample_bytree': 0.5261353954090011,
                     'reg_alpha': 0.0631139742323974,
                     'reg_lambda': 6.686183660331108}

catboost_best_params = {'iterations': 654,
                         'depth': 8,
                         'learning_rate': 0.1456532097331015,
                         'l2_leaf_reg': 2.926561373576166,
                         'bagging_temperature': 0.024573185250232735,
                         'border_count': 230,
                         'random_strength': 8.864365147004192e-05,
                         'scale_pos_weight': 1.7710717984741216,
                         'silent': True
                       }


xgb_best_params = {'n_estimators': 790,
                     'max_depth': 10,
                     'learning_rate': 0.031803109711165255,
                     'subsample': 0.813647689560639,
                     'colsample_bytree': 0.6074981565134057,
                     'gamma': 1.7120296093267298,
                     'reg_alpha': 0.7461915771713407,
                     'reg_lambda': 0.2737663612464292}


 """
 ensemble_model = VotingClassifier(estimators=[
     ('RF',RandomForestClassifier()),
     ('XGB',XGBClassifier(**xgb_best_params)),
     ('CB',CatBoostClassifier(**catboost_best_params)),
     ('LGBM',LGBMClassifier(**lgbm_best_params)),
 ])
 """


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier 

from catboost import CatBoostClassifier

## importing accuraty measures 
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,roc_auc_score
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold


from sklearn.ensemble import VotingClassifier
from sklearn.base import clone



print("Test Data: ",test_data.dtypes.unique())
print("Train Data: ", data.dtypes.unique())


data.columns


import pandas as pd


data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')


def preprocess_data(df):

    df['pdays'] = df['pdays'].replace(-1, 0)
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

# Apply the function to your training and test data
data = preprocess_data(data)
test_data = preprocess_data(test_data)


print("Test Data: ",test_data.dtypes.unique())
print("Train Data: ", data.dtypes.unique())


data.columns


y = data['y']
X = data.drop('y', axis=1)


import pandas as pd
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


from datetime import datetime
start_time = datetime.now()

num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
score_list = []
test_pred = np.zeros(test_data.shape[0])

## applying StratifiedKFold cross validation
for fold, (train_idx,test_idx) in enumerate(skf.split(X,y),1):

    fold_model = VotingClassifier(estimators=[
            ('RF',RandomForestClassifier()),
            ('XGB',XGBClassifier(**xgb_best_params)),
            ('CB',CatBoostClassifier(**catboost_best_params)),
            ('LGBM',LGBMClassifier(**lgbm_best_params)),
    ],voting='soft')
        
    X_train_fold,X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold,y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    fold_model.fit(X_train_fold,y_train_fold)
    y_test_fold_pred = fold_model.predict_proba(X_test_fold)[:,1]
        
    score = roc_auc_score(y_test_fold,y_test_fold_pred)
        
    print(f"Fold {fold}: Score : {score:.4f}")
    score_list.append(score)

    test_pred+=fold_model.predict_proba(test_data)[:,1] ## for class 1 
        
        
print(f"\n Average Score : {np.mean(score_list):.4f}\n")
   
# Average test predictions over all folds
test_pred /= num_folds

## saving prediction in submission file
sample_submission['y'] = test_pred
sample_submission.to_csv(f"ensemble_prediction.csv",index=False)
display(sample_submission.head())
print(f"File saved as ensemble_prediction.csv.....\n")

print()
end_time = datetime.now()
total_time = end_time - start_time 
print("Total Training Time:", total_time)




