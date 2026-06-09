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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train_df.head()

test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_df.head()


train_df.head()





train_df.shape


train_df.info()


external_df = pd.read_csv('/kaggle/input/bank-full/bank-full.csv',sep=';')
external_df.shape





external_df.info()


external_df.head()


# now lets concat both the dataset into one after encoding the target column in external_df


external_df['y'] = external_df['y'].map({'no':0,'yes':1})


 


updated_train_df = pd.concat([train_df,external_df],axis=0)
updated_train_df.head()


# removing the id from the dataset

updated_train_df.drop('id',axis=1,inplace=True)


updated_train_df.head()


updated_train_df.shape


updated_train_df.info()


# imputing the nan values in the balance


# checking the outliers in num columns

import matplotlib.pyplot as plt
num_cols = updated_train_df.select_dtypes(include='int').columns

number_cols = 12
item_per_row = 4 

num_rows_required = int(np.ceil(number_cols/item_per_row))
plt.figure(figsize=(20,18))
for i,col in enumerate (num_cols):
    plt.subplot(num_rows_required,item_per_row,i+1)
    plt.boxplot(updated_train_df[col],vert=False)
    plt.title(col)
plt.tight_layout()
plt.show()






updated_train_df.info()


# before applying the log1p transformation on the cols like previous, balance, campaign and duration
# we need to make sure that there should not be any -1 value and balance has 3623 values which is -1
# -1 means data is unknown, we can treat that as missing value and impute the median value in that.

#finding median excluding -1

  





test_df.isna().sum()


test_df['balance'].value_counts()


median = np.median(updated_train_df[updated_train_df['balance']>=0]['balance'])
updated_train_df['balance'] = updated_train_df['balance'].apply(lambda x:median if x<0 else x)
test_df['balance'] =  test_df['balance'].apply(lambda x:median if x<0 else x)




test_df.isna().sum()


updated_train_df.info()


updated_train_df['balance'].isna().sum()


test_df.isna().sum()


# we will be doing log1p transformation

cols_to_transformed = ['previous','balance','campaign','duration'] 

from sklearn.preprocessing import FunctionTransformer 

log_transformer = FunctionTransformer(np.log1p)

updated_train_df[cols_to_transformed] = log_transformer.transform(updated_train_df[cols_to_transformed])
test_df[cols_to_transformed] = log_transformer.transform(test_df[cols_to_transformed])






test_df.isna().sum()


updated_train_df.head()


updated_train_df['balance'].value_counts()


test_df.head()


# now lets create category from pdays column if -1 then previously it is not contacted else
# contacted...

updated_train_df['is_contacted'] = updated_train_df['pdays'].apply(lambda x:0 if x==-1 else 1)

test_df['is_contacted'] = test_df['pdays'].apply(lambda x:0 if x==-1 else 1)


updated_train_df.drop('pdays',axis=1,inplace=True)


test_df.drop('pdays',axis=1,inplace=True)


updated_train_df.head()


test_df.head()


# now we will be encoding the cat cols one by one..

for col in updated_train_df.columns:
    print(updated_train_df[col].value_counts())
    print()


updated_train_df.head()


#default,housing and loan has 2 values which is yes and no so we can use label encoder in that.Apart from this we can do label encoding on the month as well.

binary_cols  = ['default','housing','loan']

for col in binary_cols:
    updated_train_df[col] = updated_train_df[col].map({'no':0,'yes':1})
    test_df[col] = test_df[col].map({'no':0,'yes':1})
 


updated_train_df.head()


test_df.head()


updated_train_df['month'].value_counts()


# now we will do label encoding with month

month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

updated_train_df['month'] = updated_train_df['month'].map(month_map)
test_df['month'] = test_df['month'].map(month_map)



updated_train_df.head()


test_df.head()


# now we should do one hot encoder for job, marital,education,contact and poutcome right?
one_hot_cols = ['job','marital','education','contact','poutcome']
from sklearn.preprocessing import OneHotEncoder

encoder= OneHotEncoder(sparse_output=False,drop='first')
arr = encoder.fit_transform(updated_train_df[one_hot_cols])
arr_test = encoder.transform(test_df[one_hot_cols])
enc_df = pd.DataFrame(arr,columns=encoder.get_feature_names_out(one_hot_cols),index=updated_train_df.index)
enc_test_df = pd.DataFrame(arr_test,columns=encoder.get_feature_names_out(one_hot_cols),index=test_df.index)



updated_train_df.head()


enc_df.head()


 updated_train_df.drop(columns = ['job','marital','education','contact','poutcome'],inplace=True,axis=1)



test_df.drop(columns = ['job','marital','education','contact','poutcome'],inplace=True,axis=1)


updated_train_df.head()

 


pd.set_option('display.max_columns',None)
train_df_new = pd.concat([updated_train_df,enc_df],axis=1)
train_df_new.head()


test_df_new = pd.concat([test_df,enc_test_df],axis=1)
test_df_new.head()


train_df_new.head() 
target = train_df_new['y']


y= target
X = train_df_new.drop('y',axis=1)


X.head()


# we have to use standard scalar in order to scale....

from sklearn.preprocessing import StandardScaler
scaling_cols = ['balance','duration','campaign','previous','age']
scaler = StandardScaler()

X[scaling_cols] = scaler.fit_transform(X[scaling_cols])
test_df_new[scaling_cols] = scaler.transform(test_df_new[scaling_cols])
 


test_df_new.head()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold  
from sklearn.metrics import roc_auc_score  
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np

scaling_cols = ['balance','duration','campaign','previous','age']

# Setup cross-validation
skf = RepeatedStratifiedKFold(n_repeats=2, n_splits=10, random_state=42) 

# Dictionary to save ROC AUC scores for all models
roc_scores = {
    'LightGBM': [],
    'XGBoost': [],
    'LogisticRegression': [],
    'KNN': [],
}

for fold, (train_ids, valid_ids) in enumerate(skf.split(X, y), 1):

    # Split the data into training and validation sets for this fold
    X_train, X_valid = X.iloc[train_ids], X.iloc[valid_ids]
    y_train, y_valid = y.iloc[train_ids], y.iloc[valid_ids]

    # Scale only numeric columns based on training data
    scaler = StandardScaler().fit(X_train[scaling_cols])
    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()
    X_train_scaled[scaling_cols] = scaler.transform(X_train[scaling_cols])
    X_valid_scaled[scaling_cols] = scaler.transform(X_valid[scaling_cols])

    # --- Train LightGBM (no scaling needed) ---
    model_lgbm = LGBMClassifier(
        random_state=42, verbosity=-1, n_estimators=10000, learning_rate=0.03,
        min_child_samples=18, subsample=0.8, colsample_bytree=0.5,
        num_leaves=100, max_depth=10, reg_alpha=0.79, reg_lambda=3,
    )
    model_lgbm.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(100)]
    )
    pred_lgbm = model_lgbm.predict_proba(X_valid)[:, 1]
    roc_scores['LightGBM'].append(roc_auc_score(y_valid, pred_lgbm))

    # --- Train Logistic Regression (scaled data) ---
    model_lr = LogisticRegression(max_iter=2000, random_state=42)
    model_lr.fit(X_train_scaled, y_train)
    pred_lr = model_lr.predict_proba(X_valid_scaled)[:, 1]
    roc_scores['LogisticRegression'].append(roc_auc_score(y_valid, pred_lr))

    # --- Train XGBoost (no scaling needed) ---
    model_xgb = XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="logloss",
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.7, verbosity=0
    )
    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict_proba(X_valid)[:, 1]
    roc_scores['XGBoost'].append(roc_auc_score(y_valid, pred_xgb))

    # --- Train KNN (scaled data) ---
    model_knn = KNeighborsClassifier(n_neighbors=9, weights='distance')
    model_knn.fit(X_train_scaled, y_train)
    pred_knn = model_knn.predict_proba(X_valid_scaled)[:, 1]
    roc_scores['KNN'].append(roc_auc_score(y_valid, pred_knn))

# After CV, print mean ROC AUC for each model
print({model: float(np.mean(scores)) for model, scores in roc_scores.items()})

# Fit scaler on full training data for final model training and test preprocessing
scaler_full = StandardScaler().fit(X[scaling_cols])
X_scaled_full = X.copy()
X_scaled_full[scaling_cols] = scaler_full.transform(X[scaling_cols])
test_df_scaled = test_df_new.copy()
test_df_scaled[scaling_cols] = scaler_full.transform(test_df_new[scaling_cols])

# Example VotingClassifier definition and fit on full data with the same params (replace best_params_* accordingly)
voting_clf = VotingClassifier(
    estimators=[
        ('lgbm', LGBMClassifier(
            random_state=42, verbosity=-1, n_estimators=10000, learning_rate=0.03,
            min_child_samples=18, subsample=0.8, colsample_bytree=0.5,
            num_leaves=100, max_depth=10, reg_alpha=0.79, reg_lambda=3,
        )),
        ('lr', LogisticRegression(max_iter=2000, random_state=42)),
        ('xgboost', XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.7, verbosity=0
        )),
        ('knn', KNeighborsClassifier(n_neighbors=9, weights='distance'))
    ],
    voting='soft'
)

# Fit the voting classifier on the whole scaled training data
voting_clf.fit(X_scaled_full, y)

# Predict on scaled test data




test_ids = test_df_new['id']
test_ids


test_df_scaled.drop('id',axis=1,inplace=True)





pred_prob = voting_clf.predict_proba(test_df_scaled)[:, 1]








pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


submission = pd.DataFrame({'id':test_ids,'y':pred_prob})
submission.head()


submission.to_csv('submission.csv',index=False)


# Mutual information code: 

# from sklearn.feature_selection import mutual_info_regression

# def make_mi_scores(X, y, discrete_features):
#     mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
#     mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
#     mi_scores = mi_scores.sort_values(ascending=False)
#     return mi_scores

# mi_scores = make_mi_scores(X, y, discrete_features)
# mi_scores[::3]  # show a few features with their MI scores


#visualization 

# def plot_mi_scores(scores):
#     scores = scores.sort_values(ascending=True)
#     width = np.arange(len(scores))
#     ticks = list(scores.index)
#     plt.barh(width, scores)
#     plt.yticks(width, ticks)
#     plt.title("Mutual Information Scores")


# plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_scores(mi_scores)




