# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip uninstall -y imbalanced-learn scikit-learn
!pip install scikit-learn==1.2.2
!pip install imbalanced-learn==0.10.1


import sklearn
import imblearn

print("sklearn:", sklearn.__version__)
print("imblearn:", imblearn.__version__)


train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
train_identity.shape, train_transaction.shape, test_identity.shape, test_transaction.shape


train_identity.head()


train_transaction.head()


train_identity.columns


train_identity.info()


train_transaction.info()


train_transaction.columns


#join identity and transaction data
train = train_transaction.merge(train_identity, on = 'TransactionID', how = 'left')
train.head()


#join identity and transaction data for test data now
test = test_transaction.merge(test_identity, on = 'TransactionID', how = 'left')
test.head()


print(train_transaction.shape)
print(train_identity.shape)
print(train.shape)


print(test_transaction.shape)
print(test_identity.shape)
print(test.shape)


print(train.shape)
print(test.shape)


train_cols  = set(train.columns)
test_cols = set(test.columns)

#replace - with _
test_cols_rename_required = test_cols - train_cols
# test= test.rename(columns={''})

test.columns = test.columns.str.replace("-", "_", regex=False)


train = pd.concat([train, test], axis=0)
print(train.shape)


#Understand features of both identity and transaction data.
#Handle missing values separately for numerical and categorical features

#For numerical features if more than 60% of the values are missing, drop it.
#For rest, should we apply mean and replace the missing values, or do we perform group by on some column and then take mean ?

#For categorical features, same thing if more than 60% values are missing, drop it.
#For rest, should we apply mode and replace the missing values, or do we perform group by on some column and then take mode ?

#Now for categorical features, label encoding or one hot encoding?

#For numerical features, identify the important features i.e features with |corelationScore| > 0.1
#Then in the filtered features, remove one of the redundant feature where |corelaitonScore| > 0.9
#For filtered numerical features, check if it is skewed.
#Then apply normalization using log1p/box-cox on those features which requires normalization
#if target feature is skewed, then here also normalization can be applied.

#Now perform scaling fit_transform on all nuemrical features(except target feature)

#split the train and test data
#Model building
#Model Prediction
#y_predicted should be reversed as per normalization performed on target data.
#ROC curve to be applied on y_test and y_predicted


#lets check if there is some features which is specifically missing or present for fraudelent
#just to check if there is any feature whose presence or absence signals fraudelent data
for feature in train.columns:
    if train[feature].isnull().sum() > 0:
        dummy = train.groupby(['isFraud'])[feature].apply(lambda x:x.isnull().sum())
        if(dummy[0] ==0 | dummy[1] == 0):
            print(feature)
            print(dummy)
            print('-'*50)


cat = []
num = []
for feature in train.columns:
    if train[feature].dtype == 'object':
        cat.append(feature)
    else:
        num.append(feature)
print(cat)
print(num)


#handling missing values of numerical features
num_features_to_be_dropped = []
for feature in num:
    if train[feature].isnull().mean() > 0.6:
        num_features_to_be_dropped.append(feature)
    elif feature!='isFraud':
        train[feature].fillna(train[feature].mean(), inplace = True)


num_features_to_be_dropped


#droping the num features needed to be removed
train.drop(columns = num_features_to_be_dropped, inplace = True)
train.shape


#for categorical features also drop features with values more than 60% missing
#for rest replace it with mode, cause forward fill can be applied only on continuos trend data, but since its a random transaction data, we choose mode here.
cat_features_to_be_dropped = []
for feature in cat:
    if train[feature].isnull().mean() > 0.6:
        cat_features_to_be_dropped.append(feature)
    else:
        train[feature].fillna(train[feature].mode()[0], inplace = True)
cat_features_to_be_dropped


#dropping cat_features_to_be_dropped
train.drop(columns = cat_features_to_be_dropped, inplace =True)
train.shape


train.isnull().sum()[train.isnull().sum() > 0]


cat_left = list(set(cat) - set(cat_features_to_be_dropped))
for feature in cat_left:
    print(train[feature].value_counts())
    print('-'*50)


#label encoding needs to be done
from sklearn.preprocessing import LabelEncoder
for col in cat_left:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))

for feature in cat_left:
    print(train[feature].value_counts())
    print('-'*50)


#for numerical features
num_left = list(set(num)- set(num_features_to_be_dropped))
len(num_left)


#we check which features are important with respect to target feature using corelaton
corr_matrix = train.corr()
corr_matrix


imp_features = corr_matrix['isFraud'][abs(corr_matrix['isFraud']) > 0.1].index
imp_features


corr_matrix['V15']['isFraud']


#check which features in this imp_features are redundant to each other, so we remove one of them
redundant_features = []
for i in range(len(imp_features)):
    for j in range(len(imp_features)):
        if j > i:
            if abs(corr_matrix[imp_features[i]][imp_features[j]]) > 0.9:
                redundant_features.append(imp_features[i])
                break
redundant_features


imp_features_filtered = list(set(imp_features)- set(redundant_features))
imp_features_filtered


#now imp_features_filtered is checked if it has any skewed data inorder to perform normalization
#it is to be checked only on numerical features
num_left_final = list(set(num_left).intersection(imp_features_filtered))
num_left_final


num_left_final = list(set(num_left_final) - set(['isFraud']))
for feature in num_left_final:
    if train[feature].skew() > 1:
        train[feature] = np.log1p(train[feature])
train[num_left_final].skew()[train[num_left_final].skew() > 1]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[num_left_final] = scaler.fit_transform(train[num_left_final])


#final features
cat_left_final = list(set(cat_left).intersection(imp_features_filtered))
cat_left_final


final_features = num_left_final + cat_left_final + ['isFraud']
final_features


final_df = train[final_features]
train_final_df = final_df[0:590540]
test_final_df = final_df[590540:]
print(train_final_df.shape)
print(test_final_df.shape)



Y = train_final_df['isFraud']
X = train_final_df.drop(columns = ['isFraud'])
X.shape, Y.shape


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 43, stratify=Y)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


#SMOTE to handle imbalance data - only on the train data, not on the test data
#it uses interpolation to create artifical records for the minority class i.e does oversampling.
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train)


y_train_sm.value_counts()


#Model building
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_sm, y_train_sm)


y_train_pred  = lr.predict(x_train_sm)
y_train_prob = lr.predict_proba(x_train_sm)[:, 1]   # for AUC
y_test_pred  = lr.predict(x_test)
y_test_prob = lr.predict_proba(x_test)[:, 1]   # for AUC


from sklearn.metrics import roc_auc_score
print("TRAIN ROC-AUC:", roc_auc_score(y_train_sm, y_train_prob))
print("TEST ROC-AUC:", roc_auc_score(y_test, y_test_prob))


model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs'
)

model.fit(x_train_sm, y_train_sm)

y_train_prob = model.predict_proba(x_train_sm)[:, 1]   # for AUC
y_test_prob = model.predict_proba(x_test)[:, 1]   # for AUC

from sklearn.metrics import roc_auc_score
print("TRAIN ROC-AUC:", roc_auc_score(y_train_sm, y_train_prob))
print("TEST ROC-AUC:", roc_auc_score(y_test, y_test_prob))


from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

model_dt.fit(x_train, y_train)

y_train_prob = model_dt.predict_proba(x_train)[:, 1]   # for AUC
y_test_prob = model_dt.predict_proba(x_test)[:, 1]   # for AUC

from sklearn.metrics import roc_auc_score
print("TRAIN ROC-AUC:", roc_auc_score(y_train, y_train_prob))
print("TEST ROC-AUC:", roc_auc_score(y_test, y_test_prob))


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

model_rf.fit(x_train_sm, y_train_sm)

y_train_prob = model_rf.predict_proba(x_train_sm)[:, 1]   # for AUC
y_test_prob = model_rf.predict_proba(x_test)[:, 1]   # for AUC

from sklearn.metrics import roc_auc_score
print("TRAIN ROC-AUC:", roc_auc_score(y_train_sm, y_train_prob))
print("TEST ROC-AUC:", roc_auc_score(y_test, y_test_prob))


from sklearn.ensemble import GradientBoostingClassifier

model_gb = GradientBoostingClassifier(random_state=42)

model_gb.fit(x_train_sm, y_train_sm)

y_train_prob = model_gb.predict_proba(x_train_sm)[:, 1]   # for AUC
y_test_prob = model_gb.predict_proba(x_test)[:, 1]   # for AUC

from sklearn.metrics import roc_auc_score
print("TRAIN ROC-AUC:", roc_auc_score(y_train_sm, y_train_prob))
print("TEST ROC-AUC:", roc_auc_score(y_test, y_test_prob))


from xgboost import XGBClassifier

model_xgb = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)


model_xgb.fit(x_train_sm, y_train_sm)

y_train_prob = model_xgb.predict_proba(x_train_sm)[:, 1]   # for AUC
y_test_prob = model_xgb.predict_proba(x_test)[:, 1]   # for AUC

from sklearn.metrics import roc_auc_score
print("TRAIN ROC-AUC:", roc_auc_score(y_train_sm, y_train_prob))
print("TEST ROC-AUC:", roc_auc_score(y_test, y_test_prob))


import lightgbm as lgb

model_lgb = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42
)


model_lgb.fit(x_train_sm, y_train_sm)

y_train_prob = model_lgb.predict_proba(x_train_sm)[:, 1]   # for AUC
y_test_prob = model_lgb.predict_proba(x_test)[:, 1]   # for AUC

from sklearn.metrics import roc_auc_score
print("TRAIN ROC-AUC:", roc_auc_score(y_train_sm, y_train_prob))
print("TEST ROC-AUC:", roc_auc_score(y_test, y_test_prob))


from catboost import CatBoostClassifier

model_cb = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=0
)


model_cb.fit(x_train_sm, y_train_sm)

y_train_prob = model_cb.predict_proba(x_train_sm)[:, 1]   # for AUC
y_test_prob = model_cb.predict_proba(x_test)[:, 1]   # for AUC

from sklearn.metrics import roc_auc_score
print("TRAIN ROC-AUC:", roc_auc_score(y_train_sm, y_train_prob))
print("TEST ROC-AUC:", roc_auc_score(y_test, y_test_prob))


#Performing SMOTE on the entire X,Y data
sm1 = SMOTE(random_state=42)
X_sm, Y_sm = sm1.fit_resample(X, Y)


from catboost import CatBoostClassifier

model_cb = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=0
)

model_cb.fit(X_sm, Y_sm)

Y_pred_sm = model_cb.predict_proba(X_sm)[:, 1]   # for AUC
# y_test_prob = model_cb.predict_proba(x_test)[:, 1]   # for AUC

from sklearn.metrics import roc_auc_score
print("TRAIN ROC-AUC:", roc_auc_score(Y_sm, Y_pred_sm))
# print("TEST ROC-AUC:", roc_auc_score(y_test, y_test_prob))


test_final_df2 = test_final_df.drop(columns = ['isFraud'])
test_final_df2


X_sm


test_final_df2


#now test the model with test_final_df
Y_test_pred_sm = model_cb.predict_proba(test_final_df2)[:, 1]   # for AUC
Y_test_pred_sm
# from sklearn.metrics import roc_auc_score
# # print("TEST ROC-AUC:", roc_auc_score(Y_sm, Y_test_pred_sm))


submission = pd.DataFrame({'TransactionID': test['TransactionID'],'isFraud': Y_test_pred_sm})


submission.to_csv("submission.csv", index=False)




