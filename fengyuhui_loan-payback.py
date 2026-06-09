import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

# set index
train_df.set_index("id", inplace=True)
test_df.set_index('id', inplace=True)

# get shape
print(f"size of train is {train_df.shape}")
print(f"size of test is {test_df.shape}")

#sample_submit = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train_df.head()


train_df['loan_paid_back'].mean()


num_features = ['annual_income', 'debt_to_income_ratio','credit_score','loan_amount','interest_rate']
n_features = len(num_features)
fig, axes = plt.subplots(
    nrows = n_features,
    ncols = 2,
    figsize = (10, 3*n_features))

for i in range(n_features):
    feature = num_features[i]
    
    # 1. distribution
    sns.kdeplot(data=train_df, x=feature, fill=True, ax=axes[i][0])
    
    # 2. association between num_feature and target
    sns.kdeplot(data=train_df, x=feature, hue='loan_paid_back', ax=axes[i][1], fill=True, common_norm=False)


cat_features = ['gender', 'marital_status','education_level','employment_status','loan_purpose','grade_subgrade']
n_features = len(cat_features)
fig, axes = plt.subplots(
    nrows = n_features,
    ncols = 2,
    figsize = (10, 3*n_features))

for i in range(n_features):
    feature = cat_features[i]
    
    # 1. distribution
    hist = train_df.groupby(feature).size().reset_index()
    hist.columns=[feature, 'ct']
    sns.barplot(data=hist, y=feature, x='ct', ax=axes[i][0])
    
    # 2. association between num_feature and target
    target = train_df.groupby(feature).loan_paid_back.mean().reset_index()
    target.columns=[feature,'payback_percent']
    sns.barplot(data=target, y=feature, x='payback_percent', ax=axes[i][1])


# 1 - log
train_df['log_annual_income'] = np.log(train_df.annual_income)
test_df['log_annual_income'] = np.log(test_df.annual_income)

train_df['log_debt_to_income_ratio'] = np.log(train_df.debt_to_income_ratio)
test_df['log_debt_to_income_ratio'] = np.log(test_df.debt_to_income_ratio)

train_df['log_loan_amount'] = np.log(train_df.loan_amount)
test_df['log_loan_amount'] = np.log(test_df.loan_amount)

# 2 - grade
train_df['grade'] = train_df['grade_subgrade'].str[0]
test_df['grade'] = test_df['grade_subgrade'].str[0]

# 3 - One-Hot Encoding
OH_vars = {
    'gender':["Male","Female"],
    'marital_status':["Divorced","Married","Single"],
    'education_level':["Bachelor's","High School","Master's","PhD"],
    'employment_status':["Employed","Retired","Student","Unemployed"],
    'loan_purpose':["Business","Car","Debt consolidation","Education","Home","Medical","Vacation"],
    'grade':['A','B','C','D','E'],
    'grade_subgrade':['A1','A2','A3','A4','A5',
                      'B1','B2','B3','B4','B5',
                      'C1','C2','C3','C4','C5',
                      'D1','D2','D3','D4','D5',
                      'E1','E2','E3','E4','E5',
                      'F1','F2','F3','F4','F5']
}

for var, levels in OH_vars.items():
    for level in levels:
        varName = var + "_" + level
        train_df[varName] = 0
        train_df.loc[train_df[var]==level, varName] = 1

        test_df[varName] = 0
        test_df.loc[test_df[var]==level, varName] = 1
   


# Train Test Split
from sklearn.model_selection import train_test_split

X = train_df.copy()
y = X.pop('loan_paid_back')
test_X = test_df.copy()

train_X,valid_X, train_y,valid_y = train_test_split(X, y, random_state = 1)


# Calculate AUC
from sklearn.metrics import roc_auc_score

def cal_AUC(train_pred, valid_pred, train_y=train_y, valid_y=valid_y):
    train_auc = roc_auc_score(train_y, train_pred)
    valid_auc = roc_auc_score(valid_y, valid_pred)
    print(f"Train AUC {train_auc}")
    print(f"Valid AUC {valid_auc}")


from sklearn.linear_model import LogisticRegression

OH_features = []
for var, levels in OH_vars.items():
    for level in levels:
        OH_features.append(var + "_" + level)
        

lm_features = num_features + [
    'log_annual_income','log_debt_to_income_ratio','log_loan_amount'] + OH_features

# fit model
lm = LogisticRegression(random_state = 1)
lm.fit(train_X[lm_features], train_y)

# prediction
lm_train_pred = lm.predict_proba(train_X[lm_features])[:, 1]
lm_valid_pred = lm.predict_proba(valid_X[lm_features])[:, 1]

# AUC
cal_AUC(train_pred=lm_train_pred, 
        valid_pred=lm_valid_pred)


from sklearn.tree import DecisionTreeClassifier

# fit model
tree = DecisionTreeClassifier(
    random_state=1,
    min_samples_leaf=10,
    max_depth=10)

tree.fit(train_X[lm_features], train_y)

# prediction
tree_train_pred = tree.predict_proba(train_X[lm_features])[:, 1]
tree_valid_pred = tree.predict_proba(valid_X[lm_features])[:, 1]

# AUC
cal_AUC(train_pred=tree_train_pred, 
        valid_pred=tree_valid_pred)


from sklearn.ensemble import RandomForestClassifier

# fit model
rf = RandomForestClassifier(n_estimators =30,
                            max_depth = 20,
                            min_samples_split = 30,
                            min_samples_leaf =10,
                            random_state=1)

rf.fit(train_X[lm_features], train_y)

# prediction
rf_train_pred = rf.predict_proba(train_X[lm_features])[:, 1]
rf_valid_pred = rf.predict_proba(valid_X[lm_features])[:, 1]

# AUC
cal_AUC(train_pred=rf_train_pred, 
        valid_pred=rf_valid_pred)


from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators = 30,
                    max_depth = 20,
                    max_leaves = 20,
                    random_state = 1)

xgb.fit(train_X[lm_features], train_y)

# prediction
xgb_train_pred = xgb.predict_proba(train_X[lm_features])[:, 1]
xgb_valid_pred = xgb.predict_proba(valid_X[lm_features])[:, 1]

# AUC
cal_AUC(train_pred=xgb_train_pred, 
        valid_pred=xgb_valid_pred)


# submit
submission = pd.DataFrame({
    'id':test_X.index,
    'loan_paid_back':xgb.predict_proba(test_X[lm_features])[:, 1]
})
submission.to_csv('submission.csv', index=False)

