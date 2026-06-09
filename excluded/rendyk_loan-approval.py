!pip install -q optbinning ppscore


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from optbinning import BinningProcess
import ppscore as pps
import pickle
import warnings
warnings.filterwarnings('ignore')


import optbinning
print(pd.__version__)
print(np.__version__)
print(optbinning.__version__)
print(pps.__version__)
!python --version


# Load the dataset
train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')

train = train.drop(columns=['id'])

print(train.shape)
train.head()


# Check duplicates
sum(train.duplicated())
# There is no data duplicate


# Check duplicated features
sum(train[train.columns[:-1]].duplicated())
# There is no feature duplicate


train.info()
# There is no missing value


train.describe()


# Visualize numerical features
fig, ax = plt.subplots(2, 4, figsize=(12,6))

row = 0
column = 0
for col in [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length', 'loan_status'
]:
    ax[row, column].hist(train[col], edgecolor='tab:blue', fill=False)
    ax[row, column].set_title(col)
    
    column += 1
    if column == 4:
        column = 0
        row+= 1

plt.tight_layout()
plt.show()
# Outliers may be found 'person_age', 'person_income', 'person_emp_length'


# Visualize numerical features
fig, ax = plt.subplots(1, 3, figsize=(8,3))

column = 0
for col in ['person_age', 'person_income', 'person_emp_length']:
    ax[column].boxplot(train[col])
    ax[column].set_title(col)
    
    column += 1

plt.tight_layout()
plt.show()
# Remove extreme outliers:
# person_age > 100
# person_emp_length > 60


train = train[((train['person_age'] < 100) & (train['person_emp_length'] < 60))]
train.shape


train['loan_status'].value_counts()/train.shape[0]*100
# Imbalanced dataset


# Explore categorical features
# Count the unique values
for categorical in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
    num_unique = len(train[categorical].unique())
    print(f'{categorical}: {num_unique}')


# Plot the count bar
fig, ax = plt.subplots(1, 4, figsize=(12,3))

column = 0
for categorical in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
    # Only select top 8 values, if there are more than 8 unique values
    bar_plot = pd.DataFrame(train[categorical].value_counts().sort_values(ascending=False)).reset_index().head(8)
    bar_plot.columns = ['category', 'count']
    bar_plot['percentage'] = bar_plot['count'] / train.shape[0] * 100

    ax[column].bar(x=bar_plot['category'], height=bar_plot['percentage'], fill=False, edgecolor='tab:blue')
    ax[column].set_title(categorical + ' (%)')
    ax[column].set_xticklabels(bar_plot['category'], rotation=45, ha='right')

    column += 1

plt.show()
# loan_grade should not be a predictive feature. A new user will not have a loan grade.


X_train, X_val, y_train, y_val = train_test_split(
    train.drop(columns=['loan_status'], axis=0),
    train['loan_status'],
    test_size=0.2, random_state=123,
    stratify=train['loan_status']
)
print(X_train.shape)
print(X_val.shape)


X_train.head()


# Save train test split id
pd.DataFrame(y_train).to_csv('train_id.csv')
pd.DataFrame(y_val).to_csv('val_id.csv')


X_train = train.drop(columns=['loan_status'])
X_test = test
y_train = train['loan_status']


# Optimal binning
var_num = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length'
]
var_cate = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Numerical
binning_fit_params = {v:{'monotonic_trend':'auto_asc_desc'} for v in var_num}

# Binning process
bin_proc_num = BinningProcess(
    var_num,
    max_n_bins=8,
    binning_fit_params=binning_fit_params
)

# Categorical
bin_proc_cate = BinningProcess(
    var_cate,
    max_n_bins=8,
)

# Apply opt binning to train dataset
train_bin = X_train.copy()
train_bin[var_num] = bin_proc_num.fit_transform(train_bin[var_num].values, y_train)
train_bin[var_cate] = bin_proc_cate.fit_transform(train_bin[var_cate].values, y_train)

print(train_bin.shape)
train_bin.head(3)


# Summary
bin_proc_num.summary().sort_values('iv', ascending=False)


bin_proc_cate.summary().sort_values('iv', ascending=False)


bin_proc_num.get_binned_variable('loan_percent_income').binning_table.build()


bin_proc_cate.get_binned_variable('person_home_ownership').binning_table.build()


# Compile opt binning tables
# Sort the variables based on IV
var_num_sort = list(bin_proc_num.summary().sort_values('iv', ascending=False)['name'].values)
var_obj_sort = list(bin_proc_cate.summary().sort_values('iv', ascending=False)['name'].values)

# Numerical
bin_table_num = pd.DataFrame(
    {'Feature':[], 'Bin':[], 'Count':[], 'Count (%)':[], 'Sum':[], 'Std':[], 'Mean':[],
     'Min':[], 'Max':[], 'Zeros count':[], 'WoE':[], 'IV':[]}
)
for c in var_num_sort:
    df_bin = bin_proc_num.get_binned_variable(c).binning_table.build()
    df_bin = df_bin[df_bin['Count']!=0]
    df_bin.insert(0, 'Feature', c)
    bin_table_num = pd.concat([bin_table_num, df_bin], axis=0, ignore_index=True)

# Object
bin_table_cate = pd.DataFrame(
    {'Feature':[], 'Bin':[], 'Count':[], 'Count (%)':[], 'Sum':[], 'Std':[], 'Mean':[],
     'Min':[], 'Max':[], 'Zeros count':[], 'WoE':[], 'IV':[]}
)
for c in var_obj_sort:
    df_bin = bin_proc_cate.get_binned_variable(c).binning_table.build()
    df_bin = df_bin[df_bin['Count']!=0]
    df_bin.insert(0, 'Feature', c)
    bin_table_cate = pd.concat([bin_table_cate, df_bin], axis=0, ignore_index=True)


# Save opt binning
bin_table_num.to_csv('bin_table_num.csv', index=False)
bin_table_cate.to_csv('bin_table_cate.csv', index=False)


# Save opt binning pickle
pickle.dump(bin_proc_num, open('bin_proc_num.pkl', 'wb'))
pickle.dump(bin_proc_cate, open('bin_proc_cate.pkl', 'wb'))


# Apply optimal binning
X_train['loan_intent'] = train_bin['loan_intent']

# One-hot encoding for some feaures
X_train['home_own'] = np.where(X_train['person_home_ownership'] == 'OWN', 1, 0)
X_train['home_mortgage'] = np.where(X_train['person_home_ownership'] == 'MORTGAGE', 1, 0)
X_train['cb_person_default_on_file'] = np.where(X_train['cb_person_default_on_file'] == 'Y', 1, 0)

X_train = X_train.drop(columns=['loan_grade', 'person_home_ownership'])
X_train.head()


# Apply to the validation set
val_bin = X_val.copy()
val_bin[var_cate] = bin_proc_cate.transform(test_bin[var_cate].values)

# Apply optimal binning
X_val['loan_intent'] = val_bin['loan_intent']

# One-hot encoding for some feaures
X_val['home_own'] = np.where(X_val['person_home_ownership'] == 'OWN', 1, 0)
X_val['home_mortgage'] = np.where(X_val['person_home_ownership'] == 'MORTGAGE', 1, 0)
X_val['cb_person_default_on_file'] = np.where(X_val['cb_person_default_on_file'] == 'Y', 1, 0)

X_val = X_val.drop(columns=['loan_grade', 'person_home_ownership'])
X_val.head()


# Apply to the testing set
test_bin = test.copy()
test_bin[var_cate] = bin_proc_cate.transform(test_bin[var_cate].values)

# Apply optimal binning
test['loan_intent'] = test_bin['loan_intent']

# One-hot encoding for some feaures
test['home_own'] = np.where(test['person_home_ownership'] == 'OWN', 1, 0)
test['home_mortgage'] = np.where(test['person_home_ownership'] == 'MORTGAGE', 1, 0)
test['cb_person_default_on_file'] = np.where(test['cb_person_default_on_file'] == 'Y', 1, 0)

X_test = test.copy()
X_test = X_test.drop(columns=['loan_grade', 'person_home_ownership'])
X_test.head()


# Save
X_train.to_csv('X_train.csv')
X_val.to_csv('X_val.csv')
X_test.to_csv('X_test.csv')


# PPS is the substitute to the correlation test that is usually userd to find features with strong relationship
# Apply predictive power
train_pps = pps.matrix(pd.concat([X_train, y_train], axis=1))
train_pps.head()


# Plot pps
plt.figure(figsize=(6, 6))
sns.heatmap(
    train_pps[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore'),
    annot=True, fmt=".1f", annot_kws={'size': 8}, cmap="Blues"
)
plt.show()
# The plot shows that there is no feature with strong relationship with other features.
# Hence, feature selection is not needed.


# Save
train_pps.to_csv("ppscore.csv", index=False)


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import pickle
from optbinning import scorecard


# Make scorer
auc_score = make_scorer(roc_auc_score)


# Hyperparameter tuning: Bayesian Optimization
def xgb_cl_bo(n_estimators, max_depth, learning_rate, gamma, min_child_weight, subsample, colsample_bytree):
    params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric':'auc',
        'nthread':-1
    }
    params_xgb['n_estimators'] = round(n_estimators)
    params_xgb['max_depth'] = round(max_depth)
    params_xgb['learning_rate'] = learning_rate
    params_xgb['gamma'] = gamma
    params_xgb['min_child_weight'] = round(min_child_weight)
    params_xgb['subsample'] = subsample
    params_xgb['colsample_bytree'] = colsample_bytree
    params_xgb['scale_pos_weight'] = 6
        
    score = cross_val_score(
        XGBClassifier(random_state=123, **params_xgb), X_train, y_train, scoring=auc_score, cv=5
    ).mean()
    return score

# Set parameters distribution
params_xgb ={
    'n_estimators':(80, 150),
    'max_depth': (3, 50),
    'learning_rate': (0.01, 0.5),
    'gamma':(0, 10),
    'min_child_weight':(3, 100),
    'subsample':(0.8, 1),
    'colsample_bytree':(0.1, 1)
}

# Run Bayesian Optimization
xgb_bo = BayesianOptimization(xgb_cl_bo, params_xgb, random_state=111)
xgb_bo.maximize(init_points=5, n_iter=25)


# Best hyperparameters
params_xgb = xgb_bo.max['params']
params_xgb['objective'] = 'binary:logistic'
params_xgb['n_jobs'] = -1
params_xgb['colsample_bytree'] = round(params_xgb['colsample_bytree'], 2)
params_xgb['gamma'] = round(params_xgb['gamma'], 2)
params_xgb['learning_rate'] = round(params_xgb['learning_rate'], 2)
params_xgb['max_depth'] = round(params_xgb['max_depth'])
params_xgb['min_child_weight'] = round(params_xgb['min_child_weight'])
params_xgb['n_estimators'] = round(params_xgb['n_estimators'])
params_xgb['subsample'] = round(params_xgb['subsample'], 2)

params_xgb


# Fit the training data
xgb_hyp =  XGBClassifier(**params_xgb, random_state=123, nthread=-1)
xgb_hyp.fit(X_train, y_train)

# Predict the test data probability
prob_train = xgb_hyp.predict_proba(X_train)[:, 1]
prob_val = xgb_hyp.predict_proba(X_val)[:, 1]
prob_test = xgb_hyp.predict_proba(X_test)[:, 1]


# compute auc scores
ns_prob = [0 for _ in range(len(y_train))]
ns_auc = roc_auc_score(y_train, ns_prob)
xgb_auc_train = roc_auc_score(y_train, prob_train)
xgb_auc_val = roc_auc_score(y_val, prob_val)

# plot the roc curve 
ns_fpr, ns_tpr, _ = roc_curve(y_train, ns_prob)
train_fpr, train_tpr, _ = roc_curve(y_train, prob_train)
val_fpr, val_tpr, _ = roc_curve(y_val, prob_val)

fig, ax = plt.subplots(1, 2, figsize=(6,3), sharey=True)

title_list = ['Training set', 'Validation set']
for col, fpr, tpr, auc_score in zip([0, 1], [train_fpr, val_fpr], [train_tpr, val_tpr], [xgb_auc_train, xgb_auc_val]):
    ax[col].plot(ns_fpr, ns_tpr, linestyle='--', label='AUC = 0.5')
    ax[col].plot(fpr, tpr, marker='.', label='AUC = %.3f' % (auc_score))
    ax[col].set_title(title_list[col])
    ax[col].set_xlabel('False Positive Rate')
    ax[col].set_ylabel('True Positive Rate')
    ax[col].legend()

plt.show()


# Find the score threshold based on KS Plot
scorecard.plot_ks(y_train, prob_train)


threshold = 0.18

pred_train = pd.DataFrame(prob_train, index=y_train.index)
pred_train.columns = ['probability']
pred_train['true'] = y_train
pred_train['prediction'] = np.where(pred_train['probability'] > threshold, 1, 0)

pred_val = pd.DataFrame(prob_val, index=y_val.index)
pred_val.columns = ['probability']
pred_val['true'] = y_val
pred_val['prediction'] = np.where(pred_val['probability'] > threshold, 1, 0)

pred_test = pd.DataFrame(prob_test)
pred_test.columns = ['probability']
pred_test['prediction'] = np.where(pred_test['probability'] > threshold, 1, 0)


# Model Evaluation
print('==== Training set')
print('Confusion Matrix')
print(pd.DataFrame(confusion_matrix(y_train, pred_train['prediction'])))
print(classification_report(y_train, pred_train['prediction']))


# Model Evaluation
print('==== Testing set')
print('Confusion Matrix')
print(pd.DataFrame(confusion_matrix(y_val, pred_val['prediction'])))
print(classification_report(y_val, pred_val['prediction']))


# Feature importances
FeatureXgb = pd.DataFrame({'feature':X_train.columns, 'importance':list(xgb_hyp.feature_importances_)}).sort_values('importance')
plt.figure(figsize=(12,3))
plt.bar(x=FeatureXgb['feature'], height=FeatureXgb['importance'])
plt.xticks(rotation=90)
plt.show()


# Save
pickle.dump(xgb_hyp, open('xgb_risk.pkl', 'wb'))

pred_train.index = y_train.index
pred_train.to_csv('pred_train.csv')

pred_val.index = y_val.index
pred_val.to_csv('pred_val.csv')

pred_test.index = test.index
pred_test.to_csv('pred_test.csv')


submission = pred_test.drop(columns=['prediction'])
submission = submission.reset_index()
submission.columns = ['id', 'loan_status']
submission.head()


submission.to_csv('submission.csv')

