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


#  H2O Initialization and Data Conversion
import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size="12G")

# data set for H2O model
train_h2o_df = train_feature.copy()
train_h2o_df['y'] = target 

train_h2o = h2o.H2OFrame(train_h2o_df)
train_h2o['y'] = train_h2o['y'].asfactor() # classification
test_h2o = h2o.H2OFrame(test_feature)                        


# H2O AutoML training
features = [col for col in train_h2o.columns if col != 'y']

aml = H2OAutoML(
    max_runtime_secs=3600,  # 3600 (1hr default)
    nfolds=5,
    seed=42,
    sort_metric="AUC"   
)
aml.train(x=features, y='y', training_frame=train_h2o)

# H2OStackedEnsembleEstimator
# LogLoss: 0.13096497708388388
# AUC: 0.9755565947820889


# results - learderboard

lb = aml.leaderboard #trained model list
print("H2O AutoML Leaderboard :")
print(lb.head(rows=lb.nrows)) 

# the best model with the highest AUC
leader_model = aml.leader
print("Best Model :")
print(leader_model)

# model summary, model metrics, 


# Leaderboard
lb
# training time(min)    AUC    the best model
# 10	             0.9656	   StackedEnsemble
# 20	             0.9681	   StackedEnsemble
# 30	             0.9695	   StackedEnsemble
# 40                 0.9697    StackedEnsemble
# 50                 0.9700    StackedEnsemble
# 60(default)	     0.9700	   StackedEnsemble


test_feature.head()


test_h2o = h2o.H2OFrame(test_feature)
test_h2o


test_prob = leader_model.predict(test_h2o)
test_prob


test_prob_df = test_prob.as_data_frame()
test_prob_df


test_pred = (test_prob_df['p1'] >= 0.5).astype(int)
test_pred[:10]


submission = pd.DataFrame({
    'id': test['id'],  
    'y': test_pred
})

submission.to_csv('submission.csv', index=False)

submission.head()


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()

