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


import pandas as pd
import numpy as np


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", sep =",")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", sep =",")
sub_df = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv", sep =",")


test_df = pd.merge(test_df, sub_df[["id", "y"]], how = 'left', on = "id")
test_df


train_df


train_df.describe()


train_df.isnull().sum() + train_df.isin(["", np.nan, "NaN"]).sum()


train_df_good = train_df.loc[train_df["y"].isin([1, "1"])]
train_df_bad = train_df.loc[train_df["y"].isin([0, "0"])]


train_df_good.shape, train_df_bad.shape


train_df_good.describe()


train_df_bad.describe()


print(train_df_good["job"].value_counts()/train_df.shape[0])
print(train_df_bad["job"].value_counts()/train_df.shape[0])


print(train_df_good["marital"].value_counts()/train_df.shape[0])
print(train_df_bad["marital"].value_counts()/train_df.shape[0])


print(train_df_good["education"].value_counts()/train_df.shape[0])
print(train_df_bad["education"].value_counts()/train_df.shape[0])


print(train_df_good["default"].value_counts()/train_df.shape[0])
print(train_df_bad["default"].value_counts()/train_df.shape[0])


print(train_df_good["housing"].value_counts()/train_df.shape[0])
print(train_df_bad["housing"].value_counts()/train_df.shape[0])


print(train_df_good["loan"].value_counts()/train_df.shape[0])
print(train_df_bad["loan"].value_counts()/train_df.shape[0])


print(train_df_good["contact"].value_counts()/train_df.shape[0])
print(train_df_bad["contact"].value_counts()/train_df.shape[0])


print(train_df_good["day"].unique())
print(train_df_bad["day"].unique())

print(train_df_good["day"].value_counts())
print(train_df_bad["day"].value_counts())


print(train_df_good["month"].value_counts())
print(train_df_bad["month"].value_counts())


print(train_df_good["poutcome"].value_counts()/train_df.shape[0])
print(train_df_bad["poutcome"].value_counts()/train_df.shape[0])


Instance_count_df = train_df_good[["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", 
               "campaign", "pdays", "previous", "poutcome"]].value_counts().reset_index(name = "count")
Instance_count_df['count'] = Instance_count_df['count'].astype(int)
Instance_count_df.loc[Instance_count_df['count'] > 1].shape


Instance_count_df = train_df_bad[["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", 
               "campaign", "pdays", "previous", "poutcome"]].value_counts().reset_index(name = "count")
Instance_count_df['count'] = Instance_count_df['count'].astype(int)
Instance_count_df.loc[Instance_count_df['count'] > 1].shape


import matplotlib.pyplot as plt

# Count the occurrences of each class in the target variable 'y'
class_counts = train_df['y'].value_counts()

# Create a pie chart
plt.figure(figsize=(4, 4))
plt.pie(
    class_counts.values,
    labels=['Negative Outcome (0)', 'Positive Outcome (1)'],
    autopct='%1.1f%%',
    colors=['skyblue', 'salmon'],
    startangle=90
)

# Add a title
plt.title('Class Distribution of Target Variable "y"')

# Ensure the pie chart is a circle
plt.axis('equal')

# Save the plot as a PNG file
plt.savefig('class_distribution_pie.png')

print("The pie chart has been saved as 'class_distribution_pie.png'.")


train_df["default"] = np.where(train_df["default"] == "Yes", 1, 0)
train_df["housing"] = np.where(train_df["housing"] == "Yes", 1, 0)
train_df["loan"] = np.where(train_df["loan"] == "Yes", 1, 0)
train_df_all_num = pd.concat([pd.get_dummies(train_df['job'], prefix = 'job', prefix_sep = ': '),
                     pd.get_dummies(train_df['marital'], prefix = 'marital', prefix_sep = ': '),
                     pd.get_dummies(train_df['education'], prefix = 'education', prefix_sep = ': '),
                     pd.get_dummies(train_df['contact'], prefix = 'contact', prefix_sep = ': '),
                     pd.get_dummies(train_df['poutcome'], prefix = 'poutcome', prefix_sep = ': '), 
                     pd.get_dummies(train_df['month'], prefix = 'month', prefix_sep = ': ')], axis = 1)
train_df_all_num
train_df = pd.concat([train_df, train_df_all_num], axis = 1)
train_df


for col in list(train_df_all_num.columns):
    train_df[col] = np.where(train_df[col] == True, 1, 0)

train_num_df = train_df.drop(["job", "marital", "education", "contact", "month", "poutcome"], axis = 1)
train_num_df


train_num_df.describe()


train_num_df_good = train_num_df.loc[train_num_df["y"].isin([1,"1"])]
train_num_df_bad = train_num_df.loc[train_num_df["y"].isin([0,"0"])]


train_num_df_good.describe()


train_num_df_bad.describe()


IVs = pd.DataFrame(columns = ["Attributes", "Information Value"])


def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    
    dset = dset.sort_values(by='WoE')
    
    return dset, iv


train_num_df['duration_bins'] = pd.qcut(train_num_df['duration'], q=[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1], labels=[1, 2, 3, 4, 5, 6, 7, 8]) #8, 
train_num_df


duration_bin_edges = pd.qcut(train_num_df['duration'], q=[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1], retbins=True)[0]
duration_bin_ranges_list = [(interval.left, interval.right) for interval in duration_bin_edges.cat.categories]
print(duration_bin_ranges_list)


WOE, IV = calculate_woe_iv(train_num_df, "duration_bins", "y")
IVs.loc[len(IVs)] = ["duration_bins", IV]
WOE, IV


train_num_df["duration_bins:1,2,3,4,5,8"] = np.where(train_num_df["duration_bins"].isin([1,2,3,4,5,8]), 1, 0)
WOE, IV = calculate_woe_iv(train_num_df, "duration_bins:1,2,3,4,5,8", "y")
IVs.loc[len(IVs)] = ["duration_bins:1,2,3,4,5,8", IV]
WOE, IV


train_num_df['balance_bins'] = pd.qcut(train_num_df['balance'], q = [0, 0.2, 0.4, 0.6, 0.8, 1], labels=[1, 2, 3, 4, 5])
train_num_df


balance_bin_edges = pd.qcut(train_num_df['balance'], q=[0, 0.2, 0.4, 0.6, 0.8, 1], retbins=True)[0]
balance_bin_ranges_list = [(interval.left, interval.right) for interval in balance_bin_edges.cat.categories]
print(balance_bin_ranges_list)


WOE, IV = calculate_woe_iv(train_num_df, "balance_bins", "y")
IVs.loc[len(IVs)] = ["balance_bins", IV]
WOE, IV


age_q = [i / 22 for i in range(23)]
train_num_df['age_bins'] = pd.qcut(train_num_df['age'], q = age_q, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15, 16, 17, 18, 19, 20, 21, 22])
train_num_df


age_bin_edges = pd.qcut(train_num_df['age'], q=age_q, retbins=True)[0]
age_bin_ranges_list = [(interval.left, interval.right) for interval in age_bin_edges.cat.categories]
print(age_bin_ranges_list)


WOE, IV = calculate_woe_iv(train_num_df, "age_bins", "y")
IVs.loc[len(IVs)] = ["age_bins", IV]
WOE, IV


train_num_df.columns


train_num_df = pd.merge(train_num_df, train_df[["id", "job", "education"]], on = 'id', how = 'left')
train_num_df


WOE, IV = calculate_woe_iv(train_num_df, "job", "y")
IVs.loc[len(IVs)] = ["job", IV]
WOE, IV


WOE, IV = calculate_woe_iv(train_num_df, "pdays", "y")
IVs.loc[len(IVs)] = ["pdays", IV]
WOE, IV


train_num_df["job: housemaid, services, entrepreneur, blue-collar"] = np.where(train_num_df["job"].isin(["housemaid", "services", "entrepreneur", 
                                                                               "blue-collar"]), 1, 0) 


WOE, IV = calculate_woe_iv(train_num_df, "job: housemaid, services, entrepreneur, blue-collar", "y")
IVs.loc[len(IVs)] = ["job: housemaid, services, entrepreneur, blue-collar", IV]
WOE, IV


train_num_df["job: student,retired,unemployed,management,self-employed,unknown"] = np.where(train_num_df["job"].isin(["student","retired","unemployed", 
                                                                                                         "management", "self-employed", "unknown"]), 1, 0) 


WOE, IV = calculate_woe_iv(train_num_df, "job: student,retired,unemployed,management,self-employed,unknown", "y")
IVs.loc[len(IVs)] = ["job: student,retired,unemployed,management,self-employed,unknown", IV]
WOE, IV


WOE, IV = calculate_woe_iv(train_num_df, "education", "y")
IVs.loc[len(IVs)] = ["education", IV]
WOE, IV


train_num_df["education: secondary, primary"] = np.where(train_num_df["job"].isin(["secondary", "primary", "tertiary"]), 1, 0) 


WOE, IV = calculate_woe_iv(train_num_df, "education: secondary, primary", "y")
IVs.loc[len(IVs)] = ["education: secondary, primary", IV]
WOE, IV


train_df.columns


final_features = ["default", "housing", "loan", "campaign", "duration", "balance", 'age', 'pdays', "previous", 'marital: divorced', 
                  'marital: married', 'marital: single', 'education: primary', 'education: secondary', 'education: tertiary', 'education: unknown', 
                  'contact: cellular', 'contact: telephone', 'contact: unknown', 'poutcome: failure', 'poutcome: other',
                  'poutcome: success', 'poutcome: unknown', "job: housemaid, services, entrepreneur, blue-collar", 
                  "job: student,retired,unemployed,management,self-employed,unknown"]
#["duration_bins:1,2,3,4,5,8", "balance_bins", "age_bins", "pdays", "job: housemaid, services, entrepreneur, blue-collar", "job: student,retired,unemployed,management,self-employed,unknown"]
train_num_df = train_num_df[final_features + ["y"]]
IVs


!pip install scikit-learn imbalanced-learn


X = train_num_df.drop('y', axis=1)
y = train_num_df['y']

# from imblearn.over_sampling import BorderlineSMOTE
# from imblearn.combine import SMOTEENN
# from imblearn.combine import SMOTETomek
# from imblearn.over_sampling import SMOTENC


# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# train_df_SMOTE = pd.concat([X_resampled, y_resampled], axis = 1)
# train_df_SMOTE


# from imblearn.over_sampling import ADASYN

# # Applying ADASYN
# adasyn = ADASYN(sampling_strategy='minority')
# X_resampled, y_resampled = adasyn.fit_resample(X, y)

# train_df_adasyn = pd.concat([X_resampled, y_resampled], axis = 1)
# train_df_adasyn


from imblearn.over_sampling import BorderlineSMOTE

blsmote = BorderlineSMOTE(sampling_strategy='minority', kind='borderline-1')
X_resampled, y_resampled = blsmote.fit_resample(X, y)

train_df_BSMOTE = pd.concat([X_resampled, y_resampled], axis = 1)
train_df_BSMOTE


# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler()

# X_resampled, y_resampled = ros.fit_resample(X, y)

# train_df_RandO = pd.concat([X_resampled, y_resampled], axis = 1)
# train_df_RandO


# from imblearn.under_sampling import RandomUnderSampler
# ros = RandomUnderSampler()

# X_resampled, y_resampled = ros.fit_resample(X, y)

# train_df_RandU = pd.concat([X_resampled, y_resampled], axis = 1)
# train_df_RandU


test_df


test_df_all_num = pd.concat([pd.get_dummies(test_df['job'], prefix = 'job', prefix_sep = ': '),
                     pd.get_dummies(test_df['marital'], prefix = 'marital', prefix_sep = ': '),
                     pd.get_dummies(test_df['education'], prefix = 'education', prefix_sep = ': '),
                     pd.get_dummies(test_df['contact'], prefix = 'contact', prefix_sep = ': '),
                     pd.get_dummies(test_df['poutcome'], prefix = 'poutcome', prefix_sep = ': '), 
                     pd.get_dummies(test_df['month'], prefix = 'month', prefix_sep = ': ')], axis = 1)

test_df = pd.concat([test_df, test_df_all_num], axis = 1)
test_df


test_df["job: housemaid, services, entrepreneur, blue-collar"] = np.where(test_df["job"].isin(["housemaid", "services", "entrepreneur", 
                                                                               "blue-collar"]), 1, 0)
test_df["job: student,retired,unemployed,management,self-employed,unknown"] = np.where(test_df["job"].isin(["student","retired","unemployed", 
                                                                                                         "management", "self-employed", "unknown"]), 1, 0) 


test_df["default"] = np.where(test_df["default"] == "Yes", 1, 0)
test_df["housing"] = np.where(test_df["housing"] == "Yes", 1, 0)
test_df["loan"] = np.where(test_df["loan"] == "Yes", 1, 0)


bin_edges_for_cut = [duration_bin_ranges_list[0][0]] + [item[1] for item in duration_bin_ranges_list]
test_df['duration_bins'] = pd.cut(test_df['duration'], bins=bin_edges_for_cut, labels=[1, 2, 3, 4, 5, 6, 7, 8])
test_df["duration_bins:1,2,3,4,5,8"] = np.where(test_df["duration_bins"].isin([1,2,3,4,5,8]), 1, 0)
test_df


bin_edges_for_cut = [balance_bin_ranges_list[0][0]] + [item[1] for item in balance_bin_ranges_list]
test_df['balance_bins'] = pd.cut(test_df['balance'], bins=bin_edges_for_cut, labels=[1, 2, 3, 4, 5])
test_df


bin_edges_for_cut = [age_bin_ranges_list[0][0]] + [item[1] for item in age_bin_ranges_list]
test_df['age_bins'] = pd.cut(test_df['age'], bins=bin_edges_for_cut, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15, 16, 17, 18, 19, 20, 21, 22])

test_df


test_num_df = test_df[final_features + ["y"]]
test_num_df





from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

def metrics_calculation(y_true, y_pred):
    output = {}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # 
    output["True Negative"] = int(str(tn))
    output["False Positive"] = int(str(fp))
    output["False Negative"] = int(str(fn))
    output["True Positive"] = int(str(tp))
    
    train_df_results = pd.DataFrame({'Y_true': y_true, 'Y_pred': y_pred})

    # titanic_crosstab = pd.crosstab(train_df_results.Y_pred, train_df_results.Y_train)

    # print("-"*50)

    acc = accuracy_score(train_df_results.Y_true, train_df_results.Y_pred)
    prec = precision_score(train_df_results.Y_true, train_df_results.Y_pred)
    recall = recall_score(train_df_results.Y_true, train_df_results.Y_pred)

    output["Accuracy"] = acc
    output["Precision"] = prec
    output["Recall"] = recall

    print(classification_report(y_true, y_pred))
    print("-"*80)

    return output


X = train_df_BSMOTE.drop('y', axis=1)
y = train_df_BSMOTE['y']


X_test = test_num_df.drop('y', axis = 1)
y_test = test_num_df['y']


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# X, y = make_classification(n_features=4, random_state=0)
clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))
clf.fit(X, y)


y_train_hat = clf.predict(X)
metrics_calculation(y, y_train_hat)


y_test_hat_lsvc = clf.predict(X_test)


import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X, y)


y_train_hat = clf.predict(X)
metrics_calculation(y, y_train_hat)


y_test_hat_sgd = clf.predict(X_test)


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X, y)


y_train_hat = clf.predict(X)
metrics_calculation(y, y_train_hat)


y_test_hat_lr = clf.predict(X_test)


test_df["y"] = y_test_hat_lsvc
final_result = test_df[["id", "y"]]
final_result.to_csv("submission.csv", index=False)










