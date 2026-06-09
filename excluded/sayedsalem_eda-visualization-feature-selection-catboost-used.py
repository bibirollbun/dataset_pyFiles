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


!pip install feature-engine catboost


import numpy as np
import pandas as pd

train_path = '/kaggle/input/prudential-life-insurance-assessment/train.csv.zip'
train_df = pd.read_csv(train_path)
print(train_df.shape)
train_df.head()


test_path = '/kaggle/input/prudential-life-insurance-assessment/test.csv.zip'
test_df = pd.read_csv(test_path)
print(test_df.shape)
test_df.head()


sample_sub_path = '/kaggle/input/prudential-life-insurance-assessment/sample_submission.csv.zip'

sample_sub = pd.read_csv(sample_sub_path)
sample_sub.head()


catgeorical_cols = [
    "Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5",
    "Product_Info_6", "Product_Info_7", "Employment_Info_2", "Employment_Info_3",
    "Employment_Info_5", "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3",
    "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",
    "Insurance_History_1", "Insurance_History_2", "Insurance_History_3",
    "Insurance_History_4", "Insurance_History_7", "Insurance_History_8",
    "Insurance_History_9", "Family_Hist_1", "Medical_History_2",
    "Medical_History_3", "Medical_History_4", "Medical_History_5",
    "Medical_History_6", "Medical_History_7", "Medical_History_8",
    "Medical_History_9", "Medical_History_11", "Medical_History_12",
    "Medical_History_13", "Medical_History_14", "Medical_History_16",
    "Medical_History_17", "Medical_History_18", "Medical_History_19",
    "Medical_History_20", "Medical_History_21", "Medical_History_22",
    "Medical_History_23", "Medical_History_25", "Medical_History_26",
    "Medical_History_27", "Medical_History_28", "Medical_History_29",
    "Medical_History_30", "Medical_History_31", "Medical_History_33",
    "Medical_History_34", "Medical_History_35", "Medical_History_36",
    "Medical_History_37", "Medical_History_38", "Medical_History_39",
    "Medical_History_40", "Medical_History_41"
]

numerical_cols = [
    "Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1",
    "Employment_Info_4", "Employment_Info_6", "Insurance_History_5",
    "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5",
    "Medical_History_1", "Medical_History_10", "Medical_History_15",
    "Medical_History_24", "Medical_History_32"
]


# The rest columns in training
print([
    (col,str(train_df[col].dtype))
    for col in train_df.columns
    if col not in numerical_cols and col not in catgeorical_cols
])


# The rest columns in testing

print([
    (col,str(test_df[col].dtype))
    for col in test_df.columns
    if col not in numerical_cols and col not in catgeorical_cols
])


for col in catgeorical_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

for col in numerical_cols :
    train_df[col] = train_df[col].astype('float64')
    test_df[col] = test_df[col].astype('float64')


train_column_info = pd.DataFrame({
    'Column': train_df.columns,
    'Null Count': (train_df.isna().sum()/train_df.shape[0] * 100).round(2),
    'Dtype': train_df.dtypes
}).set_index('Column')

print(train_column_info)


print(train_column_info[train_column_info['Null Count']>0])


test_column_info = pd.DataFrame({
    'Column': test_df.columns,
    'Null Count': (test_df.isna().sum()/test_df.shape[0] * 100).round(2),
    'Dtype': test_df.dtypes
}).set_index('Column')

print(test_column_info[test_column_info['Null Count']>0])


train_nan_cols = set(train_df.columns[train_df.isna().any()])
test_nan_cols = set(test_df.columns[test_df.isna().any()])

cols_has_nan_in_train_only = train_nan_cols - test_nan_cols
cols_nas_nan_in_test_only = test_nan_cols - train_nan_cols

print("Columns that has nan values in train dataseto only is : ", list(cols_has_nan_in_train_only))
print("Columns that has nan values in test dataseto only is : ", list(cols_nas_nan_in_test_only))


train_df = train_df.drop('Id',axis=1)

sub_id = test_df['Id']
test_df = test_df.drop('Id',axis=1)

sub_id.head()


threshold = 50

high_nan_cols =  train_column_info[
        (train_column_info['Null Count']>threshold) |
        (test_column_info['Null Count']>threshold)
    ].index

high_nan_cols


train_df = train_df.drop(high_nan_cols, axis=1)
test_df = test_df.drop(high_nan_cols, axis=1)


rest_nan_cols = train_column_info[
        (train_column_info['Null Count']>0) &
        (train_column_info['Null Count']<threshold)
    ].index

for col in rest_nan_cols:
    print(train_df[col].value_counts())


for col in rest_nan_cols:
    mode_value = train_df[col].mode()[0]

    train_df[col] = train_df[col].fillna(mode_value)
    test_df[col] = test_df[col].fillna(mode_value)


new_category_cols = train_df.select_dtypes(include=['category']).columns
new_category_cols


import matplotlib.pyplot as plt

# Visualize the categorical columns
for col in new_category_cols:
    counts = train_df[col].value_counts().sort_index()
    fig = plt.figure(figsize=(30, 10))
    if len(counts) <=3:
        fig = plt.figure(figsize=(15, 5))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(f"Bar Plot: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
plt.show()


from feature_engine.encoding import RareLabelEncoder

minority_threshold = 0.05

rare_encoder = RareLabelEncoder(
    tol=minority_threshold,
    replace_with='Others',
    n_categories=2
)

train_df[new_category_cols] = rare_encoder.fit_transform(train_df[new_category_cols])
test_df[new_category_cols] = rare_encoder.transform(test_df[new_category_cols])

for col in new_category_cols:
    train_df[col] = train_df[col].cat.remove_unused_categories()
    test_df[col] = test_df[col].cat.remove_unused_categories()


import seaborn as sns

for col in new_category_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='Response', data=train_df)
    plt.title(f'Favorite Fruits by {col}')
    plt.legend(title='Response')
    plt.tight_layout()
    plt.show()


from scipy.stats import chi2_contingency

chi2_results = {}

for col in new_category_cols:

    contingency_table = pd.crosstab(train_df[col], train_df['Response'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    chi2_results[col] = {
        'Chi2': chi2,
        'p-value': p,
        'Degrees of Freedom': dof
    }

chi2_results_df = pd.DataFrame(chi2_results).T
chi2_results_df


alpha = 0.05

features_to_drop = chi2_results_df[chi2_results_df['p-value'] >= alpha].index
features_to_keep = chi2_results_df[chi2_results_df['p-value'] < alpha].index

print("Features to drop:", features_to_drop)
print("Features to keep:", features_to_keep)


new_numerical_cols = test_df.select_dtypes(exclude=['category']).columns.values
new_numerical_cols


n_cols = len(new_numerical_cols)
n_cols_per_row = 4
n_rows = int(np.ceil(n_cols / n_cols_per_row))

fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=(n_cols_per_row * 4, n_rows * 4))
fig.suptitle("Histograms of Numerical Features", fontsize=16)

axes = axes.flatten()

for i, col in enumerate(new_numerical_cols):
    sns.histplot(data=train_df, x=col, ax=axes[i], bins=30, color='skyblue', kde=True)
    axes[i].set_title(f"{col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


from scipy.stats import f_oneway

anova_results={}
for col in new_numerical_cols:
    groups = [train_df[col][train_df['Response'] == response] for response in train_df['Response'].unique()]
    f_value, p_value = f_oneway(*groups)

    anova_results[col] = {
        'F-value': f_value,
        'p-value': p_value
    }

anova_results_df = pd.DataFrame(anova_results).T
anova_results_df


alpha = 0.05

features_to_drop = anova_results_df[anova_results_df['p-value'] >= alpha].index
features_to_keep = anova_results_df[anova_results_df['p-value'] < alpha].index

print("Features to drop:", features_to_drop)
print("Features to keep:", features_to_keep)


train_df = train_df.drop(features_to_drop,axis=1)
test_df = test_df.drop(features_to_drop,axis=1)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

train_df['Medical_History_1'] = scaler.fit_transform(train_df[['Medical_History_1']])
test_df['Medical_History_1'] = scaler.transform(test_df[['Medical_History_1']])


new_category_cols


sum_categories = 0
for col in new_category_cols:
    sum_categories += train_df[col].nunique()-1 # -1 because I made drop='first' in OneHotEncoder
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)


from sklearn.preprocessing import OneHotEncoder

hot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

training_encoded_data = hot_encoder.fit_transform(train_df[new_category_cols])
testing_encoded_data =hot_encoder.transform(test_df[new_category_cols])

encoded_features = hot_encoder.get_feature_names_out(new_category_cols)

# Ensure that the onehotencoder is working well
assert len(encoded_features) == sum_categories

training_encoded_df = pd.DataFrame(training_encoded_data,columns=encoded_features)
testing_encoded_df = pd.DataFrame(testing_encoded_data,columns=encoded_features)

train_df = pd.concat([train_df.drop(new_category_cols,axis=1) ,training_encoded_df ],axis=1)
test_df = pd.concat([test_df.drop(new_category_cols,axis=1) ,testing_encoded_df ],axis=1)

print(train_df.shape)
print(test_df.shape)


X = train_df.drop('Response',axis=1)
y = train_df['Response']

y-=1


# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(n_estimators=100),
#     'KNN': KNeighborsClassifier(n_neighbors=3),
#     'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
#     'LightGBM': LGBMClassifier(),
#     'CatBoost': CatBoostClassifier(verbose=0)  # verbose=0 to suppress output
# }

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# results = {}

# for name, model in models.items():
#     scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

#     results[name] = {
#         'mean_accuracy': np.mean(scores),
#         'std_accuracy': np.std(scores)
#     }


# print("Cross-Validation Results (Accuracy):")
# for name, result in results.items():
#     print(f"{name}:")
#     print(f"  Mean Accuracy = {result['mean_accuracy']:.4f}")
#     print(f"  Std Dev = {result['std_accuracy']:.4f}")
#     print()


# from sklearn.model_selection import RandomizedSearchCV

# param_dist = {
#     'iterations': [50, 100, 200, 300, 500],
#     'learning_rate': np.linspace(0.01, 0.3, 10),
#     'depth': [4, 6, 8, 10],
#     'l2_leaf_reg': [1, 3, 5, 7, 9]
# }

# catboost_model = CatBoostClassifier(task_type='GPU')
# random_search = RandomizedSearchCV(catboost_model, param_dist, n_iter=20, cv=cv, scoring='accuracy')


# random_search.fit(X, y)


# print("Best Parameters:", random_search.best_params_)
# print("Best CV Accuracy (on full X):", random_search.best_score_)


best_param = {
  'learning_rate': 0.20333333333333334,
  'l2_leaf_reg': 3,
  'iterations': 500,
  'depth': 4,
  }

catboost_model = CatBoostClassifier(**best_param)


catboost_model.fit(X,y)


y_pred = catboost_model.predict(test_df) + 1

y_pred


final_sub = pd.DataFrame({
    'Id': sub_id,
    'Response': y_pred.ravel()
})
final_sub



final_sub.to_csv('submission.csv',index=False)




