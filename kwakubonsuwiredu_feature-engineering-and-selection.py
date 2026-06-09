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


data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


target = 'Personality'

cat_col = [x for x in data.columns if data[x].dtype == 'object' and x not in ['id', target]]
num_col = [x for x in data.columns if data[x].dtype != 'object' and x not in ['id', target]]

X = data.drop(['id',target], axis=1)
y = data[[target]]


data.isna().sum()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

cat_enc = OrdinalEncoder()
le_enc = LabelEncoder()


X_train[cat_col] = cat_enc.fit_transform(X_train[cat_col])
X_test[cat_col] = cat_enc.transform(X_test[cat_col])

y_train[target] = le_enc.fit_transform(y_train[target])
y_test[target] = le_enc.transform(y_test[target])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 5)

X_train_imp = pd.DataFrame()
X_train_imp[X_train.columns.tolist()] = imputer.fit_transform(X_train)


X_test_imp = pd.DataFrame()
X_test_imp[X_test.columns.tolist()] = imputer.fit_transform(X_test)


X_train_imp['I_score'] = 0.4 * X_train_imp['Stage_fear'] + 0.3 * X_train_imp['Time_spent_Alone'] + 0.3 * X_train_imp['Drained_after_socializing']
X_test_imp['I_score'] = 0.4 * X_test_imp['Stage_fear'] + 0.3 * X_test_imp['Time_spent_Alone'] + 0.3 * X_test_imp['Drained_after_socializing']


# Introversion Score (original heuristic formula)
X_train_imp['I_score'] = 0.4 * X_train_imp['Stage_fear'] + 0.3 * X_train_imp['Time_spent_Alone'] + 0.3 * X_train_imp['Drained_after_socializing']
X_test_imp['I_score'] = 0.4 * X_test_imp['Stage_fear'] + 0.3 * X_test_imp['Time_spent_Alone'] + 0.3 * X_test_imp['Drained_after_socializing']

# Social Overload Index: net social burnout
X_train_imp['Social_Overload_Index'] = X_train_imp['Drained_after_socializing'] - X_train_imp['Social_event_attendance']
X_test_imp['Social_Overload_Index'] = X_test_imp['Drained_after_socializing'] - X_test_imp['Social_event_attendance']

# Digital vs. Offline Social Lean
X_train_imp['Digital_Social_Lean'] = X_train_imp['Post_frequency'] / (X_train_imp['Social_event_attendance'] + 1e-5)
X_test_imp['Digital_Social_Lean'] = X_test_imp['Post_frequency'] / (X_test_imp['Social_event_attendance'] + 1e-5)

# Isolation Preference Score
X_train_imp['Isolation_Preference'] = X_train_imp['Time_spent_Alone'] * X_train_imp['Drained_after_socializing']
X_test_imp['Isolation_Preference'] = X_test_imp['Time_spent_Alone'] * X_test_imp['Drained_after_socializing']

# Overall Real-world Engagement Index
X_train_imp['Engagement_Index'] = X_train_imp['Going_outside'] + X_train_imp['Social_event_attendance']
X_test_imp['Engagement_Index'] = X_test_imp['Going_outside'] + X_test_imp['Social_event_attendance']

# Friend Quality Estimate (friend network efficiency)
X_train_imp['Friend_Quality_Estimate'] = X_train_imp['Friends_circle_size'] / (X_train_imp['Social_event_attendance'] + 1e-5)
X_test_imp['Friend_Quality_Estimate'] = X_test_imp['Friends_circle_size'] / (X_test_imp['Social_event_attendance'] + 1e-5)



features = [
    "I_score",
    "Stage_fear",
    "Drained_after_socializing",
    "Isolation_Preference",
    "Engagement_Index",
    "Social_Overload_Index",
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Post_frequency",
    "Friends_circle_size",
    "Digital_Social_Lean",
    "Friend_Quality_Estimate"
]


votes = {x:0 for x in features}


X_train_imp = X_train_imp[features]
X_test_imp = X_test_imp[features]


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train_imp, y_train)


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model,random_state=1).fit(X_test_imp, y_test)
eli5.show_weights(perm, feature_names = X_train_imp.columns.to_list())


all_of = {x:y for x,y in zip(X_train_imp.columns.to_list(),perm.feature_importances_)}
sorted_dict = dict(sorted(all_of.items(), key=lambda item: item[1], reverse = True))

sorted_dict

votes[list(sorted_dict.keys())[0]] += 1


sorted_dict


from sklearn.feature_selection import mutual_info_classif

scores = mutual_info_classif(X_train_imp, y_train)

all_of = {x:y for x,y in zip(X_train_imp.columns.to_list(),scores)}

sorted_dict = dict(sorted(all_of.items(), key=lambda item: item[1], reverse = True))
sorted_dict

votes[list(sorted_dict.keys())[0]] += 1


from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

for feature in X_train_imp.columns:
    feature_name = feature
    PartialDependenceDisplay.from_estimator(model, X_test_imp, [feature_name])

plt.show()


import shap  # package used to calculate Shap values

data_for_prediction = X_test_imp.iloc[1,:]  

# Create object that can calculate shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)


shap_val = explainer.shap_values(X_test_imp)
shap.summary_plot(shap_val[1], X_test_imp)


from sklearn.metrics import accuracy_score

accuracy_score(y_test, model.predict(X_test_imp))


from sklearn.feature_selection import RFECV as rfe

selector = rfe(model, step = 1, cv = 5)
selector = selector.fit(X_train_imp, y_train)


rank = selector.ranking_


rank


for x,y in zip(X_train_imp.columns.tolist(), rank):
    print(x,': ',y)


all_of = {x:y for x,y in zip(X_train_imp.columns.tolist(), rank)}

sorted_dict = dict(sorted(all_of.items(), key=lambda item: item[1], reverse = False))
sorted_dict

votes[list(sorted_dict.keys())[0]] += 1


sorted_dict


from sklearn.feature_selection import SelectFromModel
selector = SelectFromModel(model, prefit=True)


mask = selector.get_support()           


selected_feats = X_train_imp.columns[mask]
print("Automatically selected features:", list(selected_feats))


X_reduced = selector.transform(X_train_imp)



for x in list(selected_feats):
    votes[x]+=1


features = ['I_score', 'Stage_fear', 'Drained_after_socializing', 'Isolation_Preference', 'Social_Overload_Index']


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Use Logistic Regression instead of Lasso
clf = LogisticRegression(penalty='l1', solver='liblinear', C=1/0.2)  # C is inverse of alpha

# Train the model
clf.fit(X_train_imp, y_train)

# Get feature names and coefficients
names = X_train_imp.columns.to_list()
coefs = clf.coef_[0]  # For binary classification, shape is (1, n_features)

# Make predictions
y_pred = clf.predict(X_test_imp)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))



import matplotlib.pyplot as plt
plt.figure(figsize=(30, 6))
plt.bar(names, coefs)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Lasso Feature Importance')
plt.show()


all_of = {x:y for x,y in zip(X_train_imp.columns.to_list(),coefs)}

sorted_dict = dict(sorted(all_of.items(), key=lambda item: item[1], reverse = True))
sorted_dict

votes[list(sorted_dict.keys())[0]] += 1
votes[list(sorted_dict.keys())[1]] += 1


votes

