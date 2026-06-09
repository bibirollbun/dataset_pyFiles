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
pd.set_option("display.max_colwidth",999)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import shap
from sklearn.impute import KNNImputer

#Modelling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df.info()
display(train_df.head(10))
display(train_df.describe())


test_df.info()


train_df.columns


train_df.day.value_counts(sort=False)
# Here it can be ascertained that we have roughly 6 years of data to work with
# but to be sure lets go with a bar plot
days_count = train_df.day.value_counts(sort=False)
plt.figure(figsize=(14, 7))
plt.bar(days_count.values,days_count.index)
plt.xlabel('Count')
plt.ylabel('Number of Days')
plt.title('Number of Records per Day')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



train_df.hist(figsize=(20,20))
plt.show()


# lets explore pressure as it is a vital component
train_df['pressure'].value_counts().sort_index()


train_df['rainfall'].value_counts().plot(kind='bar')


# rainfall trends over the year
plt.figure(figsize=(10, 5))
train_df.groupby('day')['rainfall'].mean().plot()


# monthly rainfall trends
plt.figure(figsize=(10, 5))
train_df1 = train_df.copy()
train_df1['month'] = train_df1['day']//30 +1
train_df1.groupby('month')['rainfall'].mean().plot()


# sns.regplot(data=train_df, x='dewpoint', y='rainfall')
# sns.scatterplot(data=train_df, x='humidity', y='rainfall')


plt.figure(figsize=(32, 6))
sns.boxplot(x='cloud', y='temparature', data=train_df)
# cloud cover is inversely proportional to temperature
# for lower cloud cover the temperature variance is greater
# but from correlation below we do find that cloud cover is a central feature for rainfall


# min_temp vs max_temp
display(train_df.mintemp.value_counts().sort_index())
display(train_df.maxtemp.value_counts().sort_index())


# relation between 'humidity', 'dewpoint', 'temparature', 'rainfall'
sns.pairplot(train_df[['humidity', 'dewpoint', 'temparature', 'rainfall']], hue='rainfall')


train_df.corr()


plt.figure(figsize=(15, 10))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')


#for training df
train_fedf = train_df.copy() 
train_fedf['temp_range'] = train_fedf['maxtemp'] - train_fedf['mintemp']
# train_fedf['humidity_dew_diff'] = train_fedf['humidity'] - train_fedf['dewpoint']
# train_fedf['storm_score'] = train_fedf['cloud'] + train_fedf['humidity']
train_fedf['cloud_humidity_product'] = train_fedf['cloud'] * train_fedf['humidity'] # this already captures storm_score and its an interaction feature 
epsilon = 0.1 # avoid 0 div
train_fedf['humidity_temp_ratio'] = train_fedf['humidity'] / (train_fedf['temparature'] + epsilon)
train_fedf['Pressure_Humidity_Interaction'] = train_fedf['pressure'] * train_fedf['humidity']
train_fedf["cloud_wind_interaction"] = train_fedf["cloud"] * train_fedf["windspeed"]
# train_fedf['relative_dryness'] = 100 - train_fedf['humidity']
train_fedf['Cloud_Humidity_ratio'] = train_fedf['cloud'] / (train_fedf['humidity'] + 1e-5)
train_fedf['dewpoint_temp_diff'] = train_fedf['temparature'] - train_fedf['dewpoint']
train_df['dewpoint_humidity_product'] = train_df['dewpoint'] * train_df['humidity']



train_fedf.hist(figsize=(20,20))
plt.show()


drop_cols = ['maxtemp', 'mintemp','pressure','id','day','humidity']
train_fedf.drop(columns=drop_cols, inplace=True)


train_fedf.corr()


train_fedf.head(10)


#split
x = train_fedf.drop(['rainfall'], axis = 1)
y = train_df['rainfall']
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#scaling - Only for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)


print("Logistic Regression Report:\n")
print(classification_report(y_test, y_pred_log))



# Decision tree
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(x_train, y_train)
y_pred_tree = tree_clf.predict(x_test)

print("Decision Tree Report:\n")
print(classification_report(y_test, y_pred_tree))

# id and day doesnt affect the overall generalization and 
# day is a surface level proxy for time since cyclicity can be extracted from it 


scores = cross_val_score(tree_clf, x, y, cv=5)
print("CrossValidation Accuracy: ", scores.mean())


explainer = shap.TreeExplainer(tree_clf)
shap_values = explainer.shap_values(x_train)
#global importance
shap.summary_plot(shap_values[1], x_train, plot_type="bar")


import matplotlib.pyplot as plt
import seaborn as sns

feature_importances = tree_clf.feature_importances_
features = x.columns

sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance")
plt.show()


feat_importance = pd.DataFrame({
    'feature': x.columns,
    'importance': tree_clf.feature_importances_
})
feat_importance = feat_importance.sort_values(by='importance')

print("Least contributing features:")
print(feat_importance.head())


# Random Forest Clasifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

rfmodel = RandomForestClassifier(random_state=42)
rfmodel.fit(x_train, y_train)
y_pred = rfmodel.predict(x_test)

print("Random Forest Report:\n")
print(classification_report(y_test, y_pred))


scores = cross_val_score(rfmodel, x, y, cv=5)
print("CrossValidation Accuracy: ", scores.mean())


explainer = shap.TreeExplainer(rfmodel)
shap_values = explainer.shap_values(x_train)
#global importance
shap.summary_plot(shap_values[1], x_train, plot_type="bar")


import matplotlib.pyplot as plt
import seaborn as sns

feature_importances = rfmodel.feature_importances_
features = x.columns

sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance")
plt.show()



# xgboost
from xgboost import XGBClassifier

xgbmodel = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgbmodel.fit(x_train, y_train)
y_pred = xgbmodel.predict(x_test)
print("XGBoost:")
print(classification_report(y_test, y_pred ))


explainer = shap.Explainer(xgbmodel)
shap_values = explainer(x_train)
shap.summary_plot(shap_values, x_train)


# ensemble rfc + xgboost (soft and hard voting)
from sklearn.ensemble import VotingClassifier

ensemble_soft = VotingClassifier(estimators=[
    ('rf', rfmodel),
    ('xgb', xgbmodel)
], voting='soft')

ensemble_soft.fit(x_train, y_train)
print("Ensemble_Soft:")
print(classification_report(y_test, ensemble_soft.predict(x_test)))

ensemble_hard = VotingClassifier(estimators=[
    ('rf', rfmodel),
    ('xgb', xgbmodel)
], voting='hard')

ensemble_hard.fit(x_train, y_train)
print("Ensemble_hard:")
print(classification_report(y_test, ensemble_hard.predict(x_test)))


# lets check it up with cross validation
soft_scores = cross_val_score(ensemble_soft, x, y, cv=5)
print("Ensemble_soft CrossValidation Accuracy: ", soft_scores.mean())
hard_scores = cross_val_score(ensemble_hard, x, y, cv=5)
print("Ensemble_hard CrossValidation Accuracy: ", hard_scores.mean())

print(f"Soft Voting:  {soft_scores.mean():.4f} ± {soft_scores.std():.4f}")
print(f"Hard Voting:  {hard_scores.mean():.4f} ± {hard_scores.std():.4f}")



# now with Hyper-parameter tuning
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 7]
}

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.001,0.05,0.01,0.1,0.3,0.5]
}

rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

rf_grid.fit(x_train, y_train)
xgb_grid.fit(x_train, y_train)

print("Best RF:", rf_grid.best_params_)
print("Best XGB:", xgb_grid.best_params_)

rf_best = rf_grid.best_estimator_
xgb_best = xgb_grid.best_estimator_



ensemble_fin = VotingClassifier(
    estimators=[('rf', rf_best), 
                ('xgb', xgb_best)],
    voting='soft'
)
ensemble_fin.fit(x_train, y_train)



models = {
    'Random Forest': rf_best,
    'XGBoost': xgb_best,
    'Ensemble': ensemble_fin
}

for name, model in models.items():
    y_pred = model.predict(x_test)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))



# cross_validation
models = {
    'Random Forest': rf_best,
    'XGBoost': xgb_best,
    'Ensemble': ensemble_fin
}
for name, model in models.items():
    cv_scores = cross_val_score(model, x, y, cv=5)
    print(f"{name} CV Accuracy:", cv_scores.mean())


imputer = KNNImputer(n_neighbors=5)
test_df["winddirection"] = imputer.fit_transform(test_df[["winddirection"]])


display(test_df.head())
display(test_df.info())


test_df['Pressure_Humidity_Interaction'] = test_df['pressure'] * test_df['humidity']
test_df['cloud_wind_interaction'] = test_df['cloud'] * test_df['windspeed']
# test_df['relative_dryness'] = 100 - test_df['humidity']
test_df['Cloud_Humidity_ratio'] = test_df['cloud'] / (test_df['humidity'] + 1e-5)
test_df['dewpoint_temp_diff'] = test_df['temparature'] - test_df['dewpoint']
test_df['cloud_humidity_product'] = test_df['cloud'] * test_df['humidity']
epsilon = 0.1 # avoid 0 div
test_df['humidity_temp_ratio'] = test_df['humidity'] / (test_df['temparature'] + epsilon)
test_df['dewpoint_humidity_product'] = test_df['dewpoint'] * test_df['humidity']
test_df['temp_range'] = test_df['maxtemp'] - test_df['mintemp']


# train_fedf['temp_range'] = train_fedf['maxtemp'] - train_fedf['mintemp']
# # train_fedf['humidity_dew_diff'] = train_fedf['humidity'] - train_fedf['dewpoint']
# # train_fedf['storm_score'] = train_fedf['cloud'] + train_fedf['humidity']
# train_fedf['cloud_humidity_product'] = train_fedf['cloud'] * train_fedf['humidity'] # this already captures storm_score and its an interaction feature 
# epsilon = 0.1 # avoid 0 div
# train_fedf['humidity_temp_ratio'] = train_fedf['humidity'] / (train_fedf['temparature'] + epsilon)
# train_fedf['Pressure_Humidity_Interaction'] = train_fedf['pressure'] * train_fedf['humidity']
# train_fedf["cloud_wind_interaction"] = train_fedf["cloud"] * train_fedf["windspeed"]
# # train_fedf['relative_dryness'] = 100 - train_fedf['humidity']
# train_fedf['Cloud_Humidity_ratio'] = train_fedf['cloud'] / (train_fedf['humidity'] + 1e-5)
# train_fedf['dewpoint_temp_diff'] = train_fedf['temparature'] - train_fedf['dewpoint']
# train_df['dewpoint_humidity_product'] = train_df['dewpoint'] * train_df['humidity']


feature_columns = train_fedf.drop('rainfall', axis=1).columns
X_test_final = test_df[feature_columns]


display(train_fedf.columns)
display(X_test_final.columns)


y_test_pred = ensemble_fin.predict(X_test_final)


test_df['rainfall_pred'] = y_test_pred

# Optional: Save to CSV
test_df[['id', 'rainfall_pred']].to_csv('submission.csv', index=False)


test_df[['id', 'rainfall_pred']]

