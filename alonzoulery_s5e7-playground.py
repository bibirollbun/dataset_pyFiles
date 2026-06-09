import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3 
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from xgboost import XGBClassifier
conn = sqlite3.connect('train_data.db')
cursor = conn.cursor()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

train.to_sql('train', conn, if_exists = 'replace', index = False)
test.to_sql('test', conn, if_exists = 'replace', index = False)


train_columns = set(train.columns)
test_columns = set(test.columns)

common_columns = train_columns & test_columns
print("Common Columns", common_columns)

outlier_columns = train_columns - test_columns
print("Outlier Column(s):", outlier_columns)


train['Personality'].unique()


nan_count = train['Stage_fear'].isna().sum()
nan_count_i = train['Drained_after_socializing'].isna().sum()
nan_count_ii = train['Personality'].isna().sum()

#print(f"Stage_Fear: {nan_count} \n Drained After Socializiing: {nan_count_i} \n Personality: {nan_count_ii}")
train_i = train.copy()
train_i['Stage_fear'] = train_i['Stage_fear'].fillna('No')
train_i['Drained_after_socializing'] = train_i['Drained_after_socializing'].fillna('Yes')




for i in train_i.columns:
    if train_i[i].dtype in ['int64', 'float64']:
        nan_count = train_i[i].isna().sum()
        print(f"Column: {i}, NaN count: {nan_count}")


numerical_cols = []
for col in train_i.columns:
    if train_i[col].dtype in ['int64', 'float64']:
        numerical_cols.append(col)

for col in numerical_cols:
    mean_value = round(train_i[col].mean())
    train_i[col].fillna(mean_value, inplace = True)

train_i.to_sql('train_i', conn, if_exists = 'replace', index = False)
train_i.isnull().sum()



train_shape = train.shape
train_i_shape = train_i.shape

print(train_shape, train_i_shape)


query = """
SELECT *
FROM train_i
WHERE Personality == 'Extrovert'"""
Extroverts = pd.read_sql_query(query, conn)
Extroverts.to_sql('Extroverts', conn, if_exists = 'replace', index = False)

query = """
SELECT *
FROM train_i
WHERE Personality == 'Introvert'"""
Introverts = pd.read_sql_query(query, conn)
Introverts.to_sql('Introverts', conn, if_exists = 'replace', index = False)

num_extroverts = len(Extroverts)
num_introverts = len(Introverts)
labels = ['Extroverts', 'Introverts']
sizes = [num_extroverts, num_introverts]
colors = ['#ff9999', '#66b3ff']

plt.figure(figsize = (7,7))
plt.pie(sizes, labels = labels, autopct = '%1.1f%%', startangle = 90, colors = colors)
plt.axis('equal')
plt.title("Percentage of Extroverts vs. Introverts")
plt.show()


train_i


I_alone = Introverts['Time_spent_Alone'].value_counts()
E_alone = Extroverts['Time_spent_Alone'].value_counts()

chart_i = pd.DataFrame({
    'Introverts': I_alone,
    'Extroverts': E_alone
})

chart_i.plot(kind = 'bar', figsize = (10, 6), width = 0.8)
plt.title("Comparison of Time Spent Alone")
plt.xlabel("Alone Time")
plt.ylabel("Count")
plt.xticks(rotation = 0)
plt.tight_layout()
plt.show()


train_alter = train_i.copy()
train_alter['Personality'] = train_alter['Personality'].map({'Extrovert': 1, 'Introvert': 0})
train_alter['Drained_after_socializing'] = train_alter['Drained_after_socializing'].map({'No': 1, 'Yes': 0})
train_alter['Stage_fear'] = train_alter['Stage_fear'].map({'No': 1, 'Yes': 0})
train_alter['friends_vs_events'] = (
    train_alter['Friends_circle_size'] * train_alter['Social_event_attendance']
)
train_alter['fear_vs_expression'] = (
    (1 - train_alter['Stage_fear']) * train_alter['Post_frequency']
)
train_alter['fear_vs_expression'] = np.where(
    train_alter['fear_vs_expression'] == 0,
    train_alter['Post_frequency'],
    train_alter['fear_vs_expression']
)
train_alter


corr_matrix = train_alter.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidths = 0.5)
plt.title('Correlation Heatmap - Features to Categorize Personality')
plt.tight_layout()
plt.show()


test_alter = test.copy()
test_alter['Stage_fear'] = test_alter['Stage_fear'].fillna('No')
test_alter['Drained_after_socializing'] = test_alter['Drained_after_socializing'].fillna('Yes')

for col in numerical_cols:
    mean_value = round(train_alter[col].mean())
    test_alter[col] = test_alter[col].fillna(mean_value)
    
test_alter['Drained_after_socializing'] = test_alter['Drained_after_socializing'].map({'No': 1, 'Yes': 0})
test_alter['Stage_fear'] = test_alter['Stage_fear'].map({'No': 1, 'Yes': 0})

test_alter['friends_vs_events'] = (
    test_alter['Friends_circle_size'] * test_alter['Social_event_attendance']
)
test_alter['fear_vs_expression'] = (
    (1 - test_alter['Stage_fear']) * test_alter['Post_frequency']
)

test_alter['fear_vs_expression'] = np.where(
    test_alter['fear_vs_expression'] == 0,
    test_alter['Post_frequency'],
    test_alter['fear_vs_expression']
)

if 'Personality' in test_alter.columns:
    test_alter = test_alter.drop(columns = ['Personality'])

#test_alter


columns_to_drop = ['Friends_circle_size', 'Social_event_attendance', 'Stage_fear', 'Post_frequency']
test_alter.drop(columns = columns_to_drop, inplace = True)
train_alter.drop(columns = columns_to_drop, inplace = True)
column_check = set(test_alter.columns) - set(train_alter.columns)
print(column_check)


columns_to_drop = ['Friends_circle_size', 'Social_event_attendance', 'Stage_fear', 'Post_frequency', 'Personality']
X = train_alter.drop(columns= 'Personality')
y = train_alter['Personality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)


rf_prediction = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_prediction.fit(X_train, y_train)
rf_preds = rf_prediction.predict(X_test)

print("Random Forest Prediction Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))


rf_test_pred = rf_prediction.predict(test_alter)
rf_test_labels = ['Extrovert' if pred == 1 else 'Introvert' for pred in rf_test_pred]
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': rf_test_labels
})

submission.to_csv('submission.csv', index = False)
submission


xgb_model = xgb.XGBClassifier(
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 5,
    random_state = 42,
    use_label_encoder = False,
    eval_metric = 'logloss'
)
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))


xgb_model = XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', random_state = 42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
X_final_test = test_alter
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))

final_preds = xgb_model.predict(X_final_test)
final_preds_labels = pd.Series(final_preds).map({1: 'Extrovert', 0: 'Introvert'})

submission = pd.DataFrame({
    'id': test_alter['id'],
    'Personality':final_preds_labels
})

# submission.to_csv('submission.csv', index = False)

