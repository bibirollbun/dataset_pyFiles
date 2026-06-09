import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
test_data  = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')


train_data.shape


train_data.head()


train_data.info()


train_data.describe()


train_data.describe(exclude = np.number)


train_data.drop(['id', 'CustomerId', 'Surname'], axis = 1, inplace = True)
train_data.duplicated().sum()


test_data.drop(['id', 'CustomerId', 'Surname'], axis = 1, inplace = True)


theme_color = sns.color_palette('Paired')


fig, ax = plt.subplots(1, 2, figsize=(10, 5))

exited_count = train_data['Exited'].value_counts()
ax[0].pie(exited_count, labels = exited_count.index, autopct = '%1.1f%%', colors = theme_color)
sns.countplot(x = 'Exited', data = train_data, ax = ax[1], palette = theme_color)

plt.suptitle('Target (Exited) Analysis', fontsize = 15)
plt.show()


numerical_columns = []
categorical_columns = []

for col in train_data.columns:
    if train_data[col].dtype in ['int64', 'float64']:
        if col != 'Exited':
            numerical_columns.append(col)
    else:
        categorical_columns.append(col)

print(f'Numerical Columns: {numerical_columns}')
print(f'Categorical Columns: {categorical_columns}') 


geography_unique_counts = train_data['Geography'].value_counts()
gender_unique_counts = train_data['Gender'].value_counts()

print(geography_unique_counts)
print('----------------')
print(gender_unique_counts)


for col in numerical_columns:
    plt.figure(figsize=(10, 5))
    plt.hist(train_data[col], bins = 25, color = 'skyblue', edgecolor = 'black')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'Distribution of {col}', fontsize = 15)
    plt.show()


fig, ax = plt.subplots(1, 2, figsize = (10, 5))

ax[0].set_title('Geography')
train_data['Geography'].value_counts().plot(kind = 'pie', 
                                            autopct = '%1.1f%%', 
                                            ax = ax[0],
                                            colors = theme_color)

ax[1].set_title('Gender')
train_data['Gender'].value_counts().plot(kind = 'pie', 
                                         autopct = '%1.1f%%', 
                                         ax = ax[1],
                                         colors = theme_color)


plt.show()


plt.figure(figsize = (16, 16))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(4, 2, i)
    sns.barplot(x = 'Exited', y = col, data = train_data, palette = 'Paired')
    plt.title(f'{col} vs Exited')
    plt.xlabel('')
    plt.ylabel(col)
plt.show()


numerical_cols = numerical_columns.copy()

numerical_cols.remove('IsActiveMember')
numerical_cols.remove('Tenure')
numerical_cols.remove('NumOfProducts')
numerical_cols.remove('HasCrCard') 


for column in numerical_cols:
    fig, ax = plt.subplots(figsize = (16, 6))
    fig = sns.histplot(train_data, x = column, hue = "Exited", bins = 50, kde = True, palette = theme_color)
    plt.show()


numerical_columns.append('Exited')
correlation_matrix = train_data[numerical_columns].corr()

mask = np.zeros_like(correlation_matrix)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (16, 10))
sns.heatmap(correlation_matrix, mask = mask, annot = True, cmap = 'Blues')
plt.show()


train_data.drop_duplicates(inplace = True, keep = 'first')

train_data.shape


print(train_data['Geography'].unique())
print(train_data['Gender'].unique())


train_data = train_data.replace({
    'Geography': {'France': 1, 'Spain': 2, 'Germany': 3},
    'Gender'   : {'Male' : 0, 'Female' : 1},  
})


test_data = test_data.replace({
    'Geography': {'France': 1, 'Spain': 2, 'Germany': 3},
    'Gender'   : {'Male' : 0, 'Female' : 1},  
})


train_data.head()


numerical_columns.remove('Exited')

for column in numerical_columns:
    print(column, max(train_data[column]), min(train_data[column]))


input_data  = train_data.drop(['Exited'], axis = 1)
output_data = train_data['Exited']


x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size = 0.2, random_state = 42)


scaler = StandardScaler()

input_data['Balance'] = scaler.fit_transform(input_data[['Balance']])

input_data.head()


models = {
    'Logistic Regression': LogisticRegression(random_state = 42, max_iter = 1000),
    'Random Forest': RandomForestClassifier(random_state = 42),
    'K Nearest Neighbors': KNeighborsClassifier(n_neighbors = 9),
    'XGBoost': XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', random_state = 42)
}


idx = 1
for name, model in models.items():
    model.fit(input_data, output_data)
    print(f"Model {idx}: {name}")
    idx += 1
    model.fit(x_train, y_train)
    print(f'  - {name} Training done ✅ \n')
    


idx = 1
y_preds = []
for name, model in models.items():
    print(f"Model {idx}: {name}")
    idx += 1
    y_pred = model.predict(x_test)
    y_preds.append(y_pred)
    print(classification_report(y_test, y_pred), '\n\n')


models['Random Forest'].get_params()


param_rf_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7],
    'max_features': ['sqrt', 'log2'],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state = 42), 
                           param_rf_grid,
                           cv = 3,
                           n_jobs = -1
                          )
grid_search.fit(x_train, y_train)


best_params = grid_search.best_params_
best_auc = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best AUC Score: {best_auc}')


best_rf_model = RandomForestClassifier(**best_params, random_state = 42)
best_rf_model.fit(x_train, y_train)

rf_y_pred = best_rf_model.predict(x_test)


models['XGBoost'].get_params()


param_xgb_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(XGBClassifier(use_label_encoder = False, 
                                         eval_metric = 'logloss', 
                                         random_state = 42), 
                           param_xgb_grid, 
                           cv = 3, 
                           n_jobs = -1
                          )
grid_search.fit(x_train, y_train)


best_params = grid_search.best_params_
best_auc = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best AUC Score: {best_auc}')


best_xgb_model = XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', **best_params, random_state = 42)
best_xgb_model.fit(x_train, y_train)

xgb_y_pred = best_xgb_model.predict(x_test)


best_rf_auc = accuracy_score(y_test, rf_y_pred)
print(f'Best Random Forest AUC : {best_rf_auc * 100:.2f}%')

print('----------------------------------------')

best_xgb_auc = accuracy_score(y_test, xgb_y_pred)
print(f'Best XGBoost AUC : {best_xgb_auc * 100:.2f}%')


print(classification_report(y_test, xgb_y_pred))


cm = confusion_matrix(y_test, xgb_y_pred)
sns.heatmap(cm, annot = True, cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')


sample_submission = sample_submission.drop(['Exited'], axis=1)
submission_exited = best_xgb_model.predict_proba(test_data)[:, 1]
sample_submission['Exited'] = submission_exited

sample_submission.head(10)


sample_submission['Exited'] = (sample_submission['Exited'] > 0.5).astype(int)


sample_submission.to_csv('submission.csv', index=False)


import joblib
joblib.dump(best_xgb_model, 'best_model.pkl')

model = joblib.load('best_model.pkl')




