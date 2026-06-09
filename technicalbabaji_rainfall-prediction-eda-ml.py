import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')


train.head()


test.tail()


train.shape, test.shape


train.describe().T


train.dtypes


train.hist(bins=30, figsize=(15, 10));


train.corr()['rainfall'].sort_values(ascending=False)


# New features

train['temp_diff'] = train['maxtemp'] - train['mintemp']
train['cloud_to_sunshine'] = train['cloud'] * train['sunshine']
train['cloud_humidity'] = train['cloud'] + train['humidity']
train['humidity_sunshine'] = train['humidity'] * train['sunshine']

# Adding more features
# Dew point depression
train['dew_point_depression'] = train['temparature'] - train['dewpoint']

test['temp_diff'] = test['maxtemp'] - test['mintemp']
test['cloud_to_sunshine'] = test['cloud'] * test['sunshine']
test['cloud_humidity'] = test['cloud'] + test['humidity']
test['humidity_sunshine'] = test['humidity'] * test['sunshine']
# Dew point depression
test['dew_point_depression'] = test['temparature'] - test['dewpoint']


train.corr()['rainfall'].sort_values(ascending=False)


train.drop(columns=['temparature', 'winddirection'], inplace=True)

test.drop(columns=['temparature', 'winddirection'], inplace=True)


train.isnull().sum()


test.isnull().sum()


def remove_outliers_iqr(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

# Columns to check for outliers
columns_to_check = ['humidity', 'dewpoint', 'cloud']

# Remove outliers
train = remove_outliers_iqr(train, columns_to_check)


train['rainfall'].value_counts()


# Balance the dataset
from imblearn.over_sampling import SMOTE

X, y = train.drop(columns=['rainfall']), train['rainfall']

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)


y_resampled.value_counts()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled,
                                                  test_size=.25, 
                                                  # stratify=y
                                                 )


X_train.shape, X_val.shape


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_val_scaled = sc.transform(X_val)

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

log_model = LogisticRegression(C=0.1)
log_model.fit(X_train_scaled, y_train)


y_pred = log_model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(acc)
print(cm) 


y_pred[0]


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_model.fit(X_train_scaled, y_train)


y_pred = knn_model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(acc)
print(cm) 


from sklearn.svm import SVC

svc_model = SVC(C=0.1, kernel='rbf', gamma='auto',)
svc_model.fit(X_train_scaled, y_train)


y_pred = svc_model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(acc)
print(cm) 


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=8, min_samples_split=2, criterion='gini')
dt_model.fit(X_train_scaled, y_train)


y_pred = dt_model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(acc)
print(cm) 


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=8,
                                  max_features='sqrt',
                                 criterion='gini', bootstrap=False)
rf_model.fit(X_train_scaled, y_train)


y_pred = rf_model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(acc)
print(cm) 


from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01,
                                     max_depth=8, max_features='log2',
                                     )
gb_model.fit(X_train_scaled, y_train)


y_pred = gb_model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(acc)
print(cm) 


from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)


y_pred = nb_model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(acc)
print(cm) 


from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=7)
xgb_model.fit(X_train_scaled, y_train)


y_pred = xgb_model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(acc)
print(cm) 


from xgboost import plot_importance

plot_importance(xgb_model, max_num_features=15)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [5, 7, 8],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=param_grid,cv=5,
                           scoring='accuracy',
                           verbose=1, n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')



# Train the model with the best parameters
best_xgb_model = grid_search.best_estimator_
best_xgb_model.fit(X_train_scaled, y_train)

# Evaluate the tuned model
y_pred = best_xgb_model.predict(X_val_scaled)
print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
print(f'CM: {confusion_matrix(y_val, y_pred)}')


from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(iterations=200, learning_rate=0.2, depth=8, verbose=0)
cat_model.fit(X_train_scaled, y_train)


y_pred = cat_model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print(acc)
print(cm) 


y_pred[:5], y_val[:5].values


test_scaled = pd.DataFrame(sc.transform(test), index=test.index, columns=test.columns)
# predictions = best_xgb_model.predict(test_scaled)
predictions = cat_model.predict(test_scaled)


pred_prob = cat_model.predict_proba(test_scaled)
submission = pd.DataFrame({'id': test.index, 'rainfall': pred_prob[:, 1]})
submission.to_csv('submission.csv', index=False)




