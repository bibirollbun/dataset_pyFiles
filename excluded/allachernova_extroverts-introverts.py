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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')


train


print(train.isnull().sum())
print(train.isnull().sum().sum())



binary_cols = ['Stage_fear', 'Drained_after_socializing']
train[binary_cols] = train[binary_cols].replace({'Yes': 1, 'No': 0})
train['Personality'] = train['Personality'].replace({'Extrovert': 1, 'Introvert': 0})
test[binary_cols] = test[binary_cols].replace({'Yes': 1, 'No': 0})


X_train = train.drop(columns='Personality')  
y_train = train['Personality']  


numeric_cols = [col for col in X_train.columns if col not in binary_cols]


missing_numeric = X_train[numeric_cols].isnull().sum()
missing_binary = X_train[binary_cols].isnull().sum()

df_missing = pd.DataFrame({
    'Feature': numeric_cols + binary_cols,
    'Missing_Values': list(missing_numeric) + list(missing_binary),
    'Type': ['Numeric'] * len(numeric_cols) + ['Binary'] * len(binary_cols),
    'Imputation_Strategy': ['Median'] * len(numeric_cols) + ['Most Frequent'] * len(binary_cols)
})


plt.figure(figsize=(12, 6))
barplot = sns.barplot(
    data=df_missing,
    x='Feature',
    y='Missing_Values',
    hue='Imputation_Strategy',
    palette='coolwarm'
)

# Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ¸Ğ¼ Ğ¿Ğ¾Ğ´Ğ¿Ğ¸Ñ�Ğ¸ Ğ½Ğ° Ñ�Ñ‚Ğ¾Ğ»Ğ±Ğ¸ĞºĞ¸
for index, row in df_missing.iterrows():
    barplot.text(
        index,
        row['Missing_Values'] + 0.5,
        f"{row['Missing_Values']}",
        color='black',
        ha="center"
    )

# Ğ�Ñ„Ğ¾Ñ€Ğ¼Ğ»ĞµĞ½Ğ¸Ğµ
plt.title('Missing values', fontsize=14)
plt.xlabel('Features')
plt.ylabel('Missing values')
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title='Strategy')
plt.tight_layout()
plt.show()



imputer = SimpleImputer(strategy='median')
X_train_numeric = pd.DataFrame(imputer.fit_transform(X_train[numeric_cols]), columns=numeric_cols)
X_test_numeric = pd.DataFrame(imputer.transform(test[numeric_cols]), columns=numeric_cols)


bin_imputer = SimpleImputer(strategy='most_frequent')
X_train_binary = pd.DataFrame(bin_imputer.fit_transform(X_train[binary_cols]), columns=binary_cols)
X_test_binary = pd.DataFrame(bin_imputer.transform(test[binary_cols]), columns=binary_cols)


X_train_numeric


def feature_engineering(df):
    df['Social_x_Outside'] = df['Social_event_attendance'] * df['Going_outside']
    df['Social_x_Friends'] = df['Social_event_attendance'] * df['Friends_circle_size']
    df['Social_x_Posts'] = df['Social_event_attendance'] * df['Post_frequency']
    df['Outside_x_Friends'] = df['Going_outside'] * df['Friends_circle_size']
    df['Outside_x_Posts'] = df['Going_outside'] * df['Post_frequency']
    df['Friends_x_Posts'] = df['Friends_circle_size'] * df['Post_frequency']

    df['Alone_x_Social'] = df['Time_spent_Alone'] * df['Social_event_attendance']
    df['Alone_x_Outside'] = df['Time_spent_Alone'] * df['Going_outside']
    df['Alone_x_Friends'] = df['Time_spent_Alone'] * df['Friends_circle_size']
    df['Alone_x_Posts'] = df['Time_spent_Alone'] * df['Post_frequency']
    
feature_engineering(X_train_numeric)
feature_engineering(X_test_numeric)


X_train_numeric


plt.figure(figsize=(14, 6))
sns.boxplot(data=X_train_numeric, palette='pastel')
plt.title('Distribution of numerical features BEFORE scaling', fontsize=14)
plt.xticks(rotation=15)
plt.ylabel('Values')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_numeric), columns=X_train_numeric.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_numeric), columns=X_test_numeric.columns)

X_train_final = pd.concat([X_train_scaled, X_train_binary.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_scaled, X_test_binary.reset_index(drop=True)], axis=1)


X_train_final


X_test_final


X_test_final.isnull().sum()


X_train, X_val, y_train, y_val = train_test_split(X_train_final, y_train, test_size=0.3, stratify=y_train, random_state=42)


X_train


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))



final_predictions = model.predict(X_test_final) 
final_predictions_named = ['Introvert' if pred == 0 else 'Extrovert' for pred in final_predictions]
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = final_predictions_named
submission.to_csv('/kaggle/working/submission17.csv', index=False)
print("âœ… Submission file created successfully!")


models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"{name}: {score:.8f}")



param_grid = {
    'C': [0.001, 0.005, 0.01, 0.05],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:", grid.best_params_)
print("Ğ›ÑƒÑ‡ÑˆĞ°Ñ� Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ:", grid.best_score_)



param_grid = {
    'C': [5, 10, 15],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)



param_grid = {
    'n_estimators': [30, 50, 100, 110],
    'learning_rate': [0.005, 0.01, 0.05],
    'max_depth': [2, 3]
}

grid = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)



param_grid = {
    'n_neighbors': [3, 7, 8, 9, 10, 15]
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)



final_model = KNeighborsClassifier(n_neighbors=9)  #0.9650953580424613
# final_model = SVC(C=3, kernel='rbf', gamma='auto') #0.9650953580424613
# final_model = LogisticRegression()                 #0.9650953580424613
# final_model = LogisticRegression(C=0.01, penalty='l2', solver='liblinear') 
final_model.fit(X_train, y_train)


y_pred = final_model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))


final_predictions = final_model.predict(X_test_final) 
final_predictions_named = ['Introvert' if pred == 0 else 'Extrovert' for pred in final_predictions]
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = final_predictions_named
submission.to_csv('/kaggle/working/KNN_n=9.csv', index=False)
print("âœ… Submission file created successfully!")

