import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df.info()


df.shape


df['Personality'].value_counts()


df.head(5)


print(df['Stage_fear'].unique())
print(df['Drained_after_socializing'].unique())


mapping = {'Yes': 1, 'No': 0}
df['Stage_fear'] = df['Stage_fear'].map(mapping)

mapping1 = {'Yes': 1, 'No': 0}
df['Drained_after_socializing'] = df['Drained_after_socializing'].map(mapping1)

mapping2 = {'Extrovert': 1, 'Introvert': 0}
df['Personality'] = df['Personality'].map(mapping2)


df.head(5)


df.info()


df.isnull().sum()


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

num_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing',
            'Friends_circle_size', 'Post_frequency']

imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df.isnull().sum()


import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

num_cols = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]

X = df[num_cols]
y = df['Personality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

def objective_knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 2)  # 1=Manhattan, 2=Euclidean
    }
    model = KNeighborsClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
    return scores.mean()

def objective_svm(trial):
    params = {
        'C': trial.suggest_float('C', 0.1, 10.0, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'degree': trial.suggest_int('degree', 2, 5),
    }
    model = SVC(**params, class_weight='balanced', random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
    return scores.mean()

def objective_nb(trial):
    params = {
        'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True)
    }
    model = GaussianNB(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
    return scores.mean()

print("\nðŸš€ Tuning KNN...")
study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(objective_knn, n_trials=30, show_progress_bar=True)

print("\nðŸš€ Tuning SVM...")
study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective_svm, n_trials=30, show_progress_bar=True)

print("\nðŸš€ Tuning Naive Bayes...")
study_nb = optuna.create_study(direction='maximize')
study_nb.optimize(objective_nb, n_trials=30, show_progress_bar=True)

best_knn = KNeighborsClassifier(**study_knn.best_params)
best_svm = SVC(**study_svm.best_params, class_weight='balanced', random_state=42)
best_nb = GaussianNB(**study_nb.best_params)

best_knn.fit(X_train, y_train)
best_svm.fit(X_train, y_train)
best_nb.fit(X_train, y_train)

y_pred_knn = best_knn.predict(X_test)
y_pred_svm = best_svm.predict(X_test)
y_pred_nb = best_nb.predict(X_test)

print("\nðŸŽ¯ Best KNN Params:", study_knn.best_params)
print("ðŸŽ¯ Best SVM Params:", study_svm.best_params)
print("ðŸŽ¯ Best Naive Bayes Params:", study_nb.best_params)

results = pd.DataFrame({
    'Model': ['KNN', 'SVM', 'Naive Bayes'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_nb)
    ],
    'F1 Score (Macro)': [
        f1_score(y_test, y_pred_knn, average='macro'),
        f1_score(y_test, y_pred_svm, average='macro'),
        f1_score(y_test, y_pred_nb, average='macro')
    ]
})

print("\nðŸ“Š Model Comparison:\n")
print(results)

print("\nðŸ§© Detailed Reports:")
print("\n--- KNN ---")
print(classification_report(y_test, y_pred_knn))
print("\n--- SVM ---")
print(classification_report(y_test, y_pred_svm))
print("\n--- Naive Bayes ---")
print(classification_report(y_test, y_pred_nb))


