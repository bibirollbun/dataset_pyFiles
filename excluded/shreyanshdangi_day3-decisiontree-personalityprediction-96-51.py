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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
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

def objective(trial):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    splitter = trial.suggest_categorical('splitter', ['best', 'random'])
    max_depth = trial.suggest_int('max_depth', 3, 150)  # increased range
    min_samples_split = trial.suggest_int('min_samples_split', 2, 50)  # more flexibility
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 25)
    max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.03, step=0.002)  # pruning strength

    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)  # increased to 100 trials

print("\nðŸŽ¯ Best Hyperparameters Found:")
print(study.best_params)

best_params = study.best_params
best_model = DecisionTreeClassifier(
    **best_params,
    random_state=42
)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("\nâœ… Model Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



from sklearn.metrics import roc_auc_score
print("ROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))

