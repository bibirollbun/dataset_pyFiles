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



import pandas as pd
from sklearn.decomposition import PCA

num_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing',
            'Friends_circle_size', 'Post_frequency']

X = df[num_cols]

pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X)

loadings = pd.DataFrame(pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                        index=num_cols)

explained_var = pca.explained_variance_ratio_

feature_importance = (abs(loadings) * explained_var).sum(axis=1).sort_values(ascending=False)

print("Overall Feature Importance after 90% variance PCA:")
print(feature_importance)


import pandas as pd
from sklearn.decomposition import PCA

num_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing',
            'Friends_circle_size', 'Post_frequency']

X = df[num_cols]

pca = PCA()
X_pca = pca.fit_transform(X)

loadings = pd.DataFrame(pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                        index=num_cols)

explained_var = pca.explained_variance_ratio_

feature_importance = (abs(loadings) * explained_var).sum(axis=1).sort_values(ascending=False)

print("Overall Feature Importance after full default PCA:")
print(feature_importance)


import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

num_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing',
            'Friends_circle_size', 'Post_frequency']

X = df[num_cols]

pca_90 = PCA(n_components=0.90)
X_pca_90 = pca_90.fit_transform(X)

loadings_90 = pd.DataFrame(pca_90.components_.T,
                           columns=[f'PC{i+1}' for i in range(pca_90.n_components_)],
                           index=num_cols)

explained_var_90 = pca_90.explained_variance_ratio_

feature_importance_90 = (abs(loadings_90) * explained_var_90).sum(axis=1)

pca_full = PCA()
X_pca_full = pca_full.fit_transform(X)

loadings_full = pd.DataFrame(pca_full.components_.T,
                             columns=[f'PC{i+1}' for i in range(pca_full.n_components_)],
                             index=num_cols)

explained_var_full = pca_full.explained_variance_ratio_

feature_importance_full = (abs(loadings_full) * explained_var_full).sum(axis=1)

df_plot = pd.DataFrame({
    '90% Variance PCA': feature_importance_90,
    'Full PCA': feature_importance_full
}).sort_index()

df_plot.plot(kind='bar', figsize=(10,6))
plt.title('Feature Importance: 90% Variance vs Full PCA')
plt.ylabel('Overall Importance')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.ylim(0.30, None)
plt.show()


import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
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
    X, y, test_size=0.2, random_state=42, stratify=y
)

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-3, 1e2)
    solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
    penalty = 'l2'  
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

    model = LogisticRegression(
        C=C,
        solver=solver,
        penalty=penalty,
        class_weight=class_weight,
        max_iter=2000,
        random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest Hyperparameters:")
print(study.best_params)

best_params = study.best_params
best_model = LogisticRegression(
    C=best_params['C'],
    solver=best_params['solver'],
    penalty='l2',
    class_weight=best_params['class_weight'],
    max_iter=2000,
    random_state=42
)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("\nModel Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



from sklearn.metrics import roc_auc_score
print("ROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))

