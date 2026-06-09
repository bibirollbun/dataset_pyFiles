# Imports
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#import optuna


Dtrain = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
Dtest = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
Dtest_subm = Dtest.copy()


Dtest.head(6)


Dtrain.head(6)


Dtrain.describe()


Dtrain.info()


Dtrain = Dtrain.drop('id',axis=1)
Dtest = Dtest.drop('id',axis=1)


x_values = Dtrain.select_dtypes(include=['number'])



plt.figure(figsize=(18,8))
sns.countplot(data=Dtrain,x='Fertilizer Name',palette='cividis')
plt.title('Fertilizer Distribution')
plt.show()



corr_matrix = x_values.corr()
plt.figure(figsize=(20, 12))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(14, 28))

for i, x_value in enumerate(x_values):
    ax = axes.flatten()[i] 
    sns.boxplot(data=Dtrain, x='Fertilizer Name', y=x_value, hue='Fertilizer Name', ax=ax, palette='cividis',width=0.9)
    ax.set_title(f'{x_value.capitalize()}')
    ax.set_ylabel(x_value.capitalize())
plt.tight_layout()
plt.show()


# Separate the target variable
X = Dtrain.drop('Fertilizer Name',axis=1)
y = Dtrain['Fertilizer Name']

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

numerical_columns = list(X_train.select_dtypes(include=['float64', 'int64']).columns)
categorical_columns = list(X_train.select_dtypes(include=['object', 'category']).columns)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


#Optuna sample
y_train_series = pd.Series(y_train, index=X_train.index)

X_sample = X_train.sample(frac=0.2, random_state=42)
y_sample = y_train_series.loc[X_sample.index]


#preprocessing_pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_columns),
    ('cat', cat_pipeline, categorical_columns)
])


def objective_hist(trial):
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k=trial.suggest_categorical('k', [10, 20, 'all']))),
        ('classifier', HistGradientBoostingClassifier(
            max_iter=trial.suggest_int('max_iter', 100, 300),
            max_leaf_nodes=trial.suggest_categorical('max_leaf_nodes', [31, 50, None]),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
            random_state=42
        ))
    ])
    score = cross_val_score(pipeline, X_sample, y_sample, cv=5, scoring='accuracy', n_jobs=-1)
    return score.mean()

#study_hist = optuna.create_study(direction='maximize')
#study_hist.optimize(objective_hist, n_trials=40)


def objective_xgb(trial):
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k=trial.suggest_categorical('k', [10, 20, 'all']))),
        ('classifier', XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 200),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
            subsample=trial.suggest_float('subsample', 0.8, 1.0),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    score = cross_val_score(pipeline, X_sample, y_sample, cv=5, scoring='accuracy', n_jobs=-1)
    return score.mean()

#study_xgb = optuna.create_study(direction='maximize')
#study_xgb.optimize(objective_xgb, n_trials=40)


def objective_rf(trial):
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k=trial.suggest_categorical('k', [10, 20, 'all']))),
        ('classifier', RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 200),
            max_depth=trial.suggest_categorical('max_depth', [None, 10, 20]),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 5),
            random_state=42
        ))
    ])
    score = cross_val_score(pipeline, X_sample, y_sample, cv=5, scoring='accuracy', n_jobs=-1)
    return score.mean()

#study_rf = optuna.create_study(direction='maximize')
#study_rf.optimize(objective_rf, n_trials=40)


# HistGradientBoosting com SelectFromModel
HistGradientBoosting = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
    ('classifier', HistGradientBoostingClassifier(
        max_iter=129,
        max_leaf_nodes=50,
        learning_rate=0.07459934258227152,
        random_state=42
    ))
])

# XGBoost
XGBoost = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.08301980801070748,
        subsample=0.8534115834322589,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# RandomForest 
RandomForest = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
    ('classifier', RandomForestClassifier(
        n_estimators=162,
        max_depth=10,
        min_samples_split=4,
        random_state=42
    ))
])



RandomForest.fit(X_train, y_train)
y_pred = RandomForest.predict(X_test)
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred) 

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()


XGBoost.fit(X_train, y_train)
y_pred = XGBoost.predict(X_test)
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred) 

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()


HistGradientBoosting.fit(X_train, y_train)
y_pred = HistGradientBoosting.predict(X_test)
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred) 

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()


y_proba = XGBoost.predict_proba(Dtest)

top3_preds = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)
top3_labels_str = [' '.join(row) for row in top3_labels]

submission = pd.DataFrame({
    'id': Dtest_subm['id'],
    'Fertilizer Name': top3_labels_str
})

submission.to_csv('submission.csv', index=False)


submission

