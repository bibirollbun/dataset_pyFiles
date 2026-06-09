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


from ydata_profiling import ProfileReport

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, PowerTransformer, FunctionTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,  VotingClassifier, StackingClassifier
from xgboost import XGBClassifier

import optuna

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

import pickle

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df.sample(10)


df.shape


df.describe(include = 'all')


df.info()


df.isnull().sum().sort_values(ascending = False)


df.nunique()


df.duplicated().sum()


profile = ProfileReport(df, title = "Introvert or Extrovert EDA Report", explorative = True)
profile.to_file("introvert_or_extrovert_eda_report.html")


num_cols = df.select_dtypes(exclude = ['object']).columns.tolist()
cat_cols = df.select_dtypes(include = ['object']).columns.tolist()

print("Numerical Variables are ",num_cols)
print("Categorical Variables are ",cat_cols)


for col in num_cols:
    plt.figure(figsize = (10, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(x = col, data = df, kde = True, bins = 30, hue = 'Personality')
    plt.title(f'{col} Distribution by Personality')

    plt.subplot(1, 2, 2)
    sns.boxplot(x = 'Personality', y = col, data = df)
    plt.title(f'Boxplot for {col} by Personality')

    plt.tight_layout()
    plt.show()


for col in cat_cols:
    plt.figure(figsize = (10, 6))
    sns.countplot(data = df, x = col, hue = 'Personality')
    plt.title(f'{col} by Personality')
    plt.xticks(rotation = 30)
    plt.tight_layout()
    plt.show()


categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'Unknown')),
    ('encoder', OneHotEncoder(drop = 'first', handle_unknown = 'ignore'))
])

numeric_pipeline = Pipeline([
   ('imputer', SimpleImputer(strategy = 'median')),
   ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(transformers = [
    ('categorical_pipeline', categorical_pipeline, cat_cols[:-1]),
    ('numeric_pipeline', numeric_pipeline, num_cols[1:]),
], remainder = 'drop')


X = df.drop('Personality', axis = 1)
y = df['Personality']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

pipeline_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder = False))

])


def objective(trial):
    w1 = trial.suggest_int('lr_weight', 1, 5)
    w2 = trial.suggest_int('rf_weight', 1, 5)
    w3 = trial.suggest_int('xgb_weight', 1, 5)

    pipeline = VotingClassifier(
        estimators=[
            ('lr', pipeline_lr),
            ('rf', pipeline_rf),
            ('xgb', pipeline_xgb)
        ],
        voting = 'soft',
        weights = [w1, w2, w3]
    )

    score = cross_val_score(pipeline, X_train, y_train, n_jobs = -1, cv = 5, scoring = 'precision')
    accuracy = score.mean()

    return accuracy


optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 50)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print(f"  Value (Accuracy): {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


best_params = study.best_params
optimal_weights = [
    best_params['lr_weight'],
    best_params['rf_weight'],
    best_params['xgb_weight']
]

pipeline = VotingClassifier(
    estimators = [
        ('lr', pipeline_lr),
        ('rf', pipeline_rf),
        ('xgb', pipeline_xgb)
    ],
    voting = 'soft',
    weights = optimal_weights
)


set_config(display = 'diagram')
pipeline


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy on test set:", accuracy)


scores = cross_val_score(pipeline, X, y, cv = 5, scoring = 'accuracy')
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")


skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

stratified_scores = cross_val_score(pipeline, X, y, cv = skf, n_jobs = -1, scoring = 'accuracy')

print(f"Stratified CV Scores: {stratified_scores}")
print(f"Mean Accuracy: {np.mean(stratified_scores):.4f}")
print(f"Standard Deviation: {np.std(stratified_scores):.4f}")


class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (6, 4))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', cbar = False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()


y_probs = pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

precision, recall, _ = precision_recall_curve(y_test, y_probs)
avg_precision = average_precision_score(y_test, y_probs)

fig, axes = plt.subplots(1, 2, figsize = (10, 6))

# ROC Curve
axes[0].plot(fpr, tpr, color='darkorange', label = f'ROC Curve (AUC = {auc:.2f})')
axes[0].plot([0, 1], [0, 1], 'k--', label = 'Random Classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend(loc = 'lower right')
axes[0].grid(True)

# Precision-Recall Curve
axes[1].plot(recall, precision, color = 'blue', label = f'PR Curve (AP = {avg_precision:.2f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend(loc = 'best')
axes[1].grid(True)

plt.suptitle(f'Model Evaluation | Accuracy: {accuracy:.2f}')
plt.tight_layout()
plt.show()


with open('model_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved successfully.")


df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

X_test = df_test
y_pred = pipeline.predict(X_test)
y_pred = np.where(y_pred == 0, 'Extrovert', 'Introvert')

submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': y_pred
})

submission.to_csv('submission.csv', index = False)
print("Submission file 'submission.csv' created successfully.")

