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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer, FunctionTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import optuna

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

import pickle

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df.sample(10)


df.shape


df.describe(include = 'all')


df.info()


df.isnull().sum().sort_values(ascending = False)


df.nunique()


df.duplicated().sum()


num_cols = df.select_dtypes(exclude = ['object']).columns.tolist()
cat_cols = df.select_dtypes(include = ['object']).columns.tolist()

print("Numerical Variables are ", num_cols)
print("Categorical Variables are ", cat_cols)


profile = ProfileReport(df, title = "Bank Subscription Report", explorative = True)
profile.to_file("bank_subscription_report.html")


sns.countplot(data = df, x = 'y')


for col in num_cols[1:-1]:
    plt.figure(figsize = (10, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(x = col, data = df, kde = True, bins = 30, hue = 'y')
    plt.title(f'{col} Distribution by y')

    plt.subplot(1, 2, 2)
    sns.boxplot(x = 'y', y = col, data = df)
    plt.title(f'Boxplot for {col} by y')

    plt.tight_layout()
    plt.show()


for col in cat_cols:
    plt.figure(figsize = (10, 6))
    plt.show()
    sns.countplot(data = df, x = col, hue = 'y')
    plt.title(f'{col} by y')
    plt.xticks(rotation = 30)
    plt.tight_layout()


ordinal_encoder_pipeline = Pipeline([
    ('ordinal_encoder', OrdinalEncoder()),
    ('scaler', StandardScaler())
])

one_hot_encoder_pipeline = Pipeline([
    ('one_hot_encoder', OneHotEncoder(drop = 'first'))
])


preprocessor = ColumnTransformer(transformers = [
    ('ordinal_encoder', ordinal_encoder_pipeline, ['marital', 'education', 'month']),
    ('one_hot_encoder', one_hot_encoder_pipeline, ['job', 'default', 'housing', 'loan', 'contact', 'poutcome']),
    ('passthrough', 'passthrough', num_cols[1:-1])
], remainder = 'drop')


X = df.drop('y', axis = 1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': 42,
        'n_jobs': -1,

        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step = 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log = True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, step = 0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step = 0.1),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log = True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-8, 10.0, log = True),
        
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log = True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log = True)
    }

    classifier_obj = XGBClassifier(**params, use_label_encoder = False)
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier_obj)
    ])

    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    score = cross_val_score(full_pipeline, X, y, n_jobs = -1, cv = skf, scoring = 'roc_auc')
    accuracy = score.mean()
    
    return accuracy


optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category = UserWarning)

study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 50)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print(f"  Value (Accuracy): {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


best_params = study.best_trial.params
best_classifier = XGBClassifier(**best_params)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_classifier)
])


set_config(display = 'diagram')
pipeline


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = roc_auc_score(y_test, y_pred)
print("Model accuracy on test set:", accuracy)


scores = cross_val_score(pipeline, X, y, cv = 5, scoring = 'roc_auc')
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")


skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
stratified_scores = cross_val_score(pipeline, X, y, cv = skf, n_jobs = -1, scoring = 'roc_auc')

print(f"Stratified CV Scores: {stratified_scores}")
print(f"Mean Accuracy: {np.mean(stratified_scores):.4f}")
print(f"Standard Deviation: {np.std(stratified_scores):.4f}")


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (6, 4))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', cbar = False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()


y_probs = pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
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


df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
y_pred_probs = pipeline.predict_proba(df_test)[:, 1]

submission = pd.DataFrame({
    'id': df_test['id'],
    'y': y_pred_probs
})

submission.to_csv('submission.csv', index = False)
print("Submission file 'submission.csv' created successfully.")

