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


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


print('Train data:')
train.head(5)


print('Test data:')
test.head(5)


print('Datatypes of Train data with additional information about the columns:')
train.info()


print('Datatypes of Test data with additional information about the columns:')
test.info()


print('Descriptive statistics of Train data')
train.describe()


print('Descriptive statistics of Test data')
test.describe()


print('Missing values in Train data')
train.isna().sum()


print('Missing values in Test data')
test.isna().sum()


train_duplicates = train.duplicated().sum()
if train_duplicates == 0:
    print(f'{train_duplicates} duplicates found in Train data.')
else:
    print(f'{train_duplicates} duplicates found in Train data.')
    train.drop_duplicates()
    print(f'Duplicates removed successfully.')

test_duplicates = test.duplicated().sum()
if test_duplicates == 0:
    print(f'{test_duplicates} duplicates found in test data.')
else:
    print(f'{test_duplicates} duplicates found in test data.')
    test.drop_duplicates()
    print(f'Duplicates removed successfully.')


# Visualizing outliers using boxplot.

fig, axes = plt.subplots(4, 2, figsize=(30, 25))

sns.boxplot(data=train, x='age', ax=axes[0,0]).set_title('Age')

sns.boxplot(data=train, x='balance', ax=axes[0,1]).set_title('Balance')

sns.boxplot(data=train, x='day', ax=axes[1,0]).set_title('Day')

sns.boxplot(data=train, x='duration', ax=axes[1,1]).set_title('Duration')

sns.boxplot(data=train, x='campaign', ax=axes[2,0]).set_title('Campaign')

sns.boxplot(data=train, x='pdays', ax=axes[2,1]).set_title('pdays')

sns.boxplot(data=train, x='previous', ax=axes[3,0]).set_title('Previous')

plt.show()


# Saving the original test IDs before dropping the column
test_ids = test['id']

# Dropping irrelevant column: id

train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# Seperating feature matrix (X) and target (y) in the train data:

X = train.drop('y', axis=1)
y = train['y']


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

cat_cols = ['job','marital','education','default', 'housing', 'loan', 'contact', 'month', 'poutcome']
num_cols = ['age', 'day']
num_cols_robust = ['balance', 'duration', 'campaign', 'pdays', 'previous']

ct_preprocessing = ColumnTransformer([
    ('sscaler', StandardScaler(), num_cols),
    ('rscaler', RobustScaler(), num_cols_robust),
    ('ohe', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')

ct_preprocessing


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, f1_score


models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Linear SVM': LinearSVC(random_state=42, dual=False),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

results = [] # store results for each model
for name, model in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([
        ('preprocessor', ct_preprocessing),
        ('classifier', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    # Check for the appropriate method to get scores for ROC AUC
    if hasattr(pipe, "predict_proba"):
        # Use probability scores for models that support it
        y_scores = pipe.predict_proba(X_test)[:, 1]
    else:
        # Use decision function for models like LinearSVC
        y_scores = pipe.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    ROC_AUC = roc_auc_score(y_test, y_scores)

    # Append to results
    results.append({
        "Model": name,
        "Accuracy": acc,
        "ROC AUC": ROC_AUC
    })

# Converting the results of each each model into a dataframe
results_df = pd.DataFrame(results)

# Sorting with respect to F1 and Accuracy score
results_df = results_df.sort_values(by=['ROC AUC', 'Accuracy'], ascending=[False, False])
print("\n--- Model Comparison ---")
print(results_df)


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from lightgbm import early_stopping
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Splitting the training data again to create a validation set
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

models_and_params = {
    'LightGBM': {
        'model': LGBMClassifier(objective='binary', metric='logloss', random_state=42, verbosity=-1),
        'params': {
            'classifier__n_estimators': randint(200, 1500),
            'classifier__learning_rate': uniform(0.01, 0.2),
            'classifier__num_leaves': randint(30, 200),
            'classifier__max_depth': [-1] + list(range(5, 15)),
            'classifier__subsample': uniform(0.6, 0.4),
            'classifier__colsample_bytree': uniform(0.6, 0.4),
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(loss_function='Logloss', verbose=0, random_state=42),
        'params': {
            'classifier__iterations': randint(200, 1500),
            'classifier__learning_rate': uniform(0.01, 0.2),
            'classifier__depth': randint(4, 12),
            'classifier__l2_leaf_reg': uniform(1, 10),
            'classifier__subsample': uniform(0.7, 0.3),
        }
    },
    'XGBoost': {
        'model': XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42, verbosity=0, early_stopping_rounds=50),
        'params': {
            'classifier__n_estimators': randint(200, 1500),
            'classifier__learning_rate': uniform(0.01, 0.2),
            'classifier__max_depth': randint(4, 12),
            'classifier__subsample': uniform(0.6, 0.4),
            'classifier__colsample_bytree': uniform(0.6, 0.4),
        }
    }
}

best_estimators = {}
for model_name, mp in models_and_params.items():
    print(f"\nRunning RandomizedSearchCV for {model_name}...")

    # Creating the full pipeline
    pipeline = Pipeline([
        ('preprocessor', ct_preprocessing),
        ('classifier', mp['model'])
    ])

    # Setting up the search
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=mp['params'],
        n_iter=5,
        cv=2,
        scoring='roc_auc',
        n_jobs=2,
        verbose=0,
        random_state=42
    )

    # Preprocessing the validation set for early stopping
    # Fitting the preprocessor on the final training data and transforming the validation data.
    X_val_transformed = ct_preprocessing.fit(X_train_final).transform(X_val)

    if model_name == 'LightGBM':
        fit_params = {
            'classifier__callbacks': [early_stopping(stopping_rounds=50, verbose=False)],
            'classifier__eval_set': [(X_val_transformed, y_val)],
            'classifier__eval_metric': 'logloss'
        }
    elif model_name == 'XGBoost':
        # No need of early_stopping_rounds here
        fit_params = {
            'classifier__eval_set': [(X_val_transformed, y_val)],
            'classifier__verbose': False
        }
    else: # CatBoost
        fit_params = {
            'classifier__early_stopping_rounds': 50,
            'classifier__eval_set': [(X_val_transformed, y_val)],
            'classifier__verbose': False
        }

    # Fitting the model on the final training data
    random_search.fit(X_train_final, y_train_final, **fit_params)

    print(f"\nFinished {model_name}.")
    print(f"Best ROC AUC Score (CV): {random_search.best_score_:.4f}")
    print(f"Best Params: {random_search.best_params_}")
    
    # Storing the best estimator (the entire fitted pipeline)
    best_estimators[model_name] = random_search.best_estimator_
    print("-" * 30)


# ---  FINAL EVALUATION ON THE UNSEEN TEST SET ---
print("\n\n\nEvaluating the best models on the untouched test set...")

for model_name, estimator in best_estimators.items():
    print(f"\n--- Results for {model_name} ---")
    
    # The estimator is a pipeline, so it handles preprocessing automatically
    y_pred = estimator.predict(X_test)
    y_pred_proba = estimator.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC on Test Set: {auc_score:.4f}")


# # Model building with best parameters and fitting with the entire train data
# best_model_pipe = best_estimators['LightGBM']
# best_model_pipe.fit(X, y)


# Model building with best parameters and fitting with the entire train data
xgb_pipe = best_estimators['XGBoost']
xgb_pipe.named_steps['classifier'].set_params(early_stopping_rounds=None)

best_estimators['LightGBM'].fit(X,y)
xgb_pipe.fit(X, y)
best_estimators['CatBoost'].fit(X,y)


lgbm_preds = best_estimators['LightGBM'].predict_proba(test)[:, 1]
xgb_preds = best_estimators['XGBoost'].predict_proba(test)[:, 1]
cat_preds = best_estimators['CatBoost'].predict_proba(test)[:, 1]

# Weighted average (giving more weight to the best model)
final_preds = (0.5 * lgbm_preds) + (0.25 * xgb_preds) + (0.25 * cat_preds)


submission = pd.DataFrame({
    'id': test_ids,
    'y': final_preds
})

submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")

