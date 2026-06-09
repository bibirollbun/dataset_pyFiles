import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from optuna import create_study
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

print(train.shape)
train.head()



# Target variable distribution
sns.countplot(data=train, x='Personality')
plt.title('Target Distribution (Introvert vs Extrovert)')
plt.show()

# Missing values
print(train.isnull().sum())

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(train.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()



le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])  # 0 - Extrovert, 1 - Introvert



from sklearn.preprocessing import OrdinalEncoder

# Get list of object (categorical) columns
categorical_cols = train.select_dtypes(include='object').columns.tolist()

# Remove target column from the list if it's there
if 'Personality' in categorical_cols:
    categorical_cols.remove('Personality')

# Apply OrdinalEncoder to all object columns
oe = OrdinalEncoder()

# Fit on train, transform both train and test
train[categorical_cols] = oe.fit_transform(train[categorical_cols])
test[categorical_cols] = oe.transform(test[categorical_cols])



X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
test_ids = test['id']
X_test = test.drop('id', axis=1)



cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)



xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
cat = CatBoostClassifier(verbose=0, random_state=42)
lgb = LGBMClassifier(random_state=42)



from skopt import BayesSearchCV

search_spaces = {
    'n_estimators': (100, 1000),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3, 'log-uniform')
}

bayes_lgb = BayesSearchCV(
    estimator=lgb,
    search_spaces=search_spaces,
    cv=cv,
    n_iter=20,
    scoring='f1',
    random_state=42,
    verbose=0,
    n_jobs=-1
)

bayes_lgb.fit(X, y)
lgb_best = bayes_lgb.best_estimator_



import optuna

def objective(trial):
    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    score = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

xgb_best = XGBClassifier(**study.best_params, eval_metric='logloss', use_label_encoder=False, random_state=42)



voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_best),
        ('cat', cat),
        ('lgb', lgb_best)
    ],
    voting='soft'
)

voting_clf.fit(X, y)



def evaluate_model(model):
    acc = cross_val_score(model, X, y, scoring='accuracy', cv=cv).mean()
    f1 = cross_val_score(model, X, y, scoring='f1', cv=cv).mean()
    roc = cross_val_score(model, X, y, scoring='roc_auc', cv=cv).mean()
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc:.4f}")
    return acc, f1, roc

evaluate_model(voting_clf)



y_pred = voting_clf.predict(X_test)
y_pred_final = le.inverse_transform(y_pred)

submission = pd.DataFrame({'id': test_ids, 'Personality': y_pred_final})
submission.to_csv('submission.csv', index=False)
submission.head()



results = pd.DataFrame(columns=["Model", "Accuracy", "F1 Score", "ROC-AUC"])

for name, model in [("XGBoost", xgb_best), ("CatBoost", cat), ("LightGBM", lgb_best), ("Voting Ensemble", voting_clf)]:
    acc, f1, roc = evaluate_model(model)
    
    # Convert to DataFrame and concat
    new_row = pd.DataFrame([[name, acc, f1, roc]], columns=results.columns)
    results = pd.concat([results, new_row], ignore_index=True)

print(results)



results.set_index('Model').plot(kind='bar', figsize=(10,6), title="Model Comparison Metrics", colormap="Set2")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()





