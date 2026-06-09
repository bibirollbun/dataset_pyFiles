!pip install optuna


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score

# Load the dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')

# Display first few rows
display(train_df.head(10))
display("Train Shape", train_df.shape)
display("Test Shape", test_df.shape)

# Describe the data
display(train_df.describe())

# Display information about dtypes and missing values
display("Train Data Info:", train_df.info())

# Check target distribution
display("Target Distribution:", train_df['rainfall'].value_counts(normalize=True))

# Missing values
display("Train Missing Values:", train_df.isnull().sum().sum())
display("Test Missing Values:", test_df.isnull().sum().sum())

plt.figure(figsize=(12, 8))
sns.heatmap(data=train_df.corr(), annot=True, linewidths=0.2)
plt.show()

# Separate features and target
X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']

# Fix missing values in Test and drop columns
test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].mean())
#test_df = test_df.drop(columns=['day'])

# Define the Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        #'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
        'max_features': trial.suggest_int('max_features', 3, 15),
        'n_jobs': -1,
        'random_state': 42
    }

    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=10, scoring='roc_auc', n_jobs=-1).mean()
    return score

# Set up the study
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Get the best model
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train final model
#best_model =  RandomForestClassifier(
#    n_estimators=2366,
#    max_depth=5,
#    min_samples_split=9,
#    min_samples_leaf=8,
#    criterion='gini',
#    class_weight=None,
#    max_features='log2', 
#    n_jobs=-1,
#    random_state=42
#)

# best_model = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)
best_model.fit(X, y)

# Evaluate
y_pred = best_model.predict_proba(X)[:, 1]
print("In-sample AUC:", roc_auc_score(y, y_pred))

# Cross-validated score (more realistic)
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='roc_auc')
print(f"Cross-validated AUC-ROC score: {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")

importances = best_model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.show()

# Calibrate and predict
calibrated_model = CalibratedClassifierCV(best_model, cv=10)
calibrated_model.fit(X, y)

# Predict probabilities
y_pred_proba = calibrated_model.predict_proba(X)[:, 1]

# Cross-validated score (more realistic)
calibrated_cv_scores = cross_val_score(calibrated_model, X, y, cv=10, scoring='roc_auc')
print(f"Cross-validated AUC-ROC score: {calibrated_cv_scores.mean():.4f} Â± {calibrated_cv_scores.std():.4f}")

test_preds = calibrated_model.predict_proba(test_df)[:, 1]

# Prepare submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = test_preds
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")
display(submission)

#Best Hyperparameters: {'n_estimators': 1660, 'max_depth': 5, 'min_samples_split': 15, 'min_samples_leaf': 3, 'max_features': 'log2', 'criterion': 'entropy', 'class_weight': None}
#In-sample AUC: 0.9278136924803592
#Cross-validated AUC-ROC score: 0.8925 Â± 0.0160

