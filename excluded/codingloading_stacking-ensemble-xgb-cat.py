!pip install optuna


#  Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna


# Warnings
import warnings
warnings.filterwarnings("ignore")

#  Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')



print("Original train shape:", train.shape)
print("New data shape:", original.shape)

print("\nColumns in original train:")
print(train.columns.tolist())

print("\nColumns in new data:")
print(original.columns.tolist())



# Drop 'id' from original train before combining
train_no_id = train.drop(columns=['id'])

# Combine datasets
train = pd.concat([train, original], axis=0).reset_index(drop=True)

print("âœ… Combined train shape:", train.shape)



# Check missing values
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing



# Data types and basic stats
train.describe()



sns.countplot(x='Personality', data=train.replace({0: 'Introvert', 1: 'Extrovert'}), palette="Set2")
plt.title("Target Distribution")
plt.xlabel("Personality Type")
plt.ylabel("Count")
for p in plt.gca().patches:
    plt.text(p.get_x() + 0.3, p.get_height() + 2, int(p.get_height()), fontsize=12)
plt.show()




# KDE Plots for Numerical Features
fig, axes = plt.subplots(2, 3, figsize=(25, 15))  # 2 rows, 3 columns
axes = axes.flatten()  # Makes it easier to iterate

num_cols = ['Time_spent_Alone', 'Social_event_attendance', 
            'Going_outside', 'Friends_circle_size', 'Post_frequency']

for i, col in enumerate(num_cols):
    sns.kdeplot(data=train, x=col, hue='Personality', palette='coolwarm', fill=True, ax=axes[i])
    axes[i].set_title(f'{col} Distribution by Personality', fontsize=18)
    axes[i].set_xlabel(col, fontsize=20)
    axes[i].set_ylabel('Density', fontsize=20)
    axes[i].tick_params(axis='both', labelsize=18)
    axes[i].legend(title='Personality', fontsize=18, title_fontsize=16)

# Remove the empty last subplot if num_cols < grid size
if len(num_cols) < len(axes):
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])  # delete unused axes

plt.tight_layout()
plt.show()



#  Combined Stacked Bar Plots for Binary Features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

binary_features = ['Stage_fear', 'Drained_after_socializing']

for i, col in enumerate(binary_features):
    ct = pd.crosstab(train[col], train['Personality'], normalize='index')
    ct.plot(kind='bar', stacked=True, colormap='coolwarm', ax=axes[i])
    axes[i].set_title(f'Personality by {col}', fontsize=16)
    axes[i].set_ylabel('Proportion', fontsize=14)
    axes[i].tick_params(axis='both', labelsize=12)
    axes[i].legend(title='Personality', fontsize=12, title_fontsize=12)

plt.tight_layout()
plt.show()



# Create numeric versions of binary columns
corr_df = train[num_cols].copy()
corr_df['Stage_fear'] = train['Stage_fear'].map({'No': 0, 'Yes': 1})
corr_df['Drained_after_socializing'] = train['Drained_after_socializing'].map({'No': 0, 'Yes': 1})

# plot the heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()




# Create a copy with binary features mapped to 0/1
train_for_summary = train.copy()
train_for_summary['Stage_fear'] = train_for_summary['Stage_fear'].map({'No': 0, 'Yes': 1})
train_for_summary['Drained_after_socializing'] = train_for_summary['Drained_after_socializing'].map({'No': 0, 'Yes': 1})

# Now it's safe to take the mean across numeric columns
train_summary = train_for_summary.groupby('Personality')[num_cols + binary_features].mean().T
display(train_summary.style.background_gradient(cmap='coolwarm'))




from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

importances = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean})
importances.sort_values(by='Importance', ascending=True).plot.barh(x='Feature', figsize=(8,6), legend=False)
plt.title("Permutation Feature Importance (Random Forest)")
plt.show()



from sklearn.preprocessing import LabelEncoder

# Encode binary features
binary_map = {'Yes': 1, 'No': 0}
for col in ['Stage_fear', 'Drained_after_socializing']:
    train[col] = train[col].map(binary_map)
    test[col] = test[col].map(binary_map)

# Fill missing values with median (numeric only)
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 
            'Going_outside', 'Friends_circle_size', 'Post_frequency']

all_features = num_cols + ['Stage_fear', 'Drained_after_socializing']

for col in all_features:
    median_val = train[col].median()
    train[col].fillna(median_val, inplace=True)
    test[col].fillna(median_val, inplace=True)

# Encode target
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])  # 0=Extrovert, 1=Introvert



X = train[all_features]
y = train['Personality']
X_test = test[all_features]



import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def objective_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',  # GPU usage
        'device': 'cuda',
        'random_state': 42,
        'use_label_encoder': False,

        # Hyperparameter search space
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.08),
        'max_leaves': trial.suggest_int('max_leaves', 25, 35),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.005, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 5000, 10000),

        'subsample': trial.suggest_float('subsample', 0.85, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.85),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.55, 0.65),

        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1e-2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 2.0),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )

        preds = model.predict(X_val)
        scores.append(accuracy_score(y_val, preds))

    return np.mean(scores)

#  Start tuning
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=80, show_progress_bar=True)

#  Show results
print("\nâœ… Best XGBoost Params:")
print(study_xgb.best_params)
print("ğŸ�† Best CV Accuracy:", study_xgb.best_value)



from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import optuna

def objective_cat(trial):
    params = {
        'depth': trial.suggest_int('depth', 7, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.10, 0.14),
        'iterations': trial.suggest_int('iterations', 5000, 10050),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 70),
        'random_state': 42,
        'verbose': 0,
        'task_type': 'GPU',   # Use GPU
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy'
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=(X_val, y_val),
                  early_stopping_rounds=100,
                  verbose=False)

        preds = model.predict(X_val)
        scores.append(accuracy_score(y_val, preds))

    return np.mean(scores)

#  Run the optimization
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=True)

# Best parameters
print("\nâœ… Best CatBoost Params:")
print(study_cat.best_params)
print("ğŸ�† Best CV Accuracy:", study_cat.best_value)



from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def get_oof_predictions(model_cls, model_params, X, y, X_test):
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = model_cls(**model_params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])
        test_preds += model.predict(X_test) / skf.n_splits

    return oof_preds, test_preds


#  Patch verbose settings
xgb_best_params = study_xgb.best_params.copy()
xgb_best_params['verbosity'] = 0  # Turn off XGBoost output

cat_best_params = study_cat.best_params.copy()
cat_best_params['verbose'] = 0   # Turn off CatBoost output

# OOF predictions
xgb_oof, xgb_test = get_oof_predictions(XGBClassifier, xgb_best_params, X, y, X_test)
cat_oof, cat_test = get_oof_predictions(CatBoostClassifier, cat_best_params, X, y, X_test)

#  Stack
stack_X = pd.DataFrame({
    'xgb': xgb_oof,
    'cat': cat_oof
})
stack_test = pd.DataFrame({
    'xgb': xgb_test,
    'cat': cat_test
})

#  Logistic Regression Meta Learner
meta_model = LogisticRegression()
meta_model.fit(stack_X, y)
final_preds = meta_model.predict(stack_test)

#  Meta-model accuracy (on training stack)
stack_train_accuracy = accuracy_score(y, meta_model.predict(stack_X))
print(f"\nğŸ�¯ Meta-model (Logistic Regression) Accuracy on OOF: {stack_train_accuracy:.5f}")



from sklearn.metrics import accuracy_score

# Get predictions on OOF stacking data
stack_oof_preds = meta_model.predict(stack_X)

# Calculate accuracy
stacking_accuracy = accuracy_score(y, stack_oof_preds)
print(f"ğŸ”� Stacking Model (Logistic Regression) OOF Accuracy: {stacking_accuracy:.4f}")



submission['Personality'] = le.inverse_transform(final_preds.astype(int))
submission.to_csv("submission.csv", index=False)
submission.head()





