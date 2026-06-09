import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings


path_train = '/kaggle/input/playground-series-s5e7/train.csv'
path_test = '/kaggle/input/playground-series-s5e7/test.csv'


df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)


df_train.head()


df_train.tail()


df_train.info()


df_train.describe()


# Check for missing values
df_train.isnull().sum()


df_train['Personality'].value_counts(normalize=True)


plt.figure(figsize=(5,4))
sns.countplot(data=df_train, x='Personality', palette='coolwarm')
plt.title("Target Distribution")
plt.show()


# We want to predict the personality so we keep away this column
features = [col for col in df_train.columns if col != 'Personality']

# Separate features columns type
num_features = df_train[features].select_dtypes(include=['int64', 'float64']).columns
cat_features = df_train[features].select_dtypes(include=['object']).columns


cat_features


num_features


plt.figure(figsize=(10,9))
corr = df_train[num_features].corr()
sns.heatmap(corr, annot=False)
plt.title('Nums features correlation map - Heatmap')
plt.show()


num_features


cat_features


# Remove the column ID
df_eda = df_train.drop(columns='id')

num_features = ['Time_spent_Alone', 'Social_event_attendance', 
                'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_features = ['Stage_fear', 'Drained_after_socializing']


import missingno as msno

fig, axes = plt.subplots(
    nrows=len(num_features) + len(cat_features) + 1,
    ncols=2,
    figsize=(10, 4*(len(num_features) + len(cat_features) + 1))
)

#  Numerical distributions (KDE + Box)
for i, col in enumerate(num_features):
    # KDE
    sns.kdeplot(
        data=df_eda, x=col, hue='Personality',
        fill=True, common_norm=False, palette='coolwarm', alpha=0.5,
        ax=axes[i, 0]
    )
    axes[i, 0].set_title(f'Distribution of {col} by Personality')

    # Boxplot
    sns.boxplot(
        data=df_eda, x='Personality', y=col, palette='Set2',
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f'{col} by Personality')

# Categorical countplots
start_cat = len(num_features)
for j, col in enumerate(cat_features):
    sns.countplot(
        data=df_eda, x=col, hue='Personality', palette='pastel',
        ax=axes[start_cat + j, 0]
    )
    axes[start_cat + j, 0].set_title(f'{col} vs Personality')
    axes[start_cat + j, 1].axis('off')  # Empty col for aesthetics

# 3. Missing value matrix in the last row
msno.matrix(df_eda, ax=axes[-1, 0])
axes[-1, 0].set_title('Missing Value Matrix')
axes[-1, 1].axis('off')

# Save before showing
fig.savefig("eda_dashboard.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Remove the ids from the train dataset
train = df_train.drop(columns='id')

# Remove the ids from the test dataset and keep the ids for the submission csv
test = df_test.drop(columns='id')
test_ids = df_test['id']

# Separate featuresand target (We want to train to predict the personality column)
X_train = train.drop(columns='Personality')
y_train = train['Personality']

# Separate types columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Data imputation
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])


encoder = LabelEncoder()

for col in cat_cols + ['Personality']:
    if (col in X_train.columns):
        X_train[col] = encoder.fit_transform(X_train[col])
        test[col] = encoder.transform(test[col])
    elif col == 'Personnality':
        y_train = encoder.fit_transform(y_train)


print(f'Training shape {X_train.shape}')
print(f'Training shape {y_train.shape}')
print(f'Training shape {test.shape}')


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

x_tr, x_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                            random_state=42, stratify=y_train)


# Convert each y into 0/1
# Map "Introvert" to 0 and "Extrovert" to 1
y_tr = y_tr.map({"Introvert": 0, "Extrovert": 1})
y_val = y_val.map({"Introvert": 0, "Extrovert": 1})


import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, KFold


def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
    }
    
    model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    score = cross_val_score(model, x_tr, y_tr, n_jobs=-1, cv=KFold(n_splits=5), scoring='roc_auc').mean()
    return score


def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    
    model = CatBoostClassifier(**params, verbose=0, random_state=42)
    score = cross_val_score(model, x_tr, y_tr, n_jobs=-1, cv=KFold(n_splits=5), scoring='roc_auc').mean()
    return score


def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    model = LGBMClassifier(**params, random_state=42)
    score = cross_val_score(model, x_tr, y_tr, n_jobs=-1, cv=KFold(n_splits=5), scoring='roc_auc').mean()
    return score


# Example for XGBoost
study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler())
study_xgb.optimize(objective_xgb, n_trials=50)
best_params_xgb = study_xgb.best_params
print("Best XGBoost params:", best_params_xgb)


# Repeat for CatBoost and LightGBM
study_catboost = optuna.create_study(direction='maximize', sampler=TPESampler())
study_catboost.optimize(objective_catboost, n_trials=50)
best_params_catboost = study_catboost.best_params
print("Best CatBoost params:", best_params_catboost)


study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler())
study_lgb.optimize(objective_lgb, n_trials=50)
best_params_lgb = study_lgb.best_params
print("Best LightGBM params:", best_params_lgb)


# XGBoost
final_xgb = XGBClassifier(**best_params_xgb, random_state=42, use_label_encoder=False, eval_metric='logloss')
final_xgb.fit(x_tr, y_tr)

# CatBoost
final_catboost = CatBoostClassifier(**best_params_catboost, verbose=0, random_state=42)
final_catboost.fit(x_tr, y_tr)

# LightGBM
final_lgb = LGBMClassifier(**best_params_lgb, random_state=42)
final_lgb.fit(x_tr, y_tr)


for name, model in [('XGBoost', final_xgb), ('CatBoost', final_catboost), ('LightGBM', final_lgb)]:
    y_pred = model.predict(x_val)
    y_proba = model.predict_proba(x_val)[:, 1]
    print(f"{name} Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"{name} ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")


from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('xgb', final_xgb),
        ('catboost', final_catboost),
        ('lgb', final_lgb)
    ],
    voting='soft'  # Use 'hard' for class label voting
)

ensemble.fit(x_tr, y_tr)

y_pred = ensemble.predict(x_val)
print(f"{name} Accuracy: {accuracy_score(y_val, y_pred):.4f}")


# Predict on the final test dataset
tests_pred = ensemble.predict(test)


# Save submission
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': tests_pred
})

submission['Personality'] = submission['Personality'].map({0: "Introvert", 1: "Extrovert"})


submission_path = 'final_pred.csv'
submission.to_csv(submission_path, index=False)
print(f'Submission saved to : {submission_path}')

