import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df_train = df_train.drop('id', axis=1)
df_test = df_test.drop('id', axis=1)


le_encoder = LabelEncoder()
df_train['Personality_encoded'] = le_encoder.fit_transform(df_train['Personality'])


fear_encoder = LabelEncoder()
drained_encoder = LabelEncoder()

df_train['Stage_fear'] = fear_encoder.fit_transform(df_train['Stage_fear'])
df_train['Drained_after_socializing'] = drained_encoder.fit_transform(df_train['Drained_after_socializing'])

df_test['Stage_fear'] = fear_encoder.transform(df_test['Stage_fear'])
df_test['Drained_after_socializing'] = drained_encoder.transform(df_test['Drained_after_socializing'])


print(f'Train Shape: {df_train.shape}')
print(f'Test Shape: {df_test.shape}')


train_columns = df_train.columns
train_columns


df_train.info()


df_train.head()


def nan_info(df):
    df_total_rows = len(df)
    df_nan_count = df.isna().sum()
    df_nan_percentage = (df_nan_count / df_total_rows) * 100
    df_nan_info = pd.DataFrame({
        'nan_count': df_nan_count,
        'nan_percentage': df_nan_percentage,
        'below < 5%': df_nan_percentage < 5,
    }).sort_values(by='nan_percentage', ascending=True)

    return df_nan_info


nan_info(df_train)


def inf_info(df):
    df_number = df.select_dtypes(include=[np.number])

    return np.isinf(df_number).sum()


inf_info(df_train)


num_cols = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

df_train[num_cols] = df_train[num_cols].replace([np.inf, -np.inf], np.nan)

n_cols = 3
n_rows = int(np.ceil(len(num_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))

axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.kdeplot(
        df_train[col].dropna(),
        ax=axes[i],
        fill=True,
        alpha=0.3
    )
    axes[i].set_title(f'KDE Plot: {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


plt.figure(figsize=(14,4))
sns.pairplot(df_train.drop('Personality_encoded', axis=1), hue="Personality")
plt.tight_layout()
plt.show()


corr_matrix = df_train.drop(labels=['Personality'], axis=1).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()


imputer = IterativeImputer(random_state=42)
df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
df_test[num_cols] = imputer.transform(df_test[num_cols])


cat_cols = ['Stage_fear', 'Drained_after_socializing']

modus_cat = {}

for col in cat_cols:
    modus_value = df_train[col].mode().iloc[0]
    df_train[col] = df_train[col].fillna(modus_value)
    df_test[col] = df_test[col].fillna(modus_value)


n_cols = 3
n_rows = int(np.ceil(len(num_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))

axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.kdeplot(
        df_train[col].dropna(),
        ax=axes[i],
        fill=True,
        alpha=0.3
    )
    axes[i].set_title(f'KDE Plot: {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


df_train.drop(labels=['Personality'], axis=1).skew()


cols = ['Stage_fear', 'Time_spent_Alone', 'Drained_after_socializing']

boxcox_lambdas = {}

for col in cols:
    df_train[col], fitted_lambda = boxcox(df_train[col] + 1)
    boxcox_lambdas[col] = fitted_lambda

for col in cols:
    df_test[col] = boxcox(df_test[col] + 1, lmbda=boxcox_lambdas[col])


df_train.drop(labels=['Personality'], axis=1).skew()


plt.figure(figsize=(14,4))
sns.pairplot(df_train.drop('Personality_encoded', axis=1), hue="Personality")
plt.tight_layout()
plt.show()


corr_matrix = df_train.drop(labels=['Personality'], axis=1).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()


features = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
    'Post_frequency'
]
X = df_train[features]
y = df_train['Personality_encoded']


# create adaboost objective for hyperparameter tunning
def objective_adaboost(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0)

    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    return np.mean(scores)


# create catboost objective for hyperparameter tuning (CPU only)
def objective_catboost(trial):
    depth = trial.suggest_int('depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0)
    iterations = trial.suggest_int('iterations', 100, 1000)

    model = cb.CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        verbose=0,
        random_state=42
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    return np.mean(scores)


# create lightgbm objective for hyperparameter tuning (CPU only)
def objective_lightgbm(trial):
    num_leaves = trial.suggest_int('num_leaves', 20, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 500)

    model = lgb.LGBMClassifier(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        verbosity=-1,
        random_state=42
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    return np.mean(scores)


# create randomforest objective for hyperparameter tunning
def objective_randomforest(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    return np.mean(scores)


# create XGBoost objective for hyperparameter tuning (CPU only)
def objective_xgboost(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        random_state=42
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    return np.mean(scores)


# run optuna hyperparameter tunning
# tuning AdaBoost
study_adb = optuna.create_study(direction='maximize')
study_adb.optimize(objective_adaboost, n_trials=30)
best_params_adb = study_adb.best_params

# tuning CatBoost
study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(objective_catboost, n_trials=30)
best_params_cb = study_cb.best_params

# tuning LightGBM
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lightgbm, n_trials=30)
best_params_lgb = study_lgb.best_params

# tuning RandomForest
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_randomforest, n_trials=30)
best_params_rf = study_rf.best_params

# tuning XGBoost
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgboost, n_trials=30)
best_params_xgb = study_xgb.best_params


# get a stacking ensemble of models
def get_stacking():
    level0 = list()
    level0.append(('rf', RandomForestClassifier(**best_params_rf, random_state=42)))
    level0.append(('adb', AdaBoostClassifier(**best_params_adb, random_state=42)))
    level0.append(('cb', cb.CatBoostClassifier(**best_params_cb, verbose=0, random_state=42))) 
    level0.append(('lgb', lgb.LGBMClassifier(**best_params_lgb, verbose=-1, random_state=42))) 
    level0.append(('xgb', xgb.XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42))) 

    level1 = LogisticRegression(random_state=42)
    stacked_model = StackingClassifier(
        estimators=level0,
        final_estimator=level1,
        cv=5
    )
    return stacked_model

def get_voting():
    voting_model = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(**best_params_rf, random_state=42)),
            ('adb', AdaBoostClassifier(**best_params_adb, random_state=42)),
            ('cb', cb.CatBoostClassifier(**best_params_cb, verbose=0, random_state=42)),
            ('lgb', lgb.LGBMClassifier(**best_params_lgb, verbose=-1, random_state=42)),
            ('xgb', xgb.XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42))
        ],
        voting='soft',
        n_jobs=-1
    )
    return voting_model

# get list of models to evaluate
def get_models():
    models = dict()
    models['rf'] = RandomForestClassifier(**best_params_rf, random_state=42)
    models['adb'] = AdaBoostClassifier(**best_params_adb, random_state=42)
    models['cb'] = cb.CatBoostClassifier(**best_params_cb, verbose=0, random_state=42) 
    models['lgb'] = lgb.LGBMClassifier(**best_params_lgb, verbose=-1, random_state=42)  
    models['xgb'] = xgb.XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42)  
    models['stacking'] = get_stacking()
    models['voting'] = get_voting()
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    return scores


# get the models to evaluate
models = get_models()

# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.title('Model Performance Comparison (Accuracy)') # Added a title
plt.ylabel('Accuracy Score') # Added y-label
plt.grid(axis='y', linestyle='--', alpha=0.7) # Added grid
plt.show()


stacking_model = get_stacking()
stacking_model.fit(X, y)


voting_model = get_voting()
voting_model.fit(X, y)


X_test = df_test[features]


test_preds_encoded = voting_model.predict(X_test)

test_preds_label = le_encoder.inverse_transform(test_preds_encoded)


df_submission = df_sample_submission.copy()
df_submission['Personality'] = test_preds_label

df_submission.to_csv('submission.csv', index=False)
print('Submission saved!')

