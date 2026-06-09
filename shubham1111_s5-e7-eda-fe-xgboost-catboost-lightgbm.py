import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

import optuna
import gc

import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
OPTUNA_N_TRIALS = 50


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_features = ['Stage_fear', 'Drained_after_socializing']
target_variable = 'Personality'


plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x=target_variable, palette='viridis') # Changed palette
plt.title('How Are Personalities Distributed?', fontsize=16)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Number of Individuals', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(18, 12))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(train_df[feature].dropna(), kde=True, bins=25, color='#4CAF50', edgecolor='black') # Changed color
    plt.title(f'Spread of {feature}', fontsize=14)
    plt.xlabel(feature, fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 12))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=train_df[feature], color='#FFD700', width=0.5) # Changed color
    plt.title(f'Whispers from {feature}: Outliers and Spread', fontsize=14)
    plt.ylabel(feature, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
for i, feature in enumerate(categorical_features):
    plt.subplot(1, 2, i + 1)
    sns.countplot(data=train_df, x=feature, palette='coolwarm') # Changed palette
    plt.title(f'Categorical Counts: {feature}', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 12))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.violinplot(data=train_df, x=target_variable, y=feature, palette='plasma') # Changed palette
    plt.title(f'{feature} by Personality Type', fontsize=14)
    plt.xlabel('Personality Type', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
for i, feature in enumerate(categorical_features):
    plt.subplot(1, 2, i + 1)
    prop_df = train_df.groupby([feature, target_variable]).size().unstack(fill_value=0)
    prop_df = prop_df.apply(lambda x: x / x.sum(), axis=1)
    prop_df.plot(kind='bar', stacked=True, colormap='cividis', ax=plt.gca(), edgecolor='black')
    plt.title(f'Personality Breakdown by {feature}', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Personality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
correlation_matrix = train_df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('The Interplay: Correlation Among Numerical Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()


sns.pairplot(train_df.dropna(subset=numerical_features + [target_variable]),
             hue=target_variable, palette='tab20',
             vars=numerical_features)
plt.suptitle('Seeing Connections: Pairwise Relationships by Personality', y=1.02, fontsize=18)
plt.show()


plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
sns.violinplot(data=train_df, x='Drained_after_socializing', y='Time_spent_Alone', palette='crest') # Changed palette
plt.title('Time Spent Alone by Social Draining', fontsize=10)
plt.xlabel('Drained After Socializing', fontsize=8)
plt.ylabel('Time Spent Alone', fontsize=8)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.subplot(1, 2, 2)
sns.violinplot(data=train_df, x='Stage_fear', y='Friends_circle_size', palette='flare') # Changed palette
plt.title('Friend Circle Size by Stage Fear', fontsize=14)
plt.xlabel('Stage Fear', fontsize=12)
plt.ylabel('Friends Circle Size', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show() 


plt.figure(figsize=(10, 8))
sns.scatterplot(data=train_df.dropna(subset=['Time_spent_Alone', 'Friends_circle_size', target_variable]),
                x='Time_spent_Alone', y='Friends_circle_size', hue=target_variable, palette='deep', alpha=0.7)
plt.title('Time Alone vs. Friend Circle Size by Personality', fontsize=16)
plt.xlabel('Time Spent Alone', fontsize=12)
plt.ylabel('Friends Circle Size', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Personality')
plt.show()


le = LabelEncoder()
train_df_encoded = train_df.copy()
train_df_encoded[target_variable + '_encoded'] = le.fit_transform(train_df_encoded[target_variable])

correlations_with_target = train_df_encoded[numerical_features + [target_variable + '_encoded']].corr()[target_variable + '_encoded'].drop(target_variable + '_encoded').sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=correlations_with_target.index, y=correlations_with_target.values, palette='RdPu')
plt.title('Numerical Feature Correlation with Personality (Encoded)', fontsize=16)
plt.xlabel('Numerical Feature', fontsize=12)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


original_train_target = train_df[target_variable]
features_train_df = train_df.drop(target_variable, axis=1)
features_test_df = test_df.copy()


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ratio_median_ = ((X['Post_frequency']) / (X['Time_spent_Alone'] + X['Going_outside'] + 1e-6)).replace([np.inf, -np.inf], np.nan).median()
        return self

    def transform(self, X):
        X = X.copy()
        X['Social_Score'] = X[['Social_event_attendance', 'Going_outside', 'Friends_circle_size']].sum(axis=1)
        X['Alone_Score'] = X['Time_spent_Alone']
        X['Friends_vs_Alone'] = X['Friends_circle_size'] / (X['Time_spent_Alone'] + 1e-6)
       
        stage_fear_encoded = LabelEncoder().fit_transform(X['Stage_fear'])
        drained_encoded = LabelEncoder().fit_transform(X['Drained_after_socializing'])
        X['Combined_Fear_Drained'] = stage_fear_encoded * drained_encoded
        return X

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('scaler', StandardScaler())
    ]), numerical_features + ['Social_Score', 'Alone_Score', 'Friends_vs_Alone']),
    ('cat', Pipeline([
        ('label_enc', FunctionTransformer(lambda x: x.apply(LabelEncoder().fit_transform)))
    ]), categorical_features)
])

pipeline = Pipeline([
    ('feature_eng', FeatureEngineer()),
    ('preprocessor', preprocessor)
])

X_processed = pipeline.fit_transform(features_train_df)
X_test_processed = pipeline.transform(features_test_df)

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(original_train_target)


def optimize_model(model_class, param_suggester, study_name):
    def objective(trial):
        params = param_suggester(trial)
        model = model_class(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_processed, y_encoded):
            X_train, X_val = X_processed[train_idx], X_processed[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
            model.fit(X_train, y_train)
            preds = (model.predict_proba(X_val)[:, 1] > 0.5).astype(int)
            scores.append(accuracy_score(y_val, preds))
        return np.mean(scores)
    study = optuna.create_study(direction='maximize', study_name=study_name)
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)
    return study.best_trial.params


def suggest_xgb(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1),
        'reg_lambda': trial.suggest_float('lambda', 1e-8, 2.0, log=True),
        'reg_alpha': trial.suggest_float('alpha', 1e-8, 2.0, log=True),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0
    }


def suggest_lgb(trial):
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 2.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 2.0, log=True),
        'random_state': 42,
        'verbose': -1
    }


def suggest_cat(trial):
    return {
        'objective': 'Logloss',
        'eval_metric': 'Accuracy',
        'iterations': trial.suggest_int('iterations', 100, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 3, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-6, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
        'rsm': trial.suggest_float('rsm', 0.2, 1.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'random_seed': 42,
        'verbose': 0
    }


best_params_xgb = optimize_model(XGBClassifier, suggest_xgb, 'XGBoost_Optimization')
best_params_lgb = optimize_model(LGBMClassifier, suggest_lgb, 'LightGBM_Optimization')
best_params_cat = optimize_model(CatBoostClassifier, suggest_cat, 'CatBoost_Optimization')
best_params_cat['verbose'] = False


def cross_val_predict_save(model_class, model_name, best_params):
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_preds_folds = []
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed, y_encoded)):
        print(f"--- {model_name} FOLD {fold+1}/{n_splits} ---")
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        model = model_class(**best_params)
        model.fit(X_train, y_train)
        
        val_preds = (model.predict_proba(X_val)[:, 1] > 0.5).astype(int)
        fold_accuracies.append(accuracy_score(y_val, val_preds))
        test_pred_probs = model.predict_proba(X_test_processed)[:, 1]
        test_preds_folds.append(test_pred_probs)
        gc.collect()

    print(f"{model_name} Mean CV Accuracy: {np.mean(fold_accuracies):.4f}")
    test_preds_avg = np.mean(test_preds_folds, axis=0)
    final_test_predictions = (test_preds_avg > 0.5).astype(int)
    final_labels = target_encoder.inverse_transform(final_test_predictions)
    submission_df['Personality'] = final_labels
    submission_df.to_csv(f'submission_{model_name}.csv', index=False)
    print(submission_df.head())


cross_val_predict_save(XGBClassifier, 'XGBoost', best_params_xgb)
cross_val_predict_save(LGBMClassifier, 'LightGBM', best_params_lgb)
cross_val_predict_save(CatBoostClassifier, 'CatBoost', best_params_cat)




