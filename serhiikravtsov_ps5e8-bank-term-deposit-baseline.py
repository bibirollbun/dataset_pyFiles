import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score

import optuna 
from optuna.samplers import TPESampler


from warnings import filterwarnings
filterwarnings("ignore")


KAGGLE = True
VISUALIZE = False
OPTUNA = False
SEED = 36
if KAGGLE:
    train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
else:
    train = pd.read_csv('data/train.csv', index_col=0)
    test = pd.read_csv('data/test.csv', index_col=0)
    submission = pd.read_csv('data/sample_submission.csv')



train.head()


train.info()


train.describe().T


def plot_train_vs_test(train_df, test_df, features):
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.histplot(train_df[feature], kde=True, color='blue', label='Train', stat='density', alpha=0.6)
        if feature in test_df.columns:
            sns.histplot(test_df[feature], kde=True, color='orange', label='Test', stat='density', alpha=0.6)
        plt.title(f'Distribution of {feature} (Train vs Test)')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.show()


# visualize categorical features distribution
def plot_categorical_distribution(train_df, test_df, features):
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=train_df[feature], color='blue', label='Train', alpha=0.6)
        if feature in test_df.columns:
            sns.countplot(x=test_df[feature], color='orange', label='Test', alpha=0.6)
        plt.title(f'Distribution of {feature} (Train vs Test)')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend()
        plt.show()


def create_new_features(train, test):
    """Create new engineered features for train and test datasets."""
    
    # --- Encode categorical variables ---
    le = LabelEncoder()
    train['month'] = le.fit_transform(train['month'])
    test['month'] = le.transform(test['month'])

    for df in [train, test]:
        # --- Cyclical month encoding ---
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # --- Day & previous contacts ---
        df['day_log'] = np.log1p(df['day'])
        df['is_previous'] = (df['previous'] > 0).astype(int)
        df['previous_log'] = np.log1p(df['previous'] - df['previous'].min() + 1)

        # --- Duration, campaign, and pdays transformations ---
        df['duration_log'] = np.log1p(df['duration'] - df['duration'].min() + 1)
        df['campaign_log'] = np.log1p(df['campaign'] - df['campaign'].min() + 1)
        df['pdays_log'] = np.log1p(df['pdays'] - df['pdays'].min() + 1)
        
        df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
        df['duration_squared'] = df['duration'] ** 2
        df['duration_sqrt'] = np.sqrt(df['duration'])

        # --- Balance related features ---
        df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
        df['is_positive_balance'] = (df['balance'] > 0).astype(int)
        df['is_negative_balance'] = (df['balance'] < 0).astype(int)
        df['balance_bins'] = pd.qcut(df['balance'], q=10, labels=False, duplicates='drop')

        df['balance_to_age'] = df['balance'] / df['age']
        df['balance_x_age'] = df['balance'] * df['age']

        # --- Pdays related features ---
        df['has_previous_contact'] = (df['previous'] > 0).astype(int)
        df['pdays_active'] = (df['pdays'] != -1).astype(int)
        df['pdays_bins'] = df['pdays'].replace(-1, 999)
        df['pdays_bins'] = pd.cut(
            df['pdays_bins'], 
            bins=[-1, 0, 100, 200, 300, 400, 1000], 
            labels=False
        )

        # --- Campaign interaction ---
        df['campaign_intensity'] = df['campaign'] * df['duration']
        df['multiple_campaigns'] = (df['campaign'] > 1).astype(int)
        df['prev_to_campaign_ratio'] = df['previous'] / (df['campaign'] + 1)

        # --- Age related features ---
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 25, 35, 45, 55, 65, 100], 
            labels=['young', 'early_adult', 'mid_adult', 'mature', 'senior', 'elderly']
        )
        df['age_squared'] = df['age'] ** 2
        df['duration_to_age_ratio'] = df['duration'] / df['age']

        # --- Loan & housing combinations ---
        df['has_loan_and_housing'] = ((df['loan'] == 'yes') & (df['housing'] == 'yes')).astype(int)
        df['no_loan_no_housing'] = ((df['loan'] == 'no') & (df['housing'] == 'no')).astype(int)

        # --- Combined categorical features ---
        df['job_education'] = df['job'].astype(str) + '_' + df['education'].astype(str)

        # --- Previous outcomes ---
        df['prev_success_indicator'] = (
            (df['poutcome'] == 'success') |
            ((df['poutcome'] == 'other') & (df['previous'] > 2))
        ).astype(int)

        # --- Cleanup ---
        df.drop(columns=['month'], inplace=True)

    return train, test




train, test = create_new_features(train, test)



numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train.select_dtypes(exclude=[np.number]).columns.tolist()


if VISUALIZE:
    plot_train_vs_test(train, test, numeric_features)

    if categorical_features:
        print(f'Categorical features found: {categorical_features}')
    else:
        print('No categorical features found.')
    plot_categorical_distribution(train, test, categorical_features)


# convert categorical features to numeric using dummy encoding
train = pd.get_dummies(train, columns=categorical_features, drop_first=True)
test = pd.get_dummies(test, columns=categorical_features, drop_first=True)


# define features and target variable
X, y = train.drop(columns=['y']), train['y']


X, X_test = X.align(test, join='left', axis=1, fill_value=0)


def oof_cross_validation(model, X, y, n_splits=3):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_preds_proba = np.zeros(X.shape[0])
    oof_preds = np.zeros(X.shape[0])

    for train_index, valid_index in kf.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        # Define early stopping callback
        early_stopping_cb = lgb.early_stopping(stopping_rounds=50, verbose=False)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[early_stopping_cb]
        )           
        oof_preds_proba[valid_index] = model.predict_proba(X_valid)[:, 1]
        oof_preds[valid_index] = (oof_preds_proba[valid_index] > 0.5).astype(int)
    
    auc = roc_auc_score(y, oof_preds_proba)
    f1 = f1_score(y, oof_preds)

    return oof_preds_proba, oof_preds, auc, f1



def objective(trial):
    params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 1000,                
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),           
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),  
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),   
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),  
            "max_bin": trial.suggest_int("max_bin", 128, 511),
            "random_state": SEED,
            "n_jobs": -1,
            # handle imbalance: search around the true ratio
            # "scale_pos_weight": 7.288391830961012 * trial.suggest_float("spw_mult", 0.5, 1.5),
        }

    model = lgb.LGBMClassifier(**params)
    _, _, auc, f1 = oof_cross_validation(model, X, y)

    return auc


def optimize_hyperparameters(n_trials=50):
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study, trial.params



if OPTUNA:
    study, best_params = optimize_hyperparameters(n_trials=50)


final_params = {
    'objective': 'binary',
 'boosting_type': 'gbdt',
 'n_estimators': 1000,
 'learning_rate': 0.032584470642397445,
 'num_leaves': 187,
 'max_depth': 11,
 'min_child_samples': 180,
 'min_child_weight': 0.004740625576818533,
 'subsample': 0.9420901948196534,
 'subsample_freq': 7,
 'colsample_bytree': 0.7929434838382624,
 'reg_alpha': 4.575307338396334,
 'reg_lambda': 6.592702435020089e-08,
 'max_bin': 506,
 'random_state': 36,
 'n_jobs': -1, 
'force_col_wise':True, 
'verbose': -1
               }

if OPTUNA:
    final_params.update(best_params)
    print(final_params)



# Train the final model with the best hyperparameters
final_model = lgb.LGBMClassifier(**final_params, metric='auc')
oof_preds_proba, oof_preds, auc, f1 = oof_cross_validation(final_model, X, y)


auc, f1



# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define early stopping callback
early_stopping_cb = lgb.early_stopping(stopping_rounds=50, verbose=True)

final_model.fit(
    X, y,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping_cb]
)


# Make predictions on the test set
test_preds_proba = final_model.predict_proba(X_test)[:, 1]


submission['y'] = test_preds_proba
submission.to_csv('submission.csv', index=False)


submission

