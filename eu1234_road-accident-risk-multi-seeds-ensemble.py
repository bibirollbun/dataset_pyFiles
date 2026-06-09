import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error

# Use gpu if available
from catboost.utils import get_gpu_device_count
device = 'gpu' if get_gpu_device_count() > 0 else 'cpu'

pd.set_option('display.max_colwidth', 100)
warnings.simplefilter(action='ignore', category=FutureWarning)


# Read data
base_dir = Path('/kaggle/input/playground-series-s5e10')
train_df = pd.read_csv(base_dir / 'train.csv').drop(columns='id')
X_test = pd.read_csv(base_dir / 'test.csv').drop(columns='id')
submission_df = pd.read_csv(base_dir / 'sample_submission.csv')

# Drop duplicates
train_df = train_df.drop_duplicates()


train_df.head(5)


train_df.info()


train_df.describe().T


### Plotting functions
def plot_feats_corr(df):
    '''
    Plot the correlation between the features
    '''
    corr = df.corr(numeric_only=True)
    top_right_mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        corr,
        # mask=(corr<0.9),
        # mask=top_right_mask,
        linewidth=0.5,
        linecolor='white',
        annot=True,
        annot_kws={'size': 6},
        cmap='crest',
    )
    plt.title('Correlation between features')
    plt.tight_layout()
    plt.show()

def scatterplot_all(df, target):
    '''
    Scatterplot all the features outliers against the target variable
    '''
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
        return df[df[column].lt(lower_bound) | df[column].gt(upper_bound)]
        
    features = df.drop(columns=target).select_dtypes('number').columns
    num_cols = 4
    num_rows = (len(features) + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 4, num_rows * 4)) 

    for i, feature in enumerate(features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.scatterplot(data=df, x=feature, y=target, 
                        color='blue', label='Data', alpha=0.1)
        
        outliers = detect_outliers(df, feature)
        sns.scatterplot(data=outliers, x=feature, y=target, 
                        color='red', label='Outliers', alpha=0.1)
        
        plt.title(f'{feature} vs {target}')
        plt.xlabel(feature)
        plt.ylabel(target)

        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def boxplot_all(df):
    '''
    Boxplot all the features outliers
    '''
    features = df.select_dtypes('number').columns
    num_cols = 3
    num_rows = (len(features) + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 5, num_rows * 1))

    for i, feature in enumerate(features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(data=df, x=feature, color='blue')
        plt.title(f'{feature} outliers')
        plt.xlabel(feature)

    plt.tight_layout()
    plt.show()

def histplot_all(df):
    '''
    Plot values' distributions
    '''
    features = df.select_dtypes('number').columns
    num_cols = 4
    num_rows = (len(features) + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 4, num_rows * 3))

    for i, feature in enumerate(features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(df, x=feature, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        
    plt.tight_layout()
    plt.show()

def countplot_all(df):
    '''
    Plot categorical values distributions
    '''
    features = df.select_dtypes(['object', 'category', 'boolean']).columns
    num_cols = 4
    num_rows = (len(features) + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 4, num_rows * 3))

    for i, feature in enumerate(features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(df, x=feature)
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        
    plt.tight_layout()
    plt.show()

def plot_all(df, target):
    plot_feats_corr(df)
    scatterplot_all(df, target)
    boxplot_all(df)
    histplot_all(df)
    countplot_all(df)

# Plot data for analysis
plot_all(train_df, 'accident_risk')


# Replace problematic samples 
X = train_df.query("~accident_risk.ge(1)").copy()

# Separate target
y = X.pop('accident_risk').copy()

# Augment cliped values based on noise std from data generation function
y[y.eq(0)] = -0.2

# Convert features to categorical
cat_cols = X.select_dtypes(['object']).columns
X[cat_cols] = X[cat_cols].astype('category')
X_test[cat_cols] = X_test[cat_cols].astype('category')


def create_model(model_type, params):
    '''
    Create a specific model type with predefined parameters
    '''
    if model_type == 'lgb':
        model = lgb.LGBMRegressor(**params)
        
    elif model_type == 'lgb_class':
        model = lgb.LGBMClassifier(**params)
    
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(**params)
    
    elif model_type == 'cb':
        model = cb.CatBoostRegressor(**params)  
    
    else:
        raise ValueError('Model type is not supported')
        
    return model

def model_objective(trial, model_type):
    """
    Objective function for hyperparameter tuning using Optuna
    """
    if model_type == 'lgb':
        tune_params = {
            'n_estimators': 1500,

            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=False),
            'num_leaves': trial.suggest_int('num_leaves', 8, 256),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),

            'metric': 'rmse',
            'eval_metric': 'rmse',
            'objective': 'regression',
            'device': device,
            'verbosity': -1,
            'n_jobs': -1,
            }
    elif model_type == 'xgb':
        tune_params = {
            'n_estimators': 1500,
            
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=False),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

            'early_stopping_rounds': 50,
            'enable_categorical': True,
            'eval_metric': 'rmse',
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist' if device == 'gpu' else 'hist',
            'verbosity': 0,
            'n_jobs': -1,
        }
    elif model_type == 'cb':
        tune_params = {
            'iterations': 1500,

            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 1e-8, 10.0, log=True),

            'od_wait': 50,
            'cat_features': cat_cols.to_list(),
            'eval_metric': 'RMSE',
            'task_type': device.upper(),
            'border_count': 128 if device == 'gpu' else 254,
            'verbose': 100,
            'thread_count': -1,
        }
    # Split data in folds
    kf = KFold(n_splits=3, shuffle=True)
    cv_results = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        model = create_model(model_type, tune_params)

        if model_type == 'lgb': 
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50)],
                      )
        else:
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      )
        # Predict and accumulate results from every fold
        y_pred = model.predict(X_val)
        cv_results.append(round(mean_squared_error(y_val, y_pred, squared=False), 5))

    return np.mean(cv_results)


%%time
optuna_model = None

if optuna_model != None:
    # Optuna study
    def objective(trial):
        return model_objective(trial, model_type=optuna_model)
        
    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=50)

    # Show best results
    trial = study.best_trial

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    print('Value:', trial.value)
    print('Params:')

    for key, value in trial.params.items():
        print(f'{key}: {value}')


params_dict = {
    # Best LGBM parameters
    'lgb': {
        'n_estimators': 1000,

        'learning_rate': 0.1, 
        'num_leaves': 170, 
        'min_child_samples': 6, 
        'reg_alpha': 1.0e-06, 
        'reg_lambda': 0.003, 
        'colsample_bytree': 0.978, 
        'subsample': 0.933, 
        'subsample_freq': 7,
        
        'metric': 'rmse',
        'eval_metric': 'rmse',
        'objective': 'regression',
        'device': device,
        'verbosity': -1,
        'n_jobs': -1,
        },
    # Best XGB parameters
    'xgb': {
        'n_estimators': 1000,

        'learning_rate': 0.09, 
        'gamma': 0.0038, 
        'max_depth': 8, 
        'min_child_weight': 6, 
        'subsample': 0.76, 
        'colsample_bytree': 0.9985, 
        'reg_lambda': 1.39e-06,

        'early_stopping_rounds': 50,
        'enable_categorical': True,
        'eval_metric': 'rmse',
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist' if device == 'gpu' else 'hist',
        'verbosity': 0,
        'n_jobs': -1,
        },
    # Best CB parameters
    'cb': {
        'iterations': 1000,

        'learning_rate': 0.1, 
        'depth': 8, 
        'l2_leaf_reg': 1.65, 
        'random_strength': 0.05, 
        'bagging_temperature': 4.2e-06,

        'od_wait': 50,
        'cat_features': cat_cols.to_list(),
        'eval_metric': 'RMSE',
        'task_type': device.upper(),
        'border_count': 128 if device == 'gpu' else 254,
        'verbose': False,
        'thread_count': -1,
        },
}


def feat_importance(model_type, X, y):
    '''Plot model's feature importance'''
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Create model
    model = create_model(model_type, params_dict[model_type])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Check model type and plot
    if model_type == 'lgb':
        fig, ax = plt.subplots(figsize=(5, 3))
        lgb.plot_importance(
            model, 
            importance_type='gain', 
            max_num_features=None, 
            ignore_zero=False,
            height=0.5,
            grid=False,
            precision=0,
            title='LGBM feature importance',
            ax=ax,
            );
    elif model_type == 'xgb':
        fig, ax = plt.subplots(figsize=(5, 3))
        xgb.plot_importance(
            model, 
            max_num_features=None,
            grid=False, 
            importance_type='gain',
            height=0.5,
            title='XGB feature importance',
            show_values=False,
            ax=ax,
            );

feat_importance('lgb', X, y)


def cross_validate_models(
    X, y, X_test, params_dict, models,
    n_splits=5, random_state=42, scoring=None
):
    """
    Cross-validate one or more models with flexible scoring,
    optionally over multiple random states 

    scoring : callable
        Function with signature scoring(y_true, y_pred) returning a scalar.
        Defaults to RMSE if None.
    """

    if isinstance(models, str):
        models = [models]

    if isinstance(random_state, int):
        random_states = [random_state]
    else:
        random_states = random_state

    if scoring is None:
        def rmse(y_true, y_pred):
            return mean_squared_error(y_true, y_pred, squared=False)
        
        scoring = rmse

    oof_preds_total = np.zeros(len(X))
    test_preds_total = np.zeros(len(X_test))
    cv_results = {'stacked': [], 'per_model': {m: [] for m in models}}

    for seed in random_states:
        print(f"\n=== Random State {seed} ===")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\nFold {fold + 1}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_val_preds_list = []
            fold_test_preds_list = []

            for model_name in models:
                model = create_model(model_name, params_dict[model_name])

                if model_name == 'lgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val.clip(0, 1))],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                else:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val.clip(0, 1))],
                        verbose=False,
                    )
                val_preds = model.predict(X_val)
                test_preds_fold = model.predict(X_test)

                fold_val_preds_list.append(val_preds)
                fold_test_preds_list.append(test_preds_fold)

                score = scoring(y_val.clip(0, 1), val_preds.clip(0, 1))
                cv_results['per_model'][model_name].append(score)
                print(f"{model_name.upper()} Score: {score:.5f}")

            # Average predictions for this fold (stacked)
            fold_val_preds = np.mean(fold_val_preds_list, axis=0)
            fold_test_preds = np.mean(fold_test_preds_list, axis=0)

            oof_preds[val_idx] = fold_val_preds
            test_preds += fold_test_preds / n_splits

            stacked_score = scoring(y_val.clip(0, 1), fold_val_preds.clip(0, 1))
            cv_results['stacked'].append(stacked_score)
            print(f"Stacked Score: {stacked_score:.5f}")

        # Accumulate across random states
        oof_preds_total += oof_preds / len(random_states)
        test_preds_total += test_preds / len(random_states)

    # Print final summary
    print("\n=== CV Results Summary ===")
    print("Mean Stacked CV Score:", np.mean(cv_results['stacked']))

    print("\nMean CV Score per Model:")
    for model_name in models:
        mean_score = np.mean(cv_results['per_model'][model_name])
        print(f"{model_name.upper()}: {mean_score:.5f}")

    return oof_preds_total, test_preds_total, cv_results


# Stacked cross-validation and prediction
np.random.seed(42)
random_states = np.random.randint(0, 10000, size=5)

oof_preds, test_preds, scores = cross_validate_models(
    X, y, X_test, params_dict, ['lgb', 'xgb', 'cb'], 
    n_splits=5, random_state=random_states,
)


# Plot oof predictions vs true target (kudos to @cdeotte and @tilii7)
plt.scatter(y.clip(0, 1), oof_preds.clip(0, 1), s=0.25)
plt.plot([0,1],[0,1],'--',color='black')
plt.title("True vs Predicted")
plt.xlabel("True Target")
plt.ylabel("Predicted Target")
plt.show()


# Submit
submission_df.accident_risk = test_preds.clip(0, 1)
submission_df.to_csv('submission.csv', index=False)
submission_df.head()

