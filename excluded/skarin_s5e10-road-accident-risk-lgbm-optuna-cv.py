import pandas as pd
import numpy as np


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head()


train.describe()


import seaborn as sns
import matplotlib.pyplot as plt

print("Target variable description:")
print(train['accident_risk'].describe())
sns.histplot(train['accident_risk'], kde=True)
plt.title('Distribution of Reported Accidents')
plt.show()


train.info()


# --- Plotting Cat Features vs. Accident Risk Bins ---

train['accident_risk_binned'] = pd.cut(train['accident_risk'], bins=2, labels=False)

categorical_cols = train.select_dtypes(include='object').columns

n_cols = 2
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5), squeeze=False)
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    ax = axes[i]

    proportions = pd.crosstab(train['accident_risk_binned'], train[col], normalize='index')

    proportions.plot(kind='bar', stacked=True, ax=ax, width=0.8)

    ax.set_title(f'Proportional Distribution of {col}')
    ax.set_xlabel('Accident Risk Bin (0 = Low, 1 = High)')
    ax.set_ylabel('Proportion')
    ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


# --- Plotting Numerical Features vs. Accident Risk Bins ---
numerical_cols = ['num_lanes', 'speed_limit', 'num_reported_accidents', 'curvature']

n_cols = 2
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4), squeeze=False)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    ax = axes[i]

    sns.boxplot(x='accident_risk_binned', y=col, data=train, ax=ax)

    ax.set_title(f'Distribution of {col} by Accident Risk Bin')
    ax.set_xlabel('Accident Risk Bin (0 = Low, 1 = High)')
    ax.set_ylabel(col)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

train = train.drop('accident_risk_binned', axis=1)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

X = train.drop(['id', 'accident_risk'], axis=1)
y = train['accident_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess(df, imputer=None, num_imputer=None, ohe=None, is_train=True):
    if 'accident_risk' in df.columns:
        df = df.drop('accident_risk', axis=1)

    df = df.drop('id', axis=1, errors='ignore')


    binary_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    nominal_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ##### FEATURE ENGINEERING #####


    df['low_visibility'] = ((df['lighting'] == 'night') | (df['lighting'] == 'dim') | 
                           (df['weather'] == 'rainy') | (df['weather'] == 'foggy')).astype(int)

    df['curvature_x_visibility'] = df['curvature'] * df['low_visibility']
    df['curvature_x_accidents'] = df['curvature'] * df['num_reported_accidents']

    df[binary_cols] = df[binary_cols].astype('object')
    if is_train:
        imputer = SimpleImputer(strategy='most_frequent')
        df[binary_cols] = imputer.fit_transform(df[binary_cols])
    else:
        df[binary_cols] = imputer.transform(df[binary_cols])

    if is_train:
        num_imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = num_imputer.transform(df[numerical_cols])

    df[binary_cols] = df[binary_cols].astype(int)

    if is_train:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = ohe.fit_transform(df[nominal_cols])
    else:
        encoded_data = ohe.transform(df[nominal_cols])

    encoded_cols = ohe.get_feature_names_out(nominal_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)

    final_df = df.drop(columns=nominal_cols)
    final_df = pd.concat([final_df, encoded_df], axis=1)

    if is_train:
        return final_df, imputer, num_imputer, ohe
    else:
        return final_df


X_full, imputer_f, num_imputer_f, ohe_f = preprocess(train.copy(), is_train=True)
y_full = train['accident_risk']

test_final = preprocess(test.copy(), imputer=imputer_f, num_imputer=num_imputer_f, ohe=ohe_f, is_train=False)

test_final = test_final[X_full.columns]

print("Data preprocessing complete.")
print(f"Training data shape: {X_full.shape}")
print(f"Test data shape:  {test_final.shape}")


#import optuna
#from IPython.display import clear_output

#def print_best_callback(study, trail):
#    if study.best_trial.number == trial.number:
#        clear_output(wait=True)
#        print(f"New best trial found! Trial number: {trial.number}")
#        print(f" Value (RMSE): {trial.value}")
#        print("  Params:")
#        for key, value in trial.params.items():
#            print(f"     {key}: {value}")

#def objective(trial):
#    params = {
#        'objective': 'regression',
#        'metric': 'rmse',
#        'n_estimators': 5000,
#        'random_state': 42,
#        'device': 'gpu',
#        'verbosity': -1,
#        'num_leaves': trail.suggest_int('num_leaves', 20, 150),
#        'max_depth': trail.suggest_int('max_depth', 5, 20),
#        'learning_rate': trial.suggest_int('learning_rate', 0.01, 0.1. log=True),
#        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
#        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
#        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
#    }

#    model = lgb.LGMRegressor(**params)
#    model.fit(X_train_final, y_train,
#             eval_set=[(X_test_final, y_test)],
#             callbacks=[lgb.early_stopping(100,verbose=False)])

#    preds = model.predict(X_test_final)
#    rmse = np.sqrt(mean_squared_error(y_test, preds))

#    return rmse

#study = optuna.create_study(direction='minimize')
#study.optimize(
#    objective,
#    n_trials=100,
#    callbacks=[print_best_callback]
#)

#print("\n--- Optimization Finished ---")
#print("Best parameters found: ", study.best_params)
#print("Best RMSE found: ", study.best_value)


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


params = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 5000,
    'random_state': 42, 'verbosity': -1, 'num_leaves': 150, 'max_depth': 10,
    'learning_rate': 0.01456213255721509, 'min_child_samples': 20, 'subsample': 0.7751076296394666,
    'colsample_bytree': 0.7646810662181514, 'reg_alpha': 2.144832684543215e-07, 'reg_lambda': 1.3843359441617385e-05,
    #'device': 'gpu'
    'n_jobs': -1
}

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions = np.zeros(X_full.shape[0])
test_predictions = np.zeros(test_final.shape[0])

print(f"Starting training with {N_SPLITS}-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(kf.split(X_full, y_full)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")

    X_train_fold, X_val_fold = X_full.iloc[train_index], X_full.iloc[val_index]
    y_train_fold, y_val_fold = y_full.iloc[train_index], y_full.iloc[val_index]


    model = lgb.LGBMRegressor(**params)

    model.fit(X_train_fold, y_train_fold,
             eval_set=[(X_val_fold, y_val_fold)],
             eval_metric='rmse',
             callbacks=[lgb.early_stopping(100, verbose=False)])

    val_preds = model.predict(X_val_fold)
    oof_predictions[val_index] = val_preds

    test_predictions += model.predict(test_final) / N_SPLITS
    
    overall_rmse = root_mean_squared_error(y_full, oof_predictions)
    print(f"Root Mean Squared Eroor (RMSE): {overall_rmse}")


submission_df = sample_submission.copy()
submission_df['accident_risk'] = test_predictions
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("\nSuccessfully created submission.csv!")

