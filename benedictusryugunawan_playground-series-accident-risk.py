import numpy as np
import pandas as pd
import pickle as pkl
import missingno as msno
import warnings
import shap
import optuna

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

from ydata_profiling import ProfileReport
from sklearn.model_selection import cross_val_score, KFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline

# Global Figure Settings
rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 120

# Font and Text
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 12
rcParams['text.color'] = '#333333'

# Axes Settings
rcParams['axes.edgecolor'] = '#CCCCCC'
rcParams['axes.labelcolor'] = '#333333'
rcParams['axes.grid'] = True
rcParams['grid.color'] = '#E0E0E0'
rcParams['grid.linestyle'] = '--'
rcParams['grid.linewidth'] = 0.8

# Line and Marker
rcParams['lines.linewidth'] = 2
rcParams['lines.markersize'] = 6

# Legend
rcParams['legend.frameon'] = False
rcParams['legend.loc'] = 'best'

# Savefig
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.format'] = 'png'

# Apply
plt.style.use('dark_background')


TARGET = 'accident_risk'
SEED = 0


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

df_train = df_train.set_index("id")
df_test = df_test.set_index("id")

df_train.shape , df_test.shape


df_train.describe()


# # Assuming 'df_train' is your DataFrame
# profile = ProfileReport(df_train, title="Pandas Profiling Report for Train Data")


# Instead of profile.to_widgets()
# profile.to_notebook_iframe()


df_train = df_train.drop_duplicates()


df_train['curvature_squared'] = df_train['curvature'] ** 2
df_train['curvature_cubed'] = df_train['curvature'] ** 3
df_train['curve_speed_interaction'] = df_train['curvature'] * df_train['speed_limit']
df_train = pd.get_dummies(df_train, columns=['lighting'], prefix='lighting', drop_first=False)

df_test['curve_speed_interaction'] = df_test['curvature'] * df_test['speed_limit']
df_test['curvature_squared'] = df_test['curvature'] ** 2
df_test['curvature_cubed'] = df_test['curvature'] ** 3
df_test = pd.get_dummies(df_test, columns=['lighting'], prefix='lighting', drop_first=False)


for col in df_train.select_dtypes(include='object'):
    df_train[col] = df_train[col].astype('category')
    
for col in df_test.select_dtypes(include='object'):
    df_test[col] = df_test[col].astype('category')


X = df_train.drop(columns=[TARGET])
y = df_train[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    shuffle=True,
                                                    random_state=SEED,
                                                    test_size=0.2)


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)


import xgboost as xgb
xgb.__version__


print("Running XGBoost CV...")
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    enable_categorical=True,  # Crucial for native categorical handling
    n_jobs=-1,
    tree_method='gpu_hist'
)
xgb_scores = cross_val_score(xgb_model, X, y, cv=kf, scoring=rmse_scorer)
xgb_rmse = -xgb_scores


print("Running LightGBM CV...")
# LightGBM automatically detects and uses pandas 'category' dtype columns.
lgbm_model = LGBMRegressor(
    random_state=42,
    n_jobs=-1,
    device='gpu'
)
lgbm_scores = cross_val_score(lgbm_model, X, y, cv=kf, scoring=rmse_scorer)
lgbm_rmse = -lgbm_scores


print("Running CatBoost CV...")
# For CatBoost, you explicitly name the categorical features.
categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
cb_model = CatBoostRegressor(
    random_state=42,
    cat_features=categorical_features,
    verbose=0,  # Suppress training output for each fold
    task_type='GPU'
)
cb_scores = cross_val_score(cb_model, X, y, cv=kf, scoring=rmse_scorer)
cb_rmse = -cb_scores


print("\n" + "="*40)
print("      Model Performance Comparison")
print("="*40)

print(f"XGBoost CV RMSE:\t{xgb_rmse.mean():.4f} (+/- {xgb_rmse.std():.4f})")
print(f"LightGBM CV RMSE:\t{lgbm_rmse.mean():.4f} (+/- {lgbm_rmse.std():.4f})")
print(f"CatBoost CV RMSE:\t{cb_rmse.mean():.4f} (+/- {cb_rmse.std():.4f})")

print("-" * 40)
print("Interpretation:")
print("The 'mean' is the average RMSE across all folds.")
print("The '(+/- std)' shows the standard deviation, indicating performance consistency.")
print("A lower mean RMSE is better. A model with a lower mean and smaller std is generally preferred.")


def objective_xgb(trial):
    """Objective function for XGBoost hyperparameter tuning."""
    params = {
        'objective': 'reg:squarederror',
        'random_state': 42,
        'tree_method': 'gpu_hist',  # Use GPU
        'enable_categorical': True,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True), # L2 regularization
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),   # L1 regularization
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer, n_jobs=-1)
    return np.mean(scores)

def objective_lgbm(trial):
    """Objective function for LightGBM hyperparameter tuning."""
    params = {
        'objective': 'regression',
        'random_state': 42,
        'device': 'gpu',  # Use GPU
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2
    }
    model = LGBMRegressor(**params)
    scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer, n_jobs=-1)
    return np.mean(scores)
    
def objective_cb(trial):
    """Objective function for CatBoost hyperparameter tuning."""
    params = {
        'random_seed': 42,
        'task_type': 'GPU',  # Use GPU
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True), # L2
        'verbose': 0 # Suppress output during trials
    }
    cat_features = ['speed_limit']
    model = CatBoostRegressor(cat_features=cat_features, **params)
    scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer, n_jobs=-1)
    return np.mean(scores)

# --- 3. Run the Optimization Studies ---
N_TRIALS = 100 # Number of trials to run for each model. Increase for more thorough search.


print("--- Starting XGBoost Optimization ---")
study_xgb = optuna.create_study(direction='maximize') # We maximize because rmse_scorer is negative
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)
print(f"Best trial for XGBoost:")
print(f"  Value (RMSE): {-study_xgb.best_value:.4f}")
print(f"  Params: ")
for key, value in study_xgb.best_params.items():
    print(f"    {key}: {value}")

# --- LightGBM Study ---
print("\n--- Starting LightGBM Optimization ---")
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS)
print(f"Best trial for LightGBM:")
print(f"  Value (RMSE): {-study_lgbm.best_value:.4f}")
print(f"  Params: ")
for key, value in study_lgbm.best_params.items():
    print(f"    {key}: {value}")
    
# --- CatBoost Study ---
print("\n--- Starting CatBoost Optimization ---")
study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(objective_cb, n_trials=N_TRIALS)
print(f"Best trial for CatBoost:")
print(f"  Value (RMSE): {-study_cb.best_value:.4f}")
print(f"  Params: ")
for key, value in study_cb.best_params.items():
    print(f"    {key}: {value}")


# --- 4. Select the Best Model and Train for Submission ---
print("\n" + "="*50)
print("--- Selecting Best Model for Final Training ---")
print("="*50)

# Store results in a dictionary for easy comparison
results = {
    'XGBoost': {'score': -study_xgb.best_value, 'params': study_xgb.best_params},
    'LightGBM': {'score': -study_lgbm.best_value, 'params': study_lgbm.best_params},
    'CatBoost': {'score': -study_cb.best_value, 'params': study_cb.best_params}
}

# Find the model with the minimum RMSE score
best_model_name = min(results, key=lambda k: results[k]['score'])
best_params = results[best_model_name]['params']
best_score = results[best_model_name]['score']

print(f"The best performing model is: {best_model_name} with an estimated CV RMSE of {best_score:.4f}")

# Initialize the final model with the best parameters
if best_model_name == 'XGBoost':
    final_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        tree_method='gpu_hist',
        enable_categorical=True,
        **best_params
    )
elif best_model_name == 'LightGBM':
    final_model = LGBMRegressor(
        objective='regression',
        random_state=42,
        device='gpu',
        **best_params
    )
else: # CatBoost
    final_model = CatBoostRegressor(
        random_seed=42,
        task_type='GPU',
        cat_features=categorical_features,
        verbose=0,
        **best_params
    )


print(f"\nTraining the final {best_model_name} model on all available data...")
final_model.fit(X, y)
print("Final model training complete.")


def calculate_and_print_permutation_importance(model, model_name):
    print(f"\n--- Permutation Importance for {model_name} ---")
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    perm_sorted_idx = result.importances_mean.argsort()
    importance_df = pd.DataFrame(
        {
            "Feature": X_test.columns[perm_sorted_idx],
            "Importance": result.importances_mean[perm_sorted_idx],
            "Std Dev": result.importances_std[perm_sorted_idx],
        }
    )
    print(importance_df)

def calculate_and_plot_permutation_importance(model, model_name):
    print(f"\n--- Calculating Permutation Importance for {model_name} ---")
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    # Sort the features by importance for plotting
    perm_sorted_idx = result.importances_mean.argsort()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        X_test.columns[perm_sorted_idx],
        result.importances_mean[perm_sorted_idx],
        xerr=result.importances_std[perm_sorted_idx],
        align='center',
        alpha=0.8
    )
    ax.set_title(f"Permutation Importance for {model_name}")
    ax.set_xlabel("Performance Decrease (R-squared)")
    ax.invert_yaxis()  # Display the most important feature at the top
    fig.tight_layout()
    plt.show()

calculate_and_plot_permutation_importance(final_model, best_model_name)


def explain_and_plot_shap(model, model_name):
    print(f"\n--- Generating SHAP Summary for {model_name} ---")
    
    # Create the SHAP explainer object
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)
    
    # Create the summary plot
    plt.title(f'SHAP Summary Plot for {model_name}')
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.show()
    
    # A more detailed plot showing distribution
    plt.title(f'SHAP Feature Impact Plot for {model_name}')
    shap.summary_plot(shap_values, X_test, show=False)
    plt.show()


explain_and_plot_shap(final_model, best_model_name)


training_columns = X.columns 
df_test = df_test[training_columns]
y_pred = final_model.predict(df_test)


submission = pd.DataFrame({
    "id":df_test.index,
    "accident_risk":y_pred
})
submission.to_csv("submission.csv", index=False)

