# Warnings
import warnings
warnings.simplefilter(action='ignore', category=SyntaxWarning)
warnings.filterwarnings("ignore")

# Install and Import Libraries
import gc
import sys
import json
import platform
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from inspect import signature
from colorama import Fore, Style
from IPython.core.display import HTML
from itertools import combinations
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
gc.collect()

# Center all plots
HTML("""<style>.output_png { display: table-cell; text-align: center; vertical-align: middle;}</style>""")

# Set a consistent default figure size
plt.rcParams['figure.figsize'] = (8, 6)

# Standardized output formatting
def print_step(message):
    """Standardized step printing"""
    print(f"ğŸ“Š {message}")

def print_success(message):
    """Standardized success printing"""
    print(f"âœ… {message}")

def print_warning(message):
    """Standardized warning printing"""
    print(f"âš ï¸�  {message}")

def print_fold_header(fold):
    """Standardized fold header"""
    print(f"\n{Fore.GREEN}ğŸ�¯ {'='*15} FOLD {fold} {'='*15}{Style.RESET_ALL}")

def print_section_header(title):
    """Standardized section header"""
    print(f"\n{Fore.GREEN}{'='*20} {title} {'='*20}{Style.RESET_ALL}")

def print_header(title):
    """Prints a large, centered header banner."""
    print(f"\n{'=' * 20} {title.upper()} {'=' * 20}")

def print_section(title, symbol='-'):
    """Prints a smaller section divider using your standardized color/style."""
    try:
        # Use existing notebook function if available
        print_section_header(title) 
    except NameError:
        # Fallback if print_section_header is not defined in this scope
        print(f"\n{symbol * 5} {title} {symbol * 5}")

def print_list_nicely(data_list, items_per_row=4, prefix="* ", indent=2, sort=True):
    """Prints a list formatted with a fixed number of items per row for cleaner display."""
    if not data_list:
        return
        
    if sort:
        data_list = sorted(data_list)
        
    spacer = ' ' * indent
    num_items = len(data_list)
    
    # Calculate padding needed for aligning columns
    max_len = max(len(str(item)) for item in data_list) if data_list else 0
    col_width = max_len + len(prefix) + 2 # Prefix length + buffer

    rows = []
    for i in range(0, num_items, items_per_row):
        row_items = data_list[i:i + items_per_row]
        
        # Format each item with the prefix and fixed width
        formatted_row = [f"{prefix}{item:<{col_width - len(prefix)}}" for item in row_items]
        rows.append(spacer + "".join(formatted_row).rstrip())
    
    print('\n'.join(rows))

def print_dict_nicely(data_dict, indent=2):
    """Prints a dictionary with keys and values aligned."""
    if not data_dict:
        return
        
    spacer = ' ' * indent
    max_key_len = max(len(str(k)) for k in data_dict.keys()) if data_dict else 0
    
    for key, value in data_dict.items():
        if isinstance(value, dict):
             # Handle nested dicts (like NUMERICAL_STATS)
            print(f"{spacer}{key:<{max_key_len}}: {{", end="")
            nested_items = []
            for nk, nv in value.items():
                if isinstance(nv, (float, int)):
                    nested_items.append(f"'{nk}': {nv:.4f}")
                else:
                    nested_items.append(f"'{nk}': {json.dumps(nv)}")
            print(", ".join(nested_items), "}")
        else:
            # Handle simple key-value pairs
            print(f"{spacer}{key:<{max_key_len}}: {value}")


# Load Data

print_section_header("LOADING DATA")
print_step("Loading training, test, and external data")

# Update: Load external data with robust path handling
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print_success(f"Training data loaded: {train.shape}")
print_success(f"Test data loaded: {test.shape}")

# Constants
TARGET = 'diagnosed_diabetes'
ID_COL = 'id'

# Optional: Combine for consistent feature engineering later
# We use a sentinel value (-1) for the target in the test set
test[TARGET] = -1 
combine = pd.concat([train, test], axis=0, ignore_index=True)
print_success(f"Combined data for processing: {combine.shape}")


# Data Dimensions / Exploration

print_section_header("DATA EXPLORATION")

# Data Dimensions
print_step("Checking data structure")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Data Head
print_step("First 5 rows of Training Data (All Variables):")
# This shows the actual values for all columns, including categorical features and IDs
display(train.head())

# Statistical Summary (Training Data)
print_step("Statistical summary (Training Data):")
display(train.describe().style.format("{:.2f}"))


import warnings
warnings.filterwarnings("ignore")

# Target Distribution
print_step("Analyzing Target Balance")
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=train[TARGET], palette='viridis')
plt.title('Distribution of Diagnosed Diabetes (Target)')

# Add percentage labels on top of bars (Very helpful for imbalance check)
total = len(train)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()


import warnings
warnings.filterwarnings("ignore")

# Correlation Heatmap
print_step("Calculating Feature Correlations")

# Convert Index to list before adding the TARGET
numeric_cols = train.select_dtypes(include=[np.number]).columns.drop([ID_COL, TARGET]).tolist()
cols_to_corr = numeric_cols + [TARGET]

plt.figure(figsize=(12, 10))
# Using a smaller sample (e.g., 100k) for the heatmap is faster with 1M rows
corr_matrix = train[cols_to_corr].sample(100000).corr()

sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f', 
            linewidths=0.5, annot_kws={"size": 8})
plt.title('Feature Correlation with Diabetes Target')
plt.show()


print_section_header("FEATURE ENGINEERING: HEALTH FEATURES")
print_step("Creating health risk interaction features")

def create_features(df):
    # Copy to avoid modifying the original dataframe in place
    df = df.copy()
    # 1. Body Metrics
    df['bmi_waist_ratio'] = df['bmi'] * df['waist_to_hip_ratio']
    # 2. Lifestyle Ratios (Handling potential zeros with a small epsilon)
    df['activity_to_screen_ratio'] = df['physical_activity_minutes_per_week'] / (df['screen_time_hours_per_day'] + 1)   
    # 3. Cardiovascular Indicators
    df['bp_diff'] = df['systolic_bp'] - df['diastolic_bp']
    # 4. Cholesterol Ratios
    df['cholesterol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 0.1)
    return df

train = create_features(train)
test = create_features(test)

print_success(f"New features created. Total columns: {train.shape[1]}")


print_section_header("FEATURE ENGINEERING: TARGETED INTERACTIONS")

# 1. Define your top features based on previous importance plots
# (Adjust these names to match your specific dataset columns)
top_features = ['family_history_diabetes', 'physical_activity_minutes_per_week', 'age', 'triglycerides', 'bmi'] 

print_step(f"Creating pairwise interactions for: {', '.join(top_features)}")

# 2. Generate and add the features to both train and test
for feat1, feat2 in combinations(top_features, 2):
    new_col_name = f"{feat1}_x_{feat2}"
    
    # Apply to Train
    train[new_col_name] = train[feat1] * train[feat2]
    
    # Apply to Test
    test[new_col_name] = test[feat1] * test[feat2]
    
    print(f"   + Created: {new_col_name}")

print_success(f"Added {len(list(combinations(top_features, 2)))} new interaction features.")


print_section_header("PREPARING CATEGORICAL FEATURES")
print_step("Converting object columns to categorical type")

# Identify columns that are 'object' (strings)
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

# If ID_COL or TARGET are in there by accident, remove them
cat_cols = [c for c in cat_cols if c not in [ID_COL, TARGET]]

print(f"Found {len(cat_cols)} categorical columns: ")
print_list_nicely(cat_cols)

# Convert in both train and test
for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

print_success("Categorical conversion complete.")


# Define the models with 'tuned' parameters
cat_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 8000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 6,
    "random_strength": 1.0,     
    "bootstrap_type": "Bayesian",   
    "bagging_temperature": 0.8,    
    "min_data_in_leaf": 50,
    "od_type": "Iter",
    "od_wait": 300,
    "random_seed": 42,
    "verbose": 0,
}

xgb_params = {
    "objective": "binary:logistic",  
    "eval_metric": "auc",    
    "n_estimators": 8000,   
    "learning_rate": 0.03,
    "max_depth": 6,            
    "reg_lambda": 6,             
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": 42,
    "verbosity": 0,
}

lgbm_params = {
    "objective": "binary",       
    "metric": "auc",
    "n_estimators": 8000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "num_leaves": 63,       
    "reg_lambda": 6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "min_child_samples": 50,   
    "random_state": 42,
    "verbosity": -1,   
}

models = {
    "XGBoost": XGBClassifier(**xgb_params),
    "LightGBM": LGBMClassifier(**lgbm_params),
    "CatBoost": CatBoostClassifier(
        **cat_params,
        cat_features=cat_cols,
        allow_writing_files=False,
    ),
}

# Setup Cross-Validation
N_FOLDS = 6
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Prepare containers for OOF predictions (to optimize weights) and Test predictions
oof_preds = pd.DataFrame(index=train.index)
test_preds = pd.DataFrame(index=test.index)

# Prepare Features (Drop ID and Target)
features = [col for col in train.columns if col not in [ID_COL, TARGET]]
X = train[features]
y = train[TARGET]
X_test = test[features]

print_success(f"Framework ready for {len(features)} features.")


def _xgb_predict_proba_best(model, X_):
    best_it = getattr(model, "best_iteration", None)
    if best_it is not None:
        try:
            return model.predict_proba(X_, iteration_range=(0, best_it + 1))[:, 1]
        except TypeError:
            pass

    best_ntree = getattr(model, "best_ntree_limit", None)
    if best_ntree is not None:
        try:
            return model.predict_proba(X_, ntree_limit=best_ntree)[:, 1]
        except TypeError:
            pass

    return model.predict_proba(X_)[:, 1]

print_section_header("TRAINING")

for model_name, base_model in models.items():
    print_step(f"Starting {model_name} Cross-Validation...")

    current_oof = np.zeros(len(train))
    current_test = np.zeros(len(test))
    scores = []

    # This variable will hold the last fitted model of the CV loop
    last_fitted_model = None 

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = clone(base_model)  # fresh per fold

        if model_name == "XGBoost":
            model.set_params(early_stopping_rounds=300)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            preds_val = _xgb_predict_proba_best(model, X_val)
            preds_test = _xgb_predict_proba_best(model, X_test)

        elif model_name == "LightGBM":
            from lightgbm import early_stopping, log_evaluation
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    early_stopping(stopping_rounds=300, verbose=False),
                    log_evaluation(period=0),
                ],
            )
            best_iter = getattr(model, "best_iteration_", None)
            if best_iter:
                preds_val = model.predict_proba(X_val, num_iteration=best_iter)[:, 1]
                preds_test = model.predict_proba(X_test, num_iteration=best_iter)[:, 1]
            else:
                preds_val = model.predict_proba(X_val)[:, 1]
                preds_test = model.predict_proba(X_test)[:, 1]

        elif model_name == "CatBoost":
            # CatBoost automatically uses the best iteration for predict_proba
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            preds_val = model.predict_proba(X_val)[:, 1]
            preds_test = model.predict_proba(X_test)[:, 1]

        current_oof[val_idx] = preds_val
        current_test += preds_test / N_FOLDS

        fold_auc = roc_auc_score(y_val, preds_val)
        scores.append(fold_auc)
        print(f"    Fold {fold+1} ROC AUC: {fold_auc:.5f}")
        
        # Keep track of the model we just trained
        last_fitted_model = model

    # Save the fitted model back to the dictionary 
    # so the Feature Importance block can find it later.
    models[model_name] = last_fitted_model

    oof_preds[model_name] = current_oof
    test_preds[model_name] = current_test
    print_success(f"{model_name} CV Mean AUC: {np.mean(scores):.5f}")

print_section_header("INDIVIDUAL MODEL PERFORMANCE COMPLETE")


print_section_header("OPTIMIZING THE BLEND")
print_step("Finding the optimal weights for each model")

# Define the models we are blending
model_names = list(models.keys())

# Define the function we want to MINIMIZE (1 - AUC)
def auc_loss(weights):
    # Ensure weights sum to 1
    normalized_weights = weights / np.sum(weights)
    
    # Calculate weighted average of OOF predictions
    blend_preds = np.zeros(len(train))
    for i, name in enumerate(model_names):
        blend_preds += oof_preds[name] * normalized_weights[i]
        
    return -roc_auc_score(y, blend_preds) # Negative because we minimize

# Start with equal weights (e.g., 0.33 each)
initial_weights = [1/3, 1/3, 1/3]

# Constraints: Weights must be between 0 and 1
bounds = [(0, 1)] * len(model_names)

# Run the optimizer
res = minimize(auc_loss, initial_weights, bounds=bounds, method='SLSQP')
best_weights = res.x / np.sum(res.x)

# NOW we create the 'Final_Blend' column in both OOF and Test dataframes
oof_preds['Final_Blend'] = 0
test_preds['Final_Blend'] = 0

for i, name in enumerate(model_names):
    oof_preds['Final_Blend'] += oof_preds[name] * best_weights[i]
    test_preds['Final_Blend'] += test_preds[name] * best_weights[i]

# Show the results
print_step("Optimized Weights Found:")
for name, weight in zip(model_names, best_weights):
    print(f"   * {name}: {weight:.4f}")

print_success(f"Blended OOF ROC AUC: {roc_auc_score(y, oof_preds['Final_Blend']):.5f}")


print_section_header("POST-PROCESSING: RANK BLENDING")
print_step("Converting probabilities to ranks to stabilize the ensemble")

# 1. Prepare Rank Dataframes
model_names = list(models.keys())
oof_ranks = pd.DataFrame(index=train.index)
test_ranks = pd.DataFrame(index=test.index)

# 2. Transform each model's predictions into ranks (scaled 0 to 1)
# This is like 'percent_rank' in SQL or 'cume_dist' in R
for name in model_names:
    oof_ranks[name] = rankdata(oof_preds[name]) / len(train)
    test_ranks[name] = rankdata(test_preds[name]) / len(test)

# 3. Define the Rank-based loss function
def rank_auc_loss(weights):
    normalized_weights = weights / np.sum(weights)
    # Dot product of ranks and weights
    blend_preds = (oof_ranks[model_names].values @ normalized_weights)
    return -roc_auc_score(y, blend_preds)

# 4. Optimize the weights for the Rank Blend
res_rank = minimize(rank_auc_loss, initial_weights, bounds=bounds, method='SLSQP')
best_rank_weights = res_rank.x / np.sum(res_rank.x)

# 5. Create the Final Rank Blend Column
oof_preds['Rank_Blend'] = (oof_ranks[model_names].values @ best_rank_weights)
test_preds['Rank_Blend'] = (test_ranks[model_names].values @ best_rank_weights)

print_step("Optimized Rank Weights Found:")
for name, weight in zip(model_names, best_rank_weights):
    print(f"   * {name}: {weight:.4f}")

print_success(f"Rank-Blended OOF ROC AUC: {roc_auc_score(y, oof_preds['Rank_Blend']):.5f}")


print_section_header("FEATURE IMPORTANCE ANALYSIS")
print_step("Aggregating importance from all models")

# 1. Initialize the dataframe
feat_imp_df = pd.DataFrame({'Feature': X.columns})

# 2. Dynamically find which columns were actually added
available_models = []

for model_name in models.keys():
    try:
        # Extract importance from the model object
        if hasattr(models[model_name], 'feature_importances_'):
            feat_imp_df[model_name] = models[model_name].feature_importances_
            available_models.append(model_name)
        elif model_name == 'CatBoost':
            feat_imp_df[model_name] = models[model_name].get_feature_importance()
            available_models.append(model_name)
    except Exception as e:
        print_warning(f"Could not extract importance for {model_name}. It may not be fitted yet.")

# 3. Calculate Average only on models that successfully provided data
if available_models:
    feat_imp_df['Average'] = feat_imp_df[available_models].mean(axis=1)
    feat_imp_df = feat_imp_df.sort_values(by='Average', ascending=False)

    # Visualize
    plt.figure(figsize=(12, 8))
    # Using 'rocket' or 'viridis' for the barplot
    sns.barplot(data=feat_imp_df.head(20), x='Average', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title("Top 20 Features (Averaged across Ensembles)", fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.show()

    # Print the top 5
    print_step("Top 5 predictors of Diabetes in this dataset:")
    for i, (idx, row) in enumerate(feat_imp_df.head(5).iterrows()):
        print(f"   {i+1}. {row['Feature']:<20} (Score: {row['Average']:.4f})")
else:
    print_warning("No feature importance data found. Ensure models are fitted before running this cell.")


print_section_header("MODEL PERFORMANCE VISUALIZATION")
print_step("Generating ROC Curves with Viridis palette")

plt.figure(figsize=(8, 6))

# Define color mapping
viridis_colors = [cm.viridis(i / 3) for i in range(4)]
colors = {
    'XGBoost': viridis_colors[0],
    'LightGBM': viridis_colors[1],
    'CatBoost': viridis_colors[2],
    'Final_Blend': '#440154', # Dark Purple
    'Rank_Blend': '#fde725'    # Bright Yellow
}

for col in oof_preds.columns:
    if col not in colors: continue
    
    fpr, tpr, _ = roc_curve(y, oof_preds[col])
    auc_val = roc_auc_score(y, oof_preds[col])
    
    # Define line styles
    if col == 'Final_Blend':
        lw, ls, alpha = 3.5, '-', 1.0
    elif col == 'Rank_Blend':
        lw, ls, alpha = 3.5, '--', 0.8
    else:
        lw, ls, alpha = 1.5, ':', 0.7
        
    plt.plot(fpr, tpr, label=f'{col} (AUC = {auc_val:.4f})', 
             lw=lw, linestyle=ls, color=colors[col], alpha=alpha)

# Reference line
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='-', alpha=0.3, label='Baseline')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Final ROC Comparison', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.show()


print_section_header("CALIBRATION & RELIABILITY")
print_step("Calculating reliability curves and Brier scores")

plt.figure(figsize=(8, 6))

# 1. Define the models and colors
model_list = ['XGBoost', 'LightGBM', 'CatBoost', 'Final_Blend', 'Rank_Blend']
viridis_colors = [cm.viridis(i / 3) for i in range(4)] # Get 4 colors for the main models

colors = {
    'XGBoost': viridis_colors[0],
    'LightGBM': viridis_colors[1],
    'CatBoost': viridis_colors[2],
    'Final_Blend': '#440154', # Dark Purple
    'Rank_Blend': '#fde725'    # Bright Yellow (Viridis end-spectrum)
}

for col in oof_preds.columns:
    if col not in colors: continue
    
    # Get the calibration curve data
    fraction_of_positives, mean_predicted_value = calibration_curve(y, oof_preds[col], n_bins=10)
    
    # Calculate Brier Score
    brier = brier_score_loss(y, oof_preds[col])
    
    # Styling logic
    if col == 'Final_Blend':
        lw, ls, alpha = 3.5, '-', 1.0
    elif col == 'Rank_Blend':
        lw, ls, alpha = 3.5, '--', 0.8
    else:
        lw, ls, alpha = 1.5, ':', 0.7
    
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', markersize=5,
             label=f'{col} (Brier: {brier:.4f})', 
             lw=lw, linestyle=ls, color=colors[col], alpha=alpha)

# The "Perfectly Calibrated" reference line
plt.plot([0, 1], [0, 1], linestyle='-', color='gray', alpha=0.5, label='Perfectly Calibrated')

plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives (Actual)')
plt.title('Calibration Plot: Probability Trustworthiness', fontsize=14)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1)) # Move legend outside if it gets crowded
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


print_section_header("FINAL SUBMISSION SELECTION")

# Choose the best performing method based on Out-of-Fold (OOF) results
prob_score = roc_auc_score(y, oof_preds['Final_Blend'])
rank_score = roc_auc_score(y, oof_preds['Rank_Blend'])

if rank_score > prob_score:
    print_step(f"Rank Blending outperformed Probability Blending ({rank_score:.5f} vs {prob_score:.5f})")
    final_col = 'Rank_Blend'
else:
    print_step(f"Probability Blending remains superior ({prob_score:.5f} vs {rank_score:.5f})")
    final_col = 'Final_Blend'

# Construct the final dataframe
# Ensure ID_COL and TARGET match the competition requirements
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: test_preds[final_col]
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print_success(f"Final submission saved using the {final_col} strategy.")
print_step(f"OOF AUC Score for this submission: {max(prob_score, rank_score):.5f}")

# Final Visual Check
print_step("First 5 rows of the generated Submission File:")
display(submission.head())


# Environment summary
print_header("ENVIRONMENT SUMMARY")

env_summary = {
    "python": sys.version,
    "os": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__
}

# Clean up the python version string for cleaner output
env_summary['python'] = env_summary['python'].split('\n')[0]

print("\n# ENVIRONMENT VERSIONS")
print_dict_nicely(env_summary, indent=2)

