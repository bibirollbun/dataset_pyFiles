# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc # Garbage collector
import scipy.stats as stats # For QQ-plot

# Scikit-learn modules
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Gradient Boosting Models
import lightgbm as lgb
import xgboost as xgb

# Settings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 7) # Default figure size
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format) # Format floats


# Load datasets
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
    print("Data loaded successfully from Kaggle environment.")
    DATA_LOADED = True
except FileNotFoundError:
    print("Datasets not found in Kaggle environment. Check input paths.")
    DATA_LOADED = False
    # Optionally add local paths if running outside Kaggle
    # train_df = pd.read_csv('train.csv')
    # test_df = pd.read_csv('test.csv')
    # sample_submission = pd.read_csv('sample_submission.csv')
    # if os.path.exists('train.csv'): DATA_LOADED = True

if not DATA_LOADED:
    raise FileNotFoundError("Training or Test data not found. Please ensure files are in the correct path.")


print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

print("\nTrain Data Info:")
train_df.info()

# Check for missing values
print("\nMissing values summary (Train):")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])
if missing_train.sum() == 0: print("No missing values found in training data.")

print("\nMissing values summary (Test):")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0])
if missing_test.sum() == 0: print("No missing values found in test data.")


print("\nNumber of duplicate rows in Training Data:")
print(train_df.duplicated().sum())

# Store test IDs and drop 'id' columns
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Define target and initial features
TARGET = 'Listening_Time_minutes'
FEATURES = [col for col in train_df.columns if col != TARGET]

print(f"\nTarget variable: {TARGET}")
print(f"Number of features identified: {len(FEATURES)}")


# Descriptive Statistics
print("\nTraining Data Descriptive Statistics:")
print(train_df.describe())


plt.figure(figsize=(18, 7))

# Histogram & KDE
plt.subplot(1, 3, 1)
sns.histplot(train_df[TARGET], kde=True, bins=60, color='skyblue', line_kws={'lw': 2})
plt.title('Distribution of Listening Time', fontsize=14)
plt.xlabel('Listening Time (minutes)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Box Plot
plt.subplot(1, 3, 2)
sns.boxplot(y=train_df[TARGET], color='lightcoral', width=0.4)
plt.title('Box Plot of Listening Time', fontsize=14)
plt.ylabel('Listening Time (minutes)', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# QQ-Plot (Quantile-Quantile Plot) against Normal distribution
plt.subplot(1, 3, 3)
stats.probplot(train_df[TARGET], dist="norm", plot=plt)
plt.title('QQ-Plot vs Normal Distribution', fontsize=14)
plt.xlabel('Theoretical Quantiles', fontsize=11)
plt.ylabel('Sample Quantiles', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Quantitative analysis
target_skewness = train_df[TARGET].skew()
target_kurtosis = train_df[TARGET].kurt()
print(f"Target Variable Skewness: {target_skewness:.3f}")
print(f"Target Variable Kurtosis: {target_kurtosis:.3f}") # Excess kurtosis (relative to normal)


print(f"\nPlotting distributions for {len(FEATURES)} features...")
n_cols = 4
n_rows = int(np.ceil(len(FEATURES) / n_cols)) if FEATURES else 0

if n_rows > 0:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.array(axes).flatten() # Ensure axes is always a flat array

    for i, feature in enumerate(FEATURES):
        if i < len(axes):
            sns.histplot(train_df[feature], kde=True, ax=axes[i], bins=30, color='teal', line_kws={'lw': 1})
            axes[i].set_title(f'{feature} Distribution', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].tick_params(axis='both', which='major', labelsize=10)
        else: break # Stop if we run out of axes

    # Hide unused subplots
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])

    plt.suptitle('Distributions of Input Features', fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()
else:
     print("Error: No features were identified for plotting.")


# Filter the DataFrame to only include numeric columns
numeric_df = train_df.select_dtypes(include=['number'])

# Compute the Pearson correlation with the target variable
correlation_target = numeric_df.corr(method='pearson')[TARGET].sort_values(ascending=False)

# Plotting the correlation results
plt.figure(figsize=(10, 10))
sns.barplot(
    x=correlation_target.drop(TARGET).values,
    y=correlation_target.drop(TARGET).index,
    palette='viridis_r'
)
plt.title(f'Feature Correlation with {TARGET} (Pearson)', fontsize=16)
plt.xlabel('Correlation Coefficient (ρ)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.show()

# Print out the top correlated features
print("\nTop 5 features positively correlated with target:")
print(correlation_target.head(6))
print("\nTop 5 features negatively correlated with target:")
print(correlation_target.tail(5))


# Filter the DataFrame to only include numeric columns
numeric_df = train_df.select_dtypes(include=['number'])

plt.figure(figsize=(20, 16))
correlation_matrix = numeric_df.corr(method='pearson')
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap='coolwarm',
    fmt=".1f",
    mask=mask,
    linewidths=.5,
    cbar_kws={"shrink": .7}
)
plt.title('Pairwise Feature Correlation Matrix (Pearson)', fontsize=20, pad=20)
plt.xticks(rotation=60, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


top_corr_features = correlation_target.drop(TARGET).abs().nlargest(6).index.tolist()
print(f"\nVisualizing relationships for top correlated features: {top_corr_features}")

n_feat_plot = len(top_corr_features)
n_cols_violin = 3
n_rows_violin = int(np.ceil(n_feat_plot / n_cols_violin))

plt.figure(figsize=(n_cols_violin * 6, n_rows_violin * 5))

for i, feature in enumerate(top_corr_features):
    plt.subplot(n_rows_violin, n_cols_violin, i + 1)
    # Create bins for the feature to use as categories in the violin plot
    # Using quantiles creates bins with roughly equal numbers of samples
    train_df[f'{feature}_bins'] = pd.qcut(train_df[feature], q=5, labels=False, duplicates='drop')
    sns.violinplot(x=f'{feature}_bins', y=TARGET, data=train_df, palette='coolwarm', inner='quartile', cut=0)
    plt.title(f'{TARGET} vs {feature} (Binned)', fontsize=13)
    plt.xlabel(f'{feature} Quantile Bins', fontsize=11)
    plt.ylabel(TARGET, fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.suptitle('Target Distribution across Feature Quantiles', fontsize=18, y=1.03)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

# Clean up temporary bin columns
for feature in top_corr_features:
    train_df.drop(columns=[f'{feature}_bins'], inplace=True)


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# Select top N features based on absolute correlation
N_POLY_FEATURES = 6
poly_features_cols = correlation_target.drop(TARGET).abs().nlargest(N_POLY_FEATURES).index.tolist()
print(f"Creating polynomial features (degree 2) for: {poly_features_cols}")

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

# Initialize imputer for missing values
imputer = SimpleImputer(strategy='mean')

# Fit and transform training data for polynomial feature generation
X_poly_basis = train_df[poly_features_cols].copy()
X_poly_basis_imputed = pd.DataFrame(
    imputer.fit_transform(X_poly_basis),
    columns=X_poly_basis.columns,
    index=X_poly_basis.index
)
poly_train_feats = poly.fit_transform(X_poly_basis_imputed)
poly_feature_names = poly.get_feature_names_out(poly_features_cols)
poly_train_df = pd.DataFrame(poly_train_feats, columns=poly_feature_names, index=train_df.index)

# Transform test data using the same imputer
X_test_poly_basis = test_df[poly_features_cols].copy()
X_test_poly_basis_imputed = pd.DataFrame(
    imputer.transform(X_test_poly_basis),
    columns=X_test_poly_basis.columns,
    index=X_test_poly_basis.index
)
poly_test_feats = poly.transform(X_test_poly_basis_imputed)
poly_test_df = pd.DataFrame(poly_test_feats, columns=poly_feature_names, index=test_df.index)

# Combine original features (excluding those used for poly basis) with new polynomial features
original_features_to_keep = [f for f in FEATURES if f not in poly_features_cols]
train_df_enhanced = pd.concat([train_df[original_features_to_keep + [TARGET]], poly_train_df], axis=1)
test_df_enhanced = pd.concat([test_df[original_features_to_keep], poly_test_df], axis=1)

# Update features list
FEATURES = [col for col in train_df_enhanced.columns if col != TARGET]

print(f"\nShape after adding polynomial features:")
print(f"Train data: {train_df_enhanced.shape}")
print(f"Test data: {test_df_enhanced.shape}")
print(f"New number of features: {len(FEATURES)}")

# Memory Management
del poly_train_df, poly_test_df, poly_train_feats, poly_test_feats, X_poly_basis, X_test_poly_basis, X_poly_basis_imputed, X_test_poly_basis_imputed
gc.collect()


from sklearn.preprocessing import StandardScaler

# Identify numeric features only
numeric_features = train_df_enhanced[FEATURES].select_dtypes(include=['number']).columns.tolist()

scaler = StandardScaler()

# Prepare data for scaling (only numeric columns)
X = train_df_enhanced[numeric_features].copy()
y = train_df_enhanced[TARGET].copy()
X_test = test_df_enhanced[numeric_features].copy()

# Fit on training data and transform both train and test
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X = pd.DataFrame(X_scaled, columns=numeric_features, index=X.index)
X_test = pd.DataFrame(X_test_scaled, columns=numeric_features, index=X_test.index)

print("\nFeatures scaled using StandardScaler.")
print("Scaled Training Data Head:")
print(X.head())

# Memory Management
del X_scaled, X_test_scaled, train_df_enhanced, test_df_enhanced, train_df  # Keep only scaled data
gc.collect()


# Define RMSE evaluation metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Cross-Validation Setup
N_SPLITS = 10
SEED = 42
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Model Training Configuration
EARLY_STOPPING_ROUNDS = 500
VERBOSE_FREQUENCY = 2000 # Reduced frequency for cleaner logs

# Data Storage
oof_preds_lgb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
feature_importances_lgb = pd.DataFrame(index=FEATURES)
feature_importances_xgb = pd.DataFrame(index=FEATURES)
fold_scores_lgb = []
fold_scores_xgb = []
lgbm_models = {} # Store models for later analysis (e.g., learning curves)
xgb_models = {}


lgb_params = {
    'objective': 'regression_l1',  # MAE Loss
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 15000,
    'learning_rate': 0.004,
    'num_leaves': 48,
    'max_depth': 10,
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.65,
    'subsample': 0.65,
    'subsample_freq': 1,
    'reg_alpha': 0.05,  # L1 Regularization
    'reg_lambda': 0.05  # L2 Regularization
}

print(f"\n--- Starting LightGBM Training ({N_SPLITS}-Fold CV) ---")
feature_importances_lgb = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=VERBOSE_FREQUENCY)
        ]
    )
    lgbm_models[f'fold_{fold+1}'] = lgb_model  # Store trained model

    # Store OOF predictions
    fold_oof_preds = lgb_model.predict(X_val)
    oof_preds_lgb[val_idx] = fold_oof_preds
    test_preds_lgb += lgb_model.predict(X_test) / N_SPLITS

    # RMSE calculation
    fold_rmse = rmse(y_val, fold_oof_preds)
    fold_scores_lgb.append(fold_rmse)
    print(f"Fold {fold+1} LGBM Val RMSE: {fold_rmse:.5f}")

    # Store feature importances correctly
    fold_importance_df = pd.DataFrame({
        'Feature': X.columns,
        f'Fold_{fold+1}': lgb_model.feature_importances_
    })
    if feature_importances_lgb.empty:
        feature_importances_lgb = fold_importance_df
    else:
        feature_importances_lgb = feature_importances_lgb.merge(fold_importance_df, on='Feature', how='outer')

    del X_train, y_train, X_val, y_val
    gc.collect()

print("--- LightGBM Training Finished ---")
mean_oof_rmse_lgb = np.mean(fold_scores_lgb)
std_oof_rmse_lgb = np.std(fold_scores_lgb)
overall_oof_rmse_lgb = rmse(y, oof_preds_lgb)
print(f"\nLGBM Mean Fold RMSE: {mean_oof_rmse_lgb:.5f} +/- {std_oof_rmse_lgb:.5f}")
print(f"LGBM Overall OOF RMSE: {overall_oof_rmse_lgb:.5f}")


import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Create example data with 20 columns
np.random.seed(42)
N = 1000
n_features = 20
X = pd.DataFrame(np.random.rand(N, n_features), columns=[f'feature_{i+1}' for i in range(n_features)])
y = pd.Series(np.random.rand(N))

# Define KFold and constants
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
EARLY_STOPPING_ROUNDS = 50
VERBOSE_FREQUENCY = 100
SEED = 42

# Define XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',  # MSE loss
    'eval_metric': 'rmse',
    'eta': 0.005,                    # learning rate
    'max_depth': 9,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'gamma': 0.3,                    # minimum loss reduction for a split
    'lambda': 0.8,                   # L2 regularization
    'alpha': 0.1,                    # L1 regularization
    'seed': SEED,
    'nthread': -1
}

xgb_models = {}
oof_preds_xgb = np.zeros(X.shape[0])
test_preds_xgb = 0  # dummy; replace with your test data predictions if available
fold_scores_xgb = []
feature_importances_xgb = {}

print(f"\n--- Starting XGBoost Training ({N_SPLITS}-Fold CV) ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Create XGBoost regressor instance with early stopping
    xgb_model = xgb.XGBRegressor(
        **xgb_params,
        n_estimators=15000,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=VERBOSE_FREQUENCY
    )
    
    xgb_models[f'fold_{fold+1}'] = xgb_model  # store model
    fold_oof_preds = xgb_model.predict(X_val)
    oof_preds_xgb[val_idx] = fold_oof_preds
    # For test predictions, if you have X_test, do:
    # test_preds_xgb += xgb_model.predict(X_test) / N_SPLITS

    fold_rmse = rmse(y_val, fold_oof_preds)
    fold_scores_xgb.append(fold_rmse)
    print(f"Fold {fold+1} XGB Val RMSE: {fold_rmse:.5f}")

    # Create a pandas Series for feature importances using the training feature names.
    feature_importances_xgb[f'Fold_{fold+1}'] = pd.Series(
        xgb_model.feature_importances_,
        index=X_train.columns
    )
    
    del X_train, y_train, X_val, y_val
    gc.collect()

print("--- XGBoost Training Finished ---")
mean_oof_rmse_xgb = np.mean(fold_scores_xgb)
std_oof_rmse_xgb = np.std(fold_scores_xgb)
overall_oof_rmse_xgb = rmse(y, oof_preds_xgb)
print(f"\nXGB Mean Fold RMSE: {mean_oof_rmse_xgb:.5f} +/- {std_oof_rmse_xgb:.5f}")
print(f"XGB Overall OOF RMSE: {overall_oof_rmse_xgb:.5f}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Number of folds
N_SPLITS = 5

# Simulated fold scores for two models (each list must be of length N_SPLITS)
fold_scores_lgb = [0.85, 0.87, 0.83, 0.86, 0.84]   # example RMSE values for LightGBM
fold_scores_xgb = [0.88, 0.89, 0.87, 0.90, 0.88]   # example RMSE values for XGBoost

# Optional: Check the lengths to confirm they match N_SPLITS
print("Length of Fold list:", len([f'Fold {i+1}' for i in range(N_SPLITS)]))
print("Length of fold_scores_lgb:", len(fold_scores_lgb))
print("Length of fold_scores_xgb:", len(fold_scores_xgb))

# Create a DataFrame for fold scores
scores_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(N_SPLITS)],
    'LightGBM': fold_scores_lgb,
    'XGBoost': fold_scores_xgb
})

# Melt the DataFrame to have a long format for easier plotting with Seaborn
scores_melted = scores_df.melt(id_vars='Fold', var_name='Model', value_name='RMSE')

# Plotting the distribution of scores
plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='RMSE', data=scores_melted, palette=['skyblue', 'lightcoral'])
sns.stripplot(x='Model', y='RMSE', data=scores_melted, color=".25", size=6, jitter=True)

plt.title(f'Distribution of RMSE Scores Across {N_SPLITS} Folds', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Validation RMSE', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Print a summary (mean and standard deviation) of the fold RMSEs for each model
print("\nSummary of Fold RMSEs:")
summary = scores_df.set_index('Fold').agg(['mean', 'std']).T
print(summary)


# Plot learning curves for the first fold model
fold_to_plot = 'fold_1'
lgbm_fold1_model = lgbm_models[fold_to_plot]
xgb_fold1_model = xgb_models[fold_to_plot]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# LightGBM Learning Curve
lgb_eval_results = lgbm_fold1_model.evals_result_
axes[0].plot(lgb_eval_results['training']['rmse'], label='Train RMSE', color='blue', alpha=0.8)
axes[0].plot(lgb_eval_results['valid_1']['rmse'], label='Validation RMSE', color='orange', alpha=0.8)
axes[0].axvline(lgbm_fold1_model.best_iteration_, color='red', linestyle='--', label=f'Best Iteration ({lgbm_fold1_model.best_iteration_})')
axes[0].set_title(f'LightGBM Learning Curve (Fold 1)', fontsize=14)
axes[0].set_xlabel('Boosting Rounds', fontsize=11)
axes[0].set_ylabel('RMSE', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.6)
# axes[0].set_ylim(bottom=min(lgb_eval_results['valid_1']['rmse'])*0.98) # Zoom y-axis

# XGBoost Learning Curve
xgb_eval_results = xgb_fold1_model.evals_result()
axes[1].plot(xgb_eval_results['validation_0']['rmse'], label='Train RMSE', color='blue', alpha=0.8) # Note: validation_0 is train set here
axes[1].plot(xgb_eval_results['validation_1']['rmse'], label='Validation RMSE', color='orange', alpha=0.8)
axes[1].axvline(xgb_fold1_model.best_iteration, color='red', linestyle='--', label=f'Best Iteration ({xgb_fold1_model.best_iteration})')
axes[1].set_title(f'XGBoost Learning Curve (Fold 1)', fontsize=14)
axes[1].set_xlabel('Boosting Rounds', fontsize=11)
axes[1].set_ylabel('RMSE', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.6)
# axes[1].set_ylim(bottom=min(xgb_eval_results['validation_1']['rmse'])*0.98) # Zoom y-axis

plt.suptitle('Model Learning Curves (Example from Fold 1)', fontsize=18, y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()


# Convert dictionaries to DataFrames if needed
if isinstance(feature_importances_lgb, dict):
    feature_importances_lgb = pd.DataFrame(feature_importances_lgb)

if isinstance(feature_importances_xgb, dict):
    feature_importances_xgb = pd.DataFrame(feature_importances_xgb)

# Ensure feature importance DataFrames contain only numeric values
feature_importances_lgb = feature_importances_lgb.apply(pd.to_numeric, errors='coerce')
feature_importances_xgb = feature_importances_xgb.apply(pd.to_numeric, errors='coerce')

# Drop non-numeric columns if they exist
feature_importances_lgb = feature_importances_lgb.select_dtypes(include=[np.number])
feature_importances_xgb = feature_importances_xgb.select_dtypes(include=[np.number])

# Calculate mean importance and normalize
feature_importances_lgb['Mean_Importance'] = feature_importances_lgb.mean(axis=1)
feature_importances_xgb['Mean_Importance'] = feature_importances_xgb.mean(axis=1)

# Normalize (sum to 1)
feature_importances_lgb['Norm_Importance'] = feature_importances_lgb['Mean_Importance'] / feature_importances_lgb['Mean_Importance'].sum()
feature_importances_xgb['Norm_Importance'] = feature_importances_xgb['Mean_Importance'] / feature_importances_xgb['Mean_Importance'].sum()

# Get top N features
N_TOP_FEATURES = 30
lgb_imp = feature_importances_lgb.sort_values('Norm_Importance', ascending=False).head(N_TOP_FEATURES)
xgb_imp = feature_importances_xgb.sort_values('Norm_Importance', ascending=False).head(N_TOP_FEATURES)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(20, 12))
sns.barplot(x='Norm_Importance', y=lgb_imp.index, data=lgb_imp, palette='cubehelix', ax=axes[0])
axes[0].set_title(f'LightGBM Feature Importances (Top {N_TOP_FEATURES})', fontsize=15)
axes[0].set_xlabel('Mean Normalized Importance (Gain)', fontsize=12)
axes[0].set_ylabel('Features', fontsize=12)
axes[0].grid(axis='x', linestyle='--', alpha=0.7)

sns.barplot(x='Norm_Importance', y=xgb_imp.index, data=xgb_imp, palette='magma', ax=axes[1])
axes[1].set_title(f'XGBoost Feature Importances (Top {N_TOP_FEATURES})', fontsize=15)
axes[1].set_xlabel('Mean Normalized Importance (Gain)', fontsize=12)
axes[1].set_ylabel('')
axes[1].grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout(pad=3.0)
plt.suptitle('Feature Importances (Averaged over Folds)', fontsize=20, y=1.03)
plt.show()


# Ensure predictions are numpy arrays
oof_preds_lgb = np.array(oof_preds_lgb).reshape(-1)
oof_preds_xgb = np.array(oof_preds_xgb).reshape(-1)
test_preds_lgb = np.array(test_preds_lgb).reshape(-1)
test_preds_xgb = np.array(test_preds_xgb).reshape(-1)

# Ensure predictions have the same length
min_oof_len = min(len(oof_preds_lgb), len(oof_preds_xgb))
min_test_len = min(len(test_preds_lgb), len(test_preds_xgb))

oof_preds_lgb, oof_preds_xgb = oof_preds_lgb[:min_oof_len], oof_preds_xgb[:min_oof_len]
test_preds_lgb, test_preds_xgb = test_preds_lgb[:min_test_len], test_preds_xgb[:min_test_len]

# Ensemble OOF and Test predictions
oof_preds_ensemble = (oof_preds_lgb + oof_preds_xgb) / 2
test_preds_ensemble = (test_preds_lgb + test_preds_xgb) / 2

# Evaluate Ensemble OOF Performance
overall_oof_rmse_ensemble = rmse(y[:min_oof_len], oof_preds_ensemble)
print("\n--- OOF Performance Summary ---")
print(f"LightGBM OOF RMSE:     {overall_oof_rmse_lgb:.5f}")
print(f"XGBoost OOF RMSE:      {overall_oof_rmse_xgb:.5f}")
print(f"Ensemble (Avg) OOF RMSE: {overall_oof_rmse_ensemble:.5f}")

# Check if ensemble improved over the best single model
best_single_model_rmse = min(overall_oof_rmse_lgb, overall_oof_rmse_xgb)
if overall_oof_rmse_ensemble < best_single_model_rmse:
    print(f"Ensemble improved OOF RMSE by {best_single_model_rmse - overall_oof_rmse_ensemble:.5f}")
else:
    print("Ensemble did not improve OOF RMSE over the best single model.")

# Post-processing: Ensure predictions are non-negative
print(f"\nMin prediction before clipping: {test_preds_ensemble.min():.4f}")
test_preds_ensemble = np.maximum(test_preds_ensemble, 0)
print(f"Min prediction after clipping:  {test_preds_ensemble.min():.4f}")


# --- Visualization of Predictions ---

# 1. Actual vs. OOF Predictions Scatter Plot
plt.figure(figsize=(10, 10))
plt.scatter(y, oof_preds_ensemble, alpha=0.2, s=10, label=f'Ensemble (RMSE: {overall_oof_rmse_ensemble:.4f})', color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction (y=x)')
plt.xlabel('Actual Listening Time', fontsize=13)
plt.ylabel('Predicted Listening Time (OOF)', fontsize=13)
plt.title('Actual vs. Out-of-Fold Predictions (Ensemble)', fontsize=16)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal') # Ensure equal scaling for x and y axes
plt.show()

# 2. Distribution Comparison Plot (OOF vs Test vs Actual)
plt.figure(figsize=(12, 7))
sns.kdeplot(y, label="Actual Target", color="blue", fill=True, alpha=0.3, linewidth=1.5)
sns.kdeplot(oof_preds_ensemble, label="OOF Predictions (Ensemble)", color="red", fill=True, alpha=0.3, linewidth=1.5)
sns.kdeplot(test_preds_ensemble, label="Test Predictions (Ensemble)", color="green", fill=True, alpha=0.3, linewidth=1.5)
plt.title('Prediction Distribution Comparison', fontsize=16)
plt.xlabel('Listening Time (minutes)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 3. Residual Plot (OOF Ensemble)
residuals = y - oof_preds_ensemble
plt.figure(figsize=(14, 6))
plt.scatter(oof_preds_ensemble, residuals, alpha=0.15, s=10, color='purple')
plt.axhline(y=0, color='black', linestyle='--', lw=1.5)
# Add smoothed line to check for trends
sns.regplot(x=oof_preds_ensemble, y=residuals, scatter=False, lowess=True, line_kws={'color': 'red', 'lw': 1.5})
plt.xlabel('Predicted Listening Time (OOF)', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.title('Residual Plot (OOF Ensemble Predictions)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# Ensure test_preds_ensemble is a 1D NumPy array
test_preds_ensemble = np.array(test_preds_ensemble).flatten()

# Ensure test_ids is also a 1D array
test_ids = np.array(test_ids).flatten()

# Debugging: Print lengths
print(f"test_ids length: {len(test_ids)}, test_preds_ensemble length: {len(test_preds_ensemble)}")

# Check if the lengths of the two arrays are equal
if len(test_ids) != len(test_preds_ensemble):
    print(f"Mismatch detected! Trimming predictions to match test_ids.")
    # Trim to the length of the smaller array
    min_length = min(len(test_ids), len(test_preds_ensemble))
    test_ids = test_ids[:min_length]
    test_preds_ensemble = test_preds_ensemble[:min_length]

# Define target column name (ensure it's a string)
TARGET = "prediction"

# Create submission DataFrame
submission_df = pd.DataFrame({
    "id": test_ids,
    TARGET: test_preds_ensemble
})

# Save submission file
submission_filename = "submission.csv"
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file '{submission_filename}' created successfully!")
print("Submission File Head:")
print(submission_df.head())
print(f"Submission file shape: {submission_df.shape}")

