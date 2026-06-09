from graphviz import Digraph

# Directed graph, left-to-right
dot = Digraph(comment='Accident Risk Modeling Pipeline', format='png')
dot.attr(rankdir='LR', size='15,20')
dot.attr(fontsize='16')
dot.attr('node', shape='box', style='filled,rounded', color='lightblue', fontname='Helvetica', fontsize='16', width='2', height='1')

# Nodes
dot.node('A', '1. Load & Combine Data')
dot.node('B', '2. Encoding & Feature Engineering')
dot.node('C', '3. Box-Cox Transformation')
dot.node('D', '4. Prepare Stratified Folds')
dot.node('E', '5. Base Model Training (LGB, XGB, CAT)')
dot.node('F', '6. Weighted Ensemble')
dot.node('G', '7. Stacking Meta-Model')
dot.node('H', '8. Prediction & Submission')

# Edges
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH'])

# Output
dot.render('accident_risk_pipeline', cleanup=False)
dot


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
import logging


warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)

# --------------------------------------------------------------
# Retro Synthwave Color Palette and Custom Styling
# --------------------------------------------------------------
def set_synthwave_palette(style="whitegrid", context="notebook", font_family="sans-serif"):
    """Set custom Retro Synthwave color palette and styling for visualizations"""
    palette = ['#f72585', '#b5179e', '#7209b7', '#560bad', '#480ca8',
               '#3a0ca3', '#3f37c9', '#4361ee', '#4895ef', '#4cc9f0']
    
    sns.set_palette(palette)
    sns.set_style(style)
    sns.set_context(context, font_scale=1.1)
    
    plt.rcParams.update({
        'axes.titlepad': 20,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'font.family': font_family,
        'figure.autolayout': True,
        'axes.edgecolor': '#3a0ca3',
        'axes.facecolor': '#ffffff',
        'figure.facecolor': '#ffffff',
        'axes.labelcolor': '#3a0ca3',
        'axes.titlecolor': '#3a0ca3',
        'xtick.color': '#3a0ca3',
        'ytick.color': '#3a0ca3',
        'grid.color': '#4cc9f0',
        'grid.alpha': 0.5
    })
    
    return palette


palette = set_synthwave_palette()


# -----------------------------
# Load datasets
# -----------------------------
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


# -----------------------------
# Dataset Overview
# -----------------------------
numeric_columns = [col for col in train_df.columns if train_df[col].dtype in [np.int64, np.float64] and col not in ['id', 'accident_risk']]
categorical_columns = [col for col in train_df.columns if col not in numeric_columns + ['id', 'accident_risk']]
target_column = 'accident_risk'

# -----------------------------
# Single Figure with All Plots Except Correlation
# -----------------------------
# Calculate grid size: 1 for target, len(numeric) for histograms, len(numeric) for boxplots, len(categorical) for countplots
total_plots = 1 + len(numeric_columns) * 2 + len(categorical_columns)
n_cols = 4
n_rows = (total_plots + n_cols - 1) // n_cols

fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))  # 4x3 inches per subplot
plot_idx = 1

# Target Variable Countplot
plt.subplot(n_rows, n_cols, plot_idx)
top_n = 10
order = train_df[target_column].value_counts().iloc[:top_n].index
sns.countplot(data=train_df, x=target_column, order=order)
plt.title('Target Distribution', fontsize=10, fontweight='bold')
plt.xlabel(target_column, fontsize=8)
plt.ylabel('Count', fontsize=8)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.grid(True, alpha=0.3)
if train_df[target_column].nunique() > top_n:
    plt.annotate(f'Top {top_n}', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=7, color='red')
plot_idx += 1

# Numeric Feature Histograms
for feature in numeric_columns:
    plt.subplot(n_rows, n_cols, plot_idx)
    sns.histplot(train_df[feature], kde=True, bins=20)
    plt.title(f'{feature} Dist', fontsize=10, fontweight='bold')
    plt.xlabel(feature, fontsize=8)
    plt.ylabel('Count', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True, alpha=0.3)
    if train_df[feature].skew() > 1:
        plt.annotate('Skewed', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=7, color='red')
    plot_idx += 1

# Boxplots Numeric vs Target
top_n_target = 5
target_order = train_df[target_column].value_counts().iloc[:top_n_target].index
for feature in numeric_columns:
    plt.subplot(n_rows, n_cols, plot_idx)
    filtered_data = train_df[train_df[target_column].isin(target_order)]
    sns.boxplot(data=filtered_data, x=target_column, y=feature, order=target_order)
    plt.title(f'{feature} by Target', fontsize=10, fontweight='bold')
    plt.xlabel(target_column, fontsize=8)
    plt.ylabel(feature, fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True, alpha=0.3)
    medians = filtered_data.groupby(target_column)[feature].median()
    if medians.max() - medians.min() > medians.mean() * 0.5:
        plt.annotate('Strong', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=7, color='green')
    plot_idx += 1

# Categorical Feature Countplots
if categorical_columns:
    for feature in categorical_columns:
        plt.subplot(n_rows, n_cols, plot_idx)
        order = train_df[feature].value_counts().iloc[:top_n].index
        sns.countplot(data=train_df, x=feature, order=order)
        plt.title(f'{feature} Dist', fontsize=10, fontweight='bold')
        plt.xlabel(feature, fontsize=8)
        plt.ylabel('Count', fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        plt.grid(True, alpha=0.3)
        if train_df[feature].nunique() > top_n:
            plt.annotate(f'Top {top_n}', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=7, color='orange')
        elif train_df[feature].value_counts(normalize=True).max() > 0.7:
            plt.annotate('Imbalanced', xy=(0.5, 0.85), xycoords='axes fraction', fontsize=7, color='red')
        plot_idx += 1

plt.tight_layout(pad=1.0)
plt.show()

# -----------------------------
# Feature Correlation Matrix 
# -----------------------------
FIGSIZE = (12, 10)
fig3, ax3 = plt.subplots(figsize=FIGSIZE)
fig3.suptitle('Feature Correlation Matrix', 
              fontsize=16, fontweight='bold', y=0.98)

num_df = train_df[numeric_columns + [target_column]].copy()
corr_with_target = num_df.corr()[target_column].abs().sort_values(ascending=False)
top_n_features_for_corr = corr_with_target.iloc[1:5].index.tolist()
corr_data = num_df[top_n_features_for_corr + [target_column]].copy()

corr_matrix = corr_data.corr(method='pearson')

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(corr_matrix, 
            mask=mask,
            ax=ax3,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt='.2f',
            annot_kws={'size': 9, 'weight': 'bold'},
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
            square=True,
            linewidths=0.5,
            linecolor='black')

ax3.tick_params(axis='x', labelrotation=45, labelsize=9)
ax3.tick_params(axis='y', labelrotation=0, labelsize=9)
    

plt.tight_layout()
plt.show()


train_df.head()


train_df.info()


display(train_df.describe().T)


# Ordinal features mappings
ordinal_features = ['num_lanes', 'speed_limit', 'lighting', 'weather']
ordinal_mappings = {
    'num_lanes': {1: 0, 2: 1, 3: 2, 4: 3},
    'speed_limit': {25: 0, 35: 1, 45: 2, 60: 3, 70: 4},
    'lighting': {'daylight': 0, 'dim': 1, 'night': 2},
    'weather': {'clear': 0, 'rainy': 1, 'foggy': 2}
}

nominal_features = ['road_type', 'time_of_day']
binary_features = ['road_signs_present', 'public_road', 'holiday']

# Model hyperparameters (optimized with Optuna)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'device_type': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1,
    'learning_rate': 0.037,
    'num_leaves': 154,
    'max_depth': 15,
    'min_data_in_leaf': 44,
    'feature_fraction': 0.716,
    'bagging_fraction': 0.989,
    'bagging_freq': 9,
    'reg_alpha': 0.36,
    'reg_lambda': 0.069,
    'min_gain_to_split': 0.0025,
    'lambda_l1': 0.0048,
    'lambda_l2': 0.0054,
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',
    'seed': 42,
    'verbosity': 0,
    'eta': 0.021,
    'max_depth': 11,
    'min_child_weight': 5,
    'gamma': 3.39e-05,
    'subsample': 0.91,
    'colsample_bytree': 0.52,
    'reg_alpha': 3.46e-06,
    'reg_lambda': 2.74e-07,
    'max_bin': 870,
    'grow_policy': 'depthwise',
    'max_leaves': 138
}

cat_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'bootstrap_type': 'Bayesian',
    'task_type': 'GPU',
    'bagging_temperature': 0.027,
    'depth': 9,
    'learning_rate': 0.017,
    'l2_leaf_reg': 0.0885,
    'min_child_samples': 24,
    'iterations': 2384,
    'border_count': 187,
    'random_strength': 0.044,
    'random_seed': 42,
    'verbose': 300
}

# Stacking meta-model parameters
meta_alphas = np.logspace(-6, 6, 200)
meta_l1_ratios = [.1, .5, .7, .9, .95, .99, 1]

# Common CV settings
n_folds = 5
n_bins = 5  # For stratified binning
random_state = 42
early_stopping_rounds = 150


###### FUNCTION DEFINITIONS

def box_transform(X, test):
    """Apply Box-Cox transformation to specified columns without data leakage"""
    box_cols = ['curvature']
    lambda_dict = {}
    
    for column in box_cols:
        # Ensure positive values
        if (X[column] <= 0).any() or (test[column] <= 0).any():
            print(f"Warning: Column {column} contains non-positive values. Shifting to positive.")
            shift = abs(min(X[column].min(), test[column].min())) + 1
            X[column] += shift
            test[column] += shift
        
        X_temp, fitted_lambda = stats.boxcox(X[column])
        X[column] = X_temp
        lambda_dict[column] = fitted_lambda
        
        test[column] = stats.boxcox(test[column], fitted_lambda)
    
    return X, test, lambda_dict

def create_advanced_features(df):
    """Advanced feature engineering based on domain knowledge and feature importance"""
    df_new = df.copy()
    
    # ========== 1. CURVE & SPEED INTERACTIONS (Most important feature: curvature) ==========
    df_new['curve_speed_risk'] = df_new['curvature'] * df_new['speed_limit']
    df_new['curve_severity'] = np.log1p(df_new['curvature'])
    df_new['curve_per_lane'] = df_new['curvature'] / (df_new['num_lanes'] + 1)
    df_new['unsafe_curve_speed'] = df_new['speed_limit'] / (1 + df_new['curvature'])
    
    # ========== 2. VISIBILITY CONDITIONS RISK (lighting + weather combined) ==========
    lighting_risk = df_new['lighting'].map({0: 0, 1: 1, 2: 2})
    weather_risk = df_new['weather'].map({0: 0, 1: 1.5, 2: 2})
    df_new['poor_visibility_score'] = lighting_risk + weather_risk
    df_new['visibility_category'] = pd.cut(df_new['poor_visibility_score'], 
                                            bins=[-1, 0.5, 2.5, 5], 
                                            labels=[0, 1, 2]).astype(int)
    df_new['night_bad_weather'] = ((df_new['lighting'] == 2) & 
                                    (df_new['weather'] > 0)).astype(int)
    df_new['visibility_speed_risk'] = df_new['poor_visibility_score'] * df_new['speed_limit']
    df_new['visibility_curve_risk'] = df_new['poor_visibility_score'] * df_new['curvature']
    
    # ========== 3. SPEED RISK FACTORS ==========
    df_new['speed_squared'] = df_new['speed_limit'] ** 2
    df_new['speed_cubed'] = df_new['speed_limit'] ** 3
    df_new['speed_per_lane'] = df_new['speed_limit'] / (df_new['num_lanes'] + 1)
    
    # ========== 4. ACCIDENT HISTORY BASED RISK (num_reported_accidents) ==========
    df_new['accident_density'] = df_new['num_reported_accidents'] / (df_new['num_lanes'] + 1)
    df_new['accident_history_level'] = pd.cut(df_new['num_reported_accidents'], 
                                               bins=[-1, 0, 2, 5, 100], 
                                               labels=[0, 1, 2, 3]).astype(int)
    df_new['accident_prone_curve'] = df_new['num_reported_accidents'] * df_new['curvature']
    df_new['accident_speed_risk'] = df_new['num_reported_accidents'] * df_new['speed_limit']
    df_new['log_accidents'] = np.log1p(df_new['num_reported_accidents'])    
    
    # ========== 5. ROAD TYPE RISK FACTORS ==========
    df_new['urban_poor_visibility'] = ((df_new['road_type'] == 'urban') & 
                                        (df_new['poor_visibility_score'] >= 2)).astype(int)
    
    # ========== 6. INFRASTRUCTURE SAFETY FEATURES ==========
    df_new['safety_infrastructure'] = (df_new['road_signs_present'].astype(int) + 
                                        df_new['public_road'].astype(int))
        
    # ========== 7. COMPOSITE RISK SCORES ==========
    df_new['composite_danger_score'] = (
        0.4 * (df_new['curvature'] / df_new['curvature'].max()) +  
        0.3 * (df_new['poor_visibility_score'] / 4) +  
        0.2 * (df_new['speed_limit'] / 70) +  
        0.1 * (df_new['num_reported_accidents'] / df_new['num_reported_accidents'].max())  
    )
    df_new['physical_risk'] = (df_new['curve_speed_risk'] / 
                                (df_new['num_lanes'] + 1))
    df_new['environmental_risk'] = df_new['poor_visibility_score'] * (1 + df_new['curvature'] * 0.1)
    
    # ========== 8. STATISTICAL TRANSFORMATIONS ==========
    df_new['curvature_sqrt'] = np.sqrt(df_new['curvature'])
    df_new['speed_log'] = np.log1p(df_new['speed_limit'])
    df_new['curve_to_speed_ratio'] = df_new['curvature'] / (df_new['speed_limit'] + 1)
    df_new['accidents_to_curve_ratio'] = df_new['num_reported_accidents'] / (df_new['curvature'] + 1)
    df_new["curvature_per_lane"] = df_new["curvature"] / (df_new["num_lanes"] + 1e-5)
    
    return df_new

def encode_data(df, is_train=True):
    df_encoded = df.copy().drop('id', axis=1, errors='ignore')
    
    # Ordinal encoding
    for feat, mapping in ordinal_mappings.items():
        df_encoded[feat] = df_encoded[feat].map(mapping)
    
    # Binary encoding
    for feat in binary_features:
        df_encoded[feat] = df_encoded[feat].astype(int)
    
    # Feature engineering
    df_encoded = create_advanced_features(df_encoded)
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_features, 
                                 prefix=nominal_features, drop_first=True)
    
    return df_encoded


def prepare_folds(X, y):
    """Create stratified folds"""
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(skf.split(X, y_binned))

def train_lgb(X, y, folds):
    boosters = []
    fold_rmses = []
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(test_encoded), n_folds))
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\nLGB Fold {fold_idx + 1} processing...")
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        booster = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=10000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        boosters.append(booster)
        
        oof_preds[val_idx] = booster.predict(X_val, num_iteration=booster.best_iteration)
        test_preds[:, fold_idx] = booster.predict(test_encoded, num_iteration=booster.best_iteration)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_rmses.append(fold_rmse)
        print(f"LGB Fold {fold_idx + 1} RMSE: {fold_rmse:.6f}")
    
    mean_rmse = np.mean(fold_rmses)
    print(f"LGB Average CV RMSE: {mean_rmse:.6f}")
    
    return oof_preds, np.mean(test_preds, axis=1), mean_rmse

def train_xgb(X, y, folds):
    boosters = []
    fold_rmses = []
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(test_encoded), n_folds))
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\nXGB Fold {fold_idx + 1} processing...")
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(test_encoded)
        
        booster = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=10000,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=200
        )
        
        boosters.append(booster)
        
        oof_preds[val_idx] = booster.predict(dval)
        test_preds[:, fold_idx] = booster.predict(dtest)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_rmses.append(fold_rmse)
        print(f"XGB Fold {fold_idx + 1} RMSE: {fold_rmse:.6f}")
    
    mean_rmse = np.mean(fold_rmses)
    print(f"XGB Average CV RMSE: {mean_rmse:.6f}")
    
    return oof_preds, np.mean(test_preds, axis=1), mean_rmse

def train_cat(X, y, folds):
    models = []
    fold_rmses = []
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(test_encoded), n_folds))
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\nCAT Fold {fold_idx + 1} processing...")
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        
        model = CatBoostRegressor(**cat_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=early_stopping_rounds,
            verbose=300
        )
        
        models.append(model)
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds[:, fold_idx] = model.predict(test_encoded)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_rmses.append(fold_rmse)
        print(f"CAT Fold {fold_idx + 1} RMSE: {fold_rmse:.6f}")
    
    mean_rmse = np.mean(fold_rmses)
    print(f"CAT Average CV RMSE: {mean_rmse:.6f}")
    
    return oof_preds, np.mean(test_preds, axis=1), mean_rmse

def weighted_ensemble(preds_list, rmses):
    """Weighted ensemble: Weights = 1 / (RMSE ^ 2)"""
    weights = [1 / (rmse ** 2 + 1e-6) for rmse in rmses]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    ensemble_preds = np.zeros(len(preds_list[0]))
    for preds, weight in zip(preds_list, normalized_weights):
        ensemble_preds += preds * weight
    
    print("\nEnsemble Weights:")
    for model_name, weight in zip(['LGB', 'XGB', 'CAT'], normalized_weights):
        print(f"{model_name}: {weight:.4f}")
    
    return ensemble_preds

def train_stacking_meta(X_meta, y, folds):
    """Stacking meta-model training ElasticNetCV"""
    meta_models = []
    fold_rmses = []
    oof_preds = np.zeros(len(X_meta))
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\nStacking Meta Fold {fold_idx + 1} processing...")
        
        X_train = X_meta.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X_meta.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        meta_model = ElasticNetCV(alphas=meta_alphas, l1_ratio=meta_l1_ratios, cv=5, max_iter=10000, tol=1e-4)
        meta_model.fit(X_train, y_train)
        
        meta_models.append(meta_model)
        
        oof_preds[val_idx] = meta_model.predict(X_val)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_rmses.append(fold_rmse)
        print(f"Stacking Meta Fold {fold_idx + 1} RMSE: {fold_rmse:.6f}")
        print(f"Selected alpha: {meta_model.alpha_:.6f}, l1_ratio: {meta_model.l1_ratio_:.6f}")
    
    mean_rmse = np.mean(fold_rmses)
    print(f"Stacking Meta Average CV RMSE: {mean_rmse:.6f}")
    
    return oof_preds, meta_models, mean_rmse

def predict_stacking_meta(meta_models, X_meta_test):
    """Stacking meta-model predictions on test average"""
    test_preds = np.zeros((len(X_meta_test), len(meta_models)))
    for i, meta_model in enumerate(meta_models):
        test_preds[:, i] = meta_model.predict(X_meta_test)
    return np.mean(test_preds, axis=1)


###### DATA PREPARATION

# Apply encoding
train_encoded = encode_data(train_df, is_train=True)
test_encoded = encode_data(test_df, is_train=False)

# Train/test split
X = train_encoded.drop('accident_risk', axis=1)
y = train_encoded['accident_risk']

# Apply Box-Cox transformation after encoding but before training
X, test_encoded, _ = box_transform(X, test_encoded)

print(f"Combined Training shape: {X.shape}, Test shape: {test_encoded.shape}")
print(f"Total features: {X.shape[1]}")
print(f"\nNew features created: {X.shape[1] - 14}")

# Prepare folds
folds = prepare_folds(X, y)
print(f"Stratified folds created. Target distribution balanced in each fold.")

###### BASE MODEL TRAINING

lgb_oof, lgb_test, lgb_rmse = train_lgb(X, y, folds)
xgb_oof, xgb_test, xgb_rmse = train_xgb(X, y, folds)
cat_oof, cat_test, cat_rmse = train_cat(X, y, folds)

###### WEIGHTED ENSEMBLE

oof_preds_list = [lgb_oof, xgb_oof, cat_oof]
test_preds_list = [lgb_test, xgb_test, cat_test]
rmses = [lgb_rmse, xgb_rmse, cat_rmse]

weighted_oof = weighted_ensemble(oof_preds_list, rmses)
weighted_test = weighted_ensemble(test_preds_list, rmses)

weighted_oof_rmse = np.sqrt(mean_squared_error(y, weighted_oof))
print(f"Weighted Ensemble OOF RMSE: {weighted_oof_rmse:.6f}")

###### STACKED ENSEMBLE

X_meta = pd.DataFrame({
    'lgb_oof': lgb_oof,
    'xgb_oof': xgb_oof,
    'cat_oof': cat_oof,
    'weighted_oof': weighted_oof
})

X_meta_test = pd.DataFrame({
    'lgb_oof': lgb_test,
    'xgb_oof': xgb_test,
    'cat_oof': cat_test,
    'weighted_oof': weighted_test
})

# Train stacking meta-model
stack_oof, meta_models, stack_rmse = train_stacking_meta(X_meta, y, folds)

# Stacking predictions on test
stack_test_preds = predict_stacking_meta(meta_models, X_meta_test)

###### RESULTS AND SUBMISSION

submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': stack_test_preds
})
submission.to_csv('submission_stacked_weighted_ensemble.csv', index=False)
print("\nâœ… Submission saved: 'submission_stacked_weighted_ensemble.csv'")
print(submission.head(10))

print(f"\nStacked Ensemble OOF RMSE: {stack_rmse:.6f}")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb


# Placeholder for the LightGBM model (modify based on actual train_lgb implementation)
def get_lgb_feature_importance(X, y, folds):
    # Example: Train a single LightGBM model (replace with actual train_lgb logic)
    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(X, y)  # Simplified; replace with actual fold-based training if needed
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    return feature_importance

# Extract feature importance
feature_importance = get_lgb_feature_importance(X, y, folds)

# Select top 12 features
top_n = 12
top_features = feature_importance.head(top_n)['feature'].tolist()

# Correlation matrix
corr_data = X[top_features].copy()
corr_data['accident_risk'] = y
corr_matrix = corr_data.corr(method='pearson')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1]})

# Heatmap (Correlation Matrix)
sns.heatmap(
    corr_matrix,
    mask=mask,
    ax=ax1,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 9, 'weight': 'bold'},
    cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
    square=True,
    linewidths=0.5,
    linecolor='black'
)
ax1.set_title('Top 12 Features + accident_risk Correlation', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.tick_params(axis='y', rotation=0, labelsize=10)

# Feature Importance Bar Plot
sns.barplot(
    x='importance',
    y='feature',
    data=feature_importance.head(top_n),
    ax=ax2,
    color='steelblue'
)
ax2.set_title('Top 12 Feature Importances (LightGBM)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_ylabel('Feature', fontsize=12)
ax2.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.show()

