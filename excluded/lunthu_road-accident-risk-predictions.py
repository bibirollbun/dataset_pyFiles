import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)


# Display basic information about the training data

print(train_df.info())
train_df.head()



target_col = 'accident_risk' 

#target column distribution
if target_col in train_df.columns:
    print(train_df[target_col].value_counts(normalize=True))
    
    # Visualize target distribution
    plt.figure(figsize=(8, 5))
    train_df[target_col].value_counts().plot(kind='bar')
    plt.title('Distribution of Accident Risk')
    plt.xlabel('Accident Risk')
    plt.ylabel('Count')
    plt.xticks(rotation=20)
    plt.show()


# Identify numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

# Remove ID and target columns if present
if 'id' in numerical_cols:
    numerical_cols.remove('id')
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")


# Plot distributions of numerical features
if len(numerical_cols) > 0:
    n_cols = min(len(numerical_cols), 12)  # Limit to 12 plots
    fig, axes = plt.subplots(nrows=(n_cols+3)//4, ncols=4, figsize=(15, (n_cols+3)//4 * 3))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols[:n_cols]):
        axes[i].hist(train_df[col].dropna(), bins=30, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Hide extra subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# Analyze categorical features
if len(categorical_cols) > 0:
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        print(f"\n=== {col} Value Counts ===")
        print(train_df[col].value_counts().head(10))
        
        if train_df[col].nunique() <= 10:
            plt.figure(figsize=(10, 5))
            train_df[col].value_counts().plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


# Correlation matrix for numerical features
numerical_cols_corr = numerical_cols
numerical_cols_corr.append('accident_risk')
print(numerical_cols)
if len(numerical_cols_corr) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = train_df[numerical_cols_corr].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()


label_encoders = {}
categorical_cols_encoded = []

for col in categorical_cols:
    if col in train_df.columns and col in test_df.columns:
        le = LabelEncoder()
        
        # Fit on combined unique values from train and test
        combined_values = pd.concat([train_df[col], test_df[col]]).unique()
        le.fit(combined_values)
        
        # Transform both datasets
        train_df[col + '_encoded'] = le.transform(train_df[col])
        test_df[col + '_encoded'] = le.transform(test_df[col])
        categorical_cols_encoded.append(col+'_encoded')
        label_encoders[col] = le



#I've tried working with polynomial features - I cannot say it made the over score better - it became worse, but maybe it can be valuable for anyone

from sklearn.preprocessing import PolynomialFeatures  

numerical_cols.remove(target_col)
def create_features(df):

    poly = PolynomialFeatures(degree=2)
    poly_index = 0
    poly_features = []
    poly_df_categorical = poly.fit_transform(df[categorical_cols_encoded]).T
    for i in poly_df_categorical:
        
        buf_series = pd.Series(i, name = "poly_{}".format(poly_index))
        df = pd.concat([df, buf_series], axis=1, join="inner")
        poly_features.append("poly_{}".format(poly_index))
        poly_index+=1
        


    poly_2 = PolynomialFeatures(degree=5, interaction_only = True)
    poly_df_numerical = poly.fit_transform(df[numerical_cols]).T
    for i in poly_df_numerical:
        
        buf_series = pd.Series(i, name = "poly_{}".format(poly_index))
        df = pd.concat([df, buf_series], axis=1, join="inner")
        poly_features.append("poly_{}".format(poly_index))
        poly_index+=1
        
    print(poly_features)    
    return df, poly_features



train_df, train_features = create_features(train_df)
test_df, test_features = create_features(test_df)



train_df


# Prepare feature columns
feature_cols = []

# Add numerical columns
feature_cols.extend([col for col in numerical_cols if col in train_df.columns])

# Add encoded categorical columns
feature_cols.extend([col + '_encoded' for col in categorical_cols 
                    if col + '_encoded' in train_df.columns])


#feature_cols = train_features
print(f"Total features for modeling: {len(feature_cols)}")
print(f"Features: {feature_cols[:10]}...")  # Show first 10 features

# Prepare X and y
X = train_df[feature_cols]
y = train_df[target_col]

# Prepare test features
X_test = test_df[feature_cols]

# Store test IDs if present
test_ids = test_df['id'] if 'id' in test_df.columns else None


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

scaler = StandardScaler()

# Fit on training data and transform all sets
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate multiple regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Since target is between 0 and 1, we can also calculate percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Dictionary to store model results
model_results = {}

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_val_scaled)
# Clip predictions to [0, 1] range since target is probability
lr_pred = np.clip(lr_pred, 0, 1)
lr_metrics = evaluate_model(y_val, lr_pred, "Linear Regression")
model_results['Linear Regression'] = lr_metrics

# Ridge Regression
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = np.clip(ridge_model.predict(X_val_scaled), 0, 1)
ridge_metrics = evaluate_model(y_val, ridge_pred, "Ridge Regression")
model_results['Ridge Regression'] = ridge_metrics

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_pred = np.clip(rf_pred, 0, 1)
rf_metrics = evaluate_model(y_val, rf_pred, "Random Forest")
model_results['Random Forest'] = rf_metrics

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_val)
gb_pred = np.clip(gb_pred, 0, 1)
gb_metrics = evaluate_model(y_val, gb_pred, "Gradient Boosting")
model_results['Gradient Boosting'] = gb_metrics


xgb_params = {
    'objective': 'reg:squarederror',  # For regression
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.02,
    'n_estimators': 2000,
    'reg_alpha': 1e-3,              
    'reg_lambda': 1e-3,             # L2 regularization
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=False
)

xgb_pred = xgb_model.predict(X_val)
xgb_pred = np.clip(xgb_pred, 0, 1)
xgb_metrics = evaluate_model(y_val, xgb_pred, "XGBoost")
model_results['XGBoost'] = xgb_metrics


# LightGBM model
print("\nTraining LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1,
    'force_col_wise': True
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

lgb_pred = lgb_model.predict(X_val)
lgb_pred = np.clip(lgb_pred, 0, 1)
lgb_metrics = evaluate_model(y_val, lgb_pred, "LightGBM")
model_results['LightGBM'] = lgb_metrics


# CatBoost model
print("\nTraining CatBoost...")
# Identify categorical features for CatBoost
cat_features_indices = [i for i, col in enumerate(feature_cols) 
                        if col.endswith('_encoded')]

catboost_params = {
    'iterations': 2000,
    'learning_rate': 0.02,
    'depth': 6,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 20
}

# Create Pool objects for CatBoost
train_pool = cb.Pool(X_train, y_train, cat_features=cat_features_indices)
val_pool = cb.Pool(X_val, y_val, cat_features=cat_features_indices)

cb_model = cb.CatBoostRegressor(**catboost_params)
cb_model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True,
    verbose=False
)

cb_pred = cb_model.predict(X_val)
cb_pred = np.clip(cb_pred, 0, 1)
cb_metrics = evaluate_model(y_val, cb_pred, "CatBoost")
model_results['CatBoost'] = cb_metrics


# Create comparison dataframe
comparison_df = pd.DataFrame(model_results).T
comparison_df = comparison_df.sort_values('rmse')

print("\n=== Model Comparison (sorted by RMSE) ===")
print(comparison_df)

# Visualize model performance
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# RMSE comparison
axes[0, 0].bar(comparison_df.index, comparison_df['rmse'], color='steelblue')
axes[0, 0].set_title('Model Comparison - RMSE (Lower is Better)')
axes[0, 0].set_xlabel('Models')
axes[0, 0].set_ylabel('RMSE')
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(comparison_df['rmse']):
    axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)

# MAE comparison
axes[0, 1].bar(comparison_df.index, comparison_df['mae'], color='forestgreen')
axes[0, 1].set_title('Model Comparison - MAE (Lower is Better)')
axes[0, 1].set_xlabel('Models')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(comparison_df['mae']):
    axes[0, 1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)

# R² comparison
axes[1, 0].bar(comparison_df.index, comparison_df['r2'], color='coral')
axes[1, 0].set_title('Model Comparison - R² (Higher is Better)')
axes[1, 0].set_xlabel('Models')
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(comparison_df['r2']):
    axes[1, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)

# MAPE comparison
axes[1, 1].bar(comparison_df.index, comparison_df['mape'], color='mediumpurple')
axes[1, 1].set_title('Model Comparison - MAPE (Lower is Better)')
axes[1, 1].set_xlabel('Models')
axes[1, 1].set_ylabel('MAPE (%)')
axes[1, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(comparison_df['mape']):
    axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

plt.suptitle('Model Performance Comparison - Regression Metrics', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Select best model based on RMSE
best_model_name = comparison_df.index[0]  # First model after sorting by RMSE
print(f"\nBest performing model: {best_model_name}")
print(f"  RMSE: {comparison_df.loc[best_model_name, 'rmse']:.4f}")
print(f"  MAE:  {comparison_df.loc[best_model_name, 'mae']:.4f}")
print(f"  R²:   {comparison_df.loc[best_model_name, 'r2']:.4f}")


# Select best model based on RMSE
best_model_name = comparison_df.index[0]  # First model after sorting by RMSE
print(f"\nBest performing model: {best_model_name}")
print(f"  RMSE: {comparison_df.loc[best_model_name, 'rmse']:.4f}")
print(f"  MAE:  {comparison_df.loc[best_model_name, 'mae']:.4f}")
print(f"  R²:   {comparison_df.loc[best_model_name, 'r2']:.4f}")


# Get predictions from best model for residual analysis
if best_model_name == 'XGBoost':
    best_predictions = xgb_pred
elif best_model_name == 'Random Forest':
    best_predictions = rf_pred
elif best_model_name == 'Gradient Boosting':
    best_predictions = gb_pred
elif best_model_name == 'Ridge Regression':
    best_predictions = ridge_pred
elif best_model_name == 'LightGBM' and 'LightGBM' in model_results:
    best_predictions = lgb_pred
else:
    best_predictions = lr_pred

# Calculate residuals
residuals = y_val - best_predictions

# Residual plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residuals vs Predicted
axes[0, 0].scatter(best_predictions, residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Predicted Values')

# Histogram of residuals
axes[0, 1].hist(residuals, bins=50, edgecolor='black')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Residuals')
axes[0, 1].axvline(x=0, color='r', linestyle='--')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

# Actual vs Predicted
axes[1, 1].scatter(y_val, best_predictions, alpha=0.5)
axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Values')
axes[1, 1].set_ylabel('Predicted Values')
axes[1, 1].set_title('Actual vs Predicted Values')
axes[1, 1].legend()
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_ylim([0, 1])

plt.suptitle(f'Residual Analysis - {best_model_name}', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print(f"\nResidual Statistics:")
print(f"Mean Residual: {residuals.mean():.6f}")
print(f"Std Residual:  {residuals.std():.4f}")
print(f"Min Residual:  {residuals.min():.4f}")
print(f"Max Residual:  {residuals.max():.4f}")

### 6.7 Feature Importance Comparison

# Get feature importance from all tree-based models
importance_dict = {}

if 'Random Forest' in model_results:
    importance_dict['Random Forest'] = rf_model.feature_importances_

if 'Gradient Boosting' in model_results:
    importance_dict['Gradient Boosting'] = gb_model.feature_importances_

if 'XGBoost' in model_results:
    importance_dict['XGBoost'] = xgb_model.feature_importances_

if 'LightGBM' in model_results:
    importance_dict['LightGBM'] = lgb_model.feature_importances_

if 'CatBoost' in model_results:
    importance_dict['CatBoost'] = cb_model.feature_importances_

# Create averaged importance if multiple models
if len(importance_dict) > 0:
    # Average importance across all models
    avg_importance = np.mean(list(importance_dict.values()), axis=0)
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'avg_importance': avg_importance
    })
    
    # Add individual model importances
    for model_name, importances in importance_dict.items():
        feature_importance[f'{model_name}_importance'] = importances
    
    # Sort by average importance
    feature_importance = feature_importance.sort_values('avg_importance', ascending=False)
    
    # Plot comparison of top features across models
    top_n = 15
    top_features = feature_importance.head(top_n)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Average importance
    axes[0].barh(range(len(top_features)), top_features['avg_importance'])
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'])
    axes[0].set_xlabel('Average Feature Importance')
    axes[0].set_title(f'Top {top_n} Features - Average Importance Across Models')
    axes[0].invert_yaxis()
    
    # Heatmap of importance across models
    importance_cols = [col for col in top_features.columns if col.endswith('_importance') and col != 'avg_importance']
    if len(importance_cols) > 1:
        importance_matrix = top_features[importance_cols].values.T
        model_names = [col.replace('_importance', '') for col in importance_cols]
        
        im = axes[1].imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
        axes[1].set_yticks(range(len(model_names)))
        axes[1].set_yticklabels(model_names)
        axes[1].set_xticks(range(len(top_features)))
        axes[1].set_xticklabels(top_features['feature'].values, rotation=45, ha='right')
        axes[1].set_title(f'Feature Importance Heatmap - Top {top_n} Features')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label('Importance', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(top_features)):
                text = axes[1].text(j, i, f'{importance_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Most Important Features (averaged across models):")
    print(feature_importance[['feature', 'avg_importance']].head(10))
    
    # Show which model considers each feature most important
    print("\nModel that values each top feature the most:")
    for idx in range(min(10, len(top_features))):
        row = top_features.iloc[idx]
        feature_name = row['feature']
        model_importances = {model: row[f'{model}_importance'] 
                           for model in importance_dict.keys() 
                           if f'{model}_importance' in row.index}
        best_model = max(model_importances, key=model_importances.get)
        print(f"{feature_name:30} -> {best_model} ({model_importances[best_model]:.4f})")


# Create ensemble predictions by averaging top models
print("\n=== Creating Ensemble Predictions ===")

# Select top 3 models based on RMSE
top_models = comparison_df.head(3)
print(f"Top 3 models for ensemble: {list(top_models.index)}")

# Collect predictions from top models
ensemble_preds = []
model_weights = []

for model_name in top_models.index:
    # Weight models by their R² score (better models get more weight)
    weight = top_models.loc[model_name, 'r2']
    model_weights.append(weight)
    
    if model_name == 'XGBoost':
        preds = xgb_pred
    elif model_name == 'LightGBM':
        preds = lgb_pred
    elif model_name == 'CatBoost':
        preds = cb_pred
    elif model_name == 'Random Forest':
        preds = rf_pred
    elif model_name == 'Gradient Boosting':
        preds = gb_pred
    elif model_name == 'Ridge Regression':
        preds = ridge_pred
    else:
        preds = lr_pred
    
    ensemble_preds.append(preds)

# Normalize weights
model_weights = np.array(model_weights) / np.sum(model_weights)
print(f"Model weights: {dict(zip(top_models.index, model_weights.round(3)))}")

# Create weighted average ensemble
ensemble_pred = np.average(ensemble_preds, axis=0, weights=model_weights)
ensemble_pred = np.clip(ensemble_pred, 0, 1)

# Evaluate ensemble
ensemble_metrics = evaluate_model(y_val, ensemble_pred, "Weighted Ensemble")
model_results['Weighted Ensemble'] = ensemble_metrics

# Compare ensemble with individual models
print("\n=== Ensemble vs Individual Models ===")
print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'R²':<8}")
print("-" * 44)
for model_name in list(top_models.index) + ['Weighted Ensemble']:
    metrics = model_results[model_name]
    print(f"{model_name:<20} {metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} {metrics['r2']:<8.4f}")

# Update best model if ensemble is better
if model_results['Weighted Ensemble']['rmse'] < comparison_df.iloc[0]['rmse']:
    print("\n✓ Ensemble performs better than individual models!")
    best_model_name = 'Weighted Ensemble'
else:
    print("\n✗ Best individual model still performs better than ensemble.")

## 7. Cross-Validation for Regression

from sklearn.model_selection import KFold

# Perform cross-validation on the best model
print(f"\n=== Cross-Validation for {best_model_name} ===")

if best_model_name == 'Weighted Ensemble':
    # For ensemble, we need to train all component models
    print("Cross-validating ensemble components...")
    ensemble_cv_scores = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
        
        fold_preds = []
        # Train each model in the ensemble
        for model_name in top_models.index[:3]:  # Top 3 models
            if model_name == 'XGBoost':
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], 
                         early_stopping_rounds=20, verbose=False)
            elif model_name == 'LightGBM':
                model = lgb.LGBMRegressor(**lgb_params)
                model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)],
                         callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            elif model_name == 'CatBoost':
                train_pool = cb.Pool(X_cv_train, y_cv_train, cat_features=cat_features_indices)
                val_pool = cb.Pool(X_cv_val, y_cv_val, cat_features=cat_features_indices)
                model = cb.CatBoostRegressor(**catboost_params)
                model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
            elif model_name == 'Random Forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_cv_train, y_cv_train)
            elif model_name == 'Gradient Boosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X_cv_train, y_cv_train)
            
            fold_preds.append(model.predict(X_cv_val))
        
        # Create ensemble prediction
        ensemble_fold_pred = np.average(fold_preds, axis=0, weights=model_weights[:len(fold_preds)])
        ensemble_fold_pred = np.clip(ensemble_fold_pred, 0, 1)
        
        fold_rmse = np.sqrt(mean_squared_error(y_cv_val, ensemble_fold_pred))
        ensemble_cv_scores.append(fold_rmse)
        print(f"Fold {fold} RMSE: {fold_rmse:.4f}")
    
    cv_scores_rmse = np.array(ensemble_cv_scores)
    print(f"\nEnsemble Cross-Validation RMSE: {cv_scores_rmse.mean():.4f} (+/- {cv_scores_rmse.std() * 2:.4f})")
    
else:
    # Single model cross-validation
    if best_model_name == 'XGBoost':
        final_model = xgb.XGBRegressor(**xgb_params)
    elif best_model_name == 'Random Forest':
        final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif best_model_name == 'Gradient Boosting':
        final_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif best_model_name == 'Ridge Regression':
        final_model = Ridge(alpha=1.0, random_state=42)
    elif best_model_name == 'LightGBM':
        final_model = lgb.LGBMRegressor(**lgb_params)
    elif best_model_name == 'CatBoost':
        final_model = cb.CatBoostRegressor(**catboost_params)
    else:
        final_model = LinearRegression()
    
    # Perform 5-fold cross-validation
    cv_scores_mse = -cross_val_score(
        final_model, X, y, 
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    cv_scores_rmse = np.sqrt(cv_scores_mse)
    
    cv_scores_r2 = cross_val_score(
        final_model, X, y, 
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2',
        n_jobs=-1
    )
    
    print(f"\nCross-Validation Results for {best_model_name}:")
    print(f"RMSE scores: {cv_scores_rmse}")
    print(f"Mean RMSE: {cv_scores_rmse.mean():.4f} (+/- {cv_scores_rmse.std() * 2:.4f})")
    print(f"\nR² scores: {cv_scores_r2}")
    print(f"Mean R²: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std() * 2:.4f})")


# Train final model on entire training set
print("\n=== Training Final Model on Entire Dataset ===")
print(f"Selected model: {best_model_name}")

if best_model_name == 'Weighted Ensemble':
    print("\nTraining ensemble components...")
    final_ensemble_models = []
    final_ensemble_weights = []
    
    for model_name in top_models.index[:3]:
        print(f"Training {model_name}...")
        
        if model_name == 'XGBoost':
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X, y, verbose=False)
        elif model_name == 'LightGBM':
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
        elif model_name == 'CatBoost':
            full_pool = cb.Pool(X, y, cat_features=cat_features_indices)
            model = cb.CatBoostRegressor(**catboost_params)
            model.fit(full_pool, verbose=False)
        elif model_name == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
        elif model_name == 'Ridge Regression':
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(scaler.fit_transform(X), y)
        else:
            model = LinearRegression()
            model.fit(scaler.fit_transform(X), y)
        
        final_ensemble_models.append(model)
        # Use the same weights as before
        final_ensemble_weights.append(top_models.loc[model_name, 'r2'])
    
    # Normalize weights
    final_ensemble_weights = np.array(final_ensemble_weights) / np.sum(final_ensemble_weights)
    
    # Generate final predictions
    print("\nGenerating ensemble predictions...")
    test_preds = []
    for i, (model_name, model) in enumerate(zip(top_models.index[:3], final_ensemble_models)):
        if model_name in ['Ridge Regression', 'Linear Regression']:
            pred = model.predict(scaler.transform(X_test))
        else:
            pred = model.predict(X_test)
        test_preds.append(pred)
    
    final_predictions = np.average(test_preds, axis=0, weights=final_ensemble_weights)
    
elif best_model_name in ['Linear Regression', 'Ridge Regression']:
    if best_model_name == 'Ridge Regression':
        final_model = Ridge(alpha=1.0, random_state=42)
    else:
        final_model = LinearRegression()
    final_model.fit(scaler.fit_transform(X), y)
    final_predictions = final_model.predict(scaler.transform(X_test))
    
elif best_model_name == 'CatBoost':
    full_pool = cb.Pool(X, y, cat_features=cat_features_indices)
    test_pool = cb.Pool(X_test, cat_features=cat_features_indices)
    final_model = cb.CatBoostRegressor(**catboost_params)
    final_model.fit(full_pool, verbose=False)
    final_predictions = final_model.predict(test_pool)
    
else:
    if best_model_name == 'XGBoost':
        final_model = xgb.XGBRegressor(**xgb_params)
    elif best_model_name == 'LightGBM':
        final_model = lgb.LGBMRegressor(**lgb_params)
    elif best_model_name == 'Random Forest':
        final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif best_model_name == 'Gradient Boosting':
        final_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    final_model.fit(X, y)
    final_predictions = final_model.predict(X_test)

# Ensure predictions are within [0, 1] range
final_predictions = np.clip(final_predictions, 0, 1)

print("\nFinal model training completed!")
print(f"Prediction range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
print(f"Mean prediction: {final_predictions.mean():.4f}")
print(f"Std prediction: {final_predictions.std():.4f}")


# Create submission dataframe
submission = pd.DataFrame()

if test_ids is not None:
    submission['id'] = test_ids
else:
    submission['id'] = range(len(final_predictions))

# Add predictions (adjust column name based on competition requirements)
submission['accident_risk'] = final_predictions

# Display statistics of predictions
print("\nPrediction Statistics:")
print(submission['accident_risk'].describe())

# Visualize prediction distribution
plt.figure(figsize=(10, 5))
plt.hist(submission['accident_risk'], bins=50, edgecolor='black')
plt.title('Distribution of Predicted Accident Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# Save submission file
submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)
print(f"\nSubmission file saved as '{submission_filename}'")
print(f"Shape: {submission.shape}")
print("\nFirst 10 predictions:")
print(submission.head(10))

