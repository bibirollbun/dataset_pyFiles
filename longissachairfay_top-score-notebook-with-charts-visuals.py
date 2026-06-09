# ğŸŒ¾ Agricultural Yield Prediction Analysis with Multiple Submissions
# Complete code with visualizations, insights, and submission comparisons

# ğŸ“¦ Standard Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ğŸ“Š Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# âš™ï¸� Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

# ğŸ¤– Machine Learning Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ğŸ”® Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set style for beautiful visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== 1. DATA LOADING ====================
print("ğŸŒ¾ AGRICULTURAL YIELD PREDICTION ANALYSIS ğŸŒ¾")
print("=" * 60)

train = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
submission = pd.read_csv('/kaggle/input/agriyield-2025/sample_submission.csv')

print(f"ğŸ“Š Dataset Information:")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target variable: yield (kg/hectare)")

# ==================== 2. ORIGINAL SIMPLE MODEL (FOR MAIN SUBMISSION) ====================
print("\nğŸ�¯ CREATING MAIN SUBMISSION (Simple Approach)")
print("=" * 60)

# Fixed evaluate_model function
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.2f}")  # Fixed: was 'mse', now 'rmse'
    print(f"RÂ² Score: {r2:.2f}")
    return rmse, r2

# Simple feature preparation
X_simple = train.drop(['field_id', 'yield'], axis=1)
y_simple = train['yield']
X_test_simple = test.drop(['field_id'], axis=1)

# Train-test split
X_train_simple, X_val_simple, y_train_simple, y_val_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# Scale features
scaler_simple = StandardScaler()
X_train_scaled_simple = scaler_simple.fit_transform(X_train_simple)
X_val_scaled_simple = scaler_simple.transform(X_val_simple)
X_test_scaled_simple = scaler_simple.transform(X_test_simple)

# Train simple model (as per original code)
simple_rf = RandomForestRegressor(n_estimators=100, random_state=42)
simple_rf.fit(X_train_scaled_simple, y_train_simple)
simple_predictions = simple_rf.predict(X_test_scaled_simple)

# Create main submission file
submission['yield'] = simple_predictions
submission.to_csv('submission.csv', index=False)
print("âœ… Main submission file saved as submission.csv")

# Evaluate simple model
simple_val_pred = simple_rf.predict(X_val_scaled_simple)
print("\nSimple Random Forest Model Performance:")
evaluate_model(y_val_simple, simple_val_pred)

# Store for comparison
all_submissions = {'Simple_RF': simple_predictions}

# ==================== 3. EXPLORATORY DATA ANALYSIS ====================
print("\nğŸ“ˆ EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Basic statistics
print("\nğŸ“‹ Basic Statistics:")
print(train.describe())

# Check for missing values
print("\nğŸ”� Missing Values:")
print(train.isnull().sum())

# ==================== 4. AGRICULTURAL INSIGHTS & VISUALIZATIONS ====================
print("\nğŸŒ± AGRICULTURAL INSIGHTS")
print("=" * 60)

# Create comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Yield Distribution
ax1 = fig.add_subplot(gs[0, :])
sns.histplot(data=train, x='yield', bins=50, kde=True, ax=ax1)
ax1.set_title('Crop Yield Distribution', fontsize=16, fontweight='bold')
ax1.set_xlabel('Yield (kg/hectare)')
mean_yield = train['yield'].mean()
ax1.axvline(mean_yield, color='red', linestyle='--', label=f'Mean: {mean_yield:.1f}')
ax1.legend()

# 2. Correlation Heatmap
ax2 = fig.add_subplot(gs[1, :])
correlation_matrix = train.drop(['field_id'], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, ax=ax2, fmt='.2f')
ax2.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')

# 3. Soil pH vs Yield
ax3 = fig.add_subplot(gs[2, 0])
scatter = ax3.scatter(train['soil_ph'], train['yield'], 
                     c=train['organic_matter'], cmap='viridis', alpha=0.6)
ax3.set_xlabel('Soil pH')
ax3.set_ylabel('Yield (kg/hectare)')
ax3.set_title('Soil pH vs Yield (colored by Organic Matter)', fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Organic Matter')

# Add optimal pH range
ax3.axvspan(6.0, 7.0, alpha=0.2, color='green', label='Optimal pH Range')
ax3.legend()

# 4. NDVI vs Yield
ax4 = fig.add_subplot(gs[2, 1])
scatter2 = ax4.scatter(train['ndvi'], train['yield'], 
                      c=train['temperature'], cmap='coolwarm', alpha=0.6)
ax4.set_xlabel('NDVI (Vegetation Index)')
ax4.set_ylabel('Yield (kg/hectare)')
ax4.set_title('NDVI vs Yield (colored by Temperature)', fontweight='bold')
plt.colorbar(scatter2, ax=ax4, label='Temperature (Â°C)')

# 5. Rainfall Impact
ax5 = fig.add_subplot(gs[2, 2])
scatter3 = ax5.scatter(train['rainfall'], train['yield'], 
                      c=train['humidity'], cmap='Blues', alpha=0.6)
ax5.set_xlabel('Rainfall (mm)')
ax5.set_ylabel('Yield (kg/hectare)')
ax5.set_title('Rainfall vs Yield (colored by Humidity)', fontweight='bold')
plt.colorbar(scatter3, ax=ax5, label='Humidity (%)')

# 6. Feature Distributions
features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi']
for idx, feature in enumerate(features):
    ax = fig.add_subplot(gs[3 + idx//3, idx%3])
    train[feature].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title(f'{feature.replace("_", " ").title()} Distribution', fontweight='bold')
    ax.set_xlabel(feature.replace("_", " ").title())
    ax.set_ylabel('Frequency')
    
    # Add statistics
    mean_val = train[feature].mean()
    std_val = train[feature].std()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Î¼={mean_val:.2f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', label=f'Ïƒ={std_val:.2f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# ==================== 5. FEATURE ENGINEERING ====================
print("\nğŸ”§ FEATURE ENGINEERING")
print("=" * 60)

# Create new features based on agricultural knowledge
def create_features(df):
    df = df.copy()
    
    # Soil features
    df['soil_quality_index'] = (
        df['organic_matter'] * 0.4 + 
        (7 - abs(df['soil_ph'] - 6.5)) * 0.3 + 
        (100 - df['sand_pct']) / 20 * 0.3
    )
    
    # Water features
    df['water_stress_index'] = (df['rainfall'] / df['temperature']) * (df['humidity'] / 100)
    df['water_availability'] = df['rainfall'] * df['humidity'] / 100
    
    # Temperature features
    df['heat_stress'] = np.where(df['temperature'] > 30, df['temperature'] - 30, 0)
    df['cold_stress'] = np.where(df['temperature'] < 20, 20 - df['temperature'], 0)
    df['optimal_temp'] = np.where((df['temperature'] >= 20) & (df['temperature'] <= 30), 1, 0)
    
    # pH features
    df['ph_optimal'] = np.where((df['soil_ph'] >= 6.0) & (df['soil_ph'] <= 7.0), 1, 0)
    df['ph_squared'] = df['soil_ph'] ** 2
    
    # Interaction features
    df['ndvi_organic'] = df['ndvi'] * df['organic_matter']
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['rainfall_temp_ratio'] = df['rainfall'] / (df['temperature'] + 1)
    
    # Polynomial features for key variables
    df['organic_matter_sq'] = df['organic_matter'] ** 2
    df['ndvi_sq'] = df['ndvi'] ** 2
    
    return df

# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)

print(f"Original features: {len(train.columns) - 2}")  # Excluding field_id and yield
print(f"Engineered features: {len(train_fe.columns) - 2}")

# ==================== 6. MODEL PREPARATION ====================
print("\nğŸ�¯ ADVANCED MODEL PREPARATION")
print("=" * 60)

# Prepare data with engineered features
features_to_use = [col for col in train_fe.columns if col not in ['field_id', 'yield']]
X = train_fe[features_to_use]
y = train_fe['yield']
X_test = test_fe[features_to_use]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

# ==================== 7. ADVANCED MODEL EVALUATION FUNCTION ====================
def evaluate_model_advanced(y_true, y_pred, model_name="Model"):
    """Advanced evaluation function with additional metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RÂ² Score: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# ==================== 8. TRAINING ALL MODELS ====================
print("\nğŸ¤– TRAINING ALL MACHINE LEARNING MODELS")
print("=" * 60)

# Initialize models
models = {
    'Linear_Regression': LinearRegression(),
    'Ridge_Regression': Ridge(alpha=1.0),
    'Lasso_Regression': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1),
    'Decision_Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random_Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'Extra_Trees': ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient_Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42)
}

# Train and evaluate models
results = {}
predictions = {}
test_predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Evaluate
    metrics = evaluate_model_advanced(y_val, y_pred, name)
    results[name] = metrics
    predictions[name] = y_pred
    test_predictions[name] = y_pred_test
    
    # Save submission file
    submission_temp = submission.copy()
    submission_temp['yield'] = y_pred_test
    submission_temp.to_csv(f'submission_{name}.csv', index=False)
    all_submissions[name] = y_pred_test
    print(f"  Saved: submission_{name}.csv")

# ==================== 9. DEEP LEARNING MODEL ====================
print("\nğŸ§  TRAINING DEEP NEURAL NETWORK")
print("=" * 60)

# Build neural network
def create_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Create and train model
nn_model = create_nn_model(X_train_scaled.shape[1])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Train
history = nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Evaluate
nn_predictions = nn_model.predict(X_val_scaled).flatten()
nn_test_predictions = nn_model.predict(X_test_scaled).flatten()
nn_metrics = evaluate_model_advanced(y_val, nn_predictions, "Neural_Network")

# Save neural network submission
submission_nn = submission.copy()
submission_nn['yield'] = nn_test_predictions
submission_nn.to_csv('submission_Neural_Network.csv', index=False)
all_submissions['Neural_Network'] = nn_test_predictions
print("  Saved: submission_Neural_Network.csv")

# ==================== 10. ENSEMBLE PREDICTIONS ====================
print("\nğŸ�­ CREATING ENSEMBLE PREDICTIONS")
print("=" * 60)

# Create results dataframe including neural network
results['Neural_Network'] = nn_metrics
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('r2', ascending=False)

# Select top models for ensemble
top_models = results_df.head(5).index.tolist()
print(f"Top 5 models for ensemble: {', '.join(top_models)}")

# Create ensemble predictions with weights
weights = results_df.loc[top_models, 'r2'].values
weights = weights / weights.sum()

ensemble_test_predictions = np.zeros(len(X_test))
for model_name, weight in zip(top_models, weights):
    ensemble_test_predictions += weight * all_submissions[model_name]

# Save ensemble submission
submission_ensemble = submission.copy()
submission_ensemble['yield'] = ensemble_test_predictions
submission_ensemble.to_csv('submission_Ensemble.csv', index=False)
all_submissions['Ensemble'] = ensemble_test_predictions
print("  Saved: submission_Ensemble.csv")

# ==================== 11. SUBMISSION COMPARISON & VISUALIZATION ====================
print("\nğŸ“Š COMPARING ALL SUBMISSIONS")
print("=" * 60)

# Create comparison dataframe
submission_stats = pd.DataFrame({
    name: {
        'mean': preds.mean(),
        'std': preds.std(),
        'min': preds.min(),
        'max': preds.max(),
        'q25': np.percentile(preds, 25),
        'median': np.median(preds),
        'q75': np.percentile(preds, 75)
    }
    for name, preds in all_submissions.items()
}).T

print("\nSubmission Statistics:")
print(submission_stats.round(2))

# Visualization of all submissions
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# 1. Distribution comparison
ax1 = axes[0, 0]
for name, preds in all_submissions.items():
    ax1.hist(preds, bins=30, alpha=0.3, label=name, density=True)
ax1.set_xlabel('Predicted Yield')
ax1.set_ylabel('Density')
ax1.set_title('Prediction Distributions - All Models', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 2. Box plot comparison
ax2 = axes[0, 1]
submission_df = pd.DataFrame(all_submissions)
submission_df.boxplot(ax=ax2, rot=45)
ax2.set_ylabel('Predicted Yield')
ax2.set_title('Prediction Ranges - All Models', fontweight='bold')

# 3. Mean and std comparison
ax3 = axes[1, 0]
x_pos = np.arange(len(submission_stats))
ax3.bar(x_pos, submission_stats['mean'], yerr=submission_stats['std'], 
        capsize=5, alpha=0.7)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(submission_stats.index, rotation=45, ha='right')
ax3.set_ylabel('Mean Predicted Yield')
ax3.set_title('Mean Predictions with Standard Deviation', fontweight='bold')

# 4. Correlation between different model predictions
ax4 = axes[1, 1]
pred_corr = pd.DataFrame(all_submissions).corr()
sns.heatmap(pred_corr, annot=True, fmt='.3f', cmap='Blues', ax=ax4)
ax4.set_title('Correlation Between Model Predictions', fontweight='bold')

# 5. Scatter plot: Simple vs Best Advanced Model
best_model = results_df.index[0]
ax5 = axes[2, 0]
ax5.scatter(all_submissions['Simple_RF'], all_submissions[best_model], alpha=0.5)
ax5.plot([all_submissions['Simple_RF'].min(), all_submissions['Simple_RF'].max()], 
         [all_submissions['Simple_RF'].min(), all_submissions['Simple_RF'].max()], 
         'r--', lw=2)
ax5.set_xlabel('Simple Random Forest Predictions')
ax5.set_ylabel(f'{best_model} Predictions')
ax5.set_title(f'Simple RF vs {best_model}', fontweight='bold')

# 6. Performance comparison (validation scores)
ax6 = axes[2, 1]
performance_data = pd.DataFrame({
    'Model': list(results.keys()) + ['Simple_RF'],
    'RÂ²': [results[m]['r2'] for m in results.keys()] + [r2_score(y_val_simple, simple_val_pred)]
})
performance_data = performance_data.sort_values('RÂ²', ascending=True)
ax6.barh(performance_data['Model'], performance_data['RÂ²'])
ax6.set_xlabel('RÂ² Score')
ax6.set_title('Model Performance Comparison', fontweight='bold')
ax6.set_xlim(0, 1)

plt.tight_layout()
plt.show()

# ==================== 12. SUBMISSION DIFFERENCES ANALYSIS ====================
print("\nğŸ”� ANALYZING DIFFERENCES BETWEEN SUBMISSIONS")
print("=" * 60)

# Calculate differences from main submission
main_submission = all_submissions['Simple_RF']
differences = {}

for name, preds in all_submissions.items():
    if name != 'Simple_RF':
        diff = preds - main_submission
        differences[name] = {
            'mean_diff': diff.mean(),
            'abs_mean_diff': np.abs(diff).mean(),
            'max_increase': diff.max(),
            'max_decrease': diff.min(),
            'correlation': np.corrcoef(main_submission, preds)[0, 1]
        }

diff_df = pd.DataFrame(differences).T
print("\nDifferences from Main Submission (Simple_RF):")
print(diff_df.round(3))

# Visualize differences
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Average difference from main submission
ax1 = axes[0, 0]
ax1.bar(diff_df.index, diff_df['mean_diff'])
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_ylabel('Average Difference')
ax1.set_title('Average Difference from Main Submission', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# 2. Absolute differences
ax2 = axes[0, 1]
ax2.bar(diff_df.index, diff_df['abs_mean_diff'])
ax2.set_ylabel('Average Absolute Difference')
ax2.set_title('Average Absolute Difference from Main Submission', fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

# 3. Individual prediction differences for top models
ax3 = axes[1, 0]
top_3_models = results_df.head(3).index.tolist()
for model in top_3_models:
    if model in all_submissions:
        diff = all_submissions[model] - main_submission
        ax3.hist(diff, bins=50, alpha=0.5, label=model, density=True)
ax3.set_xlabel('Prediction Difference')
ax3.set_ylabel('Density')
ax3.set_title('Distribution of Differences from Main Submission', fontweight='bold')
ax3.legend()

# 4. Cumulative distribution of predictions
ax4 = axes[1, 1]
for name in ['Simple_RF', results_df.index[0], 'Ensemble']:
    if name in all_submissions:
        sorted_preds = np.sort(all_submissions[name])
        cumulative = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
        ax4.plot(sorted_preds, cumulative, label=name, linewidth=2)
ax4.set_xlabel('Predicted Yield')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('Cumulative Distribution of Predictions', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 13. FINAL SUMMARY ====================
print("\nğŸ“� FINAL SUMMARY")
print("=" * 60)

print("\nğŸ�† Model Rankings (by RÂ² score):")
print(results_df[['r2', 'rmse', 'mae']].round(4))

print("\nğŸ“Š Submission Files Created:")
print("1. submission.csv (Main - Simple Random Forest)")
for i, name in enumerate(all_submissions.keys(), 2):
    if name != 'Simple_RF':
        print(f"{i}. submission_{name}.csv")

print(f"\nğŸ’¡ Recommendations:")
print(f"- Best individual model: {results_df.index[0]} (RÂ² = {results_df.iloc[0]['r2']:.4f})")
print(f"- Ensemble model combines top 5 models for robust predictions")
print(f"- Main submission uses simple approach as requested")
print(f"- Consider using '{results_df.index[0]}' or 'Ensemble' for best performance")

print("\n" + "=" * 60)
print("ğŸ�‰ ANALYSIS COMPLETE! All submission files have been created.")
print("=" * 60)

