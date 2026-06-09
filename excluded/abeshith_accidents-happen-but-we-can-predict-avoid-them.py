import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.describe().T


train['accident_risk'].describe()


train.info()


train.head(8)


categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


def create_features(df):
    df_processed = df.copy()
    
    # Convert boolean columns to integers (if not already)
    for col in boolean_cols:
        df_processed[col] = df_processed[col].astype(int)
    
    # Label encoding for categorical variables (efficient for tree-based models)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # Create interaction features based on domain knowledge
    df_processed['speed_curvature_risk'] = df_processed['speed_limit'] * df_processed['curvature']
    df_processed['lanes_speed_ratio'] = df_processed['speed_limit'] / (df_processed['num_lanes'] + 1)
    
    # Weather and lighting risk combinations
    df_processed['weather_lighting_risk'] = (
        df_processed['weather_encoded'] * df_processed['lighting_encoded']
    )
    
    # Historical accident density
    df_processed['accident_density'] = df_processed['num_reported_accidents'] / (df_processed['num_lanes'] + 1)
    
    # High-risk time periods (evening/night typically more dangerous)
    df_processed['high_risk_time'] = (df_processed['time_of_day'].isin(['Evening', 'Night'])).astype(int)
    
    # Road complexity score
    df_processed['road_complexity'] = (
        df_processed['curvature'] * 
        df_processed['speed_limit'] * 
        (1 - df_processed['road_signs_present'].astype(int))
    )
    
    return df_processed, label_encoders


train_processed, encoders = create_features(train)
test_processed, _ = create_features(test)


train_processed.shape


feature_cols = (
    [col + '_encoded' for col in categorical_cols] + 
    boolean_cols + 
    numerical_cols + 
    ['speed_curvature_risk', 'lanes_speed_ratio', 'weather_lighting_risk', 
     'accident_density', 'high_risk_time', 'road_complexity']
)


X = train_processed[feature_cols]
y = train_processed['accident_risk']
X_test = test_processed[feature_cols]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5))


scaler = StandardScaler()
numerical_indices = [X.columns.get_loc(col) for col in numerical_cols if col in X.columns]


X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

if numerical_indices:
    X_train_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X_train.iloc[:, numerical_indices])
    X_val_scaled.iloc[:, numerical_indices] = scaler.transform(X_val.iloc[:, numerical_indices])
    X_test_scaled.iloc[:, numerical_indices] = scaler.transform(X_test.iloc[:, numerical_indices])


models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10, 
                                         min_samples_leaf=5, random_state=42, n_jobs=-1),
    
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, 
                                                  min_samples_split=10, min_samples_leaf=5, random_state=42),
    
    'Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=15, min_samples_split=10,
                                     min_samples_leaf=5, random_state=42, n_jobs=-1),
    
    'Ridge Regression': Ridge(alpha=1.0)
}


results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    
    # Calculate RMSE (competition metric)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
    # Additional metrics
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)
    
    results[name] = {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'model': model
    }
    
    print(f"Train RMSE: {train_rmse:.5f}")
    print(f"Validation RMSE: {val_rmse:.5f}")
    print(f"Validation MAE: {val_mae:.5f}")
    print(f"Validation R²: {val_r2:.5f}")


results_df = pd.DataFrame({
    name: {
        'Train RMSE': results[name]['train_rmse'],
        'Val RMSE': results[name]['val_rmse'],
        'Val MAE': results[name]['val_mae'],
        'Val R²': results[name]['val_r2']
    } for name in results
}).T


best_model_name = results_df['Val RMSE'].idxmin()
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name} (Val RMSE: {results_df.loc[best_model_name, 'Val RMSE']:.5f})")


if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features ({best_model_name}):")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


top_3_models = results_df.nsmallest(3, 'Val RMSE')

weights = 1 / top_3_models['Val RMSE']
weights = weights / weights.sum()


for model_name, weight in weights.items():
    print(f"{model_name}: {weight:.3f}")

# Generate ensemble predictions
ensemble_pred_val = np.zeros(len(y_val))
ensemble_pred_test = np.zeros(len(X_test_scaled))


for model_name, weight in weights.items():
    model = results[model_name]['model']
    ensemble_pred_val += weight * model.predict(X_val_scaled)
    ensemble_pred_test += weight * model.predict(X_test_scaled)

ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred_val))
print(f"\nEnsemble Validation RMSE: {ensemble_rmse:.5f}")


final_predictions = ensemble_pred_test

# Ensure predictions are within valid range [0, 1]
final_predictions = np.clip(final_predictions, 0, 1)


submission = sample_submission.copy()
submission['accident_risk'] = final_predictions


submission.to_csv('submission.csv', index=False)




