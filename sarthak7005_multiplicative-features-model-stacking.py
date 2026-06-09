import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

# Set global random seed for reproducibility
np.random.seed(42)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
import time


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")



train.head()


train.describe()


# Define numerical features
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']



# Feature engineering function
def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new


# Add feature ratios (can be useful for physiological data)
def add_feature_ratios(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(len(numerical_features)):
            if i != j:
                feature1 = numerical_features[i]
                feature2 = numerical_features[j]
                ratio_name = f"{feature1}_div_{feature2}"
                df_new[ratio_name] = df_new[feature1] / (df_new[feature2] + 1e-8)  # Avoid division by zero
    return df_new


# Add polynomial features
def add_polynomial_features(df, numerical_features, degree=2):
    df_new = df.copy()
    for feature in numerical_features:
        for d in range(2, degree + 1):
            df_new[f"{feature}_pow_{d}"] = df_new[feature] ** d
    return df_new


# Add statistical aggregations
def add_stat_features(df):
    df_new = df.copy()
    df_new['numerical_mean'] = df_new[numerical_features].mean(axis=1)
    df_new['numerical_std'] = df_new[numerical_features].std(axis=1)
    df_new['numerical_min'] = df_new[numerical_features].min(axis=1)
    df_new['numerical_max'] = df_new[numerical_features].max(axis=1)
    df_new['numerical_range'] = df_new['numerical_max'] - df_new['numerical_min']
    return df_new



# Feature engineering pipeline
def feature_engineering_pipeline(train_df, test_df):
    # Encode categorical features
    le = LabelEncoder()
    train_df['Sex'] = le.fit_transform(train_df['Sex'])
    test_df['Sex'] = le.transform(test_df['Sex'])
    
    # Convert to category
    train_df['Sex'] = train_df['Sex'].astype('category')
    test_df['Sex'] = test_df['Sex'].astype('category')
    
    # Apply feature engineering
    train_df = add_feature_cross_terms(train_df, numerical_features)
    test_df = add_feature_cross_terms(test_df, numerical_features)
    
    train_df = add_polynomial_features(train_df, numerical_features)
    test_df = add_polynomial_features(test_df, numerical_features)
    
    train_df = add_stat_features(train_df)
    test_df = add_stat_features(test_df)
    
    # More domain-specific features for exercise data
    train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
    test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)
    
    train_df['Intensity'] = train_df['Heart_Rate'] * train_df['Duration'] / 60
    test_df['Intensity'] = test_df['Heart_Rate'] * test_df['Duration'] / 60
    
    return train_df, test_df


# Apply feature engineering
train, test = feature_engineering_pipeline(train, test)


# Prepare data for modeling
X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])  # Log transform target
X_test = test.drop(columns=['id'])


# Define model parameters
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)



# Define base models
base_models = {
    'CatBoost': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3,
        verbose=100,
        random_seed=42,
        cat_features=['Sex'],
        early_stopping_rounds=100
    ),
    'XGBoost': XGBRegressor(
        max_depth=8,
        colsample_bytree=0.7,
        subsample=0.9,
        n_estimators=1000,
        learning_rate=0.02,
        gamma=0.01,
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric='rmse',
        enable_categorical=True,
        random_state=42
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        colsample_bytree=0.7,
        subsample=0.9,
        num_leaves=31,
        reg_alpha=0.3,
        reg_lambda=0.3,
        random_state=42,
        verbose=-1
    )
}



# Define meta-learner
meta_learner = Ridge(alpha=1.0)



# Create arrays for storing meta-features
train_meta_features = np.zeros((len(X), len(base_models)))
test_meta_features = np.zeros((len(X_test), len(base_models)))


# Dictionary to store results
results = {name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []} for name in list(base_models.keys()) + ['Stacked']}



# First-level training and predictions
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n=== Fold {i+1} ===")
    x_train, y_train = X.iloc[train_idx], y[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
    
    # Train each base model
    for j, (name, model) in enumerate(base_models.items()):
        print(f"\nTraining {name}...")
        start = time.time()
        
        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        else:
            model.fit(x_train, y_train)
        
        # Predict on validation data for this fold
        oof_pred = model.predict(x_valid)
        results[name]['oof'][valid_idx] = oof_pred
        
        # Predict on test data
        test_pred = model.predict(X_test)
        results[name]['pred'] += test_pred / FOLDS
        
        # Store meta-features for this fold
        train_meta_features[valid_idx, j] = oof_pred
        
        # Calculate RMSLE for this model on this fold
        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
        results[name]['rmsle'].append(rmsle)
        
        print(f"{name} - Fold {i+1} RMSLE: {rmsle:.4f}")
        print(f"Training time: {time.time() - start:.1f} sec")
    
    # Train base models on full fold training data to predict on test
    for j, (name, model) in enumerate(base_models.items()):
        if j == 0:  # Only print this once
            print("\nTraining models on full fold training data for test prediction...")
        
        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0)
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid), verbose=False)
        else:
            model.fit(x_train, y_train)
        
        # Predict on test data
        test_pred = model.predict(X_test)
        test_meta_features[:, j] += test_pred / FOLDS


# Print base model performance
print("\n=== Base Model Performance ===")
for name in base_models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")


# Create visualization for base model performance
plt.figure(figsize=(10, 6))
model_names = list(base_models.keys())
mean_rmsles = [np.mean(results[name]['rmsle']) for name in model_names]
std_rmsles = [np.std(results[name]['rmsle']) for name in model_names]

plt.bar(model_names, mean_rmsles, yerr=std_rmsles, alpha=0.7, capsize=10)
plt.ylabel('RMSLE')
plt.title('Base Model Performance')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



print("\n=== Training Meta Learner ===")
# Define a new KFold for meta-level validation
meta_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=43)
meta_oof = np.zeros(len(X))

for i, (meta_train_idx, meta_valid_idx) in enumerate(meta_kf.split(train_meta_features, y)):
    print(f"\nMeta-Learner Fold {i+1}")
    
    # Prepare meta-level data
    meta_x_train, meta_y_train = train_meta_features[meta_train_idx], y[meta_train_idx]
    meta_x_valid, meta_y_valid = train_meta_features[meta_valid_idx], y[meta_valid_idx]
    
    # Train meta-learner
    meta_learner.fit(meta_x_train, meta_y_train)
    
    # Predict on meta-validation data
    meta_oof_pred = meta_learner.predict(meta_x_valid)
    meta_oof[meta_valid_idx] = meta_oof_pred
    
    # Calculate RMSLE for meta-learner on this fold
    meta_rmsle = np.sqrt(mean_squared_log_error(np.expm1(meta_y_valid), np.expm1(meta_oof_pred)))
    results['Stacked']['rmsle'].append(meta_rmsle)
    print(f"Meta-Learner Fold {i+1} RMSLE: {meta_rmsle:.4f}")



# Final meta-learner trained on all meta-features
meta_learner.fit(train_meta_features, y)
final_pred = meta_learner.predict(test_meta_features)
results['Stacked']['pred'] = final_pred
results['Stacked']['oof'] = meta_oof


# Print final meta-learner performance
print("\n=== Final Model Performance ===")
for name in list(base_models.keys()) + ['Stacked']:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")



# Print meta-learner weights
print("\n=== Meta-Learner Coefficients ===")
for name, coef in zip(base_models.keys(), meta_learner.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {meta_learner.intercept_:.4f}")


# Visualize final performance comparison
plt.figure(figsize=(12, 6))
all_model_names = list(base_models.keys()) + ['Stacked']
all_mean_rmsles = [np.mean(results[name]['rmsle']) for name in all_model_names]
all_std_rmsles = [np.std(results[name]['rmsle']) for name in all_model_names]

plt.bar(all_model_names, all_mean_rmsles, yerr=all_std_rmsles, alpha=0.7, capsize=10)
plt.ylabel('RMSLE')
plt.title('Model Performance Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)


# Highlight the best model
best_idx = np.argmin(all_mean_rmsles)
plt.bar(all_model_names[best_idx], all_mean_rmsles[best_idx], color='green', alpha=0.7)
plt.show()



# Create submission file with stacked model predictions
y_preds = np.expm1(results['Stacked']['pred'])



# Clip predictions to reasonable range
y_preds = np.clip(y_preds, 1, 314)

submission['Calories'] = y_preds
submission.to_csv('stacked_submission.csv', index=False)



# Feature importance for individual models
for name, model in base_models.items():
    if name == 'CatBoost':
        feature_importance = model.get_feature_importance()
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'{name} Feature Importance')
        plt.tight_layout()
        plt.show()



# Print submission statistics
print("\nSubmission Statistics:")
print(f"Mean: {y_preds.mean():.2f}")
print(f"Median: {np.median(y_preds):.2f}")
print(f"Min: {y_preds.min():.2f}")
print(f"Max: {y_preds.max():.2f}")










