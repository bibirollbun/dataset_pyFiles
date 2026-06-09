import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
import warnings
warnings.simplefilter('ignore')


# === Load Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("\nSummary statistics:")
print(train.describe())


print("\nMissing values in train:")
print(train.isnull().sum())
print("\nMissing values in test:")
print(test.isnull().sum())


# === Encode 'Sex' ===
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])
train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')


# === Feature Engineering ===
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# Calculate BMI and other body metrics
for df in [train, test]:
    # BMI and variations
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['BMI_category'] = pd.cut(df['BMI'], 
                              bins=[0, 18.5, 25, 30, 100], 
                              labels=[0, 1, 2, 3]).astype('category')
    
    # BSA (Body Surface Area) using Mosteller formula
    df['BSA'] = np.sqrt((df['Height'] * df['Weight']) / 3600)
    
    # Lean Body Mass using Boer formula
    df['LBM'] = np.where(df['Sex'] == 0,
                          0.407 * df['Weight'] + 0.267 * df['Height'] - 19.2,
                          0.252 * df['Weight'] + 0.473 * df['Height'] - 48.3)
    
    # Heart rate zones
    df['HR_reserve'] = 220 - df['Age'] - df['Heart_Rate']  # Max HR - current HR
    df['HR_percentage'] = df['Heart_Rate'] / (220 - df['Age'])  # % of max HR
    
    # Metabolic equivalents approximation (rough estimate)
    df['MET_approx'] = np.where(df['HR_percentage'] < 0.5, 2,
                              np.where(df['HR_percentage'] < 0.7, 5, 
                                     np.where(df['HR_percentage'] < 0.8, 7, 10)))
    
    # Duration-based features
    df['Duration_minutes'] = df['Duration'] / 60
    df['Activity_intensity'] = df['Heart_Rate'] * df['Duration_minutes']
    
    # Create the original squared and sqrt features
    for col in numerical_features + ['BMI', 'BSA', 'LBM']:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(df[col])


# === Add Clustering Features ===
# Standardize numerical features for clustering
features_for_clustering = ['Age', 'Height', 'Weight', 'BMI', 'Heart_Rate', 'Body_Temp']
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[features_for_clustering])
test_scaled = scaler.transform(test[features_for_clustering])


# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
train['cluster'] = kmeans.fit_predict(train_scaled)
test['cluster'] = kmeans.predict(test_scaled)
train['cluster'] = train['cluster'].astype('category')
test['cluster'] = test['cluster'].astype('category')


# === Polynomial Features for Key Variables ===
poly_features = ['Weight', 'Heart_Rate', 'Duration', 'BMI', 'BSA']
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
poly_train = pd.DataFrame(
    poly.fit_transform(train[poly_features]), 
    columns=[f'poly_{i}' for i in range(poly.n_output_features_)]
)
poly_test = pd.DataFrame(
    poly.transform(test[poly_features]), 
    columns=[f'poly_{i}' for i in range(poly.n_output_features_)]
)



# Add polynomial features to the dataframes
train = pd.concat([train, poly_train], axis=1)
test = pd.concat([test, poly_test], axis=1)



# === Cross Terms for Key Features ===
def add_cross_terms(df, features):
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    return df


# Select most important features for cross terms to avoid feature explosion
important_cross_features = ['Heart_Rate', 'Duration', 'Weight', 'BMI', 'BSA', 'Activity_intensity']
train = add_cross_terms(train, important_cross_features)
test = add_cross_terms(test, important_cross_features)



# === Feature Selection ===
# Drop features with potential multicollinearity or low importance
cols_to_drop = ['id']
if 'Calories' in train.columns:
    cols_to_drop.append('Calories')

X = train.drop(columns=cols_to_drop)
y = np.log1p(train['Calories'])
X_test = test.drop(columns=['id'])


# === Model Setup with Improved Hyperparameters ===
FOLDS = 2
# Use a stratified k-fold based on binned target values to ensure similar target distribution
y_binned = pd.qcut(train['Calories'], q=5, labels=False)
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

cat_features = ['Sex', 'BMI_category', 'cluster']

models = {
    'CatBoost': CatBoostRegressor(
        iterations=3000,
        learning_rate=0.01,
        depth=8,
        l2_leaf_reg=3,
        loss_function='RMSE',
        cat_features=cat_features,
        early_stopping_rounds=100,
        verbose=0,
        random_seed=42
    ),
    'XGBoost': XGBRegressor(
        max_depth=8,
        learning_rate=0.01,
        n_estimators=3000,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        early_stopping_rounds=100,
        eval_metric='rmse',
        enable_categorical=True,
        random_state=42
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        verbose=-1
    )
}

results = {name: {'pred': np.zeros(len(test)), 'oof_pred': np.zeros(len(train)), 'rmsle': []} for name in models}



# === Training Loop with OOF Predictions ===
print("Starting model training...")
for name, model in models.items():
    print(f"\nğŸš€ Training {name}")
    oof_pred = np.zeros(len(train))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_val, y_val))
        else:  
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], 
                     categorical_feature=['Sex', 'BMI_category', 'cluster'])

        # Make predictions
        val_pred = model.predict(x_val)
        test_pred = model.predict(X_test)
        
        # Store out-of-fold predictions
        oof_pred[val_idx] = val_pred
        results[name]['pred'] += test_pred / FOLDS
        
        # Calculate RMSLE
        score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(val_pred)))
        results[name]['rmsle'].append(score)
        print(f"Fold {fold + 1} RMSLE: {score:.5f}")
    
    # Store OOF predictions for meta-learning
    results[name]['oof_pred'] = oof_pred
    
    # Print overall RMSLE
    overall_score = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_pred)))
    print(f"Overall {name} RMSLE: {overall_score:.5f}")



# === Feature Importance Analysis ===
print("\nğŸ”� Feature Importance Analysis")
# Get feature importance from CatBoost (as an example)
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    verbose=0,
    random_seed=42
)
cat_model.fit(X, y)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': cat_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print(feature_importance.head(20))  # Print top 20 features


# === Optimized Ensemble Blending ===
# Use the OOF predictions to train a meta-model (simple weighted average based on OOF performance)
oof_scores = {name: np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(results[name]['oof_pred']))) for name in models}
total_score = sum(1/score for score in oof_scores.values())
weights = {name: (1/score)/total_score for name, score in oof_scores.items()}

print("\nğŸ§ª Model Weights:")
for name, weight in weights.items():
    print(f"{name}: {weight:.4f}")

# Apply optimized weights to blend predictions
blend_preds = np.zeros(len(test))
for name, model_results in results.items():
    blend_preds += np.expm1(model_results['pred']) * weights[name]


# === Post-Processing: Clip predictions to reasonable range based on training data stats ===
min_calories = max(1, train['Calories'].min() * 0.9)  # 10% lower than min observed
max_calories = train['Calories'].max() * 1.1  # 10% higher than max observed
blend_preds = np.clip(blend_preds, min_calories, max_calories)


# === Save Final Submission ===
submission['Calories'] = blend_preds
submission.to_csv("/kaggle/working/submission_advanced_blend.csv", index=False)



# === Print Summary ===
print("\nğŸ“Š Model Performance Summary:")
for name in models:
    scores = results[name]['rmsle']
    print(f"{name} Mean RMSLE: {np.mean(scores):.5f} Â± {np.std(scores):.5f}")

print(f"\nBlended Model RMSLE (estimated): {np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(sum(results[name]['oof_pred'] * weights[name] for name in models)))):.5f}")
print("\nâœ… submission_advanced_blend.csv is saved.")




