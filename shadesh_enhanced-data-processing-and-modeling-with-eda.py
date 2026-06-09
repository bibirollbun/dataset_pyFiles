# ğŸ“Š Enhanced Data Processing and Modeling with EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# ï¿½ Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# ======================
# ğŸ§¹ DATA PREPROCESSING
# ======================

# âœ… Encode the 'Sex' column FIRST (before EDA)
train['Sex'] = train['Sex'].str.lower().map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].str.lower().map({'male': 0, 'female': 1})

# ======================
# ğŸ”� EXPLORATORY DATA ANALYSIS (EDA)
# ======================

print("ğŸ“Š Dataset Overview:")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("\nğŸ“‹ Train data info:")
train.info()

# 1. Basic statistics
print("\nğŸ“ˆ Descriptive Statistics:")
print(train.describe().T)

# 2. Target Variable Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train['Calories'], kde=True, bins=30)
plt.title('Calories Distribution')

plt.subplot(1, 2, 2)
sns.boxplot(y=train['Calories'])
plt.title('Calories Boxplot')
plt.savefig('target_distribution.png')
plt.close()

# 3. Numerical Features Distribution
numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('numerical_features_dist.png')
plt.close()

# 4. Correlation Analysis (now works after encoding Sex)
plt.figure(figsize=(10, 8))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# 5. Pairplot of Top Correlated Features with Target
top_features = corr_matrix['Calories'].abs().sort_values(ascending=False).index[1:5]
sns.pairplot(train[list(top_features) + ['Calories']], diag_kind='kde')
plt.savefig('pairplot_top_features.png')
plt.close()

# 6. Feature vs Target Relationships
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.scatterplot(x=col, y='Calories', data=train)
    plt.title(f'{col} vs Calories')
plt.tight_layout()
plt.savefig('feature_vs_target.png')
plt.close()

# 7. Outlier Detection
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(y=train[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig('outlier_detection.png')
plt.close()

# ======================
# ğŸ¤– MODELING PIPELINE
# ======================

# ğŸ�¯ Split features and target
X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']
X_test = test.drop(['id'], axis=1)

# Create preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle any missing values
    ('scaler', StandardScaler()),
    ('transformer', PowerTransformer(method='yeo-johnson')),
    ('feature_selector', SelectKBest(score_func=f_regression, k='all'))  # Use all features initially
])

# Split training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# ğŸŒ² Train RandomForestRegressor with hyperparameter tuning
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf_model = GridSearchCV(RandomForestRegressor(random_state=42), 
                       rf_params, 
                       cv=5, 
                       scoring='neg_mean_absolute_error',
                       n_jobs=-1)
rf_model.fit(X_train_processed, y_train)

# ğŸš€ Train Gradient Boosting Regressor
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), 
                       gb_params, 
                       cv=5, 
                       scoring='neg_mean_absolute_error',
                       n_jobs=-1)
gb_model.fit(X_train_processed, y_train)

# âœ… Evaluate models
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"ğŸ“‰ Validation MAE: {round(mae, 2)}")
    return mae

print("\nModel Evaluation:")
print("Random Forest:")
rf_mae = evaluate_model(rf_model, X_val_processed, y_val)
print("\nGradient Boosting:")
gb_mae = evaluate_model(gb_model, X_val_processed, y_val)

# Select best model
best_model = gb_model if gb_mae < rf_mae else rf_model
print(f"\nğŸ�† Selected Model: {'Gradient Boosting' if gb_mae < rf_mae else 'Random Forest'}")

# Feature Importance
plt.figure(figsize=(10, 6))
feature_imp = pd.Series(best_model.best_estimator_.feature_importances_,
                       index=X.columns)
feature_imp.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()

# Predict on test data
test_preds = best_model.predict(X_test_processed)

# ğŸ’¾ Save to submission.csv
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_preds
})
submission.to_csv('submission.csv', index=False)
print("\nâœ… submission.csv saved!")

