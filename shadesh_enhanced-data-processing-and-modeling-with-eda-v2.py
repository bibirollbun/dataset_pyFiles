# ğŸ“Š Optimized Data Processing and Modeling with Interactive EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# ======================
# ğŸ§¹ DATA PREPROCESSING
# ======================

# âœ… Encode the 'Sex' column
train['Sex'] = train['Sex'].str.lower().map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].str.lower().map({'male': 0, 'female': 1})

# Feature engineering - add BMI (Body Mass Index)
train['BMI'] = train['Weight'] / (train['Height']/100)**2
test['BMI'] = test['Weight'] / (test['Height']/100)**2

# ======================
# ğŸ”� INTERACTIVE EDA
# ======================

def show_eda():
    numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
    
    # 1. Target Distribution
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(train['Calories'], kde=True, bins=30)
    plt.title('Calories Distribution')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=train['Calories'])
    plt.title('Calories Boxplot')
    plt.show()
    
    # 2. Numerical Features Distribution
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        sns.histplot(train[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
    
    # 3. Correlation Analysis
    plt.figure(figsize=(12, 8))
    corr_matrix = train.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.show()
    
    # 4. Feature vs Target
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        sns.scatterplot(x=col, y='Calories', data=train, alpha=0.5)
        plt.title(f'{col} vs Calories')
    plt.tight_layout()
    plt.show()

# Display interactive EDA
show_eda()

# ======================
# ğŸ¤– ADVANCED MODELING
# ======================

# Prepare data
X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']
X_test = test.drop(['id'], axis=1)

# Create robust preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', RobustScaler()),  # Better for handling outliers than StandardScaler
    ('feature_selector', SelectKBest(score_func=mutual_info_regression, k='all'))
])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Define models with optimized parameters
models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    ),
    'HistGradientBoosting': HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nğŸ”¥ Training {name}...")
    model.fit(X_train_processed, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_processed, y_train, 
                              cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    
    # Validation score
    val_pred = model.predict(X_val_processed)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    results[name] = {
        'model': model,
        'cv_mae': cv_mae,
        'val_mae': val_mae
    }
    
    print(f"âœ… {name} - CV MAE: {cv_mae:.2f}, Val MAE: {val_mae:.2f}")

# Select best model
best_model_name = min(results, key=lambda x: results[x]['val_mae'])
best_model = results[best_model_name]['model']
print(f"\nğŸ�† Best Model: {best_model_name} with Val MAE: {results[best_model_name]['val_mae']:.2f}")

# Feature Importance
plt.figure(figsize=(10, 6))
if hasattr(best_model, 'feature_importances_'):
    feature_imp = pd.Series(best_model.feature_importances_, index=X.columns)
    feature_imp.sort_values().plot(kind='barh')
    plt.title(f'{best_model_name} Feature Importance')
    plt.show()

# Ensemble predictions (weighted average of top models)
top_models = sorted(results.items(), key=lambda x: x[1]['val_mae'])[:3]
weights = [1/(m[1]['val_mae']) for m in top_models]
weights = [w/sum(weights) for w in weights]

test_preds = sum(m[1]['model'].predict(X_test_processed) * w 
               for m, w in zip(top_models, weights))

# Save submission
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_preds
})
submission.to_csv('submission.csv', index=False)
print("\nâœ… submission.csv saved!")

