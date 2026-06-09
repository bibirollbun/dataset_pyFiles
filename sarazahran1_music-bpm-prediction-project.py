# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Model selection & evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, make_scorer

# Feature engineering
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA


# Load data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

# Basic exploration
print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print("\nTraining data info:")
train_data.info()
print("\nTest data info:")
test_data.info()

# Display first few rows
print("\nTraining data head:")
display(train_data.head())
print("\nTest data head:")
display(test_data.head())

# Statistical summary
print("\nTraining data description:")
display(train_data.describe())
print("\nTest data description:")
display(test_data.describe())

# Check for missing values
print("\nMissing values in training data:")
print(train_data.isnull().sum().sum())
print("\nMissing values in test data:")
print(test_data.isnull().sum().sum())


# Set up the visualization style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Distribution of target variable
plt.subplot(1, 2, 1)
sns.histplot(train_data['BeatsPerMinute'], kde=True)
plt.title('Distribution of BeatsPerMinute')

# Correlation heatmap
plt.subplot(1, 2, 2)
corr_matrix = train_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()

# Pairplot of selected features (first 5 for performance)
sns.pairplot(train_data.iloc[:, :6])
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

# Boxplots for outlier detection
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.drop('BeatsPerMinute')
plt.figure(figsize=(16, 8))
for i, col in enumerate(numeric_cols[:8], 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=train_data[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


# Separate features and target
X = train_data.drop(['id', 'BeatsPerMinute'], axis=1)
y = train_data['BeatsPerMinute']
X_test = test_data.drop('id', axis=1)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# Feature engineering: Add polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_processed)
X_test_poly = poly.transform(X_test_processed)

# Get the actual number of features after polynomial transformation
n_features = X_poly.shape[1]
print(f"Number of features after polynomial transformation: {n_features}")

# Adjust k for feature selection to not exceed available features
k_value = min(50, n_features)  # Use the smaller of 50 or the actual number of features

# Feature selection
selector = SelectKBest(score_func=f_regression, k=k_value)
X_selected = selector.fit_transform(X_poly, y)
X_test_selected = selector.transform(X_test_poly)

# Dimensionality reduction with PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_selected)
X_test_pca = pca.transform(X_test_selected)

print(f"Original shape: {X.shape}")
print(f"After preprocessing: {X_processed.shape}")
print(f"After polynomial features: {X_poly.shape}")
print(f"After feature selection: {X_selected.shape}")
print(f"After PCA: {X_pca.shape}")


# Define models up to Lasso
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42)
}

# Evaluate models using cross-validation
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_pca, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    results[name] = (-cv_scores).mean()
    print(f"{name}: RMSE = {(-cv_scores).mean():.4f} (卤 {cv_scores.std():.4f})")

# Find best model
best_model_name = min(results, key=results.get)
print(f"\nBest model: {best_model_name} with RMSE: {results[best_model_name]:.4f}")

# Train the best model on full data
best_model = models[best_model_name]
best_model.fit(X_pca, y)





    
   


# Make predictions on test data
test_predictions = best_model.predict(X_test_pca)
# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'BeatsPerMinute': test_predictions
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")



if best_model_name in ['Ridge', 'Lasso']:
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_pca, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f}")
    
    # Update best model with tuned parameters
    best_model = grid_search.best_estimator_
    
    # Retrain with best parameters
    best_model.fit(X_pca, y)
    
    # Make new predictions with tuned model
    test_predictions = best_model.predict(X_test_pca)
     # Create updated submission file
    submission = pd.DataFrame({
        'id': test_data['id'],
        'BeatsPerMinute': test_predictions
    })
    
    submission.to_csv('submission_tuned.csv', index=False)
    print("Tuned submission file created successfully!")

