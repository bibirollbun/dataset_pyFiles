import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import gc


!pip install --upgrade scikit-learn xgboost


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


# Apply log-transformation to Calories
train['Calories_log'] = np.log1p(train['Calories'])

# Check for invalid values in Calories_log
print("NaN in Calories_log:", train['Calories_log'].isna().sum())
print("Infinite in Calories_log:", np.isinf(train['Calories_log']).sum())


# Define features and target
X = train.drop(['Calories', 'Calories_log', 'id'], axis=1)
y = train['Calories_log']
X_test = test.drop(['id'], axis=1)

# Feature engineering on full X and X_test (before splitting)
X['Duration_Heart_Rate'] = X['Duration'] * X['Heart_Rate']
X_test['Duration_Heart_Rate'] = X_test['Duration'] * X_test['Heart_Rate']

# Check for NaN values and impute
print("NaN in X:", X.isna().sum())
print("NaN in X_test:", X_test.isna().sum())
X.fillna(X.mean(numeric_only=True), inplace=True)
X_test.fillna(X_test.mean(numeric_only=True), inplace=True)



# Verify data types
print("X dtypes:\n", X.dtypes)


# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)


# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])


# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for NaN in split data
print("NaN in X_train:", X_train.isna().sum())
print("NaN in X_val:", X_val.isna().sum())
print("NaN in y_train:", y_train.isna().sum())
print("NaN in y_val:", y_val.isna().sum())



# Define models with reduced complexity to avoid memory issues
models = {
    'Linear Regression': Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    'Random Forest': Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))]),
    'XGBoost': Pipeline([('preprocessor', preprocessor), ('regressor', XGBRegressor(max_depth=5, n_estimators=100, random_state=42))])
}



# Model training and evaluation
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_val)
    # Reverse log-transformation for evaluation
    y_pred = np.expm1(y_pred_log)
    y_val_original = np.expm1(y_val)
    rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
    print(f"{name} RMSE (original scale): {rmse:.4f}")
    # Cross-validation on log scale
    print(f"Running cross-validation for {name}...")
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='neg_root_mean_squared_error')  # Reduced to 3 folds
    print(f"{name} CV RMSE (log scale): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    gc.collect()  # Free up memory


# Choose best model (XGBoost)
best_model = models['XGBoost']



# Predict on test set (using full test set, not sampled)
test_predictions_log = best_model.predict(X_test)
test_predictions = np.expm1(test_predictions_log)

# Create submission DataFrame
submission = pd.DataFrame({'id': test['id'], 'Calories': test_predictions})

# Save to CSV with new name
submission.to_csv('/kaggle/working/calorie.csv', index=False)
print("Submission file created at /kaggle/working/calorie.csv")

# Show preview
print("\nSubmission Preview:")
print(submission.head())

# Display download link
from IPython.display import FileLink
display(FileLink('calorie.csv'))




# Feature importance
xgb_model = best_model.named_steps['regressor']
feature_names = (numerical_cols + 
                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)))
importances = pd.Series(xgb_model.feature_importances_, index=feature_names)


plt.figure(figsize=(10, 6))
importances.sort_values(ascending=False).head(10).plot.bar()
plt.title('Top 10 Feature Importances (XGBoost)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

