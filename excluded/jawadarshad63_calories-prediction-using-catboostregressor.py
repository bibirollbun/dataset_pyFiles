# Data manipulation and analysis
import pandas as pd  # For working with data in tabular form (DataFrames)
import numpy as np   # For numerical computations and array operations

# Visualization libraries
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns            # For advanced statistical visualizations

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings during execution

# Scikit-learn utilities for modeling and preprocessing
from sklearn.model_selection import KFold              # For cross-validation splitting
from sklearn.pipeline import Pipeline                  # For creating modeling pipelines
from sklearn.compose import ColumnTransformer          # For applying different preprocessing to columns
from sklearn.preprocessing import StandardScaler       # For scaling numerical features
from sklearn.linear_model import Ridge                 # Ridge regression model
from sklearn.metrics import mean_squared_log_error     # Evaluation metric (used for RMSLE)
from sklearn.impute import SimpleImputer               # For handling missing values

# Additional libraries for modeling
from catboost import CatBoostRegressor                 # Gradient boosting model optimized for categorical data
from category_encoders import TargetEncoder            # For encoding categorical variables using target mean



# Evaluation Metric: Root Mean Squared Logarithmic Error (RMSLE)
def get_rmsle(y_true, y_pred):
    # Ensure that true and predicted values are non-negative to avoid issues with log calculation
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    
    # Calculate and return the square root of the mean squared logarithmic error
    return np.sqrt(mean_squared_log_error(y_true, y_pred))



train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

print(train_df.shape)
print(test_df.shape)
train_df.head()



import matplotlib.pyplot as plt
import seaborn as sns

# Basic info
print("ðŸ”¹ Train Shape:", train_df.shape)
print("ðŸ”¹ Test Shape:", test_df.shape)
print("\nðŸ”¹ Train Columns:\n", train_df.columns)


# Check data types and missing values
print("\nðŸ”¹ Missing Values:\n", train_df.isnull().sum())
print("\nðŸ”¹ Data Types:\n", train_df.dtypes)


# Describe numeric features
print("\nðŸ”¹ Summary Statistics:")
print(train_df.describe())


# Target variable distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_df['Calories'], bins=50, kde=True, color='salmon')
plt.title('Target Variable: Calories Distribution')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


# Pairplot for interactions (optional: limit to smaller subset)
sns.pairplot(train_df[['Age', 'Height', 'Weight', 'Heart_Rate', 'Duration', 'Calories']].sample(500), corner=True)
plt.suptitle("Feature Interactions", y=1.02)
plt.tight_layout()
plt.show()





# Boxplots for potential outliers
num_features = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('Calories')

for col in num_features:
    plt.figure(figsize=(8, 3))
    sns.boxplot(data=train_df, x=col, color='lightblue')
    plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.show()


# Feature Engineering: Calculate Body Mass Index (BMI) and add it as a new feature
# BMI = weight (kg) / height (m)^2 â€” height is converted from cm to meters
train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)



# Prepare features and target

# Drop 'id' and target column 'Calories' from training data to create feature matrix X
X = train_df.drop(columns=['id', 'Calories'])

# Apply log1p (log(1 + x)) transformation to the target to reduce skewness and stabilize variance
y = np.log1p(train_df['Calories'])

# Drop 'id' from test data to create the final test feature matrix
X_test_final = test_df.drop(columns=['id'])



# Target Encoding for categoricals

# Identify categorical columns (object or category data types)
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Initialize the TargetEncoder (encodes categories using the mean of the target variable)
encoder = TargetEncoder()

# Fit the encoder on training data and transform the categorical features in training set
X[categorical_features] = encoder.fit_transform(X[categorical_features], y)

# Apply the same transformation to the test set using the fitted encoder
X_test_final[categorical_features] = encoder.transform(X_test_final[categorical_features])



# Preprocessing

# Select numeric columns (int64 and float64 types) for preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create a pipeline for numeric features:
# 1. Impute missing values using the mean
# 2. Scale features to have zero mean and unit variance
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps into a ColumnTransformer:
# Apply the numeric transformer to the selected numeric features
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features)
])



# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)



# Best model (CatBoost) with tuned parameters

model = CatBoostRegressor(
    iterations=1000,               # Maximum number of boosting iterations (trees)
    learning_rate=0.03,            # Step size for updating weights; lower values lead to slower but more precise learning
    depth=8,                       # Depth of each tree; higher depth can model more complex patterns
    l2_leaf_reg=5,                 # L2 regularization term to reduce overfitting
    loss_function='RMSE',         # Root Mean Squared Error as the optimization objective
    random_strength=1,            # Random noise added to features to reduce overfitting
    bagging_temperature=0.2,      # Controls randomness of bagging; lower = less overfitting
    od_type='Iter',               # Overfitting detector type (based on number of iterations)
    early_stopping_rounds=50,     # Stop training if validation error doesn't improve for 50 iterations
    verbose=100,                  # Print training progress every 100 iterations
    random_state=42               # Seed for reproducibility
)



# Pipeline

# Create a full modeling pipeline that includes:
# 1. Preprocessing (imputation + scaling of numeric features)
# 2. Regression model (CatBoost with tuned parameters)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Applies preprocessing steps defined earlier
    ('regressor', model)             # Trains the CatBoostRegressor on the preprocessed data
])



# Cross-validation

# List to store RMSLE score for each fold
fold_rmsle_scores = []

# Initialize array to store averaged predictions for the test set
test_preds = np.zeros(len(X_test_final))

# Loop through each fold in K-Fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # Split data into training and validation sets for this fold
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Fit the pipeline (preprocessing + model) on training data
    model_pipeline.fit(X_train, y_train)



# Final prediction

# Convert test predictions back from log scale to original scale using expm1
# Clip predictions to be within a reasonable range (0 to 5000) to avoid extreme values
final_predictions = np.clip(np.expm1(test_preds), 0, 5000)

# Create a submission DataFrame with 'id' and predicted 'Calories'
submission_df = pd.DataFrame({
    'id': test_df['id'],             # Use original test IDs
    'Calories': final_predictions    # Use clipped and transformed predictions
})



submission_df.to_csv('submission.csv', index=False)
print("\nðŸŽ¯ Submission file 'submission.csv' created with improved CatBoost model.")




