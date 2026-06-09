# Data manipulation and analysis &#128202;
import pandas as pd
import numpy as np

# Machine Learning libraries &#129302;
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Utilities &#128295;
import joblib
import warnings
warnings.filterwarnings('ignore')

print("&#9989; All libraries imported successfully!")


# Load the training dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")

print("="*60)
print(f"&#128202; Dataset Shape: {df.shape}")
print(f"&#128202; Number of samples: {df.shape[0]:,}")
print(f"&#128202; Number of features: {df.shape[1]}")
print("="*60)
print("\n&#128269; First 5 rows of the dataset:")
df.head()


# Check data types and missing values
print("&#128203; Dataset Information:")
print("="*60)
df.info()
print("="*60)


# Education Level Distribution
print("&#127891; Education Level Distribution:")
print("="*60)
print(df["education_level"].value_counts())
print("="*60)


# Employment Status Distribution
print("&#128188; Employment Status Distribution:")
print("="*60)
print(df["employment_status"].value_counts())
print("="*60)


# Loan Purpose Distribution
print("&#127919; Loan Purpose Distribution:")
print("="*60)
print(df["loan_purpose"].value_counts())
print("="*60)


# Grade/Subgrade Distribution
print("&#128200; Grade Subgrade Distribution:")
print("="*60)
print(df["grade_subgrade"].value_counts())
print("="*60)
print(f"\n&#128204; Total unique grade_subgrade values: {df['grade_subgrade'].nunique()}")


# function to grade/subgrade division and marital Status 
def marital_grade(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    df["marital_status"] = (df["marital_status"] == "Married").astype(int)
    df["grade"] = [g[0] for g in df["grade_subgrade"]]
    df["sub_grade"] = [g[1] for g in df["grade_subgrade"]]
    df.drop(columns=["grade_subgrade"], inplace=True)
    return df


# test the function : 
df = marital_grade(df)
df.head()


# Identify numerical features
NUM_FEATURES = df.select_dtypes(include=["int64", "float64"]).columns.to_list()
NUM_FEATURES.remove("loan_paid_back")  # Target variable
NUM_FEATURES.remove("id")              # Identifier column

# Identify categorical features
CAT_FEATURES = df.select_dtypes(include=["object"]).columns.to_list()

print(f"&#128202; Numerical Features ({len(NUM_FEATURES)}): {NUM_FEATURES}")
print(f"\n&#128202; Categorical Features ({len(CAT_FEATURES)}): {CAT_FEATURES}")


# Categorical preprocessing pipeline
cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

# Numerical preprocessing pipeline
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Combine both pipelines
preprocessing_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, CAT_FEATURES),
    ('num', num_pipeline, NUM_FEATURES)
])

print("&#9989; Preprocessing pipelines created!")


# Prepare features and target
df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
X = df.drop(columns=['loan_paid_back', 'id'])
y = df['loan_paid_back']


# First split: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: Split temp into validation and test (50-50)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("="*70)
print("&#128202; Data Split Summary".center(70))
print("="*70)
print(f"   {'Dataset':<20} {'Samples':>15} {'Percentage':>15}")
print("-"*70)
print(f"   {'Training set':<20} {X_train.shape[0]:>15,} {X_train.shape[0]/len(X)*100:>14.1f}%")
print(f"   {'Validation set':<20} {X_val.shape[0]:>15,} {X_val.shape[0]/len(X)*100:>14.1f}%")
print(f"   {'Test set':<20} {X_test.shape[0]:>15,} {X_test.shape[0]/len(X)*100:>14.1f}%")
print("="*70)


def preprocess_fn(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    df = marital_grade(df)
    df = preprocessing_pipeline.fit_transform(df)
    return df

# Apply preprocessing to all splits
print("&#128260; Applying preprocessing to all data splits...")
print()

X_train_prep = preprocess_fn(X_train)
X_val_prep = preprocess_fn(X_val)
X_test_prep = preprocess_fn(X_test)

print("="*70)
print("&#9989; Preprocessing applied to all splits!".center(70))
print("="*70)
print(f"\n{'Dataset':<20} {'Original Shape':>20} {'Preprocessed Shape':>25}")
print("-"*70)
print(f"{'Train':<20} {str(X_train.shape):>20} {str(X_train_prep.shape):>25}")
print(f"{'Validation':<20} {str(X_val.shape):>20} {str(X_val_prep.shape):>25}")
print(f"{'Test':<20} {str(X_test.shape):>20} {str(X_test_prep.shape):>25}")
print("="*70)
print(f"\n&#10024; Total features after encoding: {X_train_prep.shape[1]}")


# Create DMatrix objects for efficient XGBoost training
dtrain = xgb.DMatrix(X_train_prep, label=y_train)
dval = xgb.DMatrix(X_val_prep, label=y_val)
dtest = xgb.DMatrix(X_test_prep, label=y_test)

print("&#9989; DMatrix objects created!")


# Best hyperparameters from Grid Search CV
# These parameters were obtained through extensive hyperparameter tuning
params = {
    "objective": "binary:logistic",      # Binary classification
    "eval_metric": "auc",                # Optimize for ROC-AUC
    "tree_method": "hist",               # Fast histogram-based algorithm
    "device": "cuda",                    # GPU acceleration (change to 'cpu' if needed)
    
    # Optimized hyperparameters from Grid Search
    "max_depth": 6,                      # Maximum tree depth
    "eta": 0.05,                         # Learning rate
    "subsample": 0.9,                    # Row sampling ratio
    "colsample_bytree": 0.9,             # Column sampling ratio
    "reg_lambda": 1,                     # L2 regularization
    "min_child_weight": 1,               # Minimum sum of instance weight
    "seed": 42                           # Random seed for reproducibility
}

print("="*70)
print("&#9881; Model Hyperparameters Configuration".center(70))
print("="*70)
print(f"\n{'Parameter':<30} {'Value':>20}")
print("-"*70)
for key, value in params.items():
    print(f"{key:<30} {str(value):>20}")
print("="*70)
print("\n&#10024; These hyperparameters were selected via Grid Search CV!")
print("&#127919; Optimization metric: ROC-AUC")
print("&#128260; Cross-validation folds: 3")
print("="*70)


# Set up evaluation sets for monitoring
evals = [(dtrain, "train"), (dval, "val")]

# Train the model with early stopping
print("&#128640; Training XGBoost model...")
print("="*70)
print("&#9200; This may take a few minutes depending on your hardware...")
print("="*70)
print()

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,           # Maximum number of boosting rounds
    evals=evals,
    early_stopping_rounds=50,       # Stop if no improvement for 50 rounds
    verbose_eval=50                 # Print metrics every 50 rounds
)

print()
print("="*70)
print("&#9989; Model training completed!")
print(f"&#127919; Best iteration: {model.best_iteration}")
print(f"&#127919; Best validation AUC: {model.best_score:.4f}")
print("="*70)


# Generate predictions
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate metrics
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)

# Display results with beautiful formatting
print("\n" + "="*70)
print("&#128293; XGBoost Model Performance &#128293;".center(70))
print("="*70)
print(f"\n{'Metric':<30} {'Score':>20}")
print("-"*70)
print(f"{'&#128202; ROC-AUC Score':<30} {auc:>20.4f}")
print(f"{'&#128202; Accuracy Score':<30} {acc:>20.4f}")
print("="*70)
print("\n&#128203; Detailed Classification Report:")
print("="*70)
print(classification_report(y_test, y_pred, target_names=['Not Paid Back', 'Paid Back']))
print("="*70)


# Get feature importance
importance_dict = model.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': importance_dict.keys(),
    'Importance': importance_dict.values()
}).sort_values('Importance', ascending=False).head(15)

print("\n" + "="*70)
print("&#127942; Top 15 Most Important Features".center(70))
print("="*70)
print(f"\n{'Rank':<8} {'Feature':<35} {'Importance':>15}")
print("-"*70)
for idx, row in importance_df.iterrows():
    rank = importance_df.index.get_loc(idx) + 1
    print(f"{rank:<8} {row['Feature']:<35} {row['Importance']:>15.0f}")
print("="*70)


# Load test data
test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

print("="*70)
print("&#128202; Test Dataset Loaded".center(70))
print("="*70)
print(f"\n&#128202; Test dataset shape: {test_df.shape}")
print(f"&#128202; Number of predictions to generate: {test_df.shape[0]:,}")
print("\n&#128269; First 5 rows:")
test_df.head()


# Store IDs for submission
test_ids = test_df['id'].copy()

# Drop ID column for preprocessing
test_features = test_df.drop(columns=['id'])

# Apply preprocessing
print("&#128260; Preprocessing test data...")
test_preprocessed = preprocess_fn(test_features)

print("="*70)
print("&#9989; Test data preprocessed successfully!".center(70))
print("="*70)
print(f"\n&#128202; Original shape: {test_features.shape}")
print(f"&#128202; Preprocessed shape: {test_preprocessed.shape}")
print("="*70)


# Create DMatrix for predictions
dtest_submission = xgb.DMatrix(test_preprocessed)

# Generate predictions (probabilities)
print("&#127919; Generating predictions...")
predictions = model.predict(dtest_submission)

print("="*70)
print("&#9989; Predictions generated successfully!".center(70))
print("="*70)
print(f"\n&#128202; Prediction Statistics:")
print(f"   Mean probability:  {predictions.mean():.4f}")
print(f"   Std deviation:     {predictions.std():.4f}")
print(f"   Min probability:   {predictions.min():.4f}")
print(f"   Max probability:   {predictions.max():.4f}")
print("="*70)


# Create submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("="*70)
print("&#9989; Submission file created successfully!".center(70))
print("="*70)
print(f"\n&#128202; Submission file shape: {submission.shape}")
print(f"&#128202; Total predictions: {len(submission):,}")
print("\n&#128203; First 10 rows of submission:")
print("-"*70)
print(submission.head(10).to_string(index=False))
print("\n&#128203; Last 10 rows of submission:")
print("-"*70)
print(submission.tail(10).to_string(index=False))

