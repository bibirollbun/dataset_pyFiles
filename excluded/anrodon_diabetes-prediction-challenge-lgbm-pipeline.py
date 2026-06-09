import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
import numpy as np


TARGET_COLUMN = 'diagnosed_diabetes'
ID_COLUMN = 'id'
RANDOM_STATE = 42


try:
    df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
    df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Error loading 'train.csv' and 'test.csv'.")


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new, predictive features and transforms existing ones for
    the diabetes prediction task.
    """
    df_new = df.copy()

    # --- 1. FEATURE INTERACTIONS (Domain-Based Ratios) ---

    # A. Atherogenic Index (Triglycerides to HDL ratio)
    # Handle division by zero/near-zero if necessary, though hdl_cholesterol is usually > 0.
    # We add a small epsilon (1e-6) just for safety.
    df_new['trig_hdl_ratio'] = df_new['triglycerides'] / (df_new['hdl_cholesterol'] + 1e-6)

    # B. Cholesterol Ratio (Total Cholesterol to HDL ratio)
    df_new['total_hdl_ratio'] = df_new['cholesterol_total'] / (df_new['hdl_cholesterol'] + 1e-6)
    
    # C. Blood Pressure Metrics
    
    # Pulse Pressure (PP) - Measure of arterial stiffness.
    df_new['pulse_pressure'] = df_new['systolic_bp'] - df_new['diastolic_bp']
    
    # Mean Arterial Pressure (MAP) - Better measure of perfusion than SBP/DBP alone.
    df_new['mean_arterial_pressure'] = (2 * df_new['diastolic_bp'] + df_new['systolic_bp']) / 3

    # D. Body Composition Interaction
    # Interaction between BMI and Central Obesity (Waist-to-Hip Ratio)
    df_new['bmi_whr_interaction'] = df_new['bmi'] * df_new['waist_to_hip_ratio']
    

    # --- 2. TRANSFORMATION (Handling Skewness and Non-Linearity) ---

    # A. Log Transformation for Triglycerides
    # Triglycerides are often highly skewed. np.log1p is safe for values >= 0.
    df_new['log_triglycerides'] = np.log1p(df_new['triglycerides'])
    
    # B. Age Transformations
    
    # Age Squared (introduces non-linear risk growth)
    df_new['age_squared'] = df_new['age'] ** 2
    
    # Age Binning (Captures risk thresholds at different life stages)
    df_new['age_group'] = pd.cut(
        df_new['age'],
        bins=[20, 40, 55, 70, 100],
        labels=['Young', 'Middle', 'Senior', 'Elderly'],
        right=False, # Bins are [min, max)
        include_lowest=True
    )
    
    # --- 3. DROP REDUNDANT/ORIGINAL FEATURES ---
    
    # Drop originals that are now represented by stronger engineered features
    df_new = df_new.drop(columns=[
        # Dropping triglycerides since log_triglycerides and ratios are better
        'triglycerides', 
        # Optionally, drop SBP/DBP if MAP/PP prove much stronger.
        # For now, let's keep SBP/DBP until you run feature importance.
    ], errors='ignore')

    return df_new
    
# Apply to both datasets
df_train_fe = feature_engineer(df_train)
df_test_fe = feature_engineer(df_test)


X_train = df_train_fe.drop(columns=['id', TARGET_COLUMN])
X_test = df_test_fe.drop(columns=['id'])
Y_train = df_train_fe[TARGET_COLUMN]

print(f"Final training set size: {len(X_train)} samples with {len(X_train.columns)} features.")


# 4.1. Ordinal Features (Require ranking/order)
ORDINAL_FEATURES = ['education_level', 'income_level', 'age_group']

# 4.2. Nominal Features (Require One-Hot Encoding)
NOMINAL_FEATURES = ['gender', 'ethnicity', 'smoking_status', 'employment_status']

# 4.3. Scalable Features (Continuous, Discrete, and Engineered numerical features)
# This list contains all numerical columns that are NOT 0/1 binary.
SCALABLE_FEATURES = [
    # Originals
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
    
    # Engineered
    'trig_hdl_ratio', 'total_hdl_ratio', 'pulse_pressure', 'mean_arterial_pressure',
    'bmi_whr_interaction', 'log_triglycerides', 'age_squared'
]

# 4.4. Binary Features (0/1 already, no transformation needed)
BINARY_FEATURES = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']


# --- 5.1 DEFINE TRANSFORMERS ---

# 5.1.1 Numerical Transformer
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()) # Essential for linear models and regularization
])

# 5.1.2 Ordinal Transformer
income_order = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle']
education_order = ['Highschool', 'Graduate']
age_group_order = ['Young', 'Middle', 'Senior', 'Elderly']

ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(
        categories=[education_order, income_order, age_group_order],
        handle_unknown='use_encoded_value',
        unknown_value=-1 # Treats unseen categories safely
    ))
])

# 5.1.3 Nominal Transformer (One-Hot Encoding)
nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Handles unseen categories safely
])

# --- 5.2 COMBINE TRANSFORMERS (ColumnTransformer) ---

preprocessor = ColumnTransformer(
    transformers=[
        # Apply scaling to numerical features
        ('scale', numerical_transformer, SCALABLE_FEATURES),

        # Apply ordinal ranking to ordered features
        ('ordinal_enc', ordinal_transformer, ORDINAL_FEATURES),

        # Apply One-Hot Encoding to nominal features
        ('nominal_enc', nominal_transformer, NOMINAL_FEATURES)
    ],
    # 'passthrough' includes the binary (0/1) features unchanged
    remainder='passthrough'
)



model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=RANDOM_STATE, metric='auc'))
])


print("Starting 5-Fold Cross-Validation for baseline model...")

# Define CV folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

auc_scores = cross_val_score(
    model_pipeline,
    X_train,
    Y_train,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1 # Use all available cores
)

print(f"\n5-Fold CV AUC Scores: {auc_scores}")
print(f"Mean CV AUC: {np.mean(auc_scores):.4f}")
print(f"Std Dev of CV AUC: {np.std(auc_scores):.4f}")

# Train the final (untuned) model for comparison purposes if needed later.
model_pipeline.fit(X_train, Y_train)


# --- 1. DEFINE A FASTER CV FOR TUNING ---
# Using 3 folds makes the search 40% faster than 5 folds.
cv_fast = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


# --- 2. DEFINE HYPERPARAMETER SEARCH SPACE ---
param_distributions_lgbm = {
    # Number of trees (boosting rounds)
    'classifier__n_estimators': [100, 300, 500, 800], 

    # Main complexity parameter in LGBM. Higher means deeper, more complex trees.
    'classifier__num_leaves': [20, 31, 50, 70],            

    # Step size shrinkage
    'classifier__learning_rate': [0.01, 0.05, 0.1],

    # Subsample ratio (prevents overfitting)
    'classifier__subsample': [0.7, 0.9],               
    'classifier__colsample_bytree': [0.7, 0.9],
    
    # Regularization (L1 and L2)
    'classifier__reg_alpha': [0.1, 1], 
    'classifier__reg_lambda': [0.1, 1]
}

# --- 3. RUN RANDOMIZED SEARCH ---
# Total fits = 3 folds * 20 iterations = 60 fits
random_search_lgbm = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_distributions_lgbm,
    n_iter=20,  # Number of combinations to test
    scoring='roc_auc',
    cv=cv_fast, # Use the faster 3-fold CV
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("Starting Hyperparameter Tuning with LIGHTGBM (60 total fits)...\n")
random_search_lgbm.fit(X_train, Y_train)

# --- 4. EVALUATE AND SELECT BEST MODEL ---

print(f"Best CV AUC Score Found: {random_search_lgbm.best_score_:.4f}")
print("Best Parameters:")
for key, value in random_search_lgbm.best_params_.items():
    print(f"  {key}: {value}")

# Store the best model
best_model = random_search_lgbm.best_estimator_


# Predict probabilities on the test set using the best model
test_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': df_test[ID_COLUMN], 
    'diagnosed_diabetes': test_pred_proba
})

# Save the submission file
submission_file_name = 'submission.csv'
submission_df.to_csv(submission_file_name, index=False)
print(f"Successfully created submission file: {submission_file_name}")

# Display the first few rows of the submission file
print("First 5 rows of the submission file:")
print(submission_df.head().to_markdown(index=False, numalign="left", stralign="left"))

