import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier,early_stopping, log_evaluation
from catboost import CatBoostClassifier

# Validations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.preprocessing import LabelEncoder

# System Settings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')


# LOADING THE DATA FILES
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

# Checking Actual size or shape of data
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

# Checking the data
train_df.head()


# Finding NULL Values
print(f"Train_df Null values: {train_df.isnull().sum().sum()}")
print(f"Test_df Null values: {test_df.isnull().sum().sum()}")

# Describing the data
train_df.describe()


# Size of the plot
plt.figure(figsize=(8, 6))

# Countplot for Visualization
sns.countplot(x='diagnosed_diabetes', data=train_df, palette='plasma')

# Adding title and labels
plt.title('Distribution of Target Variable (Diabetes)', fontsize=15)
plt.xlabel('Diagnosed Diabetes (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Percentage of Diabetes Distribution
percentage = train_df['diagnosed_diabetes'].value_counts(normalize=True) * 100
print(f"Percentage of Diabetes Positive: {percentage[1]:.2f}%")
print(f"Percentage of Diabetes Negative: {percentage[0]:.2f}%")


def perform_feature_engineering_safe(df_train, df_test):
    # 1. Separating the data
    # We save the target col of Training data so that it does't collapse with the Testing data
    if 'diagnosed_diabetes' in df_train.columns:
        y = df_train['diagnosed_diabetes'].copy()
        df_train = df_train.drop('diagnosed_diabetes', axis=1)
    else:
        # If it's already separated, we assume it's stored in a variable 'y' outside
        # For this function, let's assume it was passed in the dataframe
        raise ValueError("Train dataframe is missing the 'diagnosed_diabetes' column!")

    # 2. Combining the features into one dataframe (all_dat)
    train_n = len(df_train)
    all_data = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    # 3. CATEGORICAL ENCODING
    cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                'smoking_status', 'employment_status']
    
    le = LabelEncoder()
    for col in cat_cols:
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        
    print("Categorical Encoding Completed")

    # 4. INTERACTION FEATURES
    all_data['Age_BMI_Interact'] = all_data['age'] * all_data['bmi']
    all_data['Cholesterol_Risk'] = all_data['cholesterol_total'] / (all_data['hdl_cholesterol'] + 0.1)
    all_data['BP_Load'] = all_data['systolic_bp'] * all_data['diastolic_bp']
    all_data['Lifestyle_Score'] = all_data['diet_score'] * all_data['physical_activity_minutes_per_week']
    all_data['Lipid_Health'] = all_data['triglycerides'] * all_data['ldl_cholesterol']

    print("Interaction Features Created")

    # 5. SPLIT BACK
    train_features = all_data.iloc[:train_n].copy()
    test_final = all_data.iloc[train_n:].copy()
    
    # 6. RE-ATTACH TARGET
    # We use .values to ignore index alignment issues, just pasting the column back in place
    train_features['diagnosed_diabetes'] = y.values
    
    return train_features, test_final

# --- EXECUTE ---
# Reload data to be safe (fresh start)
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print("Starting Robust Feature Engineering...")
train_df, test_df = perform_feature_engineering_safe(train_df, test_df)

# VERIFICATION
# Check for NaNs in the target column
nan_count = train_df['diagnosed_diabetes'].isnull().sum()
print(f"\nNaNs in Target Column: {nan_count}")

if nan_count == 0:
    print("SUCCESS: Target column is intact!")
else:
    print("CRITICAL ERROR: NaNs found. Do not proceed.")

# Optional: Check shape
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape:  {test_df.shape}")

# Final DATA
print("Training data is:\n")
train_df.head(3)


X = train_df.drop(['id','diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']

X_test = test_df.drop(['id'], axis=1)

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)


oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])

print(f"Data prepared for {FOLDS}-Fold Cross-Validation")
print(f"Training Features: {X.shape[1]}")


# We use standard high-performance parameters for these models
xgb_model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    tree_method='hist',
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc',
    early_stopping_rounds=50,
    verbosity=0
)

lgbm_model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

cat_model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.05,
    depth=7,
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=0
)


def train_and_predict(model, name):
    print(f"Training {name}...")
    
    current_oof = np.zeros(X.shape[0])
    current_test_pred = np.zeros(X_test.shape[0])
    
    # The Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if name == "LightGBM":
            # Special handling for LightGBM v4+
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[
                    early_stopping(stopping_rounds=100, verbose=False),
                    log_evaluation(period=0)
                ]
            )
        else:
            # Standard handling for XGBoost & CatBoost
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        # Predicting the Values
        val_preds = model.predict_proba(X_val)[:, 1]
        test_fold_preds = model.predict_proba(X_test)[:, 1]
        
        # Fill the arrays
        current_oof[val_idx] = val_preds
        current_test_pred += test_fold_preds / FOLDS
        
    # Print Score
    score = roc_auc_score(y, current_oof)
    print(f"{name} Overall AUC: {score:.5f}")
    return current_oof, current_test_pred

# 3. EXECUTE TRAINING
xgb_oof, xgb_test = train_and_predict(xgb_model, "XGBoost")
lgbm_oof, lgbm_test = train_and_predict(lgbm_model, "LightGBM")
cat_oof, cat_test = train_and_predict(cat_model, "CatBoost")

print("\n✅ All Models Trained!")


# XGB model performs so well thats why we give them a litle bit more weight as compare to others
final_predictions = (0.4 * xgb_test) + (0.4 * lgbm_test) + (0.2 * cat_test)
submission['diagnosed_diabetes'] = final_predictions
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully! Ready to submit.")
submission.head()

