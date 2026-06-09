import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# Validation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

# Configuration
CONFIG = {
    'SEED': 42,
    'N_FOLDS': 5,
    'TARGET': 'diagnosed_diabetes'
}

# Set random seed for reproducibility
np.random.seed(CONFIG['SEED'])
print("Setup Complete.")


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv') # Assuming you have a test file for submission

# Combine for consistent preprocessing
train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], axis=0).reset_index(drop=True)

# Drop Noise
drop_cols = ['id', 'patient_id']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

print(f"Full Dataset Shape: {df.shape}")


def create_medical_features(data):
    df = data.copy()
    
    # 1. Lipid Ratios (Crucial for Metabolic Syndrome)
    # Avoid division by zero by adding a tiny epsilon if needed, though HDL is rarely 0
    df['TG_HDL_Ratio'] = df['triglycerides'] / df['hdl_cholesterol']
    df['LDL_HDL_Ratio'] = df['ldl_cholesterol'] / df['hdl_cholesterol']
    df['Non_HDL_Cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # 2. Blood Pressure Dynamics
    df['Pulse_Pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['MAP'] = (df['systolic_bp'] + (2 * df['diastolic_bp'])) / 3
    
    # 3. Body Shape Interaction
    # "Apple shape" (high waist, high BMI) is riskier than "Pear shape"
    df['BMI_Waist_Interaction'] = df['bmi'] * df['waist_to_hip_ratio']
    
    # 4. Age Amplifiers
    # Family history becomes more relevant as you age
    df['Age_Family_Interaction'] = df['age'] * df['family_history_diabetes']
    
    # 5. Lifestyle Score
    # Combine physical activity and diet into a single "Health Effort" score
    # Normalize them first so one doesn't dominate
    df['Lifestyle_Score'] = (df['physical_activity_minutes_per_week'] / 600) + (df['diet_score'] / 10)
    
    return df

df = create_medical_features(df)
print("Added Medical Features.")
print(df[['TG_HDL_Ratio', 'Pulse_Pressure', 'Lifestyle_Score']].head())


# 1. Ordinal Encoding (Manual Mapping)
# We want the model to know that 'High' > 'Low'
edu_map = {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3}
inc_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}

df['education_level'] = df['education_level'].map(edu_map).fillna(-1)
df['income_level'] = df['income_level'].map(inc_map).fillna(-1)

# 2. Categorical Encoding
# For the remaining text columns (Gender, Ethnicity, etc.), we use Label Encoding
# because Tree-based models handle this well enough.
cat_cols = ['gender', 'ethnicity', 'smoking_status', 'employment_status']
le = LabelEncoder()

for col in cat_cols:
    if col in df.columns:
        # Convert to string to handle NaN safely
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

# Split back into Train and Test
train_df = df[df['is_train'] == 1].drop(columns=['is_train'])
test_df = df[df['is_train'] == 0].drop(columns=['is_train', CONFIG['TARGET']])

X = train_df.drop(columns=[CONFIG['TARGET']])
y = train_df[CONFIG['TARGET']]

print("Data Ready for Training.")


# 1. LightGBM (Fast, High Performance)
lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': CONFIG['SEED'],
    'n_jobs': -1,
    'verbose': -1
}
clf1 = lgb.LGBMClassifier(**lgb_params)

# 2. XGBoost (Reliable)
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': CONFIG['SEED'],
    'n_jobs': -1,
    'eval_metric': 'auc'
}
clf2 = XGBClassifier(**xgb_params)

# 3. CatBoost (Great for Categoricals)
# Note: CatBoost is slower but very accurate
cat_params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'random_seed': CONFIG['SEED'],
    'verbose': 0,
    'allow_writing_files': False
}
clf3 = CatBoostClassifier(**cat_params)

# 4. The Voting Ensemble (Soft Vote = Average of Probabilities)
ensemble = VotingClassifier(
    estimators=[
        ('lgb', clf1), 
        ('xgb', clf2), 
        ('cat', clf3)
    ],
    voting='soft',
    n_jobs=-1
)

print("Ensemble Defined.")


kf = StratifiedKFold(n_splits=CONFIG['N_FOLDS'], shuffle=True, random_state=CONFIG['SEED'])

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_df))

print(f"Starting {CONFIG['N_FOLDS']}-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Train the Ensemble
    ensemble.fit(X_train, y_train)
    
    # Predict on Validation Set
    val_p = ensemble.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_p
    
    # Predict on Test Set (Accumulate and average later)
    test_preds += ensemble.predict_proba(test_df)[:, 1] / CONFIG['N_FOLDS']
    
    # Score
    score = roc_auc_score(y_val, val_p)
    print(f"Fold {fold+1} AUC: {score:.5f}")

print("-" * 30)
print(f"Overall CV AUC: {roc_auc_score(y, oof_preds):.5f}")


# 1. Ensure we have the correct IDs from the original file
# We reload strictly the 'id' column to be 100% safe against previous drops
test_ids = pd.read_csv("data/test.csv", usecols=['id'])['id']

# 2. Create the submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_preds
})

# 3. Save
submission.to_csv('submission.csv', index=False)
print(f"✅ Submission saved with {len(submission)} rows.")

# 4. Verify the first few rows look correct (Should show actual IDs, not just 0,1,2)
print("\nFirst 5 rows of submission:")
print(submission.head())

# 5. Distribution Check
plt.figure(figsize=(6,4))
sns.histplot(submission['diagnosed_diabetes'], kde=True, color='teal')
plt.title("Final Prediction Distribution")
plt.show()

