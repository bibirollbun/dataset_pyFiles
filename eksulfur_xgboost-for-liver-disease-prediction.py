import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import numpy as np

# Load datasets
try:
    train_df = pd.read_csv('/kaggle/input/ai-for-liver-disease-outcome-prediction/train.csv')
    test_df = pd.read_csv('/kaggle/input/ai-for-liver-disease-outcome-prediction/test.csv')
    original_test_ids = test_df['id']  # Save original IDs for submission
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()


def create_features(df):
    """Create new features for the model"""
    df_copy = df.copy()
    
    # Age conversion
    df_copy['Age_years'] = df_copy['Age'] / 365.25

    # Medical ratios
    epsilon = 1e-6  # Prevent division by zero
    SGOT_ULN = 40  # Upper Limit of Normal for SGOT
    df_copy['APRI'] = (df_copy['SGOT'] / SGOT_ULN) / df_copy['Platelets'].replace(0, epsilon)
    df_copy['Bilirubin_Alk_Phos_Ratio'] = df_copy['Bilirubin'] / df_copy['Alk_Phos'].replace(0, epsilon)
    df_copy['Bilirubin_Albumin_Ratio'] = df_copy['Bilirubin'] / df_copy['Albumin'].replace(0, epsilon)

    # Interaction features
    df_copy['Age_Bilirubin_Interaction'] = df_copy['Age_years'] * df_copy['Bilirubin']
    df_copy['Stage_Age_Interaction'] = df_copy['Stage'] * df_copy['Age_years']
    
    # Polynomial features
    df_copy['Bilirubin_sq'] = df_copy['Bilirubin']**2
    df_copy['Albumin_sq'] = df_copy['Albumin']**2
    df_copy['Prothrombin_sq'] = df_copy['Prothrombin']**2

    return df_copy

train_df_featured = create_features(train_df)
test_df_featured = create_features(test_df)

# Separate target and features
y = train_df_featured['Status']
X = train_df_featured.drop(['id', 'Status'], axis=1) 
test_X = test_df_featured.drop('id', axis=1)

print(f"Number of features after engineering: {len(X.columns)}")


# Define feature types
categorical_features = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders']
ordinal_features = ['Edema', 'Stage'] 
numerical_features = [col for col in X.columns if col not in categorical_features and col not in ordinal_features]

# Process ordinal features
edema_map = {'N': 0, 'S': 1, 'Y': 2}
X['Edema'] = X['Edema'].map(edema_map)
test_X['Edema'] = test_X['Edema'].map(edema_map)

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features + ordinal_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test_X)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Data preprocessing complete")
print(f"Training data shape: {X_processed.shape}")
print(f"Test data shape: {test_processed.shape}")
print(f"Target classes: {label_encoder.classes_}")


# Model parameters
params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'booster': 'gbtree',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 4,
    'seed': 42,
    'n_jobs': -1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'early_stopping_rounds': 100
}

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), 3))
test_preds = np.zeros((len(test_df), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_encoded)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X_processed[train_idx], y_encoded[train_idx]
    X_val, y_val = X_processed[val_idx], y_encoded[val_idx]

    model = xgb.XGBClassifier(**params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    val_preds = model.predict_proba(X_val)
    oof_preds[val_idx] = val_preds
    
    fold_test_preds = model.predict_proba(test_processed)
    test_preds += fold_test_preds / N_SPLITS

oof_logloss = log_loss(y_encoded, oof_preds)
print(f"Out-of-Fold LogLoss: {oof_logloss:.5f}")


# Create submission DataFrame
submission_df = pd.DataFrame(test_preds, columns=[f'Status_{cls}' for cls in label_encoder.classes_])
submission_df['id'] = original_test_ids
submission_df = submission_df[['id'] + [col for col in submission_df.columns if col != 'id']]

# Save to file
submission_df.to_csv('/kaggle/working/submission_xgboost.csv', index=False)
print("Submission file created")

