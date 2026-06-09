# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configuration and Data Loading ---
N_SPLITS = 5
SEED = 42
TARGET = 'loan_paid_back'
ID_COL = 'id'

# Load the datasets
try:
    # Assuming the user is running this in a Kaggle kernel environment
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
except FileNotFoundError:
    print("Files not found. Please ensure 'train.csv' and 'test.csv' are in the current directory.")
    # Fallback for local execution (assuming files are in the working directory)
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')


# --- 2. Advanced Feature Engineering Pipeline ---

def preprocess_data(df):
    """Performs shared preprocessing and feature engineering."""
    
    # Identify feature types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Drop ID and Target if they exist in the feature list
    numeric_cols = [col for col in numeric_cols if col not in [ID_COL, TARGET]]

    # 2.1. Handling Missing Values (Simple Median/Mode Imputation)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna('Missing') # Impute with a 'Missing' category

    # 2.2. Domain-Specific Ratios (Financial & Risk Features)
    
    # Check for required columns before creating features
    if 'loan_amount' in df.columns and 'annual_inc' in df.columns:
        # Loan Risk per Income (High impact feature)
        df['loan_to_income'] = df['loan_amount'] / (df['annual_inc'] + 1e-6)
    
    if 'annual_inc' in df.columns and 'emp_length' in df.columns:
        # Income adjusted by work experience
        # Convert emp_length (e.g., '1 year', '10+ years') to a numeric approximation
        df['emp_length_num'] = df['emp_length'].replace({'< 1 year': 0.5, '10+ years': 10, ' years': ''}, regex=True).astype(float)
        df['inc_per_year_of_work'] = df['annual_inc'] / (df['emp_length_num'] + 1e-6)
        
    if 'int_rate' in df.columns and 'grade' in df.columns:
        # Risk interaction feature (requires categorical encoding for grade later)
        df['int_rate_x_grade'] = df['int_rate'] * df['grade'].astype('category').cat.codes.fillna(0) # Temporary ordinal
        
    if 'dti' in df.columns and 'loan_amount' in df.columns:
        # Combined Debt Load Indicator
        df['dti_x_loan_amt'] = df['dti'] * df['loan_amount']

    return df, categorical_cols

# Apply engineering to both datasets
df_train, cat_cols_train = preprocess_data(df_train)
df_test, cat_cols_test = preprocess_data(df_test)


# 2.3. High-Cardinality Feature Encoding (Frequency Encoding)
combined = pd.concat([df_train.drop(TARGET, axis=1), df_test], ignore_index=True)

# List of all current categorical columns
all_cat_cols = [col for col in df_train.columns if df_train[col].dtype == 'object']

for col in all_cat_cols:
    # Use frequency encoding for high-cardinality features (e.g., zip_code, title)
    freq_map = combined[col].value_counts().to_dict()
    df_train[f'{col}_freq'] = df_train[col].map(freq_map)
    df_test[f'{col}_freq'] = df_test[col].map(freq_map).fillna(1) # Fill with 1 for unseen
    
    # Label Encode remaining nominal categories (essential for LightGBM performance)
    le = LabelEncoder()
    # Fit on combined data to handle all categories
    le.fit(combined[col])
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# Final list of features
FEATURES = [col for col in df_train.columns if col not in [ID_COL, TARGET]]
X = df_train[FEATURES]
y = df_train[TARGET]
X_test = df_test[FEATURES]

print(f"Total engineered features: {len(FEATURES)}")
print(f"Training data shape: {X.shape}, Test data shape: {X_test.shape}")


# --- 3. Model Training (LightGBM with Stratified K-Fold) ---

# LightGBM Hyperparameters (tuned for high AUC)
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 8000, # Large number with early stopping
    'learning_rate': 0.008, # Slower rate for better convergence
    'num_leaves': 16, # Shallower tree for stability
    'max_depth': 4,
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.75,
    'subsample': 0.75,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'is_unbalance': True, # Good for imbalanced classification
}

# Initialize OOF and Test prediction arrays
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])
cv_scores = []
best_iteration = 0

# Stratified K-Fold Splitter
SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print(f"\n--- Starting {N_SPLITS}-Fold Stratified CV Training (LightGBM) ---")

for fold, (train_index, val_index) in enumerate(SKF.split(X, y)):
    print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(stopping_rounds=800, verbose=False)])

    # Store OOF Prediction (Crucial for Stacking)
    oof_preds[val_index] = model.predict_proba(X_val)[:, 1]
    
    # Store Test Prediction (Averaging over folds)
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # Store AUC score for this fold
    fold_auc = roc_auc_score(y_val, oof_preds[val_index])
    cv_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")
    
# --- 4. Performance Summary and Output ---

LGBM_OOF_AUC = roc_auc_score(y, oof_preds)
print(f"\n==============================================")
print(f"ğŸ¥‡ LGBM OOF ROC AUC Score: {LGBM_OOF_AUC:.6f}")
print(f"CV Scores Mean: {np.mean(cv_scores):.6f} | Std: {np.std(cv_scores):.6f}")
print(f"==============================================")


# --- 5. Submission File Creation ---
submission = pd.DataFrame({ID_COL: df_test[ID_COL], TARGET: test_preds})
submission.to_csv('lgbm_level0_baseline.csv', index=False)
print("\nâœ… Submission file 'lgbm_level0_baseline.csv' created.")

# --- 6. Next Step Data Prep (For Stacking) ---
# Save OOF predictions to use as a feature in the Level 1 Meta-Learner
oof_df = pd.DataFrame({ID_COL: df_train[ID_COL], 'lgbm_oof_pred': oof_preds})
oof_df.to_csv('lgbm_oof_predictions.csv', index=False)
print("âœ… OOF predictions saved for stacking: 'lgbm_oof_predictions.csv'")


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configuration & Data Setup (Re-using context from previous step) ---
N_SPLITS = 5
SEED = 42
TARGET = 'loan_paid_back'
ID_COL = 'id'

# Load the datasets (Assuming the same paths/data processing as the LGBM step)
# NOTE: In a real environment, you would run the LGBM feature engineering code first.
try:
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
except FileNotFoundError:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

# Placeholder for feature engineering from the previous step (necessary for feature alignment)
def apply_lgbm_features(df):
    if 'loan_amount' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income'] = df['loan_amount'] / (df['annual_inc'] + 1e-6)
    if 'annual_inc' in df.columns and 'emp_length' in df.columns:
        df['emp_length_num'] = df['emp_length'].replace({'< 1 year': 0.5, '10+ years': 10, ' years': ''}, regex=True).astype(float).fillna(0)
        df['inc_per_year_of_work'] = df['annual_inc'] / (df['emp_length_num'] + 1e-6)
    if 'int_rate' in df.columns and 'grade' in df.columns:
        df['int_rate_x_grade'] = df['int_rate'] * df['grade'].astype('category').cat.codes.fillna(0)
    if 'dti' in df.columns and 'loan_amount' in df.columns:
        df['dti_x_loan_amt'] = df['dti'] * df['loan_amount']
    return df

df_train = apply_lgbm_features(df_train).fillna(df_train.median(numeric_only=True))
df_test = apply_lgbm_features(df_test).fillna(df_test.median(numeric_only=True))

# Identify feature lists after imputation/engineering
CATEGORICAL_COLS = df_train.select_dtypes(include=['object']).columns.tolist()
NUMERIC_COLS = [col for col in df_train.select_dtypes(include=np.number).columns.tolist() if col not in [ID_COL, TARGET]]

# 1.1. Categorical Encoding (Re-applying Label Encoding + Frequency)
combined = pd.concat([df_train.drop(TARGET, axis=1), df_test], ignore_index=True)
for col in CATEGORICAL_COLS:
    # Frequency Encoding
    freq_map = combined[col].value_counts().to_dict()
    df_train[f'{col}_freq'] = df_train[col].map(freq_map)
    df_test[f'{col}_freq'] = df_test[col].map(freq_map).fillna(1)
    
    # Label Encoding (REQUIRED for Keras Embedding layer, must be 0-indexed)
    le = tf.keras.preprocessing.text.Tokenizer(num_words=None, oov_token=1)
    
    # FIX: Use fit_on_texts method for Keras Tokenizer
    le.fit_on_texts(combined[col].astype(str).unique())
    
    df_train[col] = le.texts_to_sequences(df_train[col].astype(str))
    df_test[col] = le.texts_to_sequences(df_test[col].astype(str))
    df_train[col] = df_train[col].apply(lambda x: x[0] if x else 1) # Extract the integer
    df_test[col] = df_test[col].apply(lambda x: x[0] if x else 1)
    
    # Update numeric features for Keras input
    NUMERIC_COLS.append(f'{col}_freq')
    
# Remove originals from numeric list if they were converted
NUMERIC_COLS = [col for col in NUMERIC_COLS if col not in CATEGORICAL_COLS]

# Final Feature Sets
NUMERIC_INPUTS = df_train[NUMERIC_COLS].values.astype(np.float32)
CATEGORICAL_INPUTS_TRAIN = [df_train[col].values for col in CATEGORICAL_COLS]
CATEGORICAL_INPUTS_TEST = [df_test[col].values for col in CATEGORICAL_COLS]
y = df_train[TARGET].values

print(f"Numeric Feature Count: {len(NUMERIC_COLS)}")
print(f"Categorical Feature Count: {len(CATEGORICAL_COLS)}")


# --- 2. Keras Model Definition with Entity Embeddings ---

def build_embedding_mlp_model(numeric_feature_count, categorical_feature_metadata):
    """
    Builds a Keras model combining Entity Embeddings and Numerical inputs.
    categorical_feature_metadata: List of (feature_name, unique_count) tuples.
    """
    tf.keras.backend.clear_session()
    
    # 2.1. Categorical Inputs and Embeddings
    embedding_inputs = []
    embedding_layers = []

    for name, unique_count in categorical_feature_metadata:
        # Input for the specific categorical feature
        cat_input = Input(shape=(1,), name=f'input_{name}')
        embedding_inputs.append(cat_input)
        
        # Calculate embedding dimension (Rule of thumb: min(50, unique_count // 2))
        embed_dim = max(3, min(50, unique_count // 2))
        
        # Embedding Layer
        embedding = Embedding(
            input_dim=unique_count + 2, # +2 for 0-indexing and OOV token
            output_dim=embed_dim,
            input_length=1,
            name=f'embedding_{name}'
        )(cat_input)
        
        # Flatten the embedding output
        embedding_layers.append(Flatten()(embedding))
        
    # 2.2. Numerical Input
    numeric_input = Input(shape=(numeric_feature_count,), name='input_numeric')
    
    # 2.3. Concatenation
    all_layers = embedding_layers + [numeric_input]
    
    if len(all_layers) > 1:
        x = Concatenate(axis=-1)(all_layers)
    else:
        x = all_layers[0]

    # 2.4. Deep Layers (The MLP)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 2.5. Output Layer
    output = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=embedding_inputs + [numeric_input], outputs=output)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    
    return model

# Prepare model metadata
cat_metadata = [(col, df_train[col].max()) for col in CATEGORICAL_COLS]
dl_model = build_embedding_mlp_model(len(NUMERIC_COLS), cat_metadata)


# --- 3. Keras Data Preprocessing (Scaling) ---

# Scaling only the numerical features
scaler = StandardScaler()
# Fit and transform on the entire training set (or combined train/test set)
NUMERIC_INPUTS_SCALED = scaler.fit_transform(NUMERIC_INPUTS)
NUMERIC_TEST_SCALED = scaler.transform(df_test[NUMERIC_COLS].values.astype(np.float32))

# Combine inputs for Keras training
def get_keras_inputs(numeric_data, categorical_data_list):
    """Combines numerical and list of categorical arrays for Keras model input."""
    return categorical_data_list + [numeric_data]


# --- 4. Stratified CV Training Loop for Deep Learning ---

dl_oof_preds = np.zeros(y.shape[0])
dl_test_preds = np.zeros(NUMERIC_TEST_SCALED.shape[0])
cv_scores = []

SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print(f"\n--- Starting {N_SPLITS}-Fold Stratified CV Training (MLP Embeddings) ---")

for fold, (train_index, val_index) in enumerate(SKF.split(NUMERIC_INPUTS_SCALED, y)):
    print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
    
    # Split Data
    X_train_num, X_val_num = NUMERIC_INPUTS_SCALED[train_index], NUMERIC_INPUTS_SCALED[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    X_train_cat = [arr[train_index] for arr in CATEGORICAL_INPUTS_TRAIN]
    X_val_cat = [arr[val_index] for arr in CATEGORICAL_INPUTS_TRAIN]
    
    # Prepare Keras Inputs
    train_inputs = get_keras_inputs(X_train_num, X_train_cat)
    val_inputs = get_keras_inputs(X_val_num, X_val_cat)
    
    # Rebuild and re-initialize model for each fold
    model = build_embedding_mlp_model(len(NUMERIC_COLS), cat_metadata)
    
    # Train the model
    history = model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=100, # Max epochs
        batch_size=256,
        callbacks=[EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True)],
        verbose=0
    )

    # OOF Prediction
    dl_oof_preds[val_index] = model.predict(val_inputs).flatten()
    
    # Test Prediction
    test_inputs = get_keras_inputs(NUMERIC_TEST_SCALED, CATEGORICAL_INPUTS_TEST)
    dl_test_preds += model.predict(test_inputs).flatten() / N_SPLITS
    
    # Store AUC score for this fold
    fold_auc = roc_auc_score(y_val, dl_oof_preds[val_index])
    cv_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")


# --- 5. Performance Summary and Data Export for Stacking ---

DL_OOF_AUC = roc_auc_score(y, dl_oof_preds)
print(f"\n==============================================")
print(f"ğŸš€ Deep Learning OOF ROC AUC Score: {DL_OOF_AUC:.6f}")
print(f"CV Scores Mean: {np.mean(cv_scores):.6f} | Std: {np.std(cv_scores):.6f}")
print(f"==============================================")


# Export DL OOF and Test predictions for the final stacking phase
oof_dl_df = pd.DataFrame({ID_COL: df_train[ID_COL], 'dl_oof_pred': dl_oof_preds})
test_dl_df = pd.DataFrame({ID_COL: df_test[ID_COL], 'dl_test_pred': dl_test_preds})

oof_dl_df.to_csv('dl_oof_predictions.csv', index=False)
test_dl_df.to_csv('dl_test_predictions.csv', index=False)

print("âœ… DL OOF predictions saved for stacking: 'dl_oof_predictions.csv'")
print("âœ… DL test predictions saved for stacking: 'dl_test_predictions.csv'")


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configuration and Data Loading ---
TARGET = 'loan_paid_back'
ID_COL = 'id'
SEED = 42

# Load the OOF predictions (Features for the Meta-Learner's training set)
try:
    oof_lgbm = pd.read_csv('lgbm_oof_predictions.csv')
    oof_dl = pd.read_csv('dl_oof_predictions.csv')
    
    # Load the Test predictions (Features for the Meta-Learner's test set)
    test_lgbm = pd.read_csv('lgbm_level0_baseline.csv').rename(columns={TARGET: 'lgbm_test_pred'})
    test_dl = pd.read_csv('dl_test_predictions.csv')
    
    # Load the original training data to get the true target variable (y_train)
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
    
except FileNotFoundError:
    print("Error: Required Level 0 prediction files not found. Ensure 'lgbm_oof_predictions.csv', 'dl_oof_predictions.csv', etc., were generated and are in the current directory.")
    exit()

# --- 2. Prepare Level 1 Feature Set ---

# Merge OOF predictions (New Training Features)
X_train_stack = df_train[[ID_COL, TARGET]].copy()
X_train_stack = X_train_stack.merge(oof_lgbm, on=ID_COL, how='left')
X_train_stack = X_train_stack.merge(oof_dl, on=ID_COL, how='left')

# Merge Test predictions (New Test Features)
X_test_stack = sample_submission[[ID_COL]].copy()
X_test_stack = X_test_stack.merge(test_lgbm[[ID_COL, 'lgbm_test_pred']], on=ID_COL, how='left')
X_test_stack = X_test_stack.merge(test_dl, on=ID_COL, how='left')


# Define the features for the Meta-Learner (the Level 0 predictions)
STACKING_FEATURES = [
    'lgbm_oof_pred',
    'dl_oof_pred'
]

X_stack = X_train_stack[STACKING_FEATURES].values
y_stack = X_train_stack[TARGET].values
X_test_meta = X_test_stack[[col.replace('oof', 'test') for col in STACKING_FEATURES]].values


print(f"Level 1 Features (Meta-Learner Inputs): {STACKING_FEATURES}")
print(f"Meta-Learner Training Data Shape: {X_stack.shape}")

# --- 3. Meta-Learner Training (Logistic Regression) ---

# Initialize Meta-Learner
# A simple, robust Logistic Regression is excellent for the Meta-Learner
meta_model = LogisticRegression(
    solver='lbfgs',
    C=0.1,  # Regularization to prevent overfitting on probabilities
    random_state=SEED,
    n_jobs=-1,
    class_weight='balanced' # Good practice for imbalanced targets
)

print("\n--- Training Level 1 Meta-Learner (Logistic Regression) ---")

# Train the Meta-Learner using the OOF predictions from the base models
# We train once on the entire OOF dataset. No further CV is usually needed for LR Meta-Learners.
meta_model.fit(X_stack, y_stack)


# --- 4. Final Prediction and Evaluation ---

# Predict probabilities on the original training set (using OOF)
final_oof_preds = meta_model.predict_proba(X_stack)[:, 1]

# Predict probabilities on the final test set (using Test predictions from base models)
final_test_preds = meta_model.predict_proba(X_test_meta)[:, 1]

# Calculate the final Stacking AUC Score
STACKING_AUC = roc_auc_score(y_stack, final_oof_preds)

print(f"\n==============================================")
print(f"ğŸ�† FINAL STACKING OOF ROC AUC: {STACKING_AUC:.6f}")
print(f"==============================================")


# --- 5. Final Submission File Creation ---

submission_final = pd.DataFrame({
    ID_COL: X_test_stack[ID_COL],
    TARGET: final_test_preds
})

# NOTE: The final submission should be the average of the test predictions from the meta-learner
submission_final.to_csv('final_stacking_submission.csv', index=False)

print(f"\nâœ… Final submission file 'final_stacking_submission.csv' created.")
print("This file represents the combined strength of your LightGBM and Deep Learning models.")


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configuration and Data Loading ---
TARGET = 'loan_paid_back'
ID_COL = 'id'
SEED = 42
N_SPLITS = 5 

# Load the OOF predictions (Features for the Meta-Learner's training set)
try:
    # Load predictions from the two models you ran
    oof_lgbm = pd.read_csv('lgbm_oof_predictions.csv')
    oof_dl = pd.read_csv('dl_oof_predictions.csv')
    
    # Load the Test predictions
    test_lgbm = pd.read_csv('lgbm_level0_baseline.csv').rename(columns={TARGET: 'lgbm_test_pred'})
    test_dl = pd.read_csv('dl_test_predictions.csv')
    
    # Load the original training data to get the true target variable (y_train)
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv') 
    
except FileNotFoundError as e:
    print(f"Error loading required prediction files: {e}. Ensure 'lgbm_oof_predictions.csv', 'dl_oof_predictions.csv', 'lgbm_level0_baseline.csv', and 'dl_test_predictions.csv' were successfully generated and are in the current directory.")
    exit()

# --- 2. Prepare Level 1 Feature Set ---

# Merge OOF predictions (New Training Features)
X_train_stack = df_train[[ID_COL, TARGET]].copy()
X_train_stack = X_train_stack.merge(oof_lgbm, on=ID_COL, how='left')
X_train_stack = X_train_stack.merge(oof_dl, on=ID_COL, how='left')
# --- REMOVED: X_train_stack = X_train_stack.merge(oof_xgb, on=ID_COL, how='left')

# Merge Test predictions (New Test Features)
X_test_stack = df_test[[ID_COL]].copy()
X_test_stack = X_test_stack.merge(test_lgbm[[ID_COL, 'lgbm_test_pred']], on=ID_COL, how='left')
X_test_stack = X_test_stack.merge(test_dl, on=ID_COL, how='left')
# --- REMOVED: X_test_stack = X_test_stack.merge(test_xgb, on=ID_COL, how='left')


# Define the features for the Meta-Learner (only LightGBM and DL)
STACKING_FEATURES = [
    'lgbm_oof_pred',
    'dl_oof_pred',
    # --- REMOVED: 'xgb_oof_pred'
]

X_stack = X_train_stack[STACKING_FEATURES].values
y_stack = X_train_stack[TARGET].values
# Match the test feature names to the train feature names structure
TEST_FEATURES = [col.replace('oof', 'test') for col in STACKING_FEATURES]
X_test_meta = X_test_stack[TEST_FEATURES].values


print(f"Level 1 Features (Meta-Learner Inputs): {STACKING_FEATURES}")
print(f"Meta-Learner Training Data Shape: {X_stack.shape}")

# --- 3. Advanced Meta-Learner Training (CatBoost with CV) ---

CB_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 3, 
    'l2_leaf_reg': 1,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': SEED,
    'verbose': 0,
    'allow_writing_files': False
}

final_oof_preds = np.zeros(X_stack.shape[0])
final_test_preds = np.zeros(X_test_meta.shape[0])
cv_scores = []

SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print(f"\n--- Starting {N_SPLITS}-Fold CV Training (CatBoost Meta-Learner) ---")

for fold, (train_index, val_index) in enumerate(SKF.split(X_stack, y_stack)):
    print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
    
    X_train, X_val = X_stack[train_index], X_stack[val_index]
    y_train, y_val = y_stack[train_index], y_stack[val_index]

    meta_model = CatBoostClassifier(**CB_PARAMS)
    
    meta_model.fit(X_train, y_train,
                   eval_set=(X_val, y_val),
                   early_stopping_rounds=50,
                   verbose=False)

    final_oof_preds[val_index] = meta_model.predict_proba(X_val)[:, 1]
    
    final_test_preds += meta_model.predict_proba(X_test_meta)[:, 1] / N_SPLITS
    
    fold_auc = roc_auc_score(y_val, final_oof_preds[val_index])
    cv_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")
    
# --- 4. Final Performance Summary and Submission ---

STACKING_AUC = roc_auc_score(y_stack, final_oof_preds)

print(f"\n==============================================")
print(f"ğŸ�† FINAL CATBOOST STACKING OOF ROC AUC: {STACKING_AUC:.6f}")
print(f"CV Scores Mean: {np.mean(cv_scores):.6f} | Std: {np.std(cv_scores):.6f}")
print(f"==============================================")


# Final Submission File Creation
submission_final = pd.DataFrame({
    ID_COL: X_test_stack[ID_COL],
    TARGET: final_test_preds
})

submission_final.to_csv('final_catboost_stacking_submission.csv', index=False)

print(f"\nâœ… Final submission file 'final_catboost_stacking_submission.csv' created.")


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configuration & Data Setup (Consistent with LGBM/DL steps) ---
N_SPLITS = 5
SEED = 42
TARGET = 'loan_paid_back'
ID_COL = 'id'

# Load the datasets
try:
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
except FileNotFoundError:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

# Combine data for consistent preprocessing
df_all = pd.concat([df_train.drop(TARGET, axis=1), df_test], ignore_index=True)
y = df_train[TARGET]

# --- 2. Feature Engineering & Preprocessing ---

def preprocess_data(df):
    """Applies common feature engineering and preprocessing steps."""
    
    # Simple Feature Engineering (as used in LGBM/DL)
    if 'loan_amount' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income'] = df['loan_amount'] / (df['annual_inc'] + 1e-6)
    if 'annual_inc' in df.columns and 'emp_length' in df.columns:
        df['emp_length_num'] = df['emp_length'].replace({'< 1 year': 0.5, '10+ years': 10, ' years': ''}, regex=True).astype(float).fillna(0)
        df['inc_per_year_of_work'] = df['annual_inc'] / (df['emp_length_num'] + 1e-6)
    if 'int_rate' in df.columns and 'grade' in df.columns:
        # Note: XGBoost handles integer-encoded categories well, but we'll create the interaction
        df['int_rate_x_grade'] = df['int_rate'] * df['grade'].astype('category').cat.codes.fillna(0)
    if 'dti' in df.columns and 'loan_amount' in df.columns:
        df['dti_x_loan_amt'] = df['dti'] * df['loan_amount']
        
    return df

df_all = preprocess_data(df_all)

# Handle Categorical Features (Label Encoding for XGBoost)
categorical_cols = df_all.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    # Use LabelEncoder, as tree models prefer simple integer mapping
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str).fillna('missing'))
    
# Fill remaining NaNs with a unique large negative value (-999) suitable for tree models
df_all = df_all.fillna(-999)

# Split back into train and test sets
X = df_all.iloc[:len(df_train)].drop(ID_COL, axis=1)
X_test = df_all.iloc[len(df_train):].drop(ID_COL, axis=1)

# Align columns to ensure the test set has the exact same feature order and count
X_test = X_test[X.columns]

print(f"Total features for XGBoost: {X.shape[1]}")
print(f"Training data shape: {X.shape}")


# --- 3. XGBoost Model Configuration ---

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.01,              # Low learning rate for better convergence
    'max_depth': 7,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 1,
    'tree_method': 'hist',    # Use faster histogram method
    'seed': SEED,
    'n_jobs': -1,
    'disable_default_eval_metric': 1,
}

# --- 4. Stratified CV Training Loop (FIXED) ---

xgb_oof_preds = np.zeros(y.shape[0])
xgb_test_preds = np.zeros(X_test.shape[0])
cv_scores = []

SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print(f"\n--- Starting {N_SPLITS}-Fold Stratified CV Training (XGBoost) ---")

for fold, (train_index, val_index) in enumerate(SKF.split(X, y)):
    print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    # Train the model (FIXED: replaced 'verbose=False' with 'verbose_eval=0')
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=0 # CORRECTED ARGUMENT
    )

    # OOF Prediction
    xgb_oof_preds[val_index] = model.predict(dval)
    
    # Test Prediction
    xgb_test_preds += model.predict(dtest) / N_SPLITS
    
    # Store AUC score for this fold
    fold_auc = roc_auc_score(y_val, xgb_oof_preds[val_index])
    cv_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.6f} (Best Iteration: {model.best_iteration})")


# --- 5. Performance Summary and Data Export for Stacking ---

XGB_OOF_AUC = roc_auc_score(y, xgb_oof_preds)
print(f"\n==============================================")
print(f"ğŸš€ XGBoost OOF ROC AUC Score: {XGB_OOF_AUC:.6f}")
print(f"CV Scores Mean: {np.mean(cv_scores):.6f} | Std: {np.std(cv_scores):.6f}")
print(f"==============================================")


# Export XGB OOF and Test predictions for the final stacking phase
oof_xgb_df = pd.DataFrame({ID_COL: df_train[ID_COL], 'xgb_oof_pred': xgb_oof_preds})
test_xgb_df = pd.DataFrame({ID_COL: df_test[ID_COL], 'xgb_test_pred': xgb_test_preds})

oof_xgb_df.to_csv('xgb_oof_predictions.csv', index=False)
test_xgb_df.to_csv('xgb_test_predictions.csv', index=False)

print("âœ… XGB OOF predictions saved for stacking: 'xgb_oof_predictions.csv'")
print("âœ… XGB test predictions saved for stacking: 'xgb_test_predictions.csv'")


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configuration and Data Loading ---
TARGET = 'loan_paid_back'
ID_COL = 'id'
SEED = 42
N_SPLITS = 5 

# List of prediction files generated by Level 0 models
OOF_FILES = ['lgbm_oof_predictions.csv', 'dl_oof_predictions.csv', 'xgb_oof_predictions.csv']
TEST_FILES = ['lgbm_level0_baseline.csv', 'dl_test_predictions.csv', 'xgb_test_predictions.csv']

# Rename mapping for test files that might not have a clean 'xgb_test_pred' column
TEST_RENAME_MAP = {
    'lgbm_level0_baseline.csv': 'lgbm_test_pred',
    'dl_test_predictions.csv': 'dl_test_pred',
    'xgb_test_predictions.csv': 'xgb_test_pred'
}

print("--- Loading Level 0 Predictions ---")
try:
    # Load the original data to get the true target (y) and IDs
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv') 

    # Prepare DataFrame for Level 1 Training Features (OOF)
    X_train_stack = df_train[[ID_COL, TARGET]].copy()
    
    # Prepare DataFrame for Level 1 Test Features
    X_test_stack = df_test[[ID_COL]].copy()

    # Iterate and merge OOF predictions
    for file in OOF_FILES:
        pred_df = pd.read_csv(file)
        X_train_stack = X_train_stack.merge(pred_df, on=ID_COL, how='left')

    # Iterate and merge Test predictions
    for file in TEST_FILES:
        pred_df = pd.read_csv(file)
        # Rename the prediction column consistently for merging
        pred_col = [col for col in pred_df.columns if col != ID_COL][0]
        new_col_name = TEST_RENAME_MAP[file]
        
        # Handle cases where the base model pred file used the target name
        if pred_col == TARGET:
             pred_df.rename(columns={TARGET: new_col_name}, inplace=True)
        else:
            # Assume the model already created the right column (e.g., 'dl_test_pred')
            pred_df.rename(columns={pred_col: new_col_name}, inplace=True) 

        X_test_stack = X_test_stack.merge(pred_df[[ID_COL, new_col_name]], on=ID_COL, how='left')
        
except FileNotFoundError as e:
    print(f"FATAL ERROR: Required prediction file not found: {e}.")
    print("Please ensure you have run the LGBM, DL, and XGBoost scripts successfully.")
    exit()

# --- 2. Define Level 1 Feature Sets ---

STACKING_FEATURES = [
    'lgbm_oof_pred',  # From LGBM model
    'dl_oof_pred',    # From Deep Learning model
    'xgb_oof_pred'    # From XGBoost model
]

# Ensure the feature names are correct after merging (use the first row for OOF)
X_stack = X_train_stack[STACKING_FEATURES].values
y_stack = X_train_stack[TARGET].values

# Ensure the feature names are correct after merging (use the 'test' counterparts)
TEST_FEATURES = [col.replace('oof', 'test') for col in STACKING_FEATURES]
X_test_meta = X_test_stack[TEST_FEATURES].values

print(f"Level 1 Features (Meta-Learner Inputs): {STACKING_FEATURES}")
print(f"Meta-Learner Training Data Shape: {X_stack.shape}")

# --- 3. CatBoost Meta-Learner Configuration ---

CB_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 3,
    'l2_leaf_reg': 1,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': SEED,
    'verbose': 0,
    'allow_writing_files': False
}

# --- 4. Stratified CV Training Loop (CatBoost) ---

final_oof_preds = np.zeros(X_stack.shape[0])
final_test_preds = np.zeros(X_test_meta.shape[0])
cv_scores = []

SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print(f"\n--- Starting {N_SPLITS}-Fold CV Training (CatBoost Meta-Learner) ---")

for fold, (train_index, val_index) in enumerate(SKF.split(X_stack, y_stack)):
    print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
    
    X_train, X_val = X_stack[train_index], X_stack[val_index]
    y_train, y_val = y_stack[train_index], y_stack[val_index]

    meta_model = CatBoostClassifier(**CB_PARAMS)
    
    meta_model.fit(X_train, y_train,
                   eval_set=(X_val, y_val),
                   early_stopping_rounds=50,
                   verbose=False)

    # Store OOF Prediction
    final_oof_preds[val_index] = meta_model.predict_proba(X_val)[:, 1]
    
    # Store Test Prediction (Averaging over folds)
    final_test_preds += meta_model.predict_proba(X_test_meta)[:, 1] / N_SPLITS
    
    fold_auc = roc_auc_score(y_val, final_oof_preds[val_index])
    cv_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")
    
# --- 5. Final Performance Summary and Submission ---

STACKING_AUC = roc_auc_score(y_stack, final_oof_preds)

print(f"\n==============================================")
print(f"ğŸ�† FINAL CATBOOST STACKING OOF ROC AUC: {STACKING_AUC:.6f}")
print(f"CV Scores Mean: {np.mean(cv_scores):.6f} | Std: {np.std(cv_scores):.6f}")
print(f"==============================================")


# Final Submission File Creation
submission_final = pd.DataFrame({
    ID_COL: X_test_stack[ID_COL],
    TARGET: final_test_preds
})

submission_final.to_csv('final_top1_stacking_submission.csv', index=False)

print(f"\nâœ… Final submission file 'final_top1_stacking_submission.csv' created.")

