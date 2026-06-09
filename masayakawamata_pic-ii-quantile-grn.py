import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import gc
import os

# --- 1. Global Configuration ---
class CFG:
    SEEDS = [42, 7, 2025] # Use 3 different seeds for averaging
    EPOCHS = 300 # Set a high number of epochs because of EarlyStopping
    BATCH_SIZE = 512
    N_SPLITS = 5
    N_INNER_SPLITS = 5 # For nested Target Encoding
    
    # --- Base Hyperparameters ---
    BASE_LR = 1e-3
    BASE_WEIGHT_DECAY = 1e-4
    
    # --- Final Architectures ---
    BEST_ARCH_LOWER = [256, 128, 128, 64, 32] # narrower_5L
    BEST_ARCH_UPPER = [1024, 512, 256, 128, 64, 32, 16] # baseline_7L

# --- 2. Utility Functions & Data Loading ---
# Utility Functions
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def set_seed(seed=CFG.SEEDS[0]):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def winkler_score(y_true, lower, upper, alpha=0.1):
    score = upper - lower
    score += np.where(y_true < lower, (2 / alpha) * (lower - y_true), 0)
    score += np.where(y_true > upper, (2 / alpha) * (y_true - upper), 0)
    return np.mean(score)

def quantile_loss(quantile):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error), axis=-1)
    return loss

# Load Data
print("--- Preparing data... ---")
try:
    # This block will try to load the competition data
    train_df_orig = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
    test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
except FileNotFoundError:
    # This block will run if the competition data is not found, creating dummy data
    print("Warning: Kaggle dataset not found. Using placeholder data for local execution.")
    train_df_orig = pd.DataFrame(np.random.rand(1000, 12), columns=[f'f{i}' for i in range(12)])
    train_df_orig[['zoning', 'subdivision', 'submarket']] = pd.DataFrame([['A','X','M'],['B','Y','N']] * 500, columns=['zoning', 'subdivision', 'submarket'])
    train_df_orig['id'] = range(1000)
    train_df_orig['sale_price'] = np.random.rand(1000) * 500000 + 100000
    train_df_orig['sale_date'] = pd.to_datetime(pd.date_range(start='2022-01-01', periods=1000, freq='D'))
    train_df_orig['year_built'] = np.random.randint(1980, 2020, 1000)
    train_df_orig['year_reno'] = train_df_orig['year_built'] + np.random.randint(0, 5, 1000)
    train_df_orig['sale_nbr'] = 1
    
    test_df = pd.DataFrame(np.random.rand(500, 12), columns=[f'f{i}' for i in range(12)])
    test_df[['zoning', 'subdivision', 'submarket']] = pd.DataFrame([['A','X','M'],['C','Z','P']] * 250, columns=['zoning', 'subdivision', 'submarket'])
    test_df['id'] = range(1000, 1500)
    test_df['sale_date'] = pd.to_datetime(pd.date_range(start='2023-01-01', periods=500, freq='D'))
    test_df['year_built'] = np.random.randint(1980, 2020, 500)
    test_df['year_reno'] = test_df['year_built'] + np.random.randint(0, 5, 500)
    test_df['sale_nbr'] = 1

test_ids = test_df['id']
y_true_orig = train_df_orig['sale_price'].values
y_train_log = pd.Series(np.log1p(train_df_orig['sale_price']).astype(np.float32), name='sale_price_log')

# --- 3. Gated Residual Network (GRN) Layer ---
def GatedResidualNetwork(units, dropout_rate=0.0):
    def apply(inputs):
        x = layers.Dense(units, activation='elu')(inputs)
        x = layers.Dense(units)(x)
        x = layers.Dropout(dropout_rate)(x)
        # The original paper uses a linear activation for the gating mechanism.
        # However, sigmoid is more common in modern implementations.
        g = layers.Dense(units, activation='sigmoid')(inputs)
        gated_x = x * g
        # Ensure residual connection has the same dimension
        if inputs.shape[-1] != units:
            residual = layers.Dense(units)(inputs)
        else:
            residual = inputs
        output = layers.LayerNormalization()(gated_x + residual)
        return output
    return apply

# --- 4. Model Build Function (using GRN) ---
def build_grn_model(quantile, architecture, X, numeric_cols, categorical_cols):
    inputs = []
    embeddings = []
    
    # Categorical features
    for col in categorical_cols:
        input_cat = layers.Input(shape=(1,), name=col)
        num_unique_values = train_df_orig[col].nunique() 
        embedding_size = min(50, int(num_unique_values / 2) + 1)
        # Add a small buffer to input_dim to handle rare categories in validation/test
        embedding = layers.Embedding(input_dim=num_unique_values + 5, output_dim=embedding_size, name=f'{col}_embed')(input_cat)
        embedding = layers.Flatten()(embedding)
        inputs.append(input_cat)
        embeddings.append(embedding)
    
    # Numerical features
    input_num = layers.Input(shape=(len(numeric_cols),), name='numerical_input')
    inputs.append(input_num)
    
    if embeddings:
        x = layers.concatenate(embeddings + [input_num])
    else:
        x = input_num
    
    # GRN blocks
    for units in architecture:
        x = GatedResidualNetwork(units=units, dropout_rate=0.0)(x)
        
    output = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=output)
    optimizer = keras.optimizers.AdamW(learning_rate=CFG.BASE_LR, weight_decay=CFG.BASE_WEIGHT_DECAY)
    model.compile(optimizer=optimizer, loss=quantile_loss(quantile))
    return model

def prepare_model_inputs(df, cat_cols, num_cols):
    """Prepares a dictionary of inputs for the Keras model."""
    inputs = {col: df[col].values for col in cat_cols}
    inputs['numerical_input'] = df[num_cols].values.astype(np.float32)
    return inputs

# --- 5. Main Training & Prediction Loop ---
oof_preds_lower = np.zeros(len(train_df_orig))
oof_preds_upper = np.zeros(len(train_df_orig))
test_preds_lower_all_folds = np.zeros((len(test_df), CFG.N_SPLITS))
test_preds_upper_all_folds = np.zeros((len(test_df), CFG.N_SPLITS))
fold_scores = [] # List to store scores for each fold

kf = KFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df_orig, y_train_log)):
    print("\n" + "="*80)
    print(f"ðŸš€ Processing Fold {fold+1}/{CFG.N_SPLITS}")
    print("="*80)
    
    # --- 5a. Preprocessing with Nested CV for Target Encoding ---
    train_fold_df = train_df_orig.iloc[train_idx].copy()
    val_fold_df = train_df_orig.iloc[val_idx].copy()
    y_train_fold = y_train_log.iloc[train_idx]
    
    # Use a fresh copy of the test set for each fold to avoid data leakage
    test_fold_df_copy = test_df.copy()
    
    # Basic Feature Engineering
    dfs = [train_fold_df, val_fold_df, test_fold_df_copy]
    for df in dfs:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['sale_year'] = df['sale_date'].dt.year
        df['sale_month'] = df['sale_date'].dt.month
        df['age_at_sale'] = df['sale_year'] - df['year_built']
        df['age_since_reno'] = np.where(df['year_reno'] > df['year_built'], df['sale_year'] - df['year_reno'], 0)
        df.drop(['sale_date', 'year_built', 'year_reno'], axis=1, inplace=True)
        df['sale_nbr'] = df['sale_nbr'].fillna(-1)

    # Leak-proof Target Encoding with Nested CV
    high_card_cols = ['zoning', 'subdivision']
    for col in high_card_cols:
        inner_kf = KFold(n_splits=CFG.N_INNER_SPLITS, shuffle=True, random_state=42)
        te_train_parts = []
        for inner_train_idx, inner_val_idx in inner_kf.split(train_fold_df, y_train_fold):
            target_map = y_train_fold.iloc[inner_train_idx].groupby(train_fold_df[col].iloc[inner_train_idx]).mean()
            part = train_fold_df[[col]].iloc[inner_val_idx]
            part[f'{col}_te'] = part[col].map(target_map)
            te_train_parts.append(part)
        
        te_train_combined = pd.concat(te_train_parts).sort_index()
        train_fold_df[f'{col}_te'] = te_train_combined[f'{col}_te']
        train_fold_df[f'{col}_te'].fillna(y_train_fold.mean(), inplace=True)

        global_target_map = y_train_fold.groupby(train_fold_df[col]).mean()
        val_fold_df[f'{col}_te'] = val_fold_df[col].map(global_target_map)
        val_fold_df[f'{col}_te'].fillna(y_train_fold.mean(), inplace=True)
        test_fold_df_copy[f'{col}_te'] = test_fold_df_copy[col].map(global_target_map)
        test_fold_df_copy[f'{col}_te'].fillna(y_train_fold.mean(), inplace=True)

    categorical_cols = [c for c in train_fold_df.select_dtypes(include=['object', 'category']).columns if c not in high_card_cols]
    for col in categorical_cols:
        le = LabelEncoder()
        all_cats = pd.concat([train_fold_df[col].astype(str), val_fold_df[col].astype(str), test_fold_df_copy[col].astype(str)]).unique()
        le.fit(all_cats)
        train_fold_df[col] = le.transform(train_fold_df[col].astype(str))
        val_fold_df[col] = le.transform(val_fold_df[col].astype(str))
        test_fold_df_copy[col] = le.transform(test_fold_df_copy[col].astype(str))
        
    train_fold_df.drop(columns=high_card_cols, inplace=True)
    val_fold_df.drop(columns=high_card_cols, inplace=True)
    test_fold_df_copy.drop(columns=high_card_cols, inplace=True)
    
    numeric_cols = train_fold_df.select_dtypes(include=np.number).columns.tolist()
    
    # --- FIX: Exclude ID and target columns from scaling ---
    numeric_cols = [c for c in numeric_cols if c not in ['id', 'sale_price']]
    
    scaler = StandardScaler()
    train_fold_df[numeric_cols] = scaler.fit_transform(train_fold_df[numeric_cols])
    val_fold_df[numeric_cols] = scaler.transform(val_fold_df[numeric_cols])
    test_fold_df_copy[numeric_cols] = scaler.transform(test_fold_df_copy[numeric_cols])
    
    # --- 5b. Seed Averaging within the fold ---
    val_preds_lower_seeds, val_preds_upper_seeds = [], []
    test_preds_lower_per_fold_seeds, test_preds_upper_per_fold_seeds = [], []

    for seed in CFG.SEEDS:
        print(f"--- Training with Seed {seed} ---")
        set_seed(seed + fold) # Vary seed per fold
        
        X_train_dict = prepare_model_inputs(train_fold_df, categorical_cols, numeric_cols)
        X_val_dict = prepare_model_inputs(val_fold_df, categorical_cols, numeric_cols)
        X_test_dict = prepare_model_inputs(test_fold_df_copy, categorical_cols, numeric_cols)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
        callbacks = [early_stopping, lr_scheduler]

        # Train lower bound model
        model_lower = build_grn_model(0.05, CFG.BEST_ARCH_LOWER, train_fold_df, numeric_cols, categorical_cols)
        model_lower.fit(X_train_dict, y_train_fold, validation_data=(X_val_dict, y_train_log.iloc[val_idx]), epochs=CFG.EPOCHS, batch_size=CFG.BATCH_SIZE, callbacks=callbacks, verbose=0)
        val_preds_lower_seeds.append(model_lower.predict(X_val_dict).flatten())
        test_preds_lower_per_fold_seeds.append(model_lower.predict(X_test_dict).flatten())
        
        # Train upper bound model
        model_upper = build_grn_model(0.95, CFG.BEST_ARCH_UPPER, train_fold_df, numeric_cols, categorical_cols)
        model_upper.fit(X_train_dict, y_train_fold, validation_data=(X_val_dict, y_train_log.iloc[val_idx]), epochs=CFG.EPOCHS, batch_size=CFG.BATCH_SIZE, callbacks=callbacks, verbose=0)
        val_preds_upper_seeds.append(model_upper.predict(X_val_dict).flatten())
        test_preds_upper_per_fold_seeds.append(model_upper.predict(X_test_dict).flatten())
        
        # Clean up memory
        del model_lower, model_upper
        keras.backend.clear_session()
        gc.collect()
        
    # Average predictions from different seeds for this fold
    oof_preds_lower[val_idx] = np.mean(val_preds_lower_seeds, axis=0)
    oof_preds_upper[val_idx] = np.mean(val_preds_upper_seeds, axis=0)
    
    test_preds_lower_all_folds[:, fold] = np.mean(test_preds_lower_per_fold_seeds, axis=0)
    test_preds_upper_all_folds[:, fold] = np.mean(test_preds_upper_per_fold_seeds, axis=0)
    
    # --- 5c. Calculate and display score for the current fold ---
    y_true_val = y_true_orig[val_idx]
    # Ensure lower bound is not greater than upper bound
    fold_lower_orig = np.expm1(oof_preds_lower[val_idx])
    fold_upper_orig = np.expm1(oof_preds_upper[val_idx])
    fold_upper_orig = np.maximum(fold_lower_orig, fold_upper_orig)
    
    fold_score = winkler_score(y_true_val, fold_lower_orig, fold_upper_orig)
    fold_scores.append(fold_score)
    print(f"  âœ… Fold {fold+1} Winkler Score: {fold_score:.5f}")

# --- 6. Final Prediction Generation and Evaluation ---
print("\n" + "="*80)
print("ðŸ“Š Final Results")
print("="*80)

# Display individual fold scores and their summary
print("--- Individual Fold Scores ---")
for i, score in enumerate(fold_scores):
    print(f"  - Fold {i+1}: {score:.5f}")
print("-" * 30)
print(f"  Average Fold Score: {np.mean(fold_scores):.5f}")
print(f"  Std Dev of Scores: {np.std(fold_scores):.5f}")
print("-" * 30)

# OOF Score
final_oof_lower_orig = np.expm1(oof_preds_lower)
final_oof_upper_orig = np.expm1(oof_preds_upper)
final_oof_upper_orig = np.maximum(final_oof_lower_orig, final_oof_upper_orig) # Final check
final_cv_score = winkler_score(y_true_orig, final_oof_lower_orig, final_oof_upper_orig)
print(f"  Total OOF Winkler Score (Leak-Proof): {final_cv_score:.5f}")

# Test Predictions (average across folds)
final_test_lower = np.mean(test_preds_lower_all_folds, axis=1)
final_test_upper = np.mean(test_preds_upper_all_folds, axis=1)
final_test_lower_orig = np.expm1(final_test_lower)
final_test_upper_orig = np.expm1(final_test_upper)
final_test_upper_orig = np.maximum(final_test_lower_orig, final_test_upper_orig)

# --- 7. Save Files ---
print("\n--- Saving final OOF and Test predictions ---")
oof_df = pd.DataFrame({
    'id': train_df_orig['id'],
    'pi_lower': final_oof_lower_orig,
    'pi_upper': final_oof_upper_orig
})
oof_df.to_csv('oof_nn_grn_final.csv', index=False)
print("  OOF predictions saved to 'oof_nn_grn_final.csv'")

test_df_preds = pd.DataFrame({
    'id': test_ids,
    'pi_lower': final_test_lower_orig,
    'pi_upper': final_test_upper_orig
})
test_df_preds.to_csv('test_nn_grn_final.csv', index=False)
print("  Test predictions saved to 'submission.csv'")
print("\nðŸŽ‰ All processes completed successfully!")





