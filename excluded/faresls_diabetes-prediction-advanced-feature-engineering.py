import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import shutil
import os

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier, Pool


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration
CONFIG = {
    'SEED': 42,
    'N_FOLDS': 10,
    'TARGET': 'diagnosed_diabetes'
}


# 1. Load Data (Correct Path)
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# 2. First Look
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# Display the first 5 rows to see what we are dealing with
display(train.head())

# Check data types and missing values
print("\n--- Info ---")
train.info()


# a. Define Target
TARGET = 'diagnosed_diabetes'

# b. Auto-Detect Feature Types
# We exclude the Target and ID from features
features = [c for c in train.columns if c not in [TARGET, 'id']]

# Get Numerical and Categorical columns automatically
num_cols = train[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = train[features].select_dtypes(include=['object']).columns.tolist()

print(f"Detected {len(num_cols)} Numerical Columns.")
print(f"Detected {len(cat_cols)} Categorical Columns: {cat_cols}")




# c. Inspect Categorical Features (Visualizing to find issues)
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    if i < len(axes):
        
        # Plot distribution
        sns.countplot(data=train, x=col, ax=axes[i], palette='viridis')
        axes[i].set_title(f"{col} (Unique: {train[col].nunique()})")
        axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# 1. Plot Distributions

fig, axes = plt.subplots(6, 3, figsize=(15, 20))
axes = axes.flatten() # Turn the grid into a list for easy looping

for i, col in enumerate(num_cols):
    sns.histplot(train[col], bins=30, kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(col)
    axes[i].set_xlabel('')

plt.tight_layout()
plt.show()



# 2. Correlation Heatmap

plt.figure(figsize=(10, 8))
corr_matrix = train[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation: Do we need Interaction Features?")
plt.show()


# 1. Define Column Groups 
# We treat the binary history columns as categorical strings for CatBoost
BINARY_COLS = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
CAT_COLS = [
    'gender', 'ethnicity', 'education_level', 
    'employment_status', 'smoking_status', 'income_level'
] + BINARY_COLS

def engineer_features(df):
    df = df.copy()
    
    # a. Fix Skewness (Log Transform) 
    # We add 1 because log(0) is infinite
    skewed_cols = ['physical_activity_minutes_per_week', 'alcohol_consumption_per_week']
    for col in skewed_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            
    # b. Medical Interactions (Based on Heatmap) 
    # 1. Blood Pressure: Pulse Pressure (Systolic - Diastolic)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # 2. Cholesterol: Non-HDL (Total - HDL) is a strong risk factor
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # 3. Lipid Ratio: LDL / HDL
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
    
    # 4. Obesity Interaction: BMI * Waist/Hip
    df['bmi_waist_interaction'] = df['bmi'] * df['waist_to_hip_ratio']
    
    # --- C. Convert Categoricals to String ---
    # Crucial for CatBoost and Target Encoding later
    for col in CAT_COLS:
        df[col] = df[col].astype(str)
        
    return df

# Apply Base Engineering
train_fe = engineer_features(train)
test_fe = engineer_features(test)

# --- D. The "Grandmaster" Binning Interaction ---
# We combine Train/Test to ensure the bins are calculated globally
def create_binning_features(df_train, df_test):
    # Combine
    n_train = len(df_train)
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    # Discretizer (Quantile strategy ensures equal-sized bins)
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    
    # Binning Age and BMI
    # We create temporary columns to hold the bin numbers
    df_all['age_bin'] = est.fit_transform(df_all[['age']]).astype(int).astype(str)
    df_all['bmi_bin'] = est.fit_transform(df_all[['bmi']]).astype(int).astype(str)
    
    # CROSS FEATURE: "Age_Group" + "BMI_Group" -> e.g., "5_9"
    # This helps trees see: "Middle Aged (5) + Very Obese (9)" = High Risk
    df_all['age_bmi_inter'] = df_all['age_bin'] + '_' + df_all['bmi_bin']
    
    # Split back
    return df_all.iloc[:n_train], df_all.iloc[n_train:]

# Apply Binning
train_fe, test_fe = create_binning_features(train_fe, test_fe)

# Update our list of categorical columns to include the new ones
ALL_CAT_COLS = CAT_COLS + ['age_bin', 'bmi_bin', 'age_bmi_inter']

print(f"Feature Engineering Complete.")
print(f"New Categorical Features: {['age_bin', 'bmi_bin', 'age_bmi_inter']}")
print(f"Train Shape: {train_fe.shape}")


def get_target_encoded_features(X_train, y_train, X_val, X_test, cat_cols, target_col='diagnosed_diabetes', smoothing=10):
    
    # 1. Get the Global Average 
    # If we don't know the category (like a new job title in test data), we use this.
    global_mean = y_train.mean()
    
    # We will store the names of the new columns here
    te_cols = []
    
    for col in cat_cols:
        # Create a new name, 
        new_col = f"{col}_TE"
        te_cols.append(new_col)
        
        # a. Calculate Stats on TRAIN Data 
        # We group by the category (e.g., Gender) and get the count and mean of diabetes
        # temp_df is just a helper to do the groupby
        temp_df = pd.DataFrame({col: X_train[col], 'target': y_train})
        stats = temp_df.groupby(col)['target'].agg(['count', 'mean'])
        
        # b. The Smoothing Formula 
        # If count is huge, 'smoothing' doesn't matter much.
        # If count is tiny, 'smoothing' drags the result to global_mean.
        score = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
        # c. Map the scores to the data 
        mapping = score.to_dict()
        
        # Apply to Train
        X_train[new_col] = X_train[col].map(mapping)
        
        # Apply to Val & Test (Fill unknown categories with global_mean)
        X_val[new_col] = X_val[col].map(mapping).fillna(global_mean)
        X_test[new_col] = X_test[col].map(mapping).fillna(global_mean)
        
    return X_train, X_val, X_test, te_cols

print("Target Encoding function defined.")


import lightgbm as lgb
from sklearn.model_selection import train_test_split

# 1. Prepare a fast verification split (80/20)
X_verify = train_fe.drop([CONFIG['TARGET'], 'id'], axis=1)
y_verify = train_fe[CONFIG['TARGET']]

X_tr_v, X_val_v, y_tr_v, y_val_v = train_test_split(X_verify, y_verify, test_size=0.2, random_state=42)
X_tr_v, X_val_v, _, te_cols_v = get_target_encoded_features(
    X_tr_v, y_tr_v, X_val_v, X_val_v.copy(), ALL_CAT_COLS, smoothing=10
)

# Combine Numerical + TE features
# We exclude raw categories (ALL_CAT_COLS) because LGBM will use the TE versions
verify_cols = [c for c in X_tr_v.columns if c not in ALL_CAT_COLS]

# 2. Train a Fast Model
print("ğŸ”� Verifying features...")
lgb_check = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)
lgb_check.fit(
    X_tr_v[verify_cols], y_tr_v, 
    eval_set=[(X_val_v[verify_cols], y_val_v)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

# 3. Plot Importance
feature_imp = pd.DataFrame({
    'Value': lgb_check.feature_importances_,
    'Feature': verify_cols
})

plt.figure(figsize=(10, 8))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title("Feature Importance (Split Gain)")
plt.tight_layout()
plt.show()

# 4. Automatic Advice
# If a feature has 0 or very low importance (< 5), it might be useless
low_impact = feature_imp[feature_imp['Value'] < 5]['Feature'].tolist()
if len(low_impact) > 0:
    print(f" Warning: These features have very low importance: {low_impact}")
else:
    print(" All features look useful!")


# 1. DÃ©finition des colonnes catÃ©gorielles de base
BINARY_COLS = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
CAT_COLS = [
    'gender', 'ethnicity', 'education_level', 
    'employment_status', 'smoking_status', 'income_level'
] + BINARY_COLS

def engineer_features(df):
    df = df.copy()
    
    # --- A. Nettoyage et Conversions ---
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # --- B. Interactions MÃ©dicales (ValidÃ©es par Feature Importance) ---
    
    # 1. Pression ArtÃ©rielle
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        # La pression diffÃ©rentielle est un indicateur clÃ© de risque cardiaque
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        # Pression artÃ©rielle moyenne
        df['map_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    
    # 2. Ratios de CholestÃ©rol
    if 'cholesterol_total' in df.columns and 'hdl_cholesterol' in df.columns:
        df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
        df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-6)
        
    if 'ldl_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
        df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
    
    # 3. Interaction ObÃ©sitÃ© (TrÃ¨s performante)
    if 'bmi' in df.columns and 'waist_to_hip_ratio' in df.columns:
        df['bmi_waist_interaction'] = df['bmi'] * df['waist_to_hip_ratio']
        
    return df

# Application de la premiÃ¨re passe
print("Applying Base Feature Engineering...")
train_fe = engineer_features(train)
test_fe = engineer_features(test)

# --- C. CrÃ©ation de l'Interaction "Grid" (Age + BMI) ---
def create_interaction_grid(df_train, df_test):
    n_train = len(df_train)
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    # On utilise KBinsDiscretizer pour crÃ©er des groupes Ã©quilibrÃ©s
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    
    if 'age' in df_all.columns and 'bmi' in df_all.columns:
        # On crÃ©e les bins temporaires
        age_bin = est.fit_transform(df_all[['age']]).astype(int).astype(str)
        bmi_bin = est.fit_transform(df_all[['bmi']]).astype(int).astype(str)
        
        # On crÃ©e L'INTERACTION (C'est celle-ci qui est importante)
        # Ex: "GroupeAge5_GroupeBMI8"
        df_all['age_bmi_inter'] = age_bin + '_' + bmi_bin
    
    return df_all.iloc[:n_train], df_all.iloc[n_train:]

print("Creating Grid Interactions...")
train_fe, test_fe = create_interaction_grid(train_fe, test_fe)

# --- D. Mise Ã  jour de la liste finale des catÃ©gories ---
# On ajoute SEULEMENT l'interaction. On ne garde pas les bins isolÃ©s.
ALL_CAT_COLS = CAT_COLS + ['age_bmi_inter']

print(f"âœ… Feature Engineering Done.")
print(f"Features CatÃ©gorielles pour le Target Encoding : {ALL_CAT_COLS}")
print(f"Train Shape: {train_fe.shape}")


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import gc

# --- CONFIG ---
CONFIG = {'SEED': 42, 'N_FOLDS': 10, 'TARGET': 'diagnosed_diabetes'}

# --- LOAD DATA ---
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# --- EMBEDDING PREP ---
# Instead of Target Encoding, we use Embeddings (Learnable Vectors)
cat_cols = ['gender', 'ethnicity', 'education_level', 'employment_status', 
            'smoking_status', 'income_level', 'family_history_diabetes', 
            'hypertension_history', 'cardiovascular_history']

# Label Encode Categoricals for Embedding Layer
cat_counts = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on both train and test to capture all categories
    all_vals = pd.concat([train[col], test[col]]).astype(str)
    le.fit(all_vals)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    cat_counts[col] = len(le.classes_)

# Normalize Numericals
num_cols = [c for c in train.columns if c not in cat_cols + [CONFIG['TARGET'], 'id']]
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

# --- MODEL ARCHITECTURE (WITH EMBEDDINGS) ---
def get_model():
    inputs = []
    embeddings = []
    
    # Create Embedding Input for each categorical feature
    for col in cat_cols:
        inp = layers.Input(shape=(1,), name=col)
        inputs.append(inp)
        # Embedding Dimension = min(50, (Categories+1)//2)
        emb_dim = min(50, (cat_counts[col] + 1) // 2)
        emb = layers.Embedding(input_dim=cat_counts[col]+1, output_dim=emb_dim)(inp)
        emb = layers.Flatten()(emb)
        embeddings.append(emb)
    
    # Numeric Input
    num_inp = layers.Input(shape=(len(num_cols),), name='numericals')
    inputs.append(num_inp)
    
    # Concatenate
    x = layers.Concatenate()(embeddings + [num_inp])
    
    # Dense Layers
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

# --- TRAINING ---
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

# Helper to format inputs for Keras (Dictionary format)
def get_inputs(df):
    inputs = {col: df[col].values for col in cat_cols}
    inputs['numericals'] = df[num_cols].values
    return inputs

print("ğŸš€ Starting Embedding NN Training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[CONFIG['TARGET']])):
    X_tr = train.iloc[train_idx]
    y_tr = train[CONFIG['TARGET']].iloc[train_idx]
    X_val = train.iloc[val_idx]
    y_val = train[CONFIG['TARGET']].iloc[val_idx]
    
    model = get_model()
    es = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)
    lr = callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', patience=5, factor=0.5)
    
    model.fit(get_inputs(X_tr), y_tr,
              validation_data=(get_inputs(X_val), y_val),
              epochs=50, batch_size=512, callbacks=[es, lr], verbose=0)
    
    val_p = model.predict(get_inputs(X_val), verbose=0).flatten()
    oof_preds[val_idx] = val_p
    test_preds += model.predict(get_inputs(test), verbose=0).flatten() / 10
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_p):.5f}")
    
    tf.keras.backend.clear_session()
    gc.collect()

print(f"\nğŸ�† Embedding NN Score: {roc_auc_score(train[CONFIG['TARGET']], oof_preds):.5f}")

# Save
sub = pd.DataFrame({'id': test['id'], 'diagnosed_diabetes': test_preds})
sub.to_csv('submission_nn_embedding.csv', index=False)



from scipy.optimize import minimize

# --- 1. MODEL CONFIGURATION ---

# LightGBM (Optuna Tuned - Shallow & Wide)
lgb_optuna_params = {
    'learning_rate': 0.088,
    'num_leaves': 282,
    'max_depth': 3,
    'min_child_samples': 92,
    'subsample': 0.87,
    'colsample_bytree': 0.67,
    'lambda_l1': 5.8e-05,
    'lambda_l2': 0.75,
    'n_estimators': 3000,
    'metric': 'auc',
    'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
    'verbose': -1, 'random_state': CONFIG['SEED']
}

# XGBoost (Standard Robust)
xgb_params = {
    'n_estimators': 2000, 
    'learning_rate': 0.03, 
    'max_depth': 5, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8,
    'eval_metric': 'auc', 
    'tree_method': 'hist', 'device': 'cuda',
    'random_state': CONFIG['SEED'], 'n_jobs': -1
}

# --- 2. PREPARATION ---
oof_lgbm = np.zeros(len(train_fe))
oof_xgb  = np.zeros(len(train_fe))
oof_nn   = np.zeros(len(train_fe))

test_lgbm = np.zeros(len(test_fe))
test_xgb  = np.zeros(len(test_fe))
test_nn   = np.zeros(len(test_fe))

# NN Features (Numerical Only)
num_cols = [c for c in train_fe.columns 
            if c not in ALL_CAT_COLS and c not in [CONFIG['TARGET'], 'id']]

# NN Architecture Helper
def get_nn_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='swish'), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(128, activation='swish'), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation='swish'),  layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['AUC'])
    return model

skf = StratifiedKFold(n_splits=CONFIG['N_FOLDS'], shuffle=True, random_state=CONFIG['SEED'])

print(f"ğŸš€ Starting Training ({CONFIG['N_FOLDS']} Folds)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_fe, train_fe[CONFIG['TARGET']])):
    print(f"\n======== FOLD {fold + 1} =========")
    
    # Split Data
    X_tr = train_fe.iloc[train_idx].copy()
    y_tr = train_fe[CONFIG['TARGET']].iloc[train_idx]
    X_val = train_fe.iloc[val_idx].copy()
    y_val = train_fe[CONFIG['TARGET']].iloc[val_idx]
    X_te = test_fe.copy()
    
    # Apply Target Encoding
    X_tr, X_val, X_te, te_cols = get_target_encoded_features(
        X_tr, y_tr, X_val, X_te, ALL_CAT_COLS, smoothing=10
    )
    
    feats_num_te = num_cols + te_cols

    # --- 1. LIGHTGBM (Optuna) ---
    lgbm = LGBMClassifier(**lgb_optuna_params)
    lgbm.fit(X_tr[feats_num_te], y_tr, eval_set=[(X_val[feats_num_te], y_val)],
             callbacks=[early_stopping(300, verbose=False)])
    
    oof_lgbm[val_idx] = lgbm.predict_proba(X_val[feats_num_te])[:, 1]
    test_lgbm += lgbm.predict_proba(X_te[feats_num_te])[:, 1] / CONFIG['N_FOLDS']
    print(f"LGBM (Optuna): {roc_auc_score(y_val, oof_lgbm[val_idx]):.5f}")
    del lgbm; gc.collect()

    # --- 2. XGBOOST ---
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr[feats_num_te], y_tr, eval_set=[(X_val[feats_num_te], y_val)], verbose=False)
    
    oof_xgb[val_idx] = xgb.predict_proba(X_val[feats_num_te])[:, 1]
    test_xgb += xgb.predict_proba(X_te[feats_num_te])[:, 1] / CONFIG['N_FOLDS']
    print(f"XGBoost:       {roc_auc_score(y_val, oof_xgb[val_idx]):.5f}")
    del xgb; gc.collect()

    # --- 3. NEURAL NETWORK ---
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr[feats_num_te].fillna(0))
    X_val_sc = scaler.transform(X_val[feats_num_te].fillna(0))
    X_te_sc = scaler.transform(X_te[feats_num_te].fillna(0))
    
    model = get_nn_model(X_tr_sc.shape[1])
    es = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)
    lr_sched = callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', patience=5, factor=0.5)
    
    model.fit(X_tr_sc, y_tr, validation_data=(X_val_sc, y_val), 
              epochs=50, batch_size=1024, callbacks=[es, lr_sched], verbose=0)
    
    oof_nn[val_idx] = model.predict(X_val_sc, verbose=0).flatten()
    test_nn += model.predict(X_te_sc, verbose=0).flatten() / CONFIG['N_FOLDS']
    print(f"Neural Net:    {roc_auc_score(y_val, oof_nn[val_idx]):.5f}")
    
    K.clear_session(); del model; gc.collect()
    del X_tr, X_val, X_te, X_tr_sc, X_val_sc, X_te_sc; gc.collect()

# --- 3. AUTO-BLENDING (Nelder-Mead) ---
print("\n===== FINDING OPTIMAL WEIGHTS (Auto-Blending) =====")

oof_preds_df = pd.DataFrame({
    'lgbm': oof_lgbm,
    'xgb': oof_xgb,
    'nn': oof_nn
})
y_true = train_fe[CONFIG['TARGET']]

# Loss Function (Negative AUC)
def auc_loss(weights):
    w = np.abs(weights)
    w = w / np.sum(w)
    final_pred = (w[0] * oof_preds_df['lgbm'] + 
                  w[1] * oof_preds_df['xgb'] + 
                  w[2] * oof_preds_df['nn'])
    return -roc_auc_score(y_true, final_pred)

# Optimize
init_weights = [0.33, 0.33, 0.33]
res = minimize(auc_loss, init_weights, method='Nelder-Mead', tol=1e-6)

# Get Best Weights
best_w = np.abs(res.x)
best_w = best_w / np.sum(best_w)

print(f"âœ… Optimal Weights Found:")
print(f"   LGBM (Optuna) : {best_w[0]:.4f}")
print(f"   XGBoost       : {best_w[1]:.4f}")
print(f"   Neural Net    : {best_w[2]:.4f}")

# Final Score
final_oof_score = -res.fun
print(f"ğŸš€ FINAL OOF SCORE (Optimized): {final_oof_score:.6f}")

# --- 4. SUBMISSION ---
final_test_pred = (best_w[0] * test_lgbm + 
                   best_w[1] * test_xgb + 
                   best_w[2] * test_nn)

df_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', usecols=['id'])
df_sub['diagnosed_diabetes'] = final_test_pred
df_sub.to_csv('submission.csv', index=False)
print("âœ… submission.csv saved successfully!")

