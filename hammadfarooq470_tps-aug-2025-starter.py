import warnings

warnings.filterwarnings("ignore")  # suppress all warnings


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("\nAll libraries imported successfully!")


# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

print("Training set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("\nTraining columns:", train_df.columns.tolist())
print("\nTarget distribution:")
print(train_df['y'].value_counts())
print("\nTarget percentage:")
print(train_df['y'].value_counts(normalize=True))

# Display basic info
print("\n=== Training Data Info ===")
print(train_df.info())
print("\n=== Missing Values ===")
print("Train missing:", train_df.isnull().sum().sum())
print("Test missing:", test_df.isnull().sum().sum())


# Separate features and target
X_train = train_df.drop(['id', 'y'], axis=1)
y_train = train_df['y']
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id']

# Identify categorical and numerical features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)
print(f"\nTotal features: {len(categorical_features) + len(numerical_features)}")


def create_features(df, is_train=False, target_encoders=None, label_encoders=None):
    """
    Create advanced features for the dataset
    """
    df = df.copy()
    
    # Create new features
    # 1. Interaction features
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['campaign_previous_ratio'] = df['campaign'] / (df['previous'] + 1)
    
    # 2. Binning numerical features
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=['young', 'adult', 'middle', 'senior', 'elderly'])
    df['balance_group'] = pd.cut(df['balance'], bins=[-np.inf, 0, 1000, 5000, np.inf], labels=['negative', 'low', 'medium', 'high'])
    df['duration_group'] = pd.cut(df['duration'], bins=[0, 100, 300, 600, np.inf], labels=['short', 'medium', 'long', 'very_long'])
    
    # 3. Flag features
    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['has_previous_success'] = (df['poutcome'] == 'success').astype(int)
    df['has_default'] = (df['default'] == 'yes').astype(int)
    df['has_housing'] = (df['housing'] == 'yes').astype(int)
    df['has_loan'] = (df['loan'] == 'yes').astype(int)
    
    # 4. Combined categorical features
    df['marital_housing'] = df['marital'] + '_' + df['housing']
    df['job_education'] = df['job'] + '_' + df['education']
    df['contact_month'] = df['contact'] + '_' + df['month']
    
    # 5. Numerical aggregations
    df['total_interactions'] = df['campaign'] + df['previous']
    df['pdays_modified'] = df['pdays'].replace(-1, 0)
    
    # 6. Month encoding (cyclical)
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_map)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    
    # 7. Day of month (cyclical)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    return df

# Apply feature engineering
X_train_fe = create_features(X_train, is_train=True)
X_test_fe = create_features(X_test, is_train=False)

print("Original features:", X_train.shape[1])
print("Engineered features:", X_train_fe.shape[1])
print("\nNew feature columns:", [col for col in X_train_fe.columns if col not in X_train.columns])


# Update categorical and numerical features after feature engineering
categorical_features = X_train_fe.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X_train_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))

# Handle categorical features - use target encoding with cross-validation
def target_encode_cv(X_train, y_train, X_test, categorical_cols, n_splits=5):
    """
    Target encoding with cross-validation to prevent overfitting
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    # Create a temporary dataframe with target for easier manipulation
    # Convert categorical columns to string to avoid dtype issues with pd.cut()
    train_temp = X_train.copy()
    for col in categorical_cols:
        if col in train_temp.columns:
            if train_temp[col].dtype.name == 'category':
                train_temp[col] = train_temp[col].astype(str)
    train_temp['__target__'] = y_train.values
    
    # Also convert test categorical columns to string
    X_test_str = X_test.copy()
    for col in categorical_cols:
        if col in X_test_str.columns:
            if X_test_str[col].dtype.name == 'category':
                X_test_str[col] = X_test_str[col].astype(str)
    
    for col in categorical_cols:
        if col in X_train.columns:
            # Initialize columns
            X_train_encoded[f'{col}_target_enc'] = 0.0
            X_test_encoded[f'{col}_target_enc'] = 0.0
            
            # Cross-validation target encoding
            for train_idx, val_idx in kf.split(X_train):
                # Calculate mean target for each category in training fold
                train_fold = train_temp.iloc[train_idx]
                train_target_mean = train_fold.groupby(col)['__target__'].mean().to_dict()
                
                # Global mean for smoothing (to handle unseen categories)
                global_mean = train_fold['__target__'].mean()
                
                # Map to validation fold
                val_col = train_temp.iloc[val_idx][col]
                val_map = val_col.map(train_target_mean)
                val_map = val_map.fillna(global_mean)
                X_train_encoded.loc[val_idx, f'{col}_target_enc'] = val_map.values
                
                # Map to test set (average across folds)
                test_col = X_test_str[col]
                test_map = test_col.map(train_target_mean)
                test_map = test_map.fillna(global_mean)
                X_test_encoded[f'{col}_target_enc'] += test_map.values / n_splits
    
    return X_train_encoded, X_test_encoded

# Apply target encoding
print("Applying target encoding...")
X_train_encoded, X_test_encoded = target_encode_cv(
    X_train_fe, y_train, X_test_fe, 
    categorical_features, n_splits=5
)

# Drop original categorical columns (only drop columns that exist)
cols_to_drop_train = [col for col in categorical_features if col in X_train_encoded.columns]
X_train_final = X_train_encoded.drop(cols_to_drop_train, axis=1)

cols_to_drop_test = [col for col in categorical_features if col in X_test_encoded.columns]
X_test_final = X_test_encoded.drop(cols_to_drop_test, axis=1)

print(f"\nFinal feature count: {X_train_final.shape[1]}")
print(f"Training shape: {X_train_final.shape}")
print(f"Test shape: {X_test_final.shape}")


# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

# Convert to DataFrames for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_final.columns, index=X_train_final.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_final.columns, index=X_test_final.index)

print("Data scaled and ready for modeling!")
print(f"Training shape: {X_train_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")


def create_deep_model(input_dim, model_type='deep'):
    """
    Create different deep learning architectures
    """
    inputs = keras.Input(shape=(input_dim,))
    
    if model_type == 'deep':
        # Deep Neural Network with residual connections
        x = layers.Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual block 1 - keep same dimensions
        residual = x
        x = layers.Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Add()([x, residual])
        x = layers.BatchNormalization()(x)
        
        # Residual block 2 - use projection for dimension change
        residual = layers.Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)  # Project residual to 128
        x = layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Add()([x, residual])
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
    elif model_type == 'wide_deep':
        # Wide & Deep architecture
        # Wide part
        wide = layers.Dense(128, activation='relu')(inputs)
        wide = layers.Dense(64, activation='relu')(wide)
        
        # Deep part
        deep = layers.Dense(256, activation='relu')(inputs)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        deep = layers.Dense(128, activation='relu')(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        deep = layers.Dense(64, activation='relu')(deep)
        
        # Concatenate
        x = layers.Concatenate()([wide, deep])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
    elif model_type == 'attention':
        # Attention-based architecture
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Self-attention
        attention = layers.Dense(256, activation='tanh')(x)
        attention = layers.Dense(1, activation='softmax')(attention)
        x = layers.Multiply()([x, attention])
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Display model architectures
print("Model architectures created!")


# Cross-validation setup
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Store predictions
oof_predictions = np.zeros(len(X_train_scaled))
test_predictions = np.zeros(len(X_test_scaled))

# Model configurations
model_configs = [
    {'type': 'deep', 'lr': 0.001, 'batch_size': 2048, 'epochs': 50},
    {'type': 'wide_deep', 'lr': 0.001, 'batch_size': 2048, 'epochs': 50},
    {'type': 'attention', 'lr': 0.0008, 'batch_size': 2048, 'epochs': 50},
]

print(f"Starting {N_FOLDS}-fold cross-validation with {len(model_configs)} model types...")


# Train models with cross-validation
all_test_preds = []

for model_idx, config in enumerate(model_configs):
    print(f"\n{'='*60}")
    print(f"Training Model {model_idx + 1}: {config['type']}")
    print(f"{'='*60}")
    
    fold_test_preds = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        print(f"\nFold {fold + 1}/{N_FOLDS}")
        
        # Split data
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create model
        model = create_deep_model(X_tr.shape[1], model_type=config['type'])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=config['lr']),
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max',
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_tr.values, y_tr.values,
            validation_data=(X_val.values, y_val.values),
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Predictions
        val_pred = model.predict(X_val.values, verbose=0).flatten()
        test_pred = model.predict(X_test_scaled.values, verbose=0).flatten()
        
        # Calculate AUC
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
        
        # Store predictions
        oof_predictions[val_idx] += val_pred / len(model_configs)
        fold_test_preds.append(test_pred)
        
        # Clear memory
        del model
        tf.keras.backend.clear_session()
    
    # Average test predictions for this model type
    avg_test_pred = np.mean(fold_test_preds, axis=0)
    all_test_preds.append(avg_test_pred)
    
    print(f"\n{config['type']} Model - CV AUC: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")

# Final OOF score
final_oof_auc = roc_auc_score(y_train, oof_predictions)
print(f"\n{'='*60}")
print(f"Overall OOF AUC: {final_oof_auc:.6f}")
print(f"{'='*60}")


# Prepare data for tree-based models (use target encoded features + label encode remaining categoricals)
X_train_tree = X_train_final.copy()
X_test_tree = X_test_final.copy()

# Check if there are any remaining categorical features that need label encoding
remaining_categorical = X_train_tree.select_dtypes(include=['object', 'category']).columns.tolist()

if remaining_categorical:
    label_encoders = {}
    for col in remaining_categorical:
        le = LabelEncoder()
        # Handle any remaining categorical features
        X_train_tree[col] = le.fit_transform(X_train_tree[col].astype(str))
        X_test_tree[col] = le.transform(X_test_tree[col].astype(str))
        label_encoders[col] = le
    print(f"Label encoded {len(remaining_categorical)} remaining categorical features")

print("Data prepared for tree-based models")
print(f"Shape: {X_train_tree.shape}")


# LightGBM with cross-validation
lgb_oof = np.zeros(len(X_train_tree))
lgb_test_preds = []

print("Training LightGBM...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tree, y_train)):
    print(f"Fold {fold + 1}/{N_FOLDS}")
    
    X_tr, X_val = X_train_tree.iloc[train_idx], X_train_tree.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # LightGBM parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=5000,
        callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(0)]
    )
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test_tree, num_iteration=model.best_iteration)
    
    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
    
    lgb_oof[val_idx] = val_pred
    lgb_test_preds.append(test_pred)

lgb_cv_auc = roc_auc_score(y_train, lgb_oof)
print(f"\nLightGBM CV AUC: {lgb_cv_auc:.6f}")
all_test_preds.append(np.mean(lgb_test_preds, axis=0))


# XGBoost with cross-validation
xgb_oof = np.zeros(len(X_train_tree))
xgb_test_preds = []

print("\nTraining XGBoost...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tree, y_train)):
    print(f"Fold {fold + 1}/{N_FOLDS}")
    
    X_tr, X_val = X_train_tree.iloc[train_idx], X_train_tree.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test_tree)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=200,
        verbose_eval=0
    )
    
    val_pred = model.predict(dval)
    test_pred = model.predict(dtest)
    
    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
    
    xgb_oof[val_idx] = val_pred
    xgb_test_preds.append(test_pred)

xgb_cv_auc = roc_auc_score(y_train, xgb_oof)
print(f"\nXGBoost CV AUC: {xgb_cv_auc:.6f}")
all_test_preds.append(np.mean(xgb_test_preds, axis=0))


# Combine OOF predictions for ensemble
all_oof_preds = [
    oof_predictions,  # Deep learning ensemble
    lgb_oof,          # LightGBM
    xgb_oof           # XGBoost
]

# Calculate weights based on individual model performance
model_scores = [
    final_oof_auc,    # Deep learning
    lgb_cv_auc,       # LightGBM
    xgb_cv_auc        # XGBoost
]

# Softmax weights (higher score = higher weight)
weights = np.exp(np.array(model_scores))
weights = weights / weights.sum()

print("Model Weights:")
print(f"Deep Learning: {weights[0]:.4f} (AUC: {model_scores[0]:.6f})")
print(f"LightGBM: {weights[1]:.4f} (AUC: {model_scores[1]:.6f})")
print(f"XGBoost: {weights[2]:.4f} (AUC: {model_scores[2]:.6f})")

# Weighted ensemble OOF
ensemble_oof = np.zeros(len(y_train))
for i, pred in enumerate(all_oof_preds):
    ensemble_oof += weights[i] * pred

ensemble_oof_auc = roc_auc_score(y_train, ensemble_oof)
print(f"\nEnsemble OOF AUC: {ensemble_oof_auc:.6f}")

# Combine deep learning models into one prediction
# all_test_preds structure: [deep, wide_deep, attention, lgb, xgb]
# We need: [dl_ensemble, lgb, xgb] to match the 3 weights
num_dl_models = len(model_configs)  # Number of deep learning models
dl_ensemble_test = np.mean(all_test_preds[:num_dl_models], axis=0)  # Average of DL models
ensemble_test_preds = [
    dl_ensemble_test,      # Combined deep learning ensemble
    all_test_preds[-2],    # LightGBM
    all_test_preds[-1]     # XGBoost
]

# Weighted ensemble test predictions
final_test_pred = np.zeros(len(X_test_scaled))
for i, pred in enumerate(ensemble_test_preds):
    final_test_pred += weights[i] * pred

print(f"\nFinal test predictions shape: {final_test_pred.shape}")
print(f"Prediction range: [{final_test_pred.min():.4f}, {final_test_pred.max():.4f}]")


# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'y': final_test_pred
})

# Ensure predictions are in valid range [0, 1]
submission['y'] = np.clip(submission['y'], 0, 1)

# Save submission
submission.to_csv('submission.csv', index=False)

print("Submission file created!")
print(f"\nSubmission shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(submission['y'].describe())
print(f"\nSubmission saved to: submission.csv")

