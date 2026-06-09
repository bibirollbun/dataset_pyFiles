import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')
SEED = 42
N_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001

np.random.seed(SEED)
tf.random.set_seed(SEED)



print("ğŸ“Š Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


def advanced_feature_engineering(df):
    df = df.copy()
    
    # Interaction features
    df['speed_x_lanes'] = df['speed_limit'] * df['num_lanes']
    df['curvature_x_speed'] = df['curvature'] * df['speed_limit']
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['risk_score'] = (df['curvature'] * df['speed_limit']) / (df['num_lanes'] + 1)
    
    # Binning features
    df['speed_category'] = pd.cut(df['speed_limit'], bins=5, labels=False)
    df['curvature_category'] = pd.cut(df['curvature'], bins=5, labels=False)
    
    # Polynomial features for key variables
    df['speed_squared'] = df['speed_limit'] ** 2
    df['curvature_squared'] = df['curvature'] ** 2
    df['accidents_squared'] = df['num_reported_accidents'] ** 2
    
    # Log transformations
    df['log_speed'] = np.log1p(df['speed_limit'])
    df['log_accidents'] = np.log1p(df['num_reported_accidents'])
    
    return df

print("\nğŸ”§ Engineering features...")
train_fe = advanced_feature_engineering(train)
test_fe = advanced_feature_engineering(test)

# Separate features and target
X = train_fe.drop(['id', 'accident_risk'], axis=1)
y = train_fe['accident_risk'].values
X_test = test_fe.drop(['id'], axis=1)

# Encode categorical variables
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = X[col].astype('category').cat.codes
    X_test[col] = X_test[col].astype('category').cat.codes

print(f"Features after engineering: {X.shape[1]}")


scaler = RobustScaler()  # More robust to outliers
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


def create_improved_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    # First block
    x = layers.Dense(256, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('selu')(x)  # SELU for self-normalization
    x = layers.Dropout(0.3)(x)
    
    # Second block with residual
    x1 = layers.Dense(192, kernel_initializer='he_normal')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('selu')(x1)
    x1 = layers.Dropout(0.25)(x1)
    
    # Third block
    x2 = layers.Dense(128, kernel_initializer='he_normal')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('selu')(x2)
    x2 = layers.Dropout(0.2)(x2)
    
    # Fourth block
    x3 = layers.Dense(64, kernel_initializer='he_normal')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('selu')(x3)
    x3 = layers.Dropout(0.15)(x3)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x3)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use Huber loss (more robust to outliers than MSE)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Better for regression
        metrics=['mae']
    )
    
    return model


print(f"\nğŸš€ Training models with {N_FOLDS}-Fold CV...")

# Storage for predictions
oof_dl = np.zeros(len(X_scaled))
oof_lgb = np.zeros(len(X_scaled))
oof_cb = np.zeros(len(X_scaled))
test_dl = np.zeros(len(X_test_scaled))
test_lgb = np.zeros(len(X_test_scaled))
test_cb = np.zeros(len(X_test_scaled))

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=0
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=0
)

cosine_decay = callbacks.LearningRateScheduler(
    lambda epoch: LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS))
)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled), 1):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}/{N_FOLDS}")
    print(f"{'='*60}")
    
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # ========== DEEP LEARNING ==========
    print("ğŸ§  Training Deep Learning model...")
    model_dl = create_improved_model(X_scaled.shape[1])
    
    history = model_dl.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    oof_dl[val_idx] = model_dl.predict(X_val_fold, verbose=0).flatten()
    test_dl += model_dl.predict(X_test_scaled, verbose=0).flatten() / N_FOLDS
    
    dl_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_dl[val_idx]))
    print(f"   DL RMSE: {dl_rmse:.6f}")
    
    # ========== LIGHTGBM ==========
    print("âš¡ Training LightGBM model...")
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': SEED,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train_fold, y_train_fold)
    val_data = lgb.Dataset(X_val_fold, y_val_fold)
    
    model_lgb = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    oof_lgb[val_idx] = model_lgb.predict(X_val_fold)
    test_lgb += model_lgb.predict(X_test) / N_FOLDS
    
    lgb_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_lgb[val_idx]))
    print(f"   LGB RMSE: {lgb_rmse:.6f}")
    
    # ========== CATBOOST ==========
    print("ğŸ�± Training CatBoost model...")
    model_cb = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=SEED,
        verbose=0
    )
    
    model_cb.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=50,
        verbose=0
    )
    
    oof_cb[val_idx] = model_cb.predict(X_val_fold)
    test_cb += model_cb.predict(X_test) / N_FOLDS
    
    cb_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_cb[val_idx]))
    print(f"   CB RMSE: {cb_rmse:.6f}")


print(f"\n{'='*60}")
print("ğŸ“Š INDIVIDUAL MODEL SCORES")
print(f"{'='*60}")

dl_score = np.sqrt(mean_squared_error(y, oof_dl))
lgb_score = np.sqrt(mean_squared_error(y, oof_lgb))
cb_score = np.sqrt(mean_squared_error(y, oof_cb))

print(f"Deep Learning OOF RMSE: {dl_score:.6f}")
print(f"LightGBM OOF RMSE:      {lgb_score:.6f}")
print(f"CatBoost OOF RMSE:      {cb_score:.6f}")

# Weighted ensemble (weights based on performance)
weights = np.array([1/dl_score, 1/lgb_score, 1/cb_score])
weights = weights / weights.sum()

print(f"\nğŸ�¯ Ensemble Weights:")
print(f"   DL:  {weights[0]:.3f}")
print(f"   LGB: {weights[1]:.3f}")
print(f"   CB:  {weights[2]:.3f}")

oof_ensemble = (oof_dl * weights[0] + 
                oof_lgb * weights[1] + 
                oof_cb * weights[2])

test_ensemble = (test_dl * weights[0] + 
                 test_lgb * weights[1] + 
                 test_cb * weights[2])

ensemble_score = np.sqrt(mean_squared_error(y, oof_ensemble))

print(f"\n{'='*60}")
print(f"ğŸ�† FINAL ENSEMBLE OOF RMSE: {ensemble_score:.6f}")
print(f"{'='*60}")


test_ensemble = np.clip(test_ensemble, 0, 1)
submission['accident_risk'] = test_ensemble
submission.to_csv('submission.csv', index=False)

print("\nâœ… Submission file created successfully!")
print(f"ğŸ“ˆ Predictions - Min: {test_ensemble.min():.6f}, Max: {test_ensemble.max():.6f}, Mean: {test_ensemble.mean():.6f}")




