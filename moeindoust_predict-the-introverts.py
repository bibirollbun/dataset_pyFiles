# --- Install Libraries ---
!pip install -q lightgbm catboost tqdm

# --- Core & Visualization ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Preprocessing & Evaluation ---
from sklearn.experimental import enable_iterative_imputer # Enable the experimental imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor # A fast model for the imputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Modeling ---
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# --- Utilities ---
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- Reproducibility ---
tf.random.set_seed(42)
np.random.seed(42)

print("All libraries and dependencies are ready.")


# --- Load Data ---
train_df_raw = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df_raw = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Create copies
train_df = train_df_raw.copy()
test_df = test_df_raw.copy()

# --- Initial Transformation for Imputer ---
# IterativeImputer needs numeric inputs. We map Yes/No to 1/0, leaving NaNs as they are.
for col in ['Stage_fear', 'Drained_after_socializing']:
    train_df[col] = train_df[col].map({'Yes': 1.0, 'No': 0.0})
    test_df[col] = test_df[col].map({'Yes': 1.0, 'No': 0.0})

# Encode the target variable as well
train_df['Personality'] = train_df['Personality'].map({'Introvert': 1, 'Extrovert': 0})

print("Data loaded and initial text-to-numeric conversion complete.")


# --- Advanced Imputation ---
features = [col for col in train_df.columns if col not in ['id', 'Personality']]
X = train_df[features]
y = train_df['Personality']
X_test = test_df[features]

# Instantiate the imputer with a fast and powerful estimator
imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
    max_iter=10, # Number of rounds to go over the data
    random_state=42,
    verbose=2 # Show progress
)

print("Starting model-based imputation...")
# Fit on training data and transform both train and test data
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)

# Convert back to pandas DataFrames
X = pd.DataFrame(X_imputed, columns=features)
X_test = pd.DataFrame(X_test_imputed, columns=features)

print("\nMissing values imputed successfully using IterativeImputer.")
# Verification
assert X.isnull().sum().sum() == 0, "Imputation failed for training data!"
assert X_test.isnull().sum().sum() == 0, "Imputation failed for test data!"
print("Verification successful: No missing values remain.")


# --- Scale Features for NN ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
print("Features scaled successfully for the Neural Network.")


# --- Train GBM Ensemble ---
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_gbm_blend = np.zeros(len(X)); test_gbm_blend = np.zeros(len(X_test))
xgb_params = {'lambda': 3.75e-05, 'alpha': 6.08e-05, 'colsample_bytree': 0.2, 'subsample': 0.73, 'learning_rate': 0.024, 'max_depth': 4}
lgb_params = {'objective': 'binary', 'n_estimators': 2000, 'learning_rate': 0.02, 'verbose': -1, 'n_jobs': -1, 'seed': 42}
cat_params = {'iterations': 2000, 'learning_rate': 0.02, 'depth': 4, 'loss_function': 'Logloss', 'random_seed': 42, 'verbose': 0, 'early_stopping_rounds': 100}

print("Generating predictions from Gradient Boosting Ensemble...")
for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, y)), total=N_SPLITS, desc="Training GBMs"):
    X_train, y_train, X_val, y_val = X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx]
    
    xgb_m = xgb.XGBClassifier(**xgb_params, n_estimators=2000, use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    lgb_m = lgb.LGBMClassifier(**lgb_params).fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    cat_m = cat.CatBoostClassifier(**cat_params).fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    oof_gbm_blend[val_idx] = (xgb_m.predict_proba(X_val)[:, 1] + lgb_m.predict_proba(X_val)[:, 1] + cat_m.predict_proba(X_val)[:, 1]) / 3
    test_gbm_blend += ((xgb_m.predict_proba(X_test)[:, 1] + lgb_m.predict_proba(X_test)[:, 1] + cat_m.predict_proba(X_test)[:, 1]) / 3) / N_SPLITS

acc_gbm_blend = accuracy_score(y, (oof_gbm_blend > 0.5).astype(int))
print(f"\nGBM Ensemble CV Accuracy: {acc_gbm_blend:.5f}")


# --- Train Neural Network ---
def build_model(input_shape):
    model=keras.Sequential([layers.Input(shape=[input_shape]), layers.Dense(64,activation='relu'), layers.Dropout(0.3), layers.Dense(32,activation='relu'), layers.Dropout(0.2), layers.Dense(1,activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']); return model

oof_nn = np.zeros(len(X)); test_nn = np.zeros(len(X_test))
callbacks_list = [callbacks.EarlyStopping(patience=15, restore_best_weights=True), callbacks.ReduceLROnPlateau(patience=5, factor=0.5)]

print("\nGenerating predictions from Neural Network...")
for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X_scaled, y)), total=N_SPLITS, desc="Training NN"):
    X_train_s, y_train, X_val_s, y_val = X_scaled.iloc[train_idx], y.iloc[train_idx], X_scaled.iloc[val_idx], y.iloc[val_idx]
    
    model = build_model(len(features))
    model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=200, batch_size=64, callbacks=callbacks_list, verbose=0)
    
    oof_nn[val_idx] = model.predict(X_val_s).flatten()
    test_nn += model.predict(X_test_scaled).flatten() / N_SPLITS
    
acc_nn = accuracy_score(y, (oof_nn > 0.5).astype(int))
print(f"\nNeural Network CV Accuracy: {acc_nn:.5f}")


# --- Final Blend Evaluation ---
blended_oof = 0.5 * oof_gbm_blend + 0.5 * oof_nn
acc_final_blend = accuracy_score(y, (blended_oof > 0.5).astype(int))

print("\n--- Final Model Evaluation ---")
print(f"GBM Ensemble CV Score (with Iterative Imputation): {acc_gbm_blend:.5f}")
print(f"Neural Network CV Score (with Iterative Imputation): {acc_nn:.5f}")
print(f"Final Blended Model CV Score: {acc_final_blend:.5f}")

# --- Create Final Submission File ---
final_blend_preds = 0.5 * test_gbm_blend + 0.5 * test_nn
final_predictions = (final_blend_preds > 0.5).astype(int)

# The fix is here: using test_df_raw['id'] instead of train_df_raw['id']
submission_df = pd.DataFrame({'id': test_df_raw['id']})
submission_df['Personality'] = np.where(final_predictions == 1, 'Introvert', 'Extrovert')

submission_df.to_csv('submission_iterative_imputer.csv', index=False)

print("\nFinal submission file 'submission_iterative_imputer.csv' created successfully!")
display(submission_df.head())


# --- We already have the predictions from the previous cells: ---
# oof_gbm_blend, test_gbm_blend
# oof_nn, test_nn
# y, test_df_raw

print("--- Strategy 1: Creating submission from the Top Performer (GBM Ensemble) ---")
# Use the predictions from the best single model component
best_single_model_preds = (test_gbm_blend > 0.5).astype(int)
submission_df_best_single = pd.DataFrame({'id': test_df_raw['id']})
submission_df_best_single['Personality'] = np.where(best_single_model_preds == 1, 'Introvert', 'Extrovert')
submission_df_best_single.to_csv('submission_best_gbm.csv', index=False)
print("File 'submission_best_gbm.csv' created successfully.")


print("\n--- Strategy 2: Creating submission from the Hyper-Optimized Blend ---")
# --- Find Optimal Weights and Threshold on OOF predictions ---
best_acc = 0
best_weights = {}
best_threshold = 0

for w_gbm in np.arange(0, 1.01, 0.01):
    w_nn = 1 - w_gbm
    blended_oof = w_gbm * oof_gbm_blend + w_nn * oof_nn
    for threshold in np.arange(0.45, 0.55, 0.001):
        preds = (blended_oof > threshold).astype(int)
        acc = accuracy_score(y, preds)
        if acc > best_acc:
            best_acc = acc
            best_weights = {'gbm': w_gbm, 'nn': w_nn}
            best_threshold = threshold

print(f"\nOptimization Found:")
print(f"  Best Possible CV Score with Blending: {best_acc:.6f}")
print(f"  Optimal Weights: GBM={best_weights['gbm']:.2f}, NN={best_weights['nn']:.2f}")
print(f"  Optimal Threshold: {best_threshold:.3f}")

# --- Apply optimal parameters to create the final submission ---
final_blend_preds = best_weights['gbm'] * test_gbm_blend + best_weights['nn'] * test_nn
final_predictions_optimized = (final_blend_preds > best_threshold).astype(int)

submission_df_optimized_blend = pd.DataFrame({'id': test_df_raw['id']})
submission_df_optimized_blend['Personality'] = np.where(final_predictions_optimized == 1, 'Introvert', 'Extrovert')
submission_df_optimized_blend.to_csv('submission.csv', index=False)
print("\nFile 'submission_optimized_blend.csv' created successfully.")

print("\n--- RECOMMENDATION ---")
print("1. Submit 'submission_best_gbm.csv' (Your highest CV score model).")
print("2. Submit 'submission_optimized_blend.csv' (Your hyper-optimized blend).")




