# =================================================================
# ■■■ Notebook: All-Sensors CatBoost Training (Corrected) ■■■
# =================================================================
import polars as pl
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. Define Constants and Helper Functions ---
BEST_CATBOOST_PARAMS = {
    'iterations': 10000, 'learning_rate': 0.02, 'depth': 7,
    'subsample': 0.8, 'rsm': 0.7, 'l2_leaf_reg': 3.0,
    'bootstrap_type': 'Bernoulli', 'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass', 'random_seed': 42, 'thread_count': -1,
}

# This function now works on the pre-aggregated dataframe
def create_final_features(df: pl.DataFrame) -> pl.DataFrame:
    """Generates final sequence-level features from pre-aggregated sensor data."""
    
    # ★★★★★ FIX IS HERE: Select ONLY numeric sensor columns for feature generation ★★★★★
    feature_cols = [col for col in df.columns if any(prefix in col for prefix in ['acc_', 'rot_', 'thm_', 'tof_'])]
    aggs = []
    
    # 1. Whole-sequence statistical features
    for col in feature_cols:
        aggs.extend([
            pl.mean(col).alias(f'{col}_mean'),
            pl.std(col).alias(f'{col}_std'),
        ])

    # 2. Difference features
    for col in feature_cols:
        aggs.extend([
            (pl.col(col).diff().fill_null(0)).mean().alias(f'{col}_diff_mean'),
            (pl.col(col).diff().fill_null(0)).std().alias(f'{col}_diff_std'),
        ])
        
    # 3. Phase-approximated features
    for part_name, part_expr in [
        ('first_30pct', pl.col('sequence_counter') < pl.max('sequence_counter') * 0.3),
        ('middle_40pct', (pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.3) & (pl.col('sequence_counter') < pl.max('sequence_counter') * 0.7)),
        ('last_30pct', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.7),
    ]:
        for col in feature_cols:
            aggs.extend([
                (pl.when(part_expr).then(pl.col(col))).mean().alias(f'{col}_mean_{part_name}'),
                (pl.when(part_expr).then(pl.col(col))).std().alias(f'{col}_std_{part_name}'),
            ])
    
    feature_df = df.group_by('sequence_id').agg(aggs).fill_null(0)
    return feature_df

def competition_metric(y_true, y_pred, sequence_type_map, le):
    """Calculates the official competition metric."""
    # (The rest of this function is unchanged)
    y_pred = y_pred.flatten()
    y_true_str = le.inverse_transform(y_true)
    y_pred_str = le.inverse_transform(y_pred)
    true_df = pd.DataFrame({'gesture': y_true_str}); true_df['sequence_type'] = true_df['gesture'].map(sequence_type_map)
    pred_df = pd.DataFrame({'gesture': y_pred_str}); pred_df['sequence_type'] = pred_df['gesture'].map(sequence_type_map)
    binary_f1 = f1_score(true_df['sequence_type'], pred_df['sequence_type'], pos_label='Target', zero_division=0)
    true_df.loc[true_df['sequence_type'] == 'Non-Target', 'gesture'] = 'non_target'
    pred_df.loc[pred_df['sequence_type'] == 'Non-Target', 'gesture'] = 'non_target'
    all_labels = np.union1d(true_df['gesture'].unique(), pred_df['gesture'].unique())
    macro_f1 = f1_score(true_df['gesture'], pred_df['gesture'], average='macro', labels=all_labels, zero_division=0)
    return (binary_f1 + macro_f1) / 2
    
# --- 2. Data Loading and Hierarchical Feature Engineering ---
print("Loading training data...")
train_df = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')

# KEY STEP 1: Filter for 'All_sensors' data ONLY
all_sensors_ids = train_df.filter(pl.col('thm_1').is_not_null())['sequence_id'].unique()
train_df = train_df.filter(pl.col('sequence_id').is_in(all_sensors_ids))
print(f"Filtered data to {len(train_df)} rows for 'All-Sensors' model.")

# KEY STEP 2: Aggregate ToF pixels for each of the 5 sensors
print("Performing hierarchical feature engineering for ToF sensors...")
for i in range(1, 6): # Loop through ToF sensors 1 to 5
    tof_cols_for_sensor_i = [f'tof_{i}_v{j}' for j in range(64)]
    train_df = train_df.with_columns([
        pl.concat_list([pl.col(c) for c in tof_cols_for_sensor_i]).list.mean().alias(f'tof_{i}_pixels_mean'),
        pl.concat_list([pl.col(c) for c in tof_cols_for_sensor_i]).list.std().alias(f'tof_{i}_pixels_std'),
        pl.concat_list([pl.col(c) for c in tof_cols_for_sensor_i]).list.max().alias(f'tof_{i}_pixels_max'),
        pl.concat_list([pl.col(c) for c in tof_cols_for_sensor_i]).list.min().alias(f'tof_{i}_pixels_min'),
    ])

# Now, drop the original 320 ToF columns
original_tof_cols = [c for c in train_df.columns if 'tof_' in c and '_pixels_' not in c]
train_df_preprocessed = train_df.drop(original_tof_cols)

# --- 3. Final Feature Creation ---
print("Creating final sequence-level features...")
train_features = create_final_features(train_df_preprocessed)
targets = train_df_preprocessed.group_by("sequence_id").agg(
    pl.first('gesture'), pl.first('subject'), pl.first('sequence_type')
)
train_features_full = train_features.join(targets, on='sequence_id', how='left')
train_features_pd = train_features_full.to_pandas()

# --- 4. Cross-Validation and Training ---
print(f"\n--- Starting 5-Fold CV for All-Sensors CatBoost Model ---")
print(f"Number of features: {len(train_features_pd.columns) - 4}")

X = train_features_pd.drop(columns=['sequence_id', 'gesture', 'subject', 'sequence_type'])
y_str = train_features_pd['gesture']
groups = train_features_pd['subject']
le = LabelEncoder()
y = le.fit_transform(y_str)
gesture_to_type_map = train_features_pd.drop_duplicates('gesture').set_index('gesture')['sequence_type'].to_dict()

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
oof_preds_cat = np.zeros((len(X), 18)) # 18 is the number of classes

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    print(f"\n--- Fold {fold+1}/5 ---")
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    model = cb.CatBoostClassifier(**BEST_CATBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=False)
    oof_preds_cat[val_idx] = model.predict_proba(X_val)
    
    preds = model.predict(X_val).flatten()
    score = competition_metric(y_val, preds, gesture_to_type_map, le)
    val_scores.append(score)
    print(f"Fold {fold+1} Score: {score:.4f}")

mean_score = np.mean(val_scores)
print(f"\n--- Average Final All-Sensors CatBoost Score: {mean_score:.4f} ---")
np.save('oof_preds_cat_all_sensors.npy', oof_preds_cat)


# --- 5. Train and Save Final Model ---
print("\n--- Training and Saving Final All-Sensors CatBoost Model ---")
final_model_all_sensors = cb.CatBoostClassifier(**BEST_CATBOOST_PARAMS)
final_model_all_sensors.fit(X, y, verbose=False)
joblib.dump(final_model_all_sensors, 'catboost_all_sensors_model.pkl')
joblib.dump(le, 'catboost_label_encoder.pkl') # Save encoder
with open('catboost_feature_list.txt', 'w') as f:
    for feature in X.columns: f.write(f"{feature}\n")

print("\nTraining notebook finished successfully.")

