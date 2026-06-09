import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import polars as pl
import kaggle_evaluation.cmi_inference_server
import joblib

import catboost
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, RobustScaler
# from sklearn.impute import KNNImputer
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import VotingClassifier


cleaned_data = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data-cleaned/cmi_bfrb_cleaned_data_full.csv')
# train = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
# train_demo = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')

# # Drop these columns from training data
# train = train.drop(['phase', 'orientation', 'behavior', 'sequence_type'])

# train = train.join(train_demo,on="subject",how="left")


def remove_gravity_from_acc(acc_data, rot_data):

    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]
             
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200): # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass
            
    return angular_vel

def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0 # Ğ˜Ğ»Ğ¸ np.nan, Ğ² Ğ·Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸ Ğ¾Ñ‚ Ğ¶ĞµĞ»Ğ°ĞµĞ¼Ğ¾Ğ³Ğ¾ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�
            continue
        try:
            # Converting quaternions to Rotation objects
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Calculating the angular distance: 2 * arccos(|real(p * q*)|)
            # where q* is the conjugate of quaternion q
            # In scipy.spatial.transform.Rotation, r1.inv() * r2 gives the relative rotation.
            # The angle of this relative rotation is the angular distance.
            relative_rotation = r1.inv() * r2
            
            # The angle of the rotation vector corresponds to the angular distance
            # The norm of the rotation vector is the angle in radians
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 # In case of invalid quaternions
            pass
            
    return angular_dist


def feature_engineering_imu(data:pl.DataFrame):
    data = data.to_pandas()
    data['acc_mag'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
    data['rot_angle'] = 2 * np.arccos(data['rot_w'].clip(-1, 1))
    data['acc_mag_jerk'] = data.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    data['rot_angle_vel'] = data.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    linear_accel_list = []
    for _, group in data.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
    
    df_linear_accel = pd.concat(linear_accel_list)
    data = pd.concat([data, df_linear_accel], axis=1)
    data['linear_acc_mag'] = np.sqrt(data['linear_acc_x']**2 + data['linear_acc_y']**2 + data['linear_acc_z']**2)
    data['linear_acc_mag_jerk'] = data.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    angular_vel_list = []
    for _, group in data.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
        angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
    
    df_angular_vel = pd.concat(angular_vel_list)
    data = pd.concat([data, df_angular_vel], axis=1)
    
    print("  Calculating angular distance between successive quaternions...")
    angular_distance_list = []
    for _, group in data.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_dist_group = calculate_angular_distance(rot_data_group)
        angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
    
    df_angular_distance = pd.concat(angular_distance_list)
    data = pd.concat([data, df_angular_distance], axis=1)
    data = pl.from_pandas(data)
    return data


# train = feature_engineering_imu(train)


# train.head(2)


def feature_engineering_stat(data:pl.DataFrame):
    non_sensor_cols = []
    if "gesture" in data.columns:
        non_sensor_cols = ["gesture"]
        
    # All numeric sensor columns (everything except id, demo, target)
    stat_cols = [
        c for c in data.columns
        if c not in non_sensor_cols + ["sequence_id", "row_id","sequence_counter","subject"]
    ]
    
    # Build aggregation expressions
    agg_exprs = []
    
    # full-stats bundle for sensor columns
    for c in stat_cols:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).first().alias(f"{c}_first"),
            pl.col(c).last().alias(f"{c}_last"),
            pl.col(c).quantile(0.25, "nearest").alias(f"{c}_t25"),
            pl.col(c).quantile(0.75, "nearest").alias(f"{c}_t75"),
            (pl.col(c).last() - pl.col(c).first()).alias(f"{c}_delta"),
            pl.corr("sequence_counter", c).alias(f"{c}_corr_time"),
            pl.col(c).diff().mean().alias(f"{c}_diff_mean"),
            pl.col(c).diff().std().alias(f"{c}_diff_std"),
            pl.col(c).skew().alias(f"{c}_skew"),
            pl.col(c).kurtosis().alias(f"{c}_kurt"),
            pl.col(c).diff().abs().gt(0).sum().alias(f"{c}_n_changes")
        ]
        agg_exprs += [
            pl.when(pl.col("sequence_counter") < 0.1 * pl.max("sequence_counter"))
              .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg1_mean"),
            pl.when(pl.col("sequence_counter") > 0.9 * pl.max("sequence_counter"))
              .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg3_mean"),
        ]
    
    # first() for demographics and target
    agg_exprs += [
        pl.col(c).first().alias(c) for c in non_sensor_cols
    ]
    
    # Group-by and aggregate
    cleaned_data = (
        data
        .group_by("sequence_id", maintain_order=True)
        .agg(agg_exprs)
    )
    return cleaned_data


# train_demographic_target_cols = [
#     "gesture"
#     ]
# cleaned_data = feature_engineering_stat(train)
# cleaned_data.shape


# cleaned_data.write_csv('cmi_bfrb_cleaned_data_full.csv')


# Assume cleaned_data is already a Polars DataFrame
target_col = "gesture"

# --- Convert Polars DataFrame to Pandas only if needed ---
# CatBoost does not yet fully support Polars directly
df = cleaned_data.to_pandas()

# --- Define X and y properly ---
X = df.drop(columns=[target_col, "sequence_id"])  # Feature matrix
y = df[target_col].values # Target
# Estimate number of classes from your target variable `y`
num_classes = len(set(y))  # or len(np.unique(y))

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

joblib.dump(le, 'Target_LabelEncoder.joblib')


#Initialize the scaler
robustscaler = RobustScaler()
# Fit on the integer columns of training data (changed to numpy array)
X_scaled = robustscaler.fit_transform(X)
#numpy array convert to panda's dataframe
X = pd.DataFrame(X_scaled, columns=X.columns)

joblib.dump(robustscaler, 'robustscaler.joblib')


params_cat = {
    'loss_function': 'MultiClass',
    'learning_rate': 0.06,
    'iterations': 3000,
    'task_type': 'GPU',
    'verbose' : False
}    

params_lgb = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_bin' : 20,
    'num_threads' : -1,
    'device': 'gpu',  # Requires LightGBM with GPU support compiled
    'verbose': -1      # Silence output; set to 1 to see progress
}

params_xgb = {
    'objective': 'multi:softprob',        # Equivalent to multiclass in LightGBM; outputs probabilities
    'num_class': num_classes,             # Required for multi-class in XGBoost
    'learning_rate': 0.08,                 # Same as LightGBM's learning_rate
    'n_estimators': 1200,                  # Same number of boosting rounds
    'max_bin': 256,                       # XGBoost default is higher; LightGBM uses 20, but XGBoost typically uses 256
    'n_jobs': -1,                         # Equivalent to num_threads in LightGBM
    'tree_method': 'hist',            # Enables GPU; requires XGBoost with GPU support
    'device': 'cuda',
    'verbosity': 0,                       # 0 = silent, 1 = info (similar to verbose=-1 in LightGBM)
    'eval_metric': 'mlogloss'             # Optional: for monitoring; multiclass logloss
}


# Initialize K-Fold
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# During cross-validation, collect best iterations
cat_best_iters = []
lgb_best_iters = []
xgb_best_iters = []

# Store OOF predictions (for training meta-model)
oof_predictions = np.zeros((len(X), num_classes * 3))  # Final OOF from base models
oof_targets = np.empty(len(y), dtype=int)  # True labels

# Meta-feature generation: Collect predictions from each model
print("Starting Stacking with Early Stopping...\n")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
    print(f"Training Fold {fold}/{n_splits}...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Convert to appropriate formats if needed
    if isinstance(X_train, pd.DataFrame):
        pass  # Already in correct format

    # --- 1. CatBoost Classifier ---
    model_cat = CatBoostClassifier(**params_cat)
    model_cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )
    pred_cat = model_cat.predict_proba(X_val)  # (n_samples, n_classes)

    # --- 2. LightGBM Classifier ---
    model_lgb = lgb.LGBMClassifier(**params_lgb)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=['valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    pred_lgb = model_lgb.predict_proba(X_val)

    # --- 3. XGBoost Classifier ---
    model_xgb = xgb.XGBClassifier(**params_xgb)
    model_xgb.set_params(early_stopping_rounds=50)  # Set before fit()
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    pred_xgb = model_xgb.predict_proba(X_val)

    # Average predictions (soft voting style) or concatenate as features
    # For stacking, we concatenate as features: shape = (n_samples, 3 * n_classes)
    stacked_features = np.hstack([pred_cat, pred_lgb, pred_xgb])  # (n, 3*C)

    # Store OOF predictions
    oof_predictions[val_idx] = stacked_features
    oof_targets[val_idx] = y_val

    # Record best iterations
    cat_best_iters.append(model_cat.get_best_iteration())
    lgb_best_iters.append(model_lgb.best_iteration_)
    xgb_best_iters.append(model_xgb.best_iteration)

# Calculate average best iterations (with safety margin)
cat_optimal = int(np.mean(cat_best_iters) * 1.15)
lgb_optimal = int(np.mean(lgb_best_iters) * 1.15)
xgb_optimal = int(np.mean(xgb_best_iters) * 1.15)

# Print with detailed explanation
print("\nğŸ“Š OPTIMAL ITERATIONS ANALYSIS")
print("================================")
print(f"CatBoost:  {np.mean(cat_best_iters):.1f} (avg from CV) â†’ {cat_optimal} (with 15% buffer)")
print(f"LightGBM:  {np.mean(lgb_best_iters):.1f} (avg from CV) â†’ {lgb_optimal} (with 15% buffer)")
print(f"XGBoost:   {np.mean(xgb_best_iters):.1f} (avg from CV) â†’ {xgb_optimal} (with 15% buffer)")
print("================================")
print("ğŸ’¡ Why 10% buffer?")
print("- Prevents underfitting when training on full dataset")
print("- Accounts for slightly different data distribution")
print("- Ensures we don't stop just before optimal performance")
print("- Still much more efficient than training to max iterations")


# Train meta-model (level-2 model) on OOF predictions
print("\nTraining meta-model (Logistic Regression)...")
meta_model = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    n_jobs=-1,
    random_state=42
)
meta_model.fit(oof_predictions, oof_targets)

# Evaluate on OOF (out-of-fold) predictions
final_oof_pred = meta_model.predict(oof_predictions)
final_oof_proba = meta_model.predict_proba(oof_predictions)

acc_oof = accuracy_score(oof_targets, final_oof_pred)
f1_oof = f1_score(oof_targets, final_oof_pred, average='macro')

print("\n====== Final OOF Results (after stacking) ======")
print(f"Accuracy:  {acc_oof:.4f}")
print(f"Macro-F1 : {f1_oof:.4f}")


params_cat_optimal = {
    'loss_function': 'MultiClass',
    'learning_rate': 0.06,
    'iterations': cat_optimal,
    'task_type': 'GPU',
    'verbose' : False,
    'early_stopping_rounds': 100
}    

params_lgb_optimal = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'learning_rate': 0.1,
    'n_estimators': lgb_optimal,
    'max_bin' : 20,
    'num_threads' : -1,
    'device': 'gpu',  # Requires LightGBM with GPU support compiled
    'verbose': -1      # Silence output; set to 1 to see progress
}

params_xgb_optimal = {
    'objective': 'multi:softprob',        # Equivalent to multiclass in LightGBM; outputs probabilities
    'num_class': num_classes,             # Required for multi-class in XGBoost
    'learning_rate': 0.08,                 # Same as LightGBM's learning_rate
    'n_estimators': xgb_optimal,           # Same number of boosting rounds
    'max_bin': 256,                       # XGBoost default is higher; LightGBM uses 20, but XGBoost typically uses 256
    'n_jobs': -1,                         # Equivalent to num_threads in LightGBM
    'tree_method': 'hist',            # Enables GPU; requires XGBoost with GPU support
    'device': 'cuda',
    'verbosity': 0,                       # 0 = silent, 1 = info (similar to verbose=-1 in LightGBM)
    'eval_metric': 'mlogloss'             # Optional: for monitoring; multiclass logloss
}


def refit_all_models(X_train_full, y_train_full):
    """
    Refit one instance of each model on the full dataset.
    """
    print("Refitting one model per algorithm on full dataset...")

    # CatBoost
    model_cat = CatBoostClassifier(**params_cat_optimal)
    model_cat.fit(X_train_full, y_train_full, verbose=False)

    # LightGBM
    model_lgb = lgb.LGBMClassifier(**params_lgb_optimal)
    model_lgb.fit(X_train_full, y_train_full)
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(**params_xgb_optimal)
    model_xgb.fit(X_train_full, y_train_full, verbose=0)

    # Wrap in lists for consistent interface
    return model_cat, model_lgb, model_xgb

# Refit All Models
model_cat, model_lgb, model_xgb = refit_all_models(X, y)


joblib.dump(model_cat, f'model_cat.joblib')
joblib.dump(model_lgb, f'model_lgb.joblib')
joblib.dump(model_xgb, f'model_xgb.joblib')
joblib.dump(meta_model, f'meta_model.joblib')


# def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
#     # data =sequence
#     data = sequence.join(demographics,on="subject",how="left")
#     # print(data.schema)
#     data = feature_engineering_imu(data)
#     cleaned_data = feature_engineering_stat(data)

#     pdf = cleaned_data.to_pandas().drop(columns=["sequence_id"])

#     pdf_scaled = robustscaler.transform(pdf)  # Shape: (1, n_features)
    
#     # Get predictions from base models (probabilities)
#     p_cat = model_cat.predict_proba(pdf_scaled)  # (1, num_classes)
#     p_lgb = model_lgb.predict_proba(pdf_scaled)
#     p_xgb = model_xgb.predict_proba(pdf_scaled)
    
#     # Stack probabilities for meta-model
#     meta_features = np.hstack([p_cat, p_lgb, p_xgb])  # (1, 3*num_classes)
    
#     # Final prediction using meta-model
#     y_pred_encoded = meta_model.predict(meta_features)  # Array of length 1
#     class_label = le.inverse_transform(y_pred_encoded)[0]
#     return class_label


# inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

# if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#     inference_server.serve()
# else:
#     inference_server.run_local_gateway(
#         data_paths=(
#             '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
#             '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
#         )
#     )

