# %%capture
# !pip install -U scikit-learn imbalanced-learn


import polars as pl
import lightgbm as lgbm
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd


# source: https://www.kaggle.com/code/nksusth/lb-0-78-quaternions-tf-bilstm-gru-attention
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

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
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

            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
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
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            relative_rotation = r1.inv() * r2
            
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
            pass
            
    return angular_dist


train = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demo = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")


train = train.drop(["sequence_type","orientation","behavior","phase"])


train.head(2)


data = train.join(train_demo,on="subject",how="left")


data.head(2)


data = data.to_pandas()


flags = []
for _,group in data.groupby('sequence_id'):
    max_count = group["sequence_counter"].max()
    group["flag"] = (group["sequence_counter"] > (max_count) * 0.1) | (group["sequence_counter"] < (max_count) * 0.9)
    flags.append(group)
data = pd.concat(flags)


data = data[data["flag"] == True]
data = data.drop(columns="flag")


# source: https://www.kaggle.com/code/nksusth/lb-0-78-quaternions-tf-bilstm-gru-attention
data['acc_mag'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
data['rot_angle'] = 2 * np.arccos(data['rot_w'].clip(-1, 1))

data['acc_mag_jerk'] = data.groupby('sequence_id')['acc_mag'].diff().fillna(0)
data['rot_angle_vel'] = data.groupby('sequence_id')['rot_angle'].diff().fillna(0)


# source: https://www.kaggle.com/code/nksusth/lb-0-78-quaternions-tf-bilstm-gru-attention
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


# source: https://www.kaggle.com/code/nksusth/lb-0-78-quaternions-tf-bilstm-gru-attention
angular_vel_list = []
for _, group in data.groupby('sequence_id'):
    rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
    angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))

df_angular_vel = pd.concat(angular_vel_list)
data = pd.concat([data, df_angular_vel], axis=1)

angular_distance_list = []
for _, group in data.groupby('sequence_id'):
    rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    angular_dist_group = calculate_angular_distance(rot_data_group)
    angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))

df_angular_distance = pd.concat(angular_distance_list)
data = pd.concat([data, df_angular_distance], axis=1)


def feature_engineering(data:pl.DataFrame):
    demographic_cols = [
    "adult_child", "age", "sex", "handedness",
    "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"
    ]
    target_col = "gesture"
    
    # All numeric sensor columns (everything except id, demo, target)
    stat_cols = [
        c for c in data.columns
        if c not in demographic_cols + [target_col, "sequence_id", "row_id","subject"]
    ]
    
    
    # Build aggregation expressions
    agg_exprs = []
    
    # full-stats bundle for sensor columns
    
    for c in stat_cols:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            # pl.col(c).mode().list.first().alias(f"{c}_mode"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
            # pl.col(c).first().alias(f"{c}_first"),
            # pl.col(c).last().alias(f"{c}_last"),
            # pl.col(c).quantile(0.25, "nearest").alias(f"{c}_t25"),
            # pl.col(c).quantile(0.75, "nearest").alias(f"{c}_t75"),
            # (pl.col(c).last() - pl.col(c).first()).alias(f"{c}_delta"),
            # pl.corr("sequence_counter", c).alias(f"{c}_corr_time"),
            # pl.col(c).diff().mean().alias(f"{c}_diff_mean"),
            # pl.col(c).diff().std().alias(f"{c}_diff_std"),
            # pl.col(c).skew().alias(f"{c}_skew"),
            # pl.col(c).kurtosis().alias(f"{c}_kurt"),
            # pl.col(c).diff().abs().gt(0).sum().alias(f"{c}_n_changes"),
            # # (pl.arg_max(c).cast(pl.Int64) / pl.count()).alias(f"{c}_t2peak"),
            # (pl.col(c) - pl.col(c).median()).abs().median().alias(f"{c}_mad"),
            # pl.col(c).pow(2).mean().sqrt().alias(f"{c}_rms"),
            # (pl.col(c).max() - pl.col(c).min()).alias(f"{c}_ptp"),
        ]
        # agg_exprs += [
        #     pl.when(pl.col("sequence_counter") < 0.25 * pl.max("sequence_counter"))
        #       .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg1_mean"),
        #     pl.when((pl.col("sequence_counter") > 0.25 * pl.max("sequence_counter")) & (pl.col("sequence_counter") < 0.5 * pl.max("sequence_counter")))
        #       .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg2_mean"),
        # pl.when((pl.col("sequence_counter") > 0.5 * pl.max("sequence_counter")) & (pl.col("sequence_counter") < 0.75 * pl.max("sequence_counter")))
        #       .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg3_mean"),
        #     pl.when(pl.col("sequence_counter") > 0.75 * pl.max("sequence_counter"))
        #       .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg4_mean"),
        # ]
    for a, b in [("acc_x", "acc_y"), ("acc_x", "acc_z"),
             ("acc_y", "acc_z"), ("angular_vel_x", "angular_vel_y"),
             ("angular_vel_x", "angular_vel_z"), ("angular_vel_y", "angular_vel_z")]:
        agg_exprs += [
            pl.corr(a, b).alias(f"{a}_{b}_corr"),
            (pl.col(a) * pl.col(b)).mean().alias(f"{a}_{b}_cov")
        ]
    # first() for demographics and target
    # agg_exprs += [
    #     pl.col(c).first().alias(c) for c in demographic_cols + [target_col]
    # ]
    agg_exprs += [
        pl.col(target_col).first().alias(target_col)
    ]
    agg_exprs += [pl.col("sequence_counter").max().alias(c)]
    # Group-by and aggregate
    cleaned_data = (
        data
        .group_by("sequence_id", maintain_order=True)
        .agg(agg_exprs)
    )
    return cleaned_data


data = pl.from_pandas(data)


cleaned_data = feature_engineering(data)


cleaned_data.head(2)


cleaned_data.write_parquet("cleaned_data.parquet")


import pandas as pd
from sklearn.preprocessing import LabelEncoder
target_col = "gesture"
pdf = cleaned_data.to_pandas() 

le = LabelEncoder()
y = le.fit_transform(pdf[target_col])
X = pdf.drop(columns=[target_col, "sequence_id"])


VALIDATE = False


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
import numpy as np
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import SMOTE
# from sklearn.impute import KNNImputer

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = dict(
    objective="multiclass",
    num_class=len(le.classes_),
    boosting_type="gbdt",
    learning_rate=0.2,
    colsample_bytree = 0.9,
    lambda_l1 = 0.5,
    lambda_l2 = 0.3,
    min_data_in_leaf=613,
    num_leaves=613,
    subsample=0.9,
    # device="gpu",
    max_depth = -1,
    n_estimators=20000,
    bagging_freq = 1,
    n_jobs=-1,
    verbose=-1,
    class_weight= "balanced",
)
if VALIDATE:
    fold_acc, fold_f1 = [], []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # imputer = KNNImputer(n_neighbors=7)
        # X_train = imputer.fit_transform(X_train)
        # sampler = RandomUnderSampler()
        # X_train, y_train = sampler.fit_resample(X_train, y_train)
  
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
    
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="macro")
    
        fold_acc.append(acc)
        fold_f1.append(f1)
    
    print("\n======  5-Fold Summary  ======")
    print(f"Accuracy:  mean={np.mean(fold_acc):.4f}  std={np.std(fold_acc):.4f}")
    print(f"Macro-F1 : mean={np.mean(fold_f1):.4f}  std={np.std(fold_f1):.4f}")

# imputer = KNNImputer(n_neighbors=7)
# X = imputer.fit_transform(X)
# sm = SMOTE(random_state=42)
# X, y = sm.fit_resample(X, y)
lgb_model = lgb.LGBMClassifier(**params)
# final_model = BaggingClassifier(estimator=lgb_model,n_estimators=10,n_jobs=-1)
lgb_model.fit(X, y)


version = 2.9
import pickle

with open(f"lgbm_gesture_model_{version}.pkl", "wb") as f:
    pickle.dump(lgb_model, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"gesture_label_encoder_{version}.pkl", "wb") as f:
    pickle.dump(le, f, protocol=pickle.HIGHEST_PROTOCOL)




