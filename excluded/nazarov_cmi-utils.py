%%writefile cmi_utils.py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score


############ Quaternion Transformation #############

def quaternion_to_euler(quat):
    # [N x,y,z,w]
    rotation = R.from_quat(quat)
    # 将旋转对象转换为欧拉角（以弧度为单位）
    euler_angles = rotation.as_euler('zyx', degrees=False) 
    euler_z,euler_y,euler_x = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    return euler_x, euler_y, euler_z  


def euler_to_quaternion(euler_x, euler_y, euler_z):
    """将欧拉角转换为四元数
    参数:
        roll (float): X轴旋转角度（弧度）
        pitch (float): Y轴旋转角度（弧度）
        yaw (float): Z轴旋转角度（弧度）
        order (str): 旋转顺序，如 'zyx', 'xyz' 等
    返回:
        np.array: 四元数 [w, x, y, z]
    """
    # 使用SciPy的Rotation类（默认顺序为 'xyz'，注意角度需为弧度）
    rotation = R.from_euler('zyx', np.array([euler_z, euler_y, euler_x]).T, degrees=False)
    q = rotation.as_quat()  # 返回 [x, y, z, w]
    return np.array([q[:,3],q[:,0],q[:,1],q[:,2]]).T  # 返回 [w, x,y, z]


def quaternion_multiply(q2, q1):
    '''
    params:  q1,q2: [N,[w,x,y,z]]
    return:  [N,[w,x,y,z]]
    '''
    if len(q2.shape)==1:
        w2, x2, y2, z2 = q2[0],q2[1],q2[2],q2[3]
    else:
        w2, x2, y2, z2 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]

    if len(q1.shape)==1:
        w1, x1, y1, z1 = q1[0],q1[1],q1[2],q1[3]
    else:
        w1, x1, y1, z1 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    return np.array([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ]).T


def quaternion_conjugate(q):
    if len(q.shape)==1:
        w, x, y, z = q[0],q[1],q[2],q[3]
    else:
        w, x, y, z = q[:,0],q[:,1],q[:,2],q[:,3]
    return np.array([w, -x, -y, -z]).T


def relative_rotation_quaternion(q1, q2, local=True):
    # q1,q2: [N,[w,x,y,z]]
    q1_conjugate = quaternion_conjugate(q1)
    if local:
        q_rel = quaternion_multiply(q1_conjugate,q2)
    else:
        q_rel = quaternion_multiply(q2, q1_conjugate)
    return q_rel


def flip_tof(x):
    x = x.reshape((len(x),5,8,8))
    x[:,[0,1,3]] =  x[:,[0,1,3],:,::-1]
    x[:,[2,4]] =  x[:,[2,4],::-1]
    x2 = x[:,2].copy()
    x[:,2],x[:,4] =  x[:,4],x2
    return x.reshape((len(x),-1))

############ Data Augmentation #############


def DataAugmentation(df, n_aug, y_grid, z_grid):
    quat_rot_y = [np.array([np.cos(np.deg2rad(d) / 2), 0, np.sin(np.deg2rad(d) / 2), 0]) 
                  for d in y_grid]
    quat_rot_z = [np.array([np.cos(np.deg2rad(d)/2),0,0,np.sin(np.deg2rad(d)/2)]) 
                  for d in z_grid]
    acc_cols = ['acc_x','acc_y','acc_z']
    cols_to_convert = df.filter(regex='^(acc|rot|tof|thm)').columns
    df[cols_to_convert] = df[cols_to_convert].astype(np.float32)
    df_new = pd.concat([df]*len(quat_rot_z), ignore_index=True)
    q1 = df[['rot_w','rot_x','rot_y','rot_z']].to_numpy(np.float32)
    for i,(q2_z,q2_y) in enumerate(zip(quat_rot_z,quat_rot_y)):
        df_tmp = df.copy()
        quat_new = quaternion_multiply(q2_z,q1)
        rot = R.from_quat(np.array([q2_y[1],q2_y[2],q2_y[3],q2_y[0]]))
        acc_new = rot.apply(df_tmp[acc_cols]).astype(np.float32)
        df_tmp[acc_cols] = acc_new
        q2_y_reverse = quaternion_conjugate(q2_y)
        quat_rot = quaternion_multiply(quat_new,q2_y_reverse)
        df_tmp[['rot_w','rot_x','rot_y','rot_z']] = quat_rot
        df_tmp['sequence_id'] = df_tmp['sequence_id']+f'__aug{i}'
        df_new.iloc[i*len(df):(i+1)*len(df)] = df_tmp
    return df_new


############ GAF #############

def gaf_cosine_similarity(time_series_2d):
    """
    Compute multivariate GAF using cosine similarity between vectors.
    
    Parameters
    ----------
    time_series_2d : array-like, shape (n_timesteps, n_features)
        Single multivariate time series
        
    Returns
    -------
    gaf_matrix : array, shape (n_timesteps, n_timesteps)
        GAF matrix with values in [-1, 1]
    """
    vectors = time_series_2d
    n_timesteps = len(vectors)
    
    # Compute cosine similarity
    dot_products = np.dot(vectors, vectors.T)
    norms = np.linalg.norm(vectors, axis=1)
    norm_matrix = np.outer(norms, norms)
    
    # Avoid division by zero
    mask = norm_matrix > 1e-10
    gaf_matrix = np.zeros_like(dot_products)
    gaf_matrix[mask] = np.clip(dot_products[mask] / norm_matrix[mask], -1, 1)
    
    return gaf_matrix

def gaf_angular_sum(time_series_2d, reference_vector=None):
    """
    Compute multivariate GAF using angular sum approach (true GAF extension).
    
    Parameters
    ----------
    time_series_2d : array-like, shape (n_timesteps, n_features)
        Single multivariate time series
    reference_vector : array-like, optional
        Reference vector for angle computation. If None, uses [1,0,0,...]
        
    Returns
    -------
    gaf_matrix : array, shape (n_timesteps, n_timesteps)
        GAF matrix with values in [-1, 1]
    """
    vectors = time_series_2d
    n_timesteps, n_features = vectors.shape
    
    # Normalize vectors to unit sphere
    norms = np.linalg.norm(vectors, axis=1)
    norms_safe = np.where(norms == 0, 1, norms)
    unit_vectors = vectors / norms_safe[:, np.newaxis]
    
    # Set reference vector
    if reference_vector is None:
        reference_vector = np.zeros(n_features)
        reference_vector[0] = 1
    else:
        reference_vector = np.array(reference_vector) / np.linalg.norm(reference_vector)
    
    # Compute angles with reference vector
    angles = []
    for i in range(len(unit_vectors)):
        dot_prod = np.dot(unit_vectors[i], reference_vector)
        dot_prod = np.clip(dot_prod, -1, 1)
        angle = np.arccos(dot_prod)
        angles.append(angle)
    
    angles = np.array(angles)
    
    # Compute cos(θ_i + θ_j) for all pairs (original GAF approach)
    gaf_matrix = np.cos(np.add.outer(angles, angles))
    
    return gaf_matrix

def gaf_gramian(time_series_2d, normalize_output=True):
    """
    Compute multivariate GAF using Gramian matrix approach.
    
    Parameters
    ----------
    time_series_2d : array-like, shape (n_timesteps, n_features)
        Single multivariate time series
    normalize_output : bool, optional (default=True)
        Whether to normalize output to [-1, 1] range
        
    Returns
    -------
    gaf_matrix : array, shape (n_timesteps, n_timesteps)
        GAF matrix (values depend on normalize_output parameter)
    """
    vectors = time_series_2d
    
    # Compute Gramian matrix (dot products)
    gaf_matrix = np.dot(vectors, vectors.T)
    
    if normalize_output:
        # Normalize to [-1, 1] range
        min_val, max_val = np.min(gaf_matrix), np.max(gaf_matrix)
        if max_val > min_val:
            gaf_matrix = 2 * (gaf_matrix - min_val) / (max_val - min_val) - 1
        else:
            gaf_matrix = np.zeros_like(gaf_matrix)
        return np.clip(gaf_matrix, -1, 1)
    else:
        return gaf_matrix

def compute_multivariate_gaf(X_3d, method='cosine', **kwargs):
    """
    Compute GAF for multivariate time series data.
    
    Parameters
    ----------
    X_3d : array-like, shape (n_samples, n_timesteps, n_features)
        Normalized multivariate time series data
    method : str, optional (default='cosine')
        GAF computation method:
        - 'cosine': Cosine similarity between vectors
        - 'angular_sum': Angular sum approach (true GAF extension)
        - 'gramian': Gramian matrix approach
    **kwargs : additional arguments passed to GAF functions
        
    Returns
    -------
    X_gaf : array, shape (n_samples, n_timesteps, n_timesteps)
        GAF matrices for each sample
    """
    
    n_samples, n_timesteps, n_features = X_3d.shape
    X_gaf = np.zeros((n_samples, n_timesteps, n_timesteps))
    
    # Select GAF function
    if method == 'cosine':
        gaf_func = gaf_cosine_similarity
    elif method == 'angular_sum':
        gaf_func = lambda x: gaf_angular_sum(x, **kwargs)
    elif method == 'gramian':
        gaf_func = lambda x: gaf_gramian(x, **kwargs)
    else:
        raise ValueError(f"Unknown GAF method: {method}")
    
    # Compute GAF for each sample
    for i in range(n_samples):
        X_gaf[i] = gaf_func(X_3d[i])
        
    return X_gaf

############# fixing outliers #############

def Fix_outlier(df,imu_only):
    outlier_idx = df.subject.isin(['SUBJ_019262','SUBJ_045235'])
    df_outlier = df.loc[outlier_idx]
    quat = df_outlier[['rot_x','rot_y','rot_z','rot_w']].fillna(
        {'rot_x':-0.119916,'rot_y':-0.059953,'rot_z':-0.188298,'rot_w':0.360375})

    d = 180
    quat_rot_z = np.array([np.cos(np.deg2rad(d)/2),0,0,np.sin(np.deg2rad(d)/2)])
    quat_rot = quaternion_multiply(quat[['rot_w','rot_x','rot_y','rot_z']].values,
                                   quat_rot_z)
    quat_rot = quat_rot*((quat_rot[:,0:1]>0)*2-1)
    df.loc[outlier_idx,['rot_w','rot_x','rot_y','rot_z']]=quat_rot
    df.loc[outlier_idx,'acc_x'] *= -1
    df.loc[outlier_idx,'acc_y'] *= -1

    if not imu_only:
        tof_columns = [f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)]
        thm_columns = [f'thm_{i}' for i in range(1,6)]
        tof = df_outlier[tof_columns].values.reshape(-1,5,8,8)
        tof = np.rot90(tof, k=2, axes=(2, 3))
        tof[:,[0,1,2,3,4]] = tof[:,[0,3,4,1,2]]
        tof = tof.reshape(-1,5*64)

        thm = df_outlier[thm_columns].values
        thm[:,[0,1,2,3,4]] =  thm[:,[0,3,4,1,2]]
        df.loc[outlier_idx,tof_columns] = tof
        df.loc[outlier_idx,thm_columns] = thm
        
    return df


########## miscellaneous ###############


def transform_to_match_distribution(x1, x2):
    # Sort both arrays
    x1_sorted = np.sort(x1)
    x2_sorted = np.sort(x2)

    # Create percentile positions
    x1_percentiles = np.linspace(0, 100, len(x1_sorted))
    x2_percentiles = np.linspace(0, 100, len(x2_sorted))

    # Interpolation function to map x2 percentiles to x1 values
    def transform(values):
        # Get percentiles for each value in x2
        # Using 'nearest' interpolation to avoid extrapolation issues
        percentiles = np.interp(values, x2_sorted, x2_percentiles, 
                                left=0, right=100)
        # Map these percentiles to x1 values
        transformed_values = np.interp(percentiles, x1_percentiles, x1_sorted, 
                                     left=x1_sorted[0], right=x1_sorted[-1])
        return transformed_values

    return transform(x2)

########### Competition metric ###############

"""
Hierarchical macro F1 metric for the CMI 2025 Challenge.

This script defines a single entry point `score(solution, submission, row_id_column_name)`
that the Kaggle metrics orchestrator will call.
It performs validation on submission IDs and computes a combined binary & multiclass F1 score.
"""

class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass


class CompetitionMetric:
    """Hierarchical macro F1 for the CMI 2025 challenge."""
    def __init__(self, data_type = 'str'):
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures
        self.non_target = 'non_target'
        if data_type != 'str':
            sorted_gestures = sorted(self.all_classes)
            self.target_gestures = [sorted_gestures.index(g) for g in self.target_gestures]
            self.non_target_gestures = [sorted_gestures.index(g) for g in self.non_target_gestures]
            self.all_classes = np.arange(len(self.all_classes))
            self.non_target = -1
        

    def calculate_hierarchical_f1(
        self,
        sol: pd.DataFrame,
        sub: pd.DataFrame
    ) -> float:

        # Validate gestures
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )

        # Compute binary F1 (Target vs Non-Target)
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
        )

        # Build multi-class labels for gestures
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else self.non_target)
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else self.non_target)

        # Compute macro F1 over all gesture classes
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )
        print(f1_binary,f1_macro)
        return 0.5 * f1_binary + 0.5 * f1_macro


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    data_type: str = 'str'
) -> float:
    """
    Compute hierarchical macro F1 for the CMI 2025 challenge.

    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'sequence_id': unique identifier for each sequence
      - 'gesture': one of the eight target gestures or "Non-Target"

    This metric averages:
    1. Binary F1 on SequenceType (Target vs Non-Target)
    2. Macro F1 on gesture (mapping non-targets to "Non-Target")

    Raises ParticipantVisibleError for invalid submissions,
    including invalid SequenceType or gesture values.


    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> solution = pd.DataFrame({'id': range(4), 'gesture': ['Eyebrow - pull hair']*4})
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Forehead - pull hairline']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.5
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Text on phone']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.0
    >>> score(solution, solution, row_id_column_name=row_id_column_name)
    1.0
    """
    # Validate required columns
    for col in (row_id_column_name, 'gesture'):
        if col not in solution.columns:
            raise ParticipantVisibleError(f"Solution file missing required column: '{col}'")
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Submission file missing required column: '{col}'")

    metric = CompetitionMetric(data_type)
    return metric.calculate_hierarchical_f1(solution, submission)

# Competition scoring function based on a specific encoding. 
# Target gestures are indexed from 0 to 7.
def score_from_int(true,pred):
    y0 = np.where(true <= 7,1,0)
    p0 = np.where(pred <= 7,1,0)
    y1 = np.where(true <= 7, true, 8)
    p1 = np.where(pred <= 7, pred, 8)
    return 0.5 * (f1_score(y0,p0) + f1_score(y1,p1,average='macro'))

########### tf79_utils ##################

import os, numpy as np, pandas as pd
KAGGLE = 'kaggle' in os.getcwd()

if KAGGLE:
    import tensorflow as tf
    from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (
        Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
        Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
        Lambda, Concatenate, GRU, GaussianNoise
    )
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.transform import Rotation as R
    from pathlib import Path
    import joblib
    
    
    #Tensor Manipulations
    def time_sum(x):
        return K.sum(x, axis=1)
    
    def squeeze_last_axis(x):
        return tf.squeeze(x, axis=-1)
    
    def expand_last_axis(x):
        return tf.expand_dims(x, axis=-1)
    
    def se_block(x, reduction=8):
        ch = x.shape[-1]
        se = GlobalAveragePooling1D()(x)
        se = Dense(ch // reduction, activation='relu')(se)
        se = Dense(ch, activation='sigmoid')(se)
        se = Reshape((1, ch))(se)
        return Multiply()([x, se])
    
    # Residual CNN Block with SE
    def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
        shortcut = x
        for _ in range(2):
            x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                       kernel_regularizer=l2(wd))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = se_block(x)
        if shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                              kernel_regularizer=l2(wd))(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = add([x, shortcut])
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size)(x)
        x = Dropout(drop)(x)
        return x
    
    def attention_layer(inputs):
        score = Dense(1, activation='tanh')(inputs)
        score = Lambda(squeeze_last_axis)(score)
        weights = Activation('softmax')(score)
        weights = Lambda(expand_last_axis)(weights)
        context = Multiply()([inputs, weights])
        context = Lambda(time_sum)(context)
        return context
    
    
    # Normalizes and cleans the time series sequence. 
    
    def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], 
                            scaler: StandardScaler):
        mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
        return scaler.transform(mat).astype('float32')
    
    
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
    
    
    def prepare_tf79_input(df_seq: pd.DataFrame):
        df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
        df_seq['rot_angle'] = 2 * np.arccos(df_seq['rot_w'].clip(-1, 1))
        df_seq['acc_mag_jerk'] = df_seq['acc_mag'].diff().fillna(0)
        df_seq['rot_angle_vel'] = df_seq['rot_angle'].diff().fillna(0)
    
        acc_cols_for_gravity_removal = ['acc_x', 'acc_y', 'acc_z']
        rot_cols_for_gravity_removal = ['rot_x', 'rot_y', 'rot_z', 'rot_w']
    
        if not all(col in df_seq.columns for col in acc_cols_for_gravity_removal + rot_cols_for_gravity_removal):
            print(f"Warning: Missing raw acc/rot columns for gravity removal in predict for sequence. Using raw acc as linear.")
            df_seq['linear_acc_x'] = df_seq.get('acc_x', 0)
            df_seq['linear_acc_y'] = df_seq.get('acc_y', 0)
            df_seq['linear_acc_z'] = df_seq.get('acc_z', 0)
        else:
            acc_data_seq = df_seq[acc_cols_for_gravity_removal]
            rot_data_seq = df_seq[rot_cols_for_gravity_removal]
            linear_accel_seq_arr = remove_gravity_from_acc(acc_data_seq, rot_data_seq)
            
            df_seq['linear_acc_x'] = linear_accel_seq_arr[:, 0]
            df_seq['linear_acc_y'] = linear_accel_seq_arr[:, 1]
            df_seq['linear_acc_z'] = linear_accel_seq_arr[:, 2]
        
        df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2)
        df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
        
        # Calculate angular velocity from quaternions in predict function
        if all(col in df_seq.columns for col in rot_cols_for_gravity_removal):
            angular_vel_seq_arr = calculate_angular_velocity_from_quat(df_seq[rot_cols_for_gravity_removal])
            df_seq['angular_vel_x'] = angular_vel_seq_arr[:, 0]
            df_seq['angular_vel_y'] = angular_vel_seq_arr[:, 1]
            df_seq['angular_vel_z'] = angular_vel_seq_arr[:, 2]
    
            # Calculate angular jerk from angular velocity
            df_seq['angular_jerk_x'] = df_seq['angular_vel_x'].diff().fillna(0)
            df_seq['angular_jerk_y'] = df_seq['angular_vel_y'].diff().fillna(0)
            df_seq['angular_jerk_z'] = df_seq['angular_vel_z'].diff().fillna(0)
    
            # Calculate angular snap from angular jerk
            df_seq['angular_snap_x'] = df_seq['angular_jerk_x'].diff().fillna(0)
            df_seq['angular_snap_y'] = df_seq['angular_jerk_y'].diff().fillna(0)
            df_seq['angular_snap_z'] = df_seq['angular_jerk_z'].diff().fillna(0)
    
        else:
            print(f"Warning: Missing quaternion columns for angular velocity, jerk, and snap calculation in predict. Filling with 0.")
            df_seq['angular_vel_x'] = 0
            df_seq['angular_vel_y'] = 0
            df_seq['angular_vel_z'] = 0
            df_seq['angular_jerk_x'] = 0
            df_seq['angular_jerk_y'] = 0
            df_seq['angular_jerk_z'] = 0
            df_seq['angular_snap_x'] = 0
            df_seq['angular_snap_y'] = 0
            df_seq['angular_snap_z'] = 0
    
        for i in range(1, 6): 
            pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
            if not all(col in df_seq.columns for col in pixel_cols_tof):
                print(f"Warning: Missing some TOF pixel columns for tof_{i} in predict. Filling aggregates with 0.")
                df_seq[f'tof_{i}_mean'] = 0
                df_seq[f'tof_{i}_std']  = 0
                df_seq[f'tof_{i}_min']  = 0
                df_seq[f'tof_{i}_max']  = 0
                continue
    
            tof_sensor_data = df_seq[pixel_cols_tof].replace(-1, np.nan)
            df_seq[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
            df_seq[f'tof_{i}_std']  = tof_sensor_data.std(axis=1)
            df_seq[f'tof_{i}_min']  = tof_sensor_data.min(axis=1)
            df_seq[f'tof_{i}_max']  = tof_sensor_data.max(axis=1)
            
        if 'tof_range_across_sensors' in final_feature_cols:
            tof_mean_cols_for_contrast = [f'tof_{i}_mean' for i in range(1, 6) if f'tof_{i}_mean' in df_seq.columns]
            thm_cols_for_contrast = [f'thm_{i}' for i in range(1, 6) if f'thm_{i}' in df_seq.columns]
    
            if tof_mean_cols_for_contrast:
                tof_values_for_contrast = df_seq[tof_mean_cols_for_contrast]
                df_seq['tof_range_across_sensors'] = tof_values_for_contrast.max(axis=1) - tof_values_for_contrast.min(axis=1)
                df_seq['tof_std_across_sensors'] = tof_values_for_contrast.std(axis=1)
            else:
                df_seq['tof_range_across_sensors'] = 0
                df_seq['tof_std_across_sensors'] = 0
    
            if thm_cols_for_contrast:
                thm_values_for_contrast = df_seq[thm_cols_for_contrast]
                df_seq['thm_range_across_sensors'] = thm_values_for_contrast.max(axis=1) - thm_values_for_contrast.min(axis=1)
                df_seq['thm_std_across_sensors'] = thm_values_for_contrast.std(axis=1)
            else:
                df_seq['thm_range_across_sensors'] = 0
                df_seq['thm_std_across_sensors'] = 0
            
        df_seq_final_features = pd.DataFrame(index=df_seq.index)
        for col_name in final_feature_cols:
            if col_name in df_seq.columns:
                df_seq_final_features[col_name] = df_seq[col_name]
            else:
                print(f"CRITICAL ERROR IN PREDICT: Feature '{col_name}' expected by model (from final_feature_cols) was NOT generated in df_seq. Filling with 0. THIS IS LIKELY A BUG.")
                df_seq_final_features[col_name] = 0 
                
        mat_unscaled = df_seq_final_features.ffill().bfill().fillna(0).values.astype('float32')
        
        mat_scaled = scaler.transform(mat_unscaled)
        
        pad_input = pad_sequences([mat_scaled], maxlen=pad_len, padding='post', truncating='post', dtype='float32')
        return pad_input
    
    
    custom_objs = {
        'time_sum': time_sum,
        'squeeze_last_axis': squeeze_last_axis,
        'expand_last_axis': expand_last_axis,
        'se_block': se_block,
        'residual_se_cnn_block': residual_se_cnn_block,
        'attention_layer': attention_layer,
    }
    
    PRETRAINED_DIR = Path("/kaggle/input/cmi-models-new/tf79-8223")
    print("tf79_utils: loading artefacts from", PRETRAINED_DIR)
    final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    
    # Re-calculate imu_dim_final based on the actual features that will be used
    imu_features_in_final_cols = [c for c in final_feature_cols if any(c.startswith(prefix) for prefix in ['linear_acc_', 'acc_', 'rot_', 'angular_vel_', 'angular_jerk_', 'angular_snap_'])]
    imu_dim_final = len(imu_features_in_final_cols)
    tof_thm_aggregated_dim_final = len(final_feature_cols) - imu_dim_final
    
    


