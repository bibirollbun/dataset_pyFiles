import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.interpolate import CubicSpline
import numpy as np
import pickle
from keras.src.utils import pad_sequences, to_categorical
import joblib
import polars as pl
from tqdm import tqdm
from sklearn.utils import compute_class_weight
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import kaggle_evaluation.cmi_inference_server


#train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv') #(574945, 341)
#train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')


TRAIN = False


#classes_count = train_df['gesture'].value_counts()
#plt.bar(classes_count.index, classes_count.values)
#plt.xlabel('Class')
#plt.ylabel('Count')
#plt.xticks(rotation=90)
#plt.tight_layout() 
#plt.title('Class Distribution')
#plt.show()


class Prepdata:
    def __init__(self, input_file: pd.DataFrame, input_demo: pd.DataFrame, augment_data: bool = False, augment_factor: int = 1, limit_factor=None, pad_percentile=75, calc_rotation_features=True):
        self.df = input_file.copy()
        self.demo = input_demo.copy()
        self.augment_data = augment_data
        self.augment_factor = augment_factor
        self.pad_percentile = pad_percentile
        self.calc_rotation_features = calc_rotation_features


        if limit_factor:
            self.df = self.df.head(limit_factor)
        
        self.excluded_cols = {
            'behavior', 'orientation',  
            'row_id', 'subject', 'phase', 'sequence_counter' 
        }

        self.is_training_mode = 'sequence_type' in self.df.columns and 'gesture' in self.df.columns

        if self.is_training_mode:
            self.targets = self.df[['sequence_id', 'sequence_type', 'gesture']].drop_duplicates()
            self.df = self.df.drop(['sequence_type', 'gesture'], axis=1)
        else:
            self.targets = None

        self.df = self.df.merge(self.demo, on='subject')

        
        self.df = self.clean_dataset(self.df)

        self.IMU_FEATURES = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
        self.BASE_TOF = [col for col in self.df.columns.to_list() if col.startswith('tof_')]


        if self.calc_rotation_features:
            self.df = self.calculate_enhanced_rotation_features(self.df)
            self.ROTATION_FEATURES = ['rot_angle', 'acc_mag', 'lin_acc_x', 'lin_acc_y' ,'lin_acc_z', 'linear_acc_mag', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'angular_dist']
            self.TOF_FEATURES = self.BASE_TOF + ['tof_mean', 'tof_std']
            self.ALL_FEATURES = self.IMU_FEATURES + self.ROTATION_FEATURES + self.TOF_FEATURES 
        else:
            self.ROTATION_FEATURES = []
            self.TOF_FEATURES = self.BASE_TOF
            self.ALL_FEATURES = self.IMU_FEATURES + self.TOF_FEATURES
        
        self.gesture_encoder = LabelEncoder()
        self.sequence_type_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        if self.is_training_mode:
            self.fit_scaler()



    def remove_gravity_from_acc(self, acc_data, rot_data):
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

    def calculate_angular_velocity_from_quat(self, rot_data, time_delta=1/200):
        if isinstance(rot_data, pd.DataFrame):
            quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
        else:
            quat_values = rot_data

        num_samples = quat_values.shape[0]
        angular_vel = np.zeros((num_samples, 3))

        for i in range(num_samples - 1):
            q_t = quat_values[i]
            q_t_plus_dt = quat_values[i+1]

            if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
                continue

            try:
                rot_t = R.from_quat(q_t)
                rot_t_plus_dt = R.from_quat(q_t_plus_dt)
                delta_rot = rot_t.inv() * rot_t_plus_dt
                angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
            except ValueError:
                pass
                
        return angular_vel

    def calculate_angular_distance(self, rot_data):
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
                
        return angular_dist

    def calculate_enhanced_rotation_features(self, input_df: pd.DataFrame):
        input_df = input_df.copy()
        
        input_df['acc_mag'] = np.sqrt(input_df['acc_x'] ** 2 + input_df['acc_y'] ** 2 + input_df['acc_z'] **2)
        input_df['rot_angle'] = 2 * np.arccos(np.clip(np.abs(input_df['rot_w']), 0, 1))
        
        acc_data = input_df[['acc_x', 'acc_y', 'acc_z']].values
        rot_data = input_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
        
        linear_accel = self.remove_gravity_from_acc(acc_data, rot_data)
        input_df['lin_acc_x'] = linear_accel[:, 0]
        input_df['lin_acc_y'] = linear_accel[:, 1]
        input_df['lin_acc_z'] = linear_accel[:, 2]
        input_df['linear_acc_mag'] = np.sqrt(linear_accel[:, 0]**2 + linear_accel[:, 1]**2 + linear_accel[:, 2]**2)
        
        if self.calc_rotation_features:
            angular_vel = self.calculate_angular_velocity_from_quat(rot_data)
            input_df['ang_vel_x'] = angular_vel[:, 0]
            input_df['ang_vel_y'] = angular_vel[:, 1]
            input_df['ang_vel_z'] = angular_vel[:, 2]
            
            angular_dist =self. calculate_angular_distance(rot_data)
            input_df['angular_dist'] = angular_dist
        
        input_df['tof_mean'] = np.mean(input_df[self.BASE_TOF], axis=1)
        input_df['tof_std'] = np.std(input_df[self.BASE_TOF], axis=1)
        
        return input_df
   
    def fit_scaler(self):
        cleaned_df = self.clean_dataset(self.df)
        feature_data = cleaned_df[self.ALL_FEATURES].values
        self.scaler.fit(feature_data)
        print(f"StandardScaler fitted on {len(self.ALL_FEATURES)} features")
    
    def time_warp_series(self, input_data, sigma=0.2, knot=4):
        T = input_data.shape[0]
        orig_steps = np.arange(T)

        warp_factors = np.random.normal(loc=1.0, scale=sigma, size=(knot,))
        warp_steps = np.linspace(0, T-1, num=knot)
        warp_curve = CubicSpline(warp_steps, warp_factors)(orig_steps)

        warped_time = np.cumsum(warp_curve)
        warped_time = warped_time / warped_time[-1] * (T - 1)

        warped_series = np.zeros_like(input_data)
        for d in range(input_data.shape[1]):
            cs = CubicSpline(orig_steps, input_data[:, d])
            warped_series[:, d] = cs(warped_time)

        return warped_series
    
    def augment_sequence(self, scaled_data):
        augmented_sequences = []
        
        if np.isnan(scaled_data).sum() == 0:
            for _ in range(self.augment_factor):
                warped_data = self.time_warp_series(scaled_data)
                augmented_sequences.append(warped_data)
        
        return augmented_sequences
    
    def clean_dataset(self, df: pd.DataFrame):
        cols_to_drop = [col for col in self.excluded_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        df = df.replace(-1.0, np.nan)
        df = df.ffill().bfill().fillna(0)
        return df

    def encode_targets(self, input_targets):
        if input_targets is None:
            return None, {}, {}
        
        encoded_targets = input_targets.copy()
        
        encoded_targets['gesture'] = self.gesture_encoder.fit_transform(input_targets['gesture'])
        gesture_2_id = dict(zip(self.gesture_encoder.classes_, 
                               self.gesture_encoder.transform(self.gesture_encoder.classes_)))
        
        encoded_targets['sequence_type'] = self.sequence_type_encoder.fit_transform(input_targets['sequence_type'])
        sequence_type_2_id = dict(zip(self.sequence_type_encoder.classes_, 
                                    self.sequence_type_encoder.transform(self.sequence_type_encoder.classes_)))
        
        return encoded_targets, gesture_2_id, sequence_type_2_id
    
    def preprocess_sequence(self, sequence_data):
        feature_data = sequence_data[self.ALL_FEATURES].values
        scaled_data = self.scaler.transform(feature_data)
        return scaled_data
    
    def prepare_data(self):
        cleaned_df = self.df
        X_sequences, Y, sequence_lengths = [], [], []
        
        if self.is_training_mode:
            encoded_targets, gesture_2_id, sequence_type_2_id = self.encode_targets(self.targets)
            
            for seq_id, group in tqdm(cleaned_df.groupby('sequence_id'), desc='Preprocessing data w/ scaling & augmentation: '):
                
                scaled_sequence = self.preprocess_sequence(group)
                
                if scaled_sequence is not None:
                    X_sequences.append(scaled_sequence)
                    sequence_lengths.append(len(scaled_sequence))
                    
                    target_gesture = encoded_targets[encoded_targets['sequence_id'] == seq_id]
                    if not target_gesture.empty:
                        gesture_label = target_gesture['gesture'].iloc[0]
                        Y.append(gesture_label)
                        
                        if self.augment_data:
                            augmented_sequences = self.augment_sequence(scaled_sequence)
                            for aug_seq in augmented_sequences:
                                X_sequences.append(aug_seq)
                                sequence_lengths.append(len(aug_seq))
                                Y.append(gesture_label)
                    else:
                        print(f'Found no target for sequence_id: {seq_id}')
        else:
            encoded_targets, gesture_2_id, sequence_type_2_id = None, {}, {}
            
            for seq_id, group in tqdm(cleaned_df.groupby('sequence_id'), desc='Preprocessing inference data: '):
                
                scaled_sequence = self.preprocess_sequence(group)
                
                if scaled_sequence is not None:
                    X_sequences.append(scaled_sequence)
                    sequence_lengths.append(len(scaled_sequence))

        if sequence_lengths:
            pad_length = int(np.percentile(sequence_lengths, self.pad_percentile))
            print(f"Padding sequences to length: {pad_length} ({self.pad_percentile}th percentile)")
            
            X_padded = pad_sequences(X_sequences, maxlen=pad_length, padding='post', truncating='post', dtype='float32')
        else:
            X_padded = np.array([])
            pad_length = 0

        if self.is_training_mode and Y:
            n_classes = len(gesture_2_id)
            Y_categorical = to_categorical(Y, num_classes=n_classes)
            print(f"Labels converted to categorical format with {n_classes} classes")
        else:
            Y_categorical = np.array([])
        return X_padded, Y_categorical, gesture_2_id, sequence_type_2_id, pad_length
    
    def save_preprocessing_artifacts(self, export_dir="./"):
        if self.is_training_mode:
            joblib.dump(self.scaler, f"{export_dir}/scaler.pkl")

            with open(f"{export_dir}/gesture_map.pkl", "wb") as f:
                pickle.dump(gesture_2_id, f)
            
            with open(f"{export_dir}/gesture_encoder.pkl", "wb") as f:
                pickle.dump(self.gesture_encoder, f)
            
            with open(f"{export_dir}/sequence_type_encoder.pkl", "wb") as f:
                pickle.dump(self.sequence_type_encoder, f)
            
            np.save(f"{export_dir}/feature_cols.npy", np.array(self.ALL_FEATURES))
            
            print(f"Preprocessing artifacts saved to {export_dir}")

    def load_preprocessing_artifacts(self, export_dir="/kaggle/input/decoding-environment/"):
        if not self.is_training_mode:
            self.scaler = joblib.load(f"{export_dir}/scaler.pkl")
            
            with open(f"{export_dir}/gesture_encoder.pkl", "rb") as f:
                self.gesture_encoder = pickle.load(f)
            
            with open(f"{export_dir}/sequence_type_encoder.pkl", "rb") as f:
                self.sequence_type_encoder = pickle.load(f)
            
            print(f"Preprocessing artifacts loaded from {export_dir}")


if TRAIN:     
    prep = Prepdata(train_df, train_demo,  augment_data=True, augment_factor=1)
    X_padded, Y_categorical, gesture_2_id, sequence_type_2_id, pad_length = prep.prepare_data()

    np.save("X_padded_array.npy", X_padded)
    np.save("Y_categorical_array.npy", Y_categorical)


    prep.save_preprocessing_artifacts()

    with open("gesture_map.pkl", "rb") as f:
        gesture_2_id = pickle.load(f)


    predicted_class = np.argmax(Y_categorical, axis=1)
    classes = np.unique(predicted_class)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=predicted_class)

    class_weight_dict = {int(class_num): float(weight) for class_num, weight in zip(classes, class_weights)}

    
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_padded, Y_categorical, random_state=42, test_size=0.2)

    X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2)

    
    Y_train = Y_train.squeeze() 
    Y_test = Y_test.squeeze() 
    
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"Y_train: {Y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"Y_test: {Y_test.shape}")
    
    n_classes = len(gesture_2_id)
    print(f"Number of classes: {n_classes}")

    print(np.var(X_train))
    print(np.var(X_test))



import tensorflow as tf
import keras
from keras.src.models import Model
from tensorflow.keras.utils import Sequence
from keras.src.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GRU, GaussianNoise, Layer
)
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau    , ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from keras.src import backend as K



EPOCHS = 50
BATCH_SIZE = 32
IMU_DIM = 17
TOF_DIM = 322
n_classes = 18


@keras.saving.register_keras_serializable()
class ComputeSlope(Layer):
    def __init__(self, **kwargs):
        super(ComputeSlope, self).__init__(**kwargs)
    
    def call(self, x):
        time_steps = tf.shape(x)[1]
        time_indices = tf.range(time_steps, dtype=tf.float32)
        time_indices = tf.reshape(time_indices, [1, -1, 1])
        
        n = tf.cast(time_steps, tf.float32)
        sum_t = tf.reduce_sum(time_indices, axis=1, keepdims=True)
        sum_x = tf.reduce_sum(x, axis=1, keepdims=True)
        sum_tx = tf.reduce_sum(x * time_indices, axis=1, keepdims=True)
        sum_t2 = tf.reduce_sum(time_indices * time_indices, axis=1, keepdims=True)
        
        numerator = n * sum_tx - sum_t * sum_x
        denominator = n * sum_t2 - sum_t * sum_t
        slope = tf.where(tf.abs(denominator) > 1e-8, numerator / denominator, tf.zeros_like(numerator))
        
        return slope
    
    def get_config(self):
        return super().get_config()

def time_sum(x):
    return tf.reduce_sum(x, axis=1)

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
    score = Lambda(squeeze_last_axis, output_shape=lambda input_shape: (input_shape[0], input_shape[1]))(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis, output_shape=lambda input_shape: (input_shape[0], input_shape[1], 1))(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum, output_shape=lambda input_shape: (input_shape[0], input_shape[2]))(context)
    return context

class MixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
    
    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx], self.y[idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        return X_mix, y_mix
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

def build_two_branch_model(imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(None, imu_dim+tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)

    merged = Concatenate()([x1, x2])

    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation='elu')(xc)
    
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)

if TRAIN:
    model = build_two_branch_model(IMU_DIM, TOF_DIM, n_classes)
    model.summary()


saving_path = "fold_models"
n_folds = 3
n_ensemble_models = 2
if TRAIN:
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    stk = StratifiedKFold(n_folds)

    Y_train_indicies = np.argmax(Y_train, axis=1)

    fold_predictions = []
    fold_models = []
    fold_scores = []
    print("\nStarting Training...")


    for i, (train_idx, test_idx) in enumerate(stk.split(X_train, Y_train_indicies)):
        X_fold_train, Y_fold_train = X_train[train_idx], Y_train[train_idx]
        X_fold_test, Y_fold_test = X_train[test_idx], Y_train[test_idx]

        print(f"Fold {i + 1} - Train: {X_fold_train.shape[0]}, Val: {X_fold_test.shape[0]}")


        fold_ensemble_models = []
        fold_ensemble_predictions = []

        for model_idx in range(n_ensemble_models):
            print(f"\nTraining model {model_idx + 1}/{n_ensemble_models} for fold {i + 1}")

            model = build_two_branch_model(IMU_DIM, TOF_DIM, n_classes)
            model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
                ModelCheckpoint(f'{saving_path}/model_fold_{i}_ensemble_{model_idx}.keras', 
                              monitor='val_loss', save_best_only=True, save_weights_only=False)
            ]

            history = model.fit(
                X_fold_train, Y_fold_train,
                epochs=EPOCHS,
                validation_data=(X_fold_test, Y_fold_test),
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )

            model.load_weights(f'{saving_path}/model_fold_{i}_ensemble_{model_idx}.keras')
            fold_pred = model.predict(X_fold_test, verbose=0)
            fold_ensemble_predictions.append(fold_pred)
            fold_ensemble_models.append(model)
    
        ensemble_pred = np.mean(fold_ensemble_predictions, axis=0)
        fold_predictions.append(ensemble_pred)
        fold_models.append(fold_ensemble_models)

        fold_score = accuracy_score(np.argmax(Y_fold_test, axis=1), np.argmax(ensemble_pred, axis=1))
        fold_scores.append(fold_score)
        print(f"Fold {i + 1} ensemble accuracy: {fold_score:.4f}")


    print(f"\nCross-validation scores: {fold_scores}")
    print(f"Mean CV score: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores) * 2:.4f})")





!zip -r /kaggle/working/fold_models.zip /kaggle/working/fold_models


from sklearn.metrics import classification_report, confusion_matrix

if TRAIN:
   test_predictions = []
   
   for fold_idx in range(n_folds):
       fold_ensemble_predictions = []
       
       for model_idx in range(n_ensemble_models):
           model = build_two_branch_model(IMU_DIM, TOF_DIM, n_classes)
           model.load_weights(f'{saving_path}/model_fold_{fold_idx}_ensemble_{model_idx}.keras')
           
           pred = model.predict(X_test, verbose=0)
           fold_ensemble_predictions.append(pred)
       
       fold_ensemble_pred = np.mean(fold_ensemble_predictions, axis=0)
       test_predictions.append(fold_ensemble_pred)
   
   final_predictions = np.mean(test_predictions, axis=0)
   predicted_classes = np.argmax(final_predictions, axis=1)

   true_labels = np.argmax(Y_test, axis=1)
   print("classificaiton report: \n", classification_report(true_labels, predicted_classes))

   print("\nconfusion matrix: ", confusion_matrix(true_labels, predicted_classes))
   


   
#    print(f"X_test shape: {X_test.shape}")
#    print(f"Test predictions shape: {final_predictions.shape}")
#    print(f"Predicted classes: {predicted_classes}")

    


PREPROCESSING_ARTIFACTS = None

def load_artifacts_once(artifacts_dir="/kaggle/input/decoding-environment/"):
    global PREPROCESSING_ARTIFACTS
    
    if PREPROCESSING_ARTIFACTS is None:
        PREPROCESSING_ARTIFACTS = {}
        
        PREPROCESSING_ARTIFACTS['scaler'] = joblib.load(f"{artifacts_dir}/scaler.pkl")
        
        with open(f"{artifacts_dir}/gesture_encoder.pkl", "rb") as f:
            PREPROCESSING_ARTIFACTS['gesture_encoder'] = pickle.load(f)
            
        with open(f"{artifacts_dir}/sequence_type_encoder.pkl", "rb") as f:
            PREPROCESSING_ARTIFACTS['sequence_type_encoder'] = pickle.load(f)
        
        PREPROCESSING_ARTIFACTS['feature_cols'] = np.load(f"{artifacts_dir}/feature_cols.npy")
        
        gesture_encoder = PREPROCESSING_ARTIFACTS['gesture_encoder']
        PREPROCESSING_ARTIFACTS['gesture_2_id'] = dict(zip(
            gesture_encoder.classes_, 
            gesture_encoder.transform(gesture_encoder.classes_)
        ))
        PREPROCESSING_ARTIFACTS['id_2_gesture'] = {v: k for k, v in PREPROCESSING_ARTIFACTS['gesture_2_id'].items()}
        
        print(f"Gestures: {PREPROCESSING_ARTIFACTS['gesture_2_id']}")
        print(f"Expected features: {len(PREPROCESSING_ARTIFACTS['feature_cols'])}")
    
    return PREPROCESSING_ARTIFACTS

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    try:
        artifacts = load_artifacts_once()
        
        sequence_pd = sequence.to_pandas()
        demographics_pd = demographics.to_pandas()
        
        prep = Prepdata(sequence_pd, demographics_pd, augment_data=False, augment_factor=1, calc_rotation_features=True)
        
        prep.load_preprocessing_artifacts()
        X_padded, _, _, _, pad_length = prep.prepare_data()
        
        if len(X_padded) == 0 or X_padded.size == 0:
            fallback_gesture = list(artifacts['gesture_2_id'].keys())[0]
            print(f"Empty data, returning fallback: {fallback_gesture}")
            return fallback_gesture
        
        X_padded = np.array(X_padded, dtype=np.float32)
        if len(X_padded.shape) == 2:
            X_padded = np.expand_dims(X_padded, axis=0)
        
        print(f"Input shape for prediction: {X_padded.shape}")
        
        all_predictions = []

        for fold_idx in range(n_folds):
            fold_predictions = []
            
            for model_idx in range(n_ensemble_models):
                try:
                    ensemble_model = build_two_branch_model(IMU_DIM, TOF_DIM, n_classes)

                    model_path = f'/kaggle/input/model_test_1/keras/default/1/model_fold_{fold_idx}_ensemble_{model_idx}.keras'
                    ensemble_model.load_weights(model_path)
                    
                    pred = ensemble_model.predict(X_padded, verbose=0)
                    fold_predictions.append(pred)
                    
                except Exception as e:
                    print(f"Warning: Could not load model fold_{fold_idx}_ensemble_{model_idx}: {e}")
                    continue
            
            if fold_predictions:
                fold_ensemble_pred = np.mean(fold_predictions, axis=0)
                all_predictions.append(fold_ensemble_pred)
        
        if not all_predictions:
            print("Warning: No ensemble models could be loaded, using fallback")
            fallback_gesture = list(artifacts['gesture_2_id'].keys())[0]
            return fallback_gesture
        
        final_prediction = np.mean(all_predictions, axis=0)
        
        if len(final_prediction.shape) > 1:
            predicted_class_idx = np.argmax(final_prediction[0])
        else:
            predicted_class_idx = np.argmax(final_prediction)
        
        predicted_gesture = artifacts['id_2_gesture'][predicted_class_idx]
        
        print(f"Predicted gesture: {predicted_gesture} (from {len(all_predictions)} folds)")
        return predicted_gesture
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            artifacts = load_artifacts_once()
            fallback = list(artifacts['gesture_2_id'].keys())[0]
        except:
            fallback = "unknown"
        
        print(f"Returning fallback gesture: {fallback}")
        return fallback

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

