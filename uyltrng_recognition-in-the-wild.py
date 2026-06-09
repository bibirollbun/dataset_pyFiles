import os
import shutil
import pyarrow.parquet as pq
import tensorflow as tf
import json
import matplotlib
# Đảm bảo matplotlib sử dụng backend không tương tác nếu chạy trong môi trường không có GUI
# matplotlib.use('Agg') # Bỏ comment nếu cần
import matplotlib.pyplot as plt
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd

# --- 0. GLOBAL CONSTANTS AND CONFIGURATION ---

LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE
X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]
FEATURE_COLUMNS = X + Y + Z

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "right_hand" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "left_hand" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "pose" in col and int(col.split('_')[-1]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "pose" in col and int(col.split('_')[-1]) in LPOSE]

FRAME_LEN = 128
TARGET_MAXLEN = 64 # Độ dài tối đa của chuỗi nhãn sau khi padding

CHAR_TO_NUM_PATH = "/kaggle/input/asl-fingerspelling/character_to_prediction_index.json"
if not os.path.exists(CHAR_TO_NUM_PATH):
    print(f"Warning: Character map file not found at {CHAR_TO_NUM_PATH}. Using a dummy map.")
    char_to_num_orig = {chr(ord('a') + i): i for i in range(26)}
    char_to_num_orig[' '] = 26
else:
    with open (CHAR_TO_NUM_PATH, "r") as f:
        char_to_num_orig = json.load(f)

# --- Vocabulary cho CTC ---
_char_to_num_intermediate = {}
current_max_original_idx = -1
for char, idx_val in char_to_num_orig.items():
    try:
        idx = int(idx_val)
        _char_to_num_intermediate[char] = idx
        current_max_original_idx = max(current_max_original_idx, idx)
    except ValueError:
        print(f"Warning: Could not convert index '{idx_val}' for char '{char}' to int. Skipping.")

pad_token = 'P'
unknown_token = '?'

unique_chars_from_orig = list(_char_to_num_intermediate.keys())
if unknown_token not in unique_chars_from_orig:
    unique_chars_from_orig.append(unknown_token)
sorted_predictable_chars = sorted(list(set(unique_chars_from_orig)))

ctc_char_to_num = {}
for i, char in enumerate(sorted_predictable_chars):
    ctc_char_to_num[char] = i

ctc_num_to_char = {v: k for k, v in ctc_char_to_num.items()}

NUM_PREDICTABLE_CHARS = len(ctc_char_to_num)
BLANK_INDEX = NUM_PREDICTABLE_CHARS
CTC_NUM_CLASSES = NUM_PREDICTABLE_CHARS + 1
pad_token_idx_for_padding = -1

print(f"--- Vocabulary Debug for CTC ---")
print(f"ctc_char_to_num (ký tự dự đoán được map tới 0..N-1): {ctc_char_to_num}")
print(f"ctc_num_to_char: {ctc_num_to_char}")
print(f"unknown_token is mapped to ctc_char_to_num index: {ctc_char_to_num.get(unknown_token, 'NOT IN VOCAB (ERROR)')}")
print(f"NUM_PREDICTABLE_CHARS (N): {NUM_PREDICTABLE_CHARS}")
print(f"BLANK_INDEX (sẽ là N): {BLANK_INDEX}")
print(f"CTC_NUM_CLASSES (N+1, cho lớp Dense cuối, bao gồm blank): {CTC_NUM_CLASSES}")
print(f"pad_token_idx_for_padding (dùng để pad nhãn): {pad_token_idx_for_padding}")
print(f"--- End Vocabulary Debug ---")

default_lookup_value = ctc_char_to_num.get(unknown_token)
if default_lookup_value is None:
    print(f"CRITICAL WARNING: unknown_token '{unknown_token}' not found in ctc_char_to_num. Defaulting lookup to 0.")
    default_lookup_value = 0 if NUM_PREDICTABLE_CHARS > 0 else -1

ctc_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=list(ctc_char_to_num.keys()),
        values=tf.constant(list(ctc_char_to_num.values()), dtype=tf.int32),
    ),
    default_value=tf.constant(default_lookup_value, dtype=tf.int32),
    name="ctc_char_to_num_lookup"
)

# --- 1. DATA PREPROCESSING FUNCTIONS ---
def pre_process(x):
    x_rh = tf.gather(x, indices=RHAND_IDX[:21], axis=1)
    y_rh = tf.gather(x, indices=RHAND_IDX[21:42], axis=1)
    z_rh = tf.gather(x, indices=RHAND_IDX[42:63], axis=1)
    rhand = tf.stack([x_rh, y_rh, z_rh], axis=-1)
    x_lh = tf.gather(x, indices=LHAND_IDX[:21], axis=1)
    y_lh = tf.gather(x, indices=LHAND_IDX[21:42], axis=1)
    z_lh = tf.gather(x, indices=LHAND_IDX[42:63], axis=1)
    lhand = tf.stack([x_lh, y_lh, z_lh], axis=-1)
    x_rp = tf.gather(x, indices=RPOSE_IDX[:5], axis=1)
    y_rp = tf.gather(x, indices=RPOSE_IDX[5:10], axis=1)
    z_rp = tf.gather(x, indices=RPOSE_IDX[10:15], axis=1)
    rpose_data = tf.stack([x_rp, y_rp, z_rp], axis=-1)
    x_lp = tf.gather(x, indices=LPOSE_IDX[:5], axis=1)
    y_lp = tf.gather(x, indices=LPOSE_IDX[5:10], axis=1)
    z_lp = tf.gather(x, indices=LPOSE_IDX[10:15], axis=1)
    lpose_data = tf.stack([x_lp, y_lp, z_lp], axis=-1)

    total_rhand_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(rhand), tf.float32))
    total_lhand_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(lhand), tf.float32))
    is_left_dominant = total_rhand_nans > total_lhand_nans

    dominant_hand_data_raw = tf.cond(is_left_dominant, lambda: tf.concat([1.0 - lhand[..., :1], lhand[..., 1:]], axis=-1), lambda: rhand)
    dominant_pose_data_raw = tf.cond(is_left_dominant, lambda: tf.concat([1.0 - lpose_data[..., :1], lpose_data[..., 1:]], axis=-1), lambda: rpose_data)

    dominant_hand_data_filled = tf.where(tf.math.is_nan(dominant_hand_data_raw), tf.zeros_like(dominant_hand_data_raw), dominant_hand_data_raw)
    dominant_pose_data_filled = tf.where(tf.math.is_nan(dominant_pose_data_raw), tf.zeros_like(dominant_pose_data_raw), dominant_pose_data_raw)

    mean_hand = tf.math.reduce_mean(dominant_hand_data_filled, axis=1, keepdims=True)
    std_hand = tf.math.reduce_std(dominant_hand_data_filled, axis=1, keepdims=True)
    hand_normalized = (dominant_hand_data_filled - mean_hand) / (std_hand + 1e-6)
    mean_pose = tf.math.reduce_mean(dominant_pose_data_filled, axis=1, keepdims=True)
    std_pose = tf.math.reduce_std(dominant_pose_data_filled, axis=1, keepdims=True)
    pose_normalized = (dominant_pose_data_filled - mean_pose) / (std_pose + 1e-6)

    processed_hand = tf.where(tf.math.is_nan(hand_normalized), tf.zeros_like(hand_normalized), hand_normalized)
    processed_pose = tf.where(tf.math.is_nan(pose_normalized), tf.zeros_like(pose_normalized), pose_normalized)
    combined_features = tf.concat([processed_hand, processed_pose], axis=1)

    current_frames = tf.shape(combined_features)[0]
    target_len = FRAME_LEN
    expected_landmarks = 21 + 5
    expected_coords = 3
    
    reshaped_for_resize = tf.reshape(combined_features, [1, current_frames, expected_landmarks * expected_coords])
    resized_features = tf.image.resize(reshaped_for_resize, [1, target_len], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    final_features_flat = tf.reshape(resized_features, [target_len, expected_landmarks * expected_coords])
    final_features_flat.set_shape([target_len, expected_landmarks * expected_coords])
    return final_features_flat

def decode_fn(record_bytes):
    schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in FEATURE_COLUMNS}
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)
    features = tf.io.parse_single_example(record_bytes, schema)
    phrase = features["phrase"]
    landmarks_list = [tf.sparse.to_dense(features[col]) for col in FEATURE_COLUMNS]
    landmarks = tf.transpose(tf.stack(landmarks_list))
    return landmarks, phrase

def convert_fn_ctc(landmarks, phrase_str):
    processed_landmarks = pre_process(landmarks)
    phrase_chars = tf.strings.bytes_split(phrase_str)
    phrase_ids_ctc = ctc_table.lookup(phrase_chars)
    label_length = tf.shape(phrase_ids_ctc)[0]
    padding_size = TARGET_MAXLEN - label_length
    padding_size = tf.maximum(0, padding_size)
    phrase_padded_ctc = tf.pad(phrase_ids_ctc, paddings=[[0, padding_size]], mode='CONSTANT', constant_values=tf.cast(pad_token_idx_for_padding, dtype=tf.int32))
    phrase_padded_ctc = phrase_padded_ctc[:TARGET_MAXLEN]
    phrase_padded_ctc.set_shape([TARGET_MAXLEN])
    input_length_val = FRAME_LEN // 4
    return processed_landmarks, phrase_padded_ctc, tf.cast(label_length, dtype=tf.int32), tf.cast(input_length_val, dtype=tf.int32)

# --- 2. MODEL ARCHITECTURE CLASSES ---
class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid, dropout_rate=0.2):
        super().__init__()
        self.input_projection = layers.Dense(num_hid, activation=None, use_bias=False, name="landmark_input_proj")
        self.layer_norm_proj = layers.LayerNormalization(name="landmark_proj_layernorm")
        self.proj_activation = layers.Activation('relu', name="landmark_proj_relu")
        self.conv1 = tf.keras.layers.Conv1D(num_hid, kernel_size=5, strides=2, padding="same", activation=None, use_bias=False, name="landmark_conv1")
        self.norm1 = layers.LayerNormalization(name="landmark_conv1_layernorm")
        self.act1 = layers.Activation('relu', name="landmark_conv1_relu")
        self.drop1 = layers.Dropout(dropout_rate, name="landmark_conv1_dropout")
        self.conv2 = tf.keras.layers.Conv1D(num_hid, kernel_size=5, strides=2, padding="same", activation=None, use_bias=False, name="landmark_conv2")
        self.norm2 = layers.LayerNormalization(name="landmark_conv2_layernorm")
        self.act2 = layers.Activation('relu', name="landmark_conv2_relu")
        self.drop2 = layers.Dropout(dropout_rate, name="landmark_conv2_dropout")
        self.conv3 = tf.keras.layers.Conv1D(num_hid, kernel_size=3, strides=1, padding="same", activation=None, use_bias=False, name="landmark_conv3_stride1")
        self.norm3 = layers.LayerNormalization(name="landmark_conv3_layernorm")
        self.act3 = layers.Activation('relu', name="landmark_conv3_relu")
        self.drop3 = layers.Dropout(dropout_rate, name="landmark_conv3_dropout")
    def call(self, x, training=False):
        x = self.input_projection(x); x = self.layer_norm_proj(x, training=training); x = self.proj_activation(x)
        x = self.conv1(x); x = self.norm1(x, training=training); x = self.act1(x); x = self.drop1(x, training=training)
        x = self.conv2(x); x = self.norm2(x, training=training); x = self.act2(x); x = self.drop2(x, training=training)
        x = self.conv3(x); x = self.norm3(x, training=training); x = self.act3(x); x = self.drop3(x, training=training)
        return x

class CTCPredictionModel(keras.Model):
    def __init__(self, num_hid, ctc_num_classes, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_hid = num_hid
        self.ctc_num_classes = ctc_num_classes
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.edit_dist_metric = keras.metrics.Mean(name="edit_dist")
        self.landmark_embedding = LandmarkEmbedding(num_hid=num_hid, dropout_rate=dropout_rate)
        self.encoder_bilstm1 = layers.Bidirectional(layers.LSTM(num_hid, return_sequences=True, dropout=dropout_rate), name="encoder_bilstm_1")
        self.encoder_bilstm2 = layers.Bidirectional(layers.LSTM(num_hid, return_sequences=True, dropout=dropout_rate), name="encoder_bilstm_2")
        self.ctc_output_dense = layers.Dense(self.ctc_num_classes, name="ctc_classifier_dense")
    def call(self, source_input, training=False):
        embedded_source = self.landmark_embedding(source_input, training=training)
        encoder_out = self.encoder_bilstm1(embedded_source, training=training)
        encoder_out = self.encoder_bilstm2(encoder_out, training=training)
        ctc_logits = self.ctc_output_dense(encoder_out)
        return ctc_logits

    def train_step(self, batch):
        source, y_true, label_length, input_length = batch
        y_true = tf.cast(y_true, dtype=tf.int32)
        input_length_for_loss = tf.cast(tf.reshape(input_length, [-1, 1]), dtype=tf.int32)
        label_length_for_loss = tf.cast(tf.reshape(label_length, [-1, 1]), dtype=tf.int32)

        with tf.GradientTape() as tape:
            y_pred_logits = self(source, training=True) 
            y_pred_logits_time_major = tf.transpose(y_pred_logits, perm=[1, 0, 2])
            y_pred_softmax_batch_major = tf.nn.softmax(y_pred_logits, axis=-1)

            raw_loss = tf.keras.backend.ctc_batch_cost(
                y_true, y_pred_softmax_batch_major, input_length_for_loss, label_length_for_loss
            )
            finite_loss_values = tf.boolean_mask(raw_loss, tf.math.is_finite(raw_loss))
            
            loss = tf.cond(
                tf.size(finite_loss_values) > 0,
                lambda: tf.reduce_mean(finite_loss_values),
                lambda: tf.constant(0.0) 
            )

        grads = tape.gradient(loss, self.trainable_variables)
        if not any(g is None for g in grads):
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_metric.update_state(loss)

        input_length_for_decode = tf.reshape(input_length, [-1])
        
        decoded_st_tuple_list, _ = tf.nn.ctc_greedy_decoder(
            inputs=y_pred_logits_time_major,
            sequence_length=input_length_for_decode,
            merge_repeated=True
        )
        decoded_predictions_st_tuple = decoded_st_tuple_list[0]
        decoded_predictions_sparse = tf.SparseTensor(
            indices=decoded_predictions_st_tuple.indices,
            values=decoded_predictions_st_tuple.values,
            dense_shape=decoded_predictions_st_tuple.dense_shape
        )
        
        y_true_for_sparse = tf.cast(y_true, dtype=tf.int64)
        label_length_for_sparse = tf.reshape(label_length, [-1])
        indices = tf.where(tf.sequence_mask(label_length_for_sparse, maxlen=tf.shape(y_true_for_sparse)[1]))
        values = tf.gather_nd(y_true_for_sparse, indices)
        
        max_len_in_batch = tf.cond(tf.reduce_sum(label_length_for_sparse) > 0,
                                   lambda: tf.reduce_max(label_length_for_sparse),
                                   lambda: tf.constant(0, dtype=tf.int32))

        is_empty_batch_labels = tf.reduce_sum(label_length_for_sparse) == 0
        
        def get_dense_shape_with_labels():
            return tf.stack([tf.cast(tf.shape(y_true_for_sparse)[0], dtype=tf.int64), 
                             tf.cast(max_len_in_batch, dtype=tf.int64)])
        def get_dense_shape_empty_labels():
            return tf.stack([tf.cast(tf.shape(y_true_for_sparse)[0], dtype=tf.int64), 
                             tf.constant(0, dtype=tf.int64)])

        dense_shape = tf.cond(is_empty_batch_labels, get_dense_shape_empty_labels, get_dense_shape_with_labels)
        y_true_sparse = tf.SparseTensor(indices, values, dense_shape)

        edit_dist = tf.edit_distance(decoded_predictions_sparse, y_true_sparse, normalize=False)
        
        finite_edit_dist_indices = tf.where(tf.math.is_finite(edit_dist))
        valid_edit_dist_values = tf.gather_nd(edit_dist, finite_edit_dist_indices)
        
        mean_edit_dist = tf.cond(
            tf.size(valid_edit_dist_values) > 0,
            lambda: tf.reduce_mean(valid_edit_dist_values),
            lambda: tf.constant(0.0, dtype=tf.float32) # Đảm bảo dtype
        )
        
        should_update_metric = tf.size(valid_edit_dist_values) > 0
        def update_op():
            self.edit_dist_metric.update_state(mean_edit_dist)
            return tf.constant(True) 
        def no_update_op():
            return tf.constant(False) 
        _ = tf.cond(should_update_metric, update_op, no_update_op)


        return {"loss": self.loss_metric.result(), "edit_dist": self.edit_dist_metric.result()}

    def test_step(self, batch):
        source, y_true, label_length, input_length = batch
        y_true = tf.cast(y_true, dtype=tf.int32)
        input_length_for_loss = tf.cast(tf.reshape(input_length, [-1, 1]), dtype=tf.int32)
        label_length_for_loss = tf.cast(tf.reshape(label_length, [-1, 1]), dtype=tf.int32)

        y_pred_logits = self(source, training=False)
        y_pred_logits_time_major = tf.transpose(y_pred_logits, perm=[1, 0, 2])
        y_pred_softmax_batch_major = tf.nn.softmax(y_pred_logits, axis=-1)
        
        raw_loss = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred_softmax_batch_major, input_length_for_loss, label_length_for_loss
        )
        finite_loss_values = tf.boolean_mask(raw_loss, tf.math.is_finite(raw_loss))
        
        loss = tf.cond(
            tf.size(finite_loss_values) > 0,
            lambda: tf.reduce_mean(finite_loss_values),
            lambda: tf.constant(0.0)
        )
        self.loss_metric.update_state(loss)

        input_length_for_decode = tf.reshape(input_length, [-1])
        decoded_st_tuple_list, _ = tf.nn.ctc_greedy_decoder(
            inputs=y_pred_logits_time_major,
            sequence_length=input_length_for_decode,
            merge_repeated=True
        )
        decoded_predictions_st_tuple = decoded_st_tuple_list[0]
        decoded_predictions_sparse = tf.SparseTensor(
            indices=decoded_predictions_st_tuple.indices,
            values=decoded_predictions_st_tuple.values,
            dense_shape=decoded_predictions_st_tuple.dense_shape
        )

        y_true_for_sparse = tf.cast(y_true, dtype=tf.int64)
        label_length_for_sparse = tf.reshape(label_length, [-1])
        indices = tf.where(tf.sequence_mask(label_length_for_sparse, maxlen=tf.shape(y_true_for_sparse)[1]))
        values = tf.gather_nd(y_true_for_sparse, indices)

        max_len_in_batch = tf.cond(tf.reduce_sum(label_length_for_sparse) > 0,
                                   lambda: tf.reduce_max(label_length_for_sparse),
                                   lambda: tf.constant(0, dtype=tf.int32))
                                   
        is_empty_batch_labels = tf.reduce_sum(label_length_for_sparse) == 0
        
        def get_dense_shape_with_labels():
            return tf.stack([tf.cast(tf.shape(y_true_for_sparse)[0], dtype=tf.int64), 
                             tf.cast(max_len_in_batch, dtype=tf.int64)])
        def get_dense_shape_empty_labels():
            return tf.stack([tf.cast(tf.shape(y_true_for_sparse)[0], dtype=tf.int64), 
                             tf.constant(0, dtype=tf.int64)])
        
        dense_shape = tf.cond(is_empty_batch_labels, get_dense_shape_empty_labels, get_dense_shape_with_labels)
        y_true_sparse = tf.SparseTensor(indices, values, dense_shape)
        
        edit_dist = tf.edit_distance(decoded_predictions_sparse, y_true_sparse, normalize=False)
        finite_edit_dist_indices = tf.where(tf.math.is_finite(edit_dist))
        valid_edit_dist_values = tf.gather_nd(edit_dist, finite_edit_dist_indices)

        mean_edit_dist = tf.cond(
            tf.size(valid_edit_dist_values) > 0,
            lambda: tf.reduce_mean(valid_edit_dist_values),
            lambda: tf.constant(0.0, dtype=tf.float32) # Đảm bảo dtype
        )
        
        should_update_metric = tf.size(valid_edit_dist_values) > 0
        def update_op_test():
            self.edit_dist_metric.update_state(mean_edit_dist)
            return tf.constant(True) 
        def no_update_op_test():
            return tf.constant(False) 
        _ = tf.cond(should_update_metric, update_op_test, no_update_op_test)
                
        return {"loss": self.loss_metric.result(), "edit_dist": self.edit_dist_metric.result()}

    def ctc_generate_greedy(self, source_input, ctc_num_to_char_map_local, blank_idx_local):
        batch_size_val = tf.shape(source_input)[0]
        static_batch_size = tf.get_static_value(batch_size_val)
        if static_batch_size is not None and static_batch_size == 0:
             return [""] * 0
        
        y_pred_logits = self(source_input, training=False)
        y_pred_logits_time_major = tf.transpose(y_pred_logits, perm=[1, 0, 2])

        input_len_val = FRAME_LEN // 4
        input_lengths_for_decode = tf.ones(shape=(batch_size_val,), dtype=tf.int32) * input_len_val
        
        decoded_st_tuple_list, _ = tf.nn.ctc_greedy_decoder(
            inputs=y_pred_logits_time_major,
            sequence_length=input_lengths_for_decode,
            merge_repeated=True
        )
        decoded_sparse = tf.SparseTensor(
            indices=decoded_st_tuple_list[0].indices,
            values=decoded_st_tuple_list[0].values,
            dense_shape=decoded_st_tuple_list[0].dense_shape
        )
        
        non_char_default_value = tf.cast(blank_idx_local + 100, dtype=tf.int64)
        
        try:
            decoded_dense_np = tf.sparse.to_dense(decoded_sparse, default_value=non_char_default_value).numpy()
        except AttributeError:
            if static_batch_size is not None:
                return ["<graph_mode_decode_issue_numpy>"] * static_batch_size
            else:
                return ["<graph_mode_decode_issue_numpy_unknown_batch>"]

        output_texts = []
        for i in range(decoded_dense_np.shape[0]):
            default_val_for_compare = non_char_default_value.numpy() if hasattr(non_char_default_value, 'numpy') else non_char_default_value
            sequence_indices = [idx for idx in decoded_dense_np[i] if idx != blank_idx_local and idx != default_val_for_compare]
            sequence_chars = [ctc_num_to_char_map_local.get(idx, '?') for idx in sequence_indices]
            output_texts.append("".join(sequence_chars))
        return output_texts
    @property
    def metrics(self): return [self.loss_metric, self.edit_dist_metric]

# --- 3. CALLBACKS ---
class DisplayOutputsCTC(keras.callbacks.Callback):
    def __init__(self, batch_data, ctc_n2c_map, blank_idx):
        super().__init__()
        if batch_data is not None:
            self.source_data = batch_data[0]; self.target_labels_padded = batch_data[1].numpy(); self.target_label_lengths = batch_data[2].numpy()
        else: self.source_data = None; self.target_labels_padded = None; self.target_label_lengths = None
        self.ctc_num_to_char_map_cb = ctc_n2c_map; self.blank_idx_cb = blank_idx
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0: return
        if self.source_data is None or self.target_labels_padded is None or self.target_label_lengths is None:
            print("DisplayOutputsCTC: No batch data for display."); return
        
        source_shape = tf.shape(self.source_data)
        batch_size_disp_tensor = source_shape[0]
        batch_size_disp_val = tf.get_static_value(batch_size_disp_tensor)
        if batch_size_disp_val is None:
            try:
                batch_size_disp_val = batch_size_disp_tensor.numpy()
            except AttributeError:
                print("DisplayOutputsCTC: Could not determine batch size for display.")
                return

        if batch_size_disp_val == 0 : return
        
        predictions_text = self.model.ctc_generate_greedy(self.source_data, self.ctc_num_to_char_map_cb, self.blank_idx_cb)
        print(f"\n--- Epoch {epoch+1} Sample Predictions (CTC) ---")
        for i in range(min(batch_size_disp_val, 3)):
            target_text = ""
            if self.target_label_lengths[i] > 0:
                true_label_indices = self.target_labels_padded[i, :self.target_label_lengths[i]]
                target_text_list = [self.ctc_num_to_char_map_cb.get(val, '?') for val in true_label_indices if val != pad_token_idx_for_padding and val < self.blank_idx_cb]
                target_text = "".join(target_text_list)
            
            if i < len(predictions_text):
                prediction_text = predictions_text[i]
                print(f"Target    : {target_text}"); print(f"Prediction: {prediction_text}\n")
            else:
                print(f"Target    : {target_text}"); print(f"Prediction: <error_generating_prediction_for_sample_{i}>\n")

        print(f"--- End of Sample Predictions (CTC) ---\n")

# --- 4. DATASET LOADING AND PREPARATION ---
TRAIN_CSV_PATH = '/kaggle/input/asl-fingerspelling/train.csv'
TFRECORDS_BASE_PATH = '/kaggle/input/pre-data-fsp/new_data/'
if not os.path.exists(TRAIN_CSV_PATH): exit("ERROR: Train CSV not found")
if not os.path.exists(TFRECORDS_BASE_PATH) or (os.path.isdir(TFRECORDS_BASE_PATH) and not os.listdir(TFRECORDS_BASE_PATH)): exit("ERROR: TFRecords directory not found or empty")
dataset_df = pd.read_csv(TRAIN_CSV_PATH); tf_records_paths = dataset_df.file_id.map(lambda x: os.path.join(TFRECORDS_BASE_PATH, f'{x}.tfrecord')).unique()
tf_records_paths = [path for path in tf_records_paths if os.path.exists(path)]
if not tf_records_paths: exit("No TFRecord files found.")
print(f"Found {len(tf_records_paths)} TFRecord files."); random.shuffle(tf_records_paths)
BATCH_SIZE = 64; TRAIN_SPLIT_RATIO = 0.9; EPOCHS = 100; LEARNING_RATE = 1e-3; NUM_HIDDEN_UNITS = 384; DROPOUT_RATE = 0.3
train_len_count = int(TRAIN_SPLIT_RATIO * len(tf_records_paths))
if len(tf_records_paths) > 1:
    if train_len_count == len(tf_records_paths) and train_len_count > 0: train_len_count -=1
    if train_len_count == 0 and len(tf_records_paths) > 0 : train_len_count = 1
elif len(tf_records_paths) == 1: train_len_count = 1
else: train_len_count = 0
print(f"Using {train_len_count} files for training, {len(tf_records_paths) - train_len_count} for validation.")
padded_shapes_ctc = ([FRAME_LEN, 26 * 3], [TARGET_MAXLEN], [], [])
train_ds = None
if train_len_count > 0:
    train_ds = tf.data.TFRecordDataset(tf_records_paths[:train_len_count]).shuffle(buffer_size=min(train_len_count * 10, 2048)).map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE).map(convert_fn_ctc, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(BATCH_SIZE, padded_shapes=padded_shapes_ctc, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
valid_ds = None; num_val_files = len(tf_records_paths) - train_len_count
if num_val_files > 0:
    valid_ds = tf.data.TFRecordDataset(tf_records_paths[train_len_count:]).map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE).map(convert_fn_ctc, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(BATCH_SIZE, padded_shapes=padded_shapes_ctc, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

# --- 5. MODEL TRAINING ---
callbacks_list_train = [TerminateOnNaN()]; batch_for_display_cb_ctc = None
if valid_ds:
    try: batch_for_display_cb_ctc = next(iter(valid_ds.take(1)))
    except (StopIteration, tf.errors.OutOfRangeError): print("Warning: Validation dataset empty or could not get sample for DisplayOutputsCTC."); pass
if batch_for_display_cb_ctc is None and train_ds:
    try: batch_for_display_cb_ctc = next(iter(train_ds.take(1)))
    except (StopIteration, tf.errors.OutOfRangeError): print("Warning: Training dataset also empty or could not get sample for DisplayOutputsCTC."); pass
if batch_for_display_cb_ctc is not None:
    display_cb_instance_ctc = DisplayOutputsCTC(batch_for_display_cb_ctc, ctc_num_to_char, BLANK_INDEX); callbacks_list_train.append(display_cb_instance_ctc)
else: print("DisplayOutputsCTC callback disabled as no suitable batch could be obtained.")
monitor_metric = 'val_edit_dist' if valid_ds else 'edit_dist'
callbacks_list_train.append(EarlyStopping(monitor=monitor_metric, patience=20, restore_best_weights=True, verbose=1, mode='min'))
callbacks_list_train.append(ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=7, min_lr=1e-6, verbose=1, mode='min'))
model_instance_ctc = CTCPredictionModel(num_hid=NUM_HIDDEN_UNITS, ctc_num_classes=CTC_NUM_CLASSES, dropout_rate=DROPOUT_RATE)
optimizer_instance = keras.optimizers.Adam(learning_rate=LEARNING_RATE); model_instance_ctc.compile(optimizer=optimizer_instance)
sample_for_build_ctc = None
if train_ds:
    try: sample_batch_train = next(iter(train_ds.take(1))); sample_for_build_ctc = sample_batch_train[0]
    except (StopIteration, tf.errors.OutOfRangeError): print("Warning: Training dataset empty for model build."); pass
    except Exception as e: print(f"Warning: Error taking sample from training dataset for model build: {e}"); pass
if sample_for_build_ctc is None and valid_ds:
    try: sample_batch_valid = next(iter(valid_ds.take(1))); sample_for_build_ctc = sample_batch_valid[0]
    except (StopIteration, tf.errors.OutOfRangeError): print("Warning: Validation dataset empty for model build."); pass
    except Exception as e: print(f"Warning: Error taking sample from validation dataset for model build: {e}"); pass
if sample_for_build_ctc is not None:
    try: model_instance_ctc(sample_for_build_ctc); model_instance_ctc.summary()
    except Exception as e: print(f"Error during model build or summary: {e}")
else: print("Could not obtain a sample to build the CTC model and print summary.")
print(f"CTC Model compiled. Training: Batch={BATCH_SIZE}, LR={LEARNING_RATE}, Hidden={NUM_HIDDEN_UNITS}, CTC Classes={CTC_NUM_CLASSES}")
if train_ds:
    history_obj_ctc = model_instance_ctc.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, callbacks=callbacks_list_train, verbose=1)
    print("\nCTC Training finished.")
    if history_obj_ctc and history_obj_ctc.history:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.plot(history_obj_ctc.history['loss'], label='Train Loss')
        if valid_ds and 'val_loss' in history_obj_ctc.history: plt.plot(history_obj_ctc.history['val_loss'], label='Val Loss')
        plt.title('CTC Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2); plt.plot(history_obj_ctc.history['edit_dist'], label='Train Edit Dist')
        if valid_ds and 'val_edit_dist' in history_obj_ctc.history: plt.plot(history_obj_ctc.history['val_edit_dist'], label='Val Edit Dist')
        plt.title('CTC Edit Distance'); plt.xlabel('Epoch'); plt.ylabel('Edit Dist'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        try: plt.savefig("ctc_training_history.png"); print("CTC Training history plot saved to ctc_training_history.png")
        except Exception as e: print(f"Could not save CTC training history plot: {e}")
    model_instance_ctc.save_weights("ctc_model_final.weights.h5"); print("CTC Model weights saved.")
else: print("Training dataset is not available. Cannot start CTC training.")
print("Script finished.")

