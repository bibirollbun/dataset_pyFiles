# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import shutil
import pyarrow.parquet as pq
import tensorflow as tf
import json
import matplotlib
import matplotlib.pyplot as plt
import random

from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.notebook import tqdm
from matplotlib import animation, rc


import numpy as np # linear algebra
import pandas as pd 


dataset_df = pd.read_csv('/kaggle/input/asl-fingerspelling/train.csv')
tf_records = dataset_df.file_id.map(lambda x: f'/kaggle/input/pre-data-fsp/new_data/{x}.tfrecord').unique()
print(f"List of {len(tf_records)} TFRecord files.")


LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE
X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]
FEATURE_COLUMNS = X + Y + Z
RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]


print("Sá»‘ cá»™t (features):", len(FEATURE_COLUMNS) + 1)  # +1 vÃ¬ cÃ³ 'phrase'
print("TÃªn cÃ¡c cá»™t:", FEATURE_COLUMNS + ["phrase"])


# HÃ m Ä‘á»ƒ parse má»™t record
def _parse_function(example_proto):
    # Táº¡o dict schema Ä‘á»ƒ giáº£i mÃ£
    feature_description = {
        col: tf.io.VarLenFeature(tf.float32) for col in FEATURE_COLUMNS
    }
    feature_description["phrase"] = tf.io.FixedLenFeature([], tf.string)

    return tf.io.parse_single_example(example_proto, feature_description)

# Ä�á»�c file .tfrecord (thay báº±ng tÃªn file thá»±c táº¿ cá»§a báº¡n)
tfrecord_path = '/kaggle/input/pre-data-fsp/new_data/1019715464.tfrecord'
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
parsed_dataset = raw_dataset.map(_parse_function)
# In dá»¯ liá»‡u
for i, parsed_record in enumerate(parsed_dataset.take(1)): 
    print(f"\nğŸ§¾ Record {i+1}")
    print(len(parsed_record.keys()))
    for key in FEATURE_COLUMNS:
        values = tf.sparse.to_dense(parsed_record[key])
        print(f"{key}: {values.numpy().shape}")
    print(f"phrase: {parsed_record['phrase'].numpy().decode('utf-8')}")


FRAME_LEN = 128


import json
import tensorflow as tf # Ensure tf is imported for StaticHashTable later

# Load original character to number mapping
with open("/kaggle/input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    char_to_num_orig = json.load(f)
VOCAB_SIZE = len(char_to_num_orig)
BLANK_LABEL_IDX = VOCAB_SIZE  
NUM_CTC_CLASSES = VOCAB_SIZE + 1 # e.g., 60

# Create num_to_char based on the original mapping
num_to_char = {j:i for i,j in char_to_num_orig.items()}

# For padding target labels in convert_fn, we can use a value like 0 (space)
# as long as label_length is accurate.
# Or, define a specific PAD_VALUE for labels if space is critical and distinct from padding.
# For this example, let's assume label_length correctly handles it, and padding with 0 is fine.
# The char_to_num_orig already contains ' ' mapped to 0.

# char_to_num will be the original mapping
char_to_num = char_to_num_orig

print(f"Original Vocabulary Size (VOCAB_SIZE): {VOCAB_SIZE}")
print(f"CTC Blank Label Index (BLANK_LABEL_IDX): {BLANK_LABEL_IDX}")
print(f"Total CTC Prediction Classes (NUM_CTC_CLASSES): {NUM_CTC_CLASSES}")
print("num_to_char mapping (first 5 entries):")
for i in range(min(5, len(num_to_char))):
    print(f"  {i}: {num_to_char.get(i, 'N/A')}")
# Max phrase length for padding labels
TARGET_MAXLEN = 64
PAD_TOKEN_LABEL_VALUE = 0 # Using space (index 0) for padding labels, ensure label_length is accurate


def resize_pad(x):
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN-tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x

# Detect the dominant hand from the number of NaN values.
# Dominant hand will have less NaN values since it is in frame moving.
def pre_process(x):
    print(x.shape)
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)
    
    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)
    
    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)
    
    # For dominant hand
    if rnans > lnans:
        hand = lhand
        pose = lpose
        
        hand_x = hand[:, 0*(len(LHAND_IDX)//3) : 1*(len(LHAND_IDX)//3)]
        hand_y = hand[:, 1*(len(LHAND_IDX)//3) : 2*(len(LHAND_IDX)//3)]
        hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
        hand = tf.concat([1-hand_x, hand_y, hand_z], axis=1)
        
        pose_x = pose[:, 0*(len(LPOSE_IDX)//3) : 1*(len(LPOSE_IDX)//3)]
        pose_y = pose[:, 1*(len(LPOSE_IDX)//3) : 2*(len(LPOSE_IDX)//3)]
        pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
        pose = tf.concat([1-pose_x, pose_y, pose_z], axis=1)
    else:
        hand = rhand
        pose = rpose
    
    hand_x = hand[:, 0*(len(LHAND_IDX)//3) : 1*(len(LHAND_IDX)//3)]
    hand_y = hand[:, 1*(len(LHAND_IDX)//3) : 2*(len(LHAND_IDX)//3)]
    hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
    hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]], axis=-1)
    
    mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
    std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
    hand = (hand - mean) / std

    pose_x = pose[:, 0*(len(LPOSE_IDX)//3) : 1*(len(LPOSE_IDX)//3)]
    pose_y = pose[:, 1*(len(LPOSE_IDX)//3) : 2*(len(LPOSE_IDX)//3)]
    pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
    pose = tf.concat([pose_x[..., tf.newaxis], pose_y[..., tf.newaxis], pose_z[..., tf.newaxis]], axis=-1)
    
    x = tf.concat([hand, pose], axis=1)
    x = resize_pad(x)
    
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
    print(x.shape)
    return x


def decode_fn(record_bytes):
    # step 1: create schema
    schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in FEATURE_COLUMNS}
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)

    # step 2: Parse record
    features = tf.io.parse_single_example(record_bytes, schema)
    print(features["x_left_hand_0"])
    # step 3: get sequences
    phrase = features["phrase"]


    # step 4:  SparseTensor -> Dense
    landmarks = [tf.sparse.to_dense(features[COL]) for COL in FEATURE_COLUMNS]
    print("_____________________________________________________________________")
    print(landmarks[0])
    # step 5: 
    landmarks = tf.transpose(landmarks)
    return landmarks, phrase


table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=list(char_to_num.keys()), # Use the original char_to_num
        values=list(char_to_num.values()),
    ),
    default_value=tf.constant(-1), # Or a more robust error value if a char is not found
    name="character_lookup_table"
)

def convert_fn(landmarks, phrase_str_tensor):
    # phrase_str_tensor is a scalar string tensor
    phrase_chars = tf.strings.bytes_split(phrase_str_tensor)
    
    # Calculate actual label length BEFORE padding
    label_length = tf.shape(phrase_chars)[0]
    
    phrase_indices = table.lookup(phrase_chars)
    
    # Pad phrase_indices to TARGET_MAXLEN
    # Important: CTC loss needs labels to be in [0, NUM_CTC_CLASSES - 2]
    # Pad with a value that can be handled by label_length, e.g., 0 (space)
    phrase_indices_padded = tf.pad(phrase_indices,
                                   paddings=[[0, TARGET_MAXLEN - tf.shape(phrase_indices)[0]]],
                                   mode='CONSTANT',
                                   constant_values=PAD_TOKEN_LABEL_VALUE) # e.g., space character index 0
    
    # Apply pre_process function to the landmarks.
    processed_landmarks = pre_process(landmarks) # pre_process must be defined and accessible
    
    return processed_landmarks, phrase_indices_padded, tf.cast(label_length, tf.int32)


a = tf.strings.bytes_split("test bytes split")
print(a)
a = table.lookup(a)
print(a)


indices = [[0],[2]]
values = [1.0, 3.0]
dense_shape = [3]
# Táº¡o SparseTensor
sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
# Chuyá»ƒn SparseTensor thÃ nh DenseTensor
dense_tensor = tf.sparse.to_dense(sparse_tensor)
# In káº¿t quáº£
print(dense_tensor)


batch_size = 64
train_len = int(0.8 * len(tf_records))

# AUTOTUNE for prefetch
AUTOTUNE = tf.data.AUTOTUNE

# Note: decode_fn processes TFRecord into (landmarks, phrase_string)
# convert_fn processes (landmarks, phrase_string) into (processed_landmarks, phrase_indices_padded, label_length)

train_ds = tf.data.TFRecordDataset(tf_records[:train_len]) \
    .map(decode_fn, num_parallel_calls=AUTOTUNE) \
    .map(convert_fn, num_parallel_calls=AUTOTUNE) \
    .batch(batch_size) \
    .prefetch(buffer_size=AUTOTUNE) \
    .cache()

valid_ds = tf.data.TFRecordDataset(tf_records[train_len:]) \
    .map(decode_fn, num_parallel_calls=AUTOTUNE) \
    .map(convert_fn, num_parallel_calls=AUTOTUNE) \
    .batch(batch_size) \
    .prefetch(buffer_size=AUTOTUNE) \
    .cache()


print(train_ds)
print(valid_ds)



for batch in train_ds.take(1):
    print("Landmarks shape:", batch[0].shape)
    print("Phrase indices shape:", batch[1].shape)
    print("Label lengths shape:", batch[2].shape)

# Cell 17 (example modification)
for i, batch in enumerate(train_ds.take(5)):
    inputs, label_indices, label_lengths = batch
    print(f"Batch {i + 1}:")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Label indices shape: {label_indices.shape}")
    print(f"  Label lengths shape: {label_lengths.shape}")
    print(f"  Number of elements in batch: {inputs.shape[0]}")
    print(f"  Processed landmark sequence length: {inputs.shape[1]}") # Should be FRAME_LEN (e.g. 128)
    print(f"  Processed landmark feature dim: {inputs.shape[2]}") # Should be 78
    print(f"  Padded label max length: {label_indices.shape[1]}") # Should be TARGET_MAXLEN (e.g. 64)
    print(f"  Example label lengths in batch: {label_lengths[:5].numpy()}")
    print("-" * 50)


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=FRAME_LEN):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D( # Giáº£m chiá»�u dÃ i 1 láº§n
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=1, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=1, padding="same", activation="relu"
        )
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


# Táº¡o má»™t input máº«u (batch size = 1, sequence length = 5)
sample_input = tf.constant([[5, 2, 8, 0, 1],[1, 2, 3, 4, 5]])

# Khá»Ÿi táº¡o lá»›p embedding
token_emb_layer = TokenEmbedding(num_vocab=1000, maxlen=100, num_hid=16)  # 16 chiá»�u dá»… quan sÃ¡t

# Cháº¡y forward pass
output = token_emb_layer(sample_input)

# In káº¿t quáº£
print("Shape cá»§a output:", output.shape)
print("Output (Ä‘oáº¡n Ä‘áº§u):\n", output[0, :2])  # In 2 token Ä‘áº§u Ä‘á»ƒ xem rÃµ


ENCODER_OUTPUT_SEQ_LEN = FRAME_LEN // 2
FRAME_LEN = 128
BLANK_LABEL_IDX = 59 # Giáº£ sá»­ VOCAB_SIZE = 59
NUM_CTC_CLASSES = 60



class TransformerEncoder(layers.Layer): # Báº¡n cáº§n Ä‘á»‹nh nghÄ©a lá»›p nÃ y tá»« code gá»‘c
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerEncoderStack(keras.layers.Layer):
    def __init__(self, num_layers, num_hid, num_head, num_feed_forward):
        super().__init__()
        self.enc_layers = [
            TransformerEncoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers)
        ]

    def call(self, x, training=False):
        for layer in self.enc_layers:
            x = layer(x, training=training)
        return x
class CTCModel(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=FRAME_LEN,
        encoder_output_seqlen_param=ENCODER_OUTPUT_SEQ_LEN,
        num_layers_enc=4,
        num_ctc_classes=NUM_CTC_CLASSES,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="ctc_loss")
        self.cer_metric = keras.metrics.Mean(name="cer")

        self.landmark_embedding = LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.pos_emb_encoder = layers.Embedding(input_dim=encoder_output_seqlen_param, output_dim=num_hid)
        self.encoder_output_actual_seqlen = encoder_output_seqlen_param

        self.encoder = TransformerEncoderStack(
            num_layers=num_layers_enc,
            num_hid=num_hid,
            num_head=num_head,
            num_feed_forward=num_feed_forward
        )
        self.classifier = layers.Dense(num_ctc_classes)

    def call(self, source, training=False):
        x = self.landmark_embedding(source)
        batch_size = tf.shape(x)[0]
        current_encoder_seq_len = tf.shape(x)[1]

        positions = tf.range(start=0, limit=current_encoder_seq_len, delta=1)
        positions = tf.broadcast_to(positions, [batch_size, current_encoder_seq_len])
        pos_embeddings = self.pos_emb_encoder(positions)
        x = x + pos_embeddings

        x_encoded = self.encoder(x, training=training)
        logits = self.classifier(x_encoded)
        return logits

    @property
    def metrics(self):
        return [self.loss_metric, self.cer_metric]

    def ctc_batch_loss(self, y_true, y_pred, input_length, label_length):
        y_pred_time_major = tf.transpose(y_pred, perm=[1, 0, 2])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred_time_major,
            label_length=label_length,
            logit_length=input_length,
            blank_index=BLANK_LABEL_IDX,
            logits_time_major=True
        )
        label_length_float = tf.cast(label_length, dtype=loss.dtype)
        normalized_loss = tf.math.divide_no_nan(loss, label_length_float)
        return tf.reduce_mean(normalized_loss)

    def _dense_to_sparse(self, dense_tensor, sequence_lengths):
        mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(dense_tensor)[1])
        indices = tf.cast(tf.where(mask), tf.int64)
        values = tf.cast(tf.boolean_mask(dense_tensor, mask), tf.int64)
        dense_shape = tf.cast(tf.shape(dense_tensor), tf.int64)
        return tf.SparseTensor(indices, values, dense_shape)

    def train_step(self, batch):
        source, target_labels, target_label_lengths = batch
        batch_size_tf = tf.shape(source)[0]
        ctc_input_length = tf.fill([batch_size_tf], self.encoder_output_actual_seqlen)
        ctc_input_length = tf.cast(ctc_input_length, tf.int32)
        target_label_lengths_int32 = tf.cast(target_label_lengths, tf.int32)

        with tf.GradientTape() as tape:
            preds_logits = self(source, training=True)
            loss = self.ctc_batch_loss(target_labels, preds_logits, ctc_input_length, target_label_lengths_int32)
        
        # Khá»‘i code nÃ y Ä‘Ã£ Ä‘Æ°á»£c sá»­a lá»—i thá»¥t lá»�
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        
        decoded_sparse_hyp, _ = tf.nn.ctc_greedy_decoder(
            tf.transpose(preds_logits, perm=[1, 0, 2]),
            ctc_input_length
        )
        sparse_true_truth = self._dense_to_sparse(target_labels, target_label_lengths_int32)
        
        edit_dist = tf.edit_distance(decoded_sparse_hyp[0], sparse_true_truth, normalize=True)
        self.cer_metric.update_state(tf.reduce_mean(edit_dist))
        
        return {"ctc_loss": self.loss_metric.result(), "cer": self.cer_metric.result()}

    def test_step(self, batch):
        source, target_labels, target_label_lengths = batch
        batch_size_tf = tf.shape(source)[0]
        ctc_input_length = tf.fill([batch_size_tf], self.encoder_output_actual_seqlen)
        ctc_input_length = tf.cast(ctc_input_length, tf.int32)
        target_label_lengths_int32 = tf.cast(target_label_lengths, tf.int32)

        preds_logits = self(source, training=False)
        loss = self.ctc_batch_loss(target_labels, preds_logits, ctc_input_length, target_label_lengths_int32)
        self.loss_metric.update_state(loss)

        decoded_sparse_hyp, _ = tf.nn.ctc_greedy_decoder(
            tf.transpose(preds_logits, perm=[1, 0, 2]), 
            ctc_input_length
        )
        sparse_true_truth = self._dense_to_sparse(target_labels, target_label_lengths_int32)
        edit_dist = tf.edit_distance(decoded_sparse_hyp[0], sparse_true_truth, normalize=True)
        self.cer_metric.update_state(tf.reduce_mean(edit_dist))
        return {"ctc_loss": self.loss_metric.result(), "cer": self.cer_metric.result()}

    def generate(self, source):
        preds_logits = self(source, training=False)
        batch_size = tf.shape(source)[0]
        ctc_input_length = tf.fill([batch_size], self.encoder_output_actual_seqlen)
        ctc_input_length = tf.cast(ctc_input_length, tf.int32)
        
        decoded_sparse, _ = tf.nn.ctc_greedy_decoder(
            tf.transpose(preds_logits, perm=[1, 0, 2]),
            sequence_length=ctc_input_length
        )
        decoded_dense = tf.sparse.to_dense(decoded_sparse[0], default_value=tf.cast(BLANK_LABEL_IDX, tf.int64))
        return decoded_dense



class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [batch_size[..., tf.newaxis], tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target, training):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att, training = training))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out, training = training) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out, training = training))
        return ffn_out_norm


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch_data, idx_to_token_map):
        # batch_data is (source, target_labels, target_label_lengths)
        self.batch_source = batch_data[0]
        self.batch_target_labels = batch_data[1]
        self.batch_target_lengths = batch_data[2] # Store target lengths
        self.idx_to_char = idx_to_token_map
        # self.target_start_token_idx is no longer needed for CTC generation

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 4 != 0:
            return
        
        source_for_display = self.batch_source # Use the stored batch source
        
        # --- CORRECTED CALL TO GENERATE ---
        preds_indices_dense = self.model.generate(source_for_display) # Now takes only source
        # preds_indices_dense is a dense tensor (batch_size, max_decoded_len)
        # Values are character indices, padded with BLANK_LABEL_IDX (e.g., 59)
        # --- END CORRECTION ---

        preds_indices_np = preds_indices_dense.numpy()

        print(f"\n--- Epoch {epoch+1} Sample Predictions ---")
        # Use min(5, actual_batch_size_of_display_batch) for looping
        num_samples_to_display = min(5, source_for_display.shape[0]) 

        for i in range(num_samples_to_display):
            # Get true label
            target_len = self.batch_target_lengths[i].numpy()
            target_text_indices = self.batch_target_labels[i, :target_len].numpy()
            target_text = "".join([self.idx_to_char.get(idx, '?') for idx in target_text_indices])
            
            # Get predicted label from dense tensor
            prediction_row = preds_indices_np[i]
            
            # Filter out BLANK_LABEL_IDX (and potentially PAD_TOKEN_LABEL_VALUE if it somehow appears)
            # to form the predicted string
            predicted_chars = []
            for idx in prediction_row:
                if idx == BLANK_LABEL_IDX: # Stop if we hit padding from ctc_decoder to_dense
                    break 
                char = self.idx_to_char.get(idx)
                if char is not None: # Ensure char exists (handles case where an unexpected index appears)
                    predicted_chars.append(char)
            prediction_text = "".join(predicted_chars)

            print(f"Target:     {target_text}")
            print(f"Prediction: {prediction_text}\n")
        print(f"--- End Epoch {epoch+1} Sample Predictions ---\n")


from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf 

# Prepare a batch for DisplayOutputs
val_iter = iter(valid_ds)
display_batch_data = next(val_iter) 

display_cb = DisplayOutputs(display_batch_data, num_to_char) 

# ---- MODIFICATION HERE ----
earlystop_cb = EarlyStopping(
    monitor='val_ctc_loss', 
    patience=40,            
    restore_best_weights=True,
    mode='min'  # Explicitly tell EarlyStopping to minimize this metric
)
# ---- END MODIFICATION ----

ctc_model = CTCModel(
    num_hid=200,
    num_head=4,
    num_feed_forward=400,
    source_maxlen=FRAME_LEN,
    encoder_output_seqlen_param=ENCODER_OUTPUT_SEQ_LEN, # Truyá»�n vÃ o giÃ¡ trá»‹ Ä‘Ã£ cáº­p nháº­t
    num_layers_enc=2,
    num_ctc_classes=NUM_CTC_CLASSES
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

ctc_model.compile(optimizer=optimizer, jit_compile=False) 

history = ctc_model.fit(train_ds, 
                        validation_data=valid_ds, 
                        callbacks=[display_cb, earlystop_cb], 
                        epochs=100)


# Váº½ biá»ƒu Ä‘á»“ cho metric CER
plt.plot(history.history['cer'])
plt.plot(history.history['val_cer'])
plt.title('Model CER (Character Error Rate)')
plt.ylabel('CER')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Báº¡n cÅ©ng cÃ³ thá»ƒ váº½ biá»ƒu Ä‘á»“ cho loss náº¿u muá»‘n
plt.plot(history.history['ctc_loss'])
plt.plot(history.history['val_ctc_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


dataset_df = pd.read_csv('/kaggle/input/asl-fingerspelling/supplemental_metadata.csv')
tf_records = dataset_df.file_id.map(lambda x: f'/kaggle/input/test-fsp/test/{x}.tfrecord').unique()
print(f"List of {len(tf_records)} TFRecord files.")


test_ds = tf.data.TFRecordDataset(tf_records).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()


results = ctc_model.evaluate(test_ds)
print("Loss:", results[0])
print("Edit distance:", results[1])


