import gc
import json
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa


def get_strategy(device='TPU-VM'):
    if "TPU" in device:
        try:
            tpu = 'local' if device == 'TPU-VM' else None
            print("Connecting to TPU...")
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            IS_TPU = True
            print("Connected to TPU.")
        except:
            print("TPU not available. Falling back to GPU...")
            device = "GPU"

    if device == "GPU":
        IS_TPU = False
        ngpu = len(tf.config.experimental.list_physical_devices('GPU'))
        if ngpu > 1:
            print("Using multiple GPUs.")
            strategy = tf.distribute.MirroredStrategy()
        elif ngpu == 1:
            print("Using single GPU.")
            strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        else:
            print("No GPU available. Falling back to CPU...")
            device = "CPU"

    if device == "CPU":
        print("Using CPU.")
        strategy = tf.distribute.OneDeviceStrategy("CPU")
        IS_TPU = False

    AUTO = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f"REPLICAS: {REPLICAS}")

    return strategy, REPLICAS, IS_TPU

STRATEGY, N_REPLICAS, IS_TPU = get_strategy()

LOAD_WEIGHTS = False


with open ("/kaggle/input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    char_to_num = json.load(f)

pad_token = '^'
pad_token_idx = 59

char_to_num[pad_token] = pad_token_idx

num_to_char = {j:i for i,j in char_to_num.items()}
df = pd.read_csv('/kaggle/input/asl-fingerspelling/train.csv')

LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE] + [f'x_face_{i}' for i in LIP]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE] + [f'y_face_{i}' for i in LIP]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE] + [f'z_face_{i}' for i in LIP]

SEL_COLS = X + Y + Z
FRAME_LEN = 128
MAX_PHRASE_LENGTH = 64

LIP_IDX_X   = [i for i, col in enumerate(SEL_COLS)  if  "face" in col and "x" in col]
RHAND_IDX_X = [i for i, col in enumerate(SEL_COLS)  if "right" in col and "x" in col]
LHAND_IDX_X = [i for i, col in enumerate(SEL_COLS)  if  "left" in col and "x" in col]
RPOSE_IDX_X = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in RPOSE and "x" in col]
LPOSE_IDX_X = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in LPOSE and "x" in col]

LIP_IDX_Y   = [i for i, col in enumerate(SEL_COLS)  if  "face" in col and "y" in col]
RHAND_IDX_Y = [i for i, col in enumerate(SEL_COLS)  if "right" in col and "y" in col]
LHAND_IDX_Y = [i for i, col in enumerate(SEL_COLS)  if  "left" in col and "y" in col]
RPOSE_IDX_Y = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in RPOSE and "y" in col]
LPOSE_IDX_Y = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in LPOSE and "y" in col]

LIP_IDX_Z   = [i for i, col in enumerate(SEL_COLS)  if  "face" in col and "z" in col]
RHAND_IDX_Z = [i for i, col in enumerate(SEL_COLS)  if "right" in col and "z" in col]
LHAND_IDX_Z = [i for i, col in enumerate(SEL_COLS)  if  "left" in col and "z" in col]
RPOSE_IDX_Z = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in RPOSE and "z" in col]
LPOSE_IDX_Z = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in LPOSE and "z" in col]


SUPPLEMENTAL_RHM = np.load("/kaggle/input/supplemently-data-set/mean_std/rh_mean.npy")
SUPPLEMENTAL_LHM = np.load("/kaggle/input/supplemently-data-set/mean_std/lh_mean.npy")
SUPPLEMENTAL_RPM = np.load("/kaggle/input/supplemently-data-set/mean_std/rp_mean.npy")
SUPPLEMENTAL_LPM = np.load("/kaggle/input/supplemently-data-set/mean_std/lp_mean.npy")
SUPPLEMENTAL_LIPM = np.load("/kaggle/input/supplemently-data-set/mean_std/lip_mean.npy")

SUPPLEMENTAL_RHS = np.load("/kaggle/input/supplemently-data-set/mean_std/rh_std.npy")
SUPPLEMENTAL_LHS = np.load("/kaggle/input/supplemently-data-set/mean_std/lh_std.npy")
SUPPLEMENTAL_RPS = np.load("/kaggle/input/supplemently-data-set/mean_std/rp_std.npy")
SUPPLEMENTAL_LPS = np.load("/kaggle/input/supplemently-data-set/mean_std/lp_std.npy")
SUPPLEMENTAL_LIPS = np.load("/kaggle/input/supplemently-data-set/mean_std/lip_std.npy")

TRAIN_RHM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/rh_mean.npy")
TRAIN_LHM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lh_mean.npy")
TRAIN_RPM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/rp_mean.npy")
TRAIN_LPM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lp_mean.npy")
TRAIN_LIPM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lip_mean.npy")

TRAIN_RHS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/rh_std.npy")
TRAIN_LHS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lh_std.npy")
TRAIN_RPS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/rp_std.npy")
TRAIN_LPS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lp_std.npy")
TRAIN_LIPS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lip_std.npy")


RHM = (TRAIN_RHM + SUPPLEMENTAL_RHM) / 2
LHM = (TRAIN_LHM + SUPPLEMENTAL_LHM) / 2
RPM = (TRAIN_RPM + SUPPLEMENTAL_RPM) / 2
LPM = (TRAIN_LPM + SUPPLEMENTAL_LPM) / 2
LIPM = (TRAIN_LIPM + SUPPLEMENTAL_LIPM) / 2

RHS = (TRAIN_RHS + SUPPLEMENTAL_RHS) / 2
LHS = (TRAIN_LHS + SUPPLEMENTAL_LHS) / 2
RPS = (TRAIN_RPS + SUPPLEMENTAL_RPS) / 2
LPS = (TRAIN_LPS + SUPPLEMENTAL_LPS) / 2
LIPS = (TRAIN_LIPS + SUPPLEMENTAL_LIPS) / 2





"""RHM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/rh_mean.npy")
LHM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lh_mean.npy")
RPM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/rp_mean.npy")
LPM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lp_mean.npy")
LIPM = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lip_mean.npy")

RHS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/rh_std.npy")
LHS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lh_std.npy")
RPS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/rp_std.npy")
LPS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lp_std.npy")
LIPS = np.load("/kaggle/input/aslfr-dataset-tfrecords/mean_std/lip_std.npy")"""




def load_relevant_data_subset(pq_path):
    return pd.read_parquet(pq_path, columns=SEL_COLS)

file_id = df.file_id.iloc[0]
inpdir = "/kaggle/input/asl-fingerspelling/train_landmarks"
pqfile = f"{inpdir}/{file_id}.parquet"
seq_refs = df.loc[df.file_id == file_id]
seqs = load_relevant_data_subset(pqfile)

seq_id = seq_refs.sequence_id.iloc[0]
frames = seqs.iloc[seqs.index == seq_id]
phrase = str(df.loc[df.sequence_id == seq_id].phrase.iloc[0])




@tf.function()
def resize_pad(x):
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN-tf.shape(x)[0]], [0, 0], [0, 0]]), constant_values=float("NaN"))
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x

@tf.function(jit_compile=True)
def pre_process0(x):
    lip_x = tf.gather(x, LIP_IDX_X, axis=1)
    lip_y = tf.gather(x, LIP_IDX_Y, axis=1)
    lip_z = tf.gather(x, LIP_IDX_Z, axis=1)

    rhand_x = tf.gather(x, RHAND_IDX_X, axis=1)
    rhand_y = tf.gather(x, RHAND_IDX_Y, axis=1)
    rhand_z = tf.gather(x, RHAND_IDX_Z, axis=1)
    
    lhand_x = tf.gather(x, LHAND_IDX_X, axis=1)
    lhand_y = tf.gather(x, LHAND_IDX_Y, axis=1)
    lhand_z = tf.gather(x, LHAND_IDX_Z, axis=1)

    rpose_x = tf.gather(x, RPOSE_IDX_X, axis=1)
    rpose_y = tf.gather(x, RPOSE_IDX_Y, axis=1)
    rpose_z = tf.gather(x, RPOSE_IDX_Z, axis=1)
    
    lpose_x = tf.gather(x, LPOSE_IDX_X, axis=1)
    lpose_y = tf.gather(x, LPOSE_IDX_Y, axis=1)
    lpose_z = tf.gather(x, LPOSE_IDX_Z, axis=1)
    
    lip   = tf.concat([lip_x[..., tf.newaxis], lip_y[..., tf.newaxis], lip_z[..., tf.newaxis]], axis=-1)
    rhand = tf.concat([rhand_x[..., tf.newaxis], rhand_y[..., tf.newaxis], rhand_z[..., tf.newaxis]], axis=-1)
    lhand = tf.concat([lhand_x[..., tf.newaxis], lhand_y[..., tf.newaxis], lhand_z[..., tf.newaxis]], axis=-1)
    rpose = tf.concat([rpose_x[..., tf.newaxis], rpose_y[..., tf.newaxis], rpose_z[..., tf.newaxis]], axis=-1)
    lpose = tf.concat([lpose_x[..., tf.newaxis], lpose_y[..., tf.newaxis], lpose_z[..., tf.newaxis]], axis=-1)
    
    hand = tf.concat([rhand, lhand], axis=1)
    hand = tf.where(tf.math.is_nan(hand), 0.0, hand)
    mask = tf.math.not_equal(tf.reduce_sum(hand, axis=[1, 2]), 0.0)

    lip = lip[mask]
    rhand = rhand[mask]
    lhand = lhand[mask]
    rpose = rpose[mask]
    lpose = lpose[mask]

    return rhand, lhand, lpose, rpose, lip

tf.function()
def pre_process1(rhand, lhand, lpose, rpose, lip):
    lip   = (resize_pad(lip) - LIPM) / LIPS
    rhand = (resize_pad(rhand) - RHM) / RHS
    lhand = (resize_pad(lhand) - LHM) / LHS
    rpose = (resize_pad(rpose) - RPM) / RPS
    lpose = (resize_pad(lpose) - LPM) / LPS


    #if augment:
        #print('augment')
        #rhand, lhand, lpose, rpose, lip = augment_fn((rhand, lhand, lpose, rpose, lip), always=always, max_len=max_len)
    
    x = tf.concat([rhand, lhand, lpose, rpose, lip], axis=1)
    s = tf.shape(x)
    #x = tf.reshape(x, (s[0], s[1]*s[2]))
    x = tf.concat([x[..., 0], x[..., 1], x[..., 2]], axis=-1)
    x = tf.where(tf.math.is_nan(x), 0.0, x)
    return x





"""def compute_velocity(x):
    # Calculate velocity: difference between consecutive frames
    dx = x[:, 1:] - x[:, :-1]
    # Pad zeros to maintain original shape
    dx_padded = tf.pad(dx, [[0, 0], [0, 1], [0, 0]])

    return dx_padded

def compute_acceleration(x):
    # Calculate acceleration: difference between consecutive velocities
    dv = x[:, 1:] - x[:, :-1]
    # Pad zeros to maintain original shape
    dv_padded = tf.pad(dv, [[0, 0], [0, 1], [0, 0]])

    return dv_padded

def get_vel_acc(landmarks):
    velocity = compute_velocity(landmarks)
    acceleration = compute_acceleration(velocity)

    # Concatenate original landmarks, velocity, and acceleration
    processed_landmarks = tf.concat([landmarks, velocity, acceleration], axis=1)

    return processed_landmarks




def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis,  keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))



@tf.function()
def pre_process1(rhand, lhand, lpose, rpose, lip):
    lip   = (resize_pad(lip) - tf_nan_mean(lip,axis=[0,1], keepdims=True)) / tf_nan_std(lip,axis=[0,1], keepdims=True)
    rhand = (resize_pad(rhand) - tf_nan_mean(rhand,axis=[0,1], keepdims=True)) / tf_nan_std(rhand,axis=[0,1], keepdims=True)
    lhand = (resize_pad(lhand) - tf_nan_mean(lhand,axis=[0,1], keepdims=True)) / tf_nan_std(lhand,axis=[0,1], keepdims=True)
    rpose = (resize_pad(rpose) - tf_nan_mean(rpose,axis=[0,1], keepdims=True)) / tf_nan_std(rpose,axis=[0,1], keepdims=True)
    lpose = (resize_pad(lpose) - tf_nan_mean(lpose,axis=[0,1], keepdims=True)) / tf_nan_std(lpose,axis=[0,1], keepdims=True)
    
    #if augment:
        #print('augment')
        #rhand, lhand, lpose, rpose, lip = augment_fn((rhand, lhand, lpose, rpose, lip), always=always, max_len=max_len)
    
    x = tf.concat([rhand, lhand, lpose, rpose, lip], axis=1)
    #x = get_vel_acc(x)
    s = tf.shape(x)
    #x = tf.reshape(x, (s[0], s[1]*s[2]))
    x = tf.concat([x[..., 0], x[..., 1], x[..., 2]], axis=-1)
    x = tf.where(tf.math.is_nan(x), 0.0, x)
    return x"""

input_shpae=[12,3]


def decode_fn(record_bytes):
    schema = {
        "lip": tf.io.VarLenFeature(tf.float32),
        "rhand": tf.io.VarLenFeature(tf.float32),
        "lhand": tf.io.VarLenFeature(tf.float32),
        "rpose": tf.io.VarLenFeature(tf.float32),
        "lpose": tf.io.VarLenFeature(tf.float32),
        "phrase": tf.io.VarLenFeature(tf.int64)
    }
    x = tf.io.parse_single_example(record_bytes, schema)

    lip = tf.reshape(tf.sparse.to_dense(x["lip"]), (-1, 40, 3))
    rhand = tf.reshape(tf.sparse.to_dense(x["rhand"]), (-1, 21, 3))
    lhand = tf.reshape(tf.sparse.to_dense(x["lhand"]), (-1, 21, 3))
    rpose = tf.reshape(tf.sparse.to_dense(x["rpose"]), (-1, 5, 3))
    lpose = tf.reshape(tf.sparse.to_dense(x["lpose"]), (-1, 5, 3))
    phrase = tf.sparse.to_dense(x["phrase"])

    return lip, rhand, lhand, rpose, lpose, phrase

def pre_process_fn(lip, rhand, lhand, rpose, lpose, phrase):
    phrase = tf.pad(phrase, [[0, MAX_PHRASE_LENGTH-tf.shape(phrase)[0]]], constant_values=pad_token_idx)
    return pre_process1(rhand, lhand, lpose, rpose, lip), phrase
    
tffiles = [f"/kaggle/input/aslfr-dataset-tfrecords/tfds/{file_id}.tfrecord" for file_id in df.file_id.unique()]
val_len = 1#int(0.05 * len(pqfiles))
train_batch_size = 32
val_batch_size = 32

train_dataset =  tf.data.TFRecordDataset(tffiles[val_len:]).prefetch(tf.data.AUTOTUNE).shuffle(5000).map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE).map(pre_process_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(train_batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset =  tf.data.TFRecordDataset(tffiles[:val_len]).prefetch(tf.data.AUTOTUNE).map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE).map(pre_process_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(val_batch_size).prefetch(tf.data.AUTOTUNE)

batch = next(iter(val_dataset))
batch[0].shape, batch[1].shape


foo=next(iter(train_dataset))











LAYER_NORM_EPS = 1e-6

# Dropout
EMBEDDING_DROPOUT = 0.15

MLP_RATIO = 2

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)
# Activations
GELU = tf.keras.activations.gelu




def scaled_dot_product(q,k,v, softmax, attention_mask):
    #calculates Q . K(transpose)
    qkt = tf.matmul(q,k,transpose_b=True)
    #caculates scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1],dtype=tf.float32))
    scaled_qkt = qkt/dk
    softmax = softmax(scaled_qkt, mask=attention_mask)
    z = tf.matmul(softmax,v)
    #shape: (m,Tx,depth), same shape as q,k,v
    return z

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_of_heads,dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, q, k, v, attention_mask=None):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq(q)
            K = self.wk(k)
            V = self.wv(v)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax, attention_mask))

        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        multi_head_attention = self.dropout(multi_head_attention)
        return multi_head_attention


class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, 
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer='glorot_uniform',
        name='', **kwargs):
        super().__init__(name=name,**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

def Conv1DBlock(channel_size,
          kernel_size,
          dilation_rate=1,
          drop_rate=0.0,
          expand_ratio=2,
          se_ratio=0.25,
          activation='swish',
          name=None):
    '''
    efficient conv1d block, @hoyso48
    '''
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def TransformerBlock(dim=256, num_heads=6, expand=4, attn_dropout=0.3, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadSelfAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Dense(inputs.shape[-1], activation='relu')(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Dense(dim*expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(inputs.shape[-1], use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    
    return apply

def positional_encoding(maxlen, num_hid):
        depth = num_hid/2
        positions = tf.range(maxlen, dtype = tf.float32)[..., tf.newaxis]
        depths = tf.range(depth, dtype = tf.float32)[np.newaxis, :]/depth
        angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
        angle_rads = tf.linalg.matmul(positions, angle_rates)
        pos_encoding = tf.concat(
          [tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
          axis=-1)
        return pos_encoding


def CTCLoss(labels, logits):
    label_length = tf.reduce_sum(tf.cast(labels != pad_token_idx, tf.int32), axis=-1)
    logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
    loss = tf.nn.ctc_loss(
            labels=labels,
            logits=logits,
            label_length=label_length,
            logit_length=logit_length,
            blank_index=pad_token_idx,
            logits_time_major=False
        )
    loss = tf.reduce_mean(loss)
    return loss


INPUT_SHAPE = [128,276]


def get_model(dim = 384):
    inp = tf.keras.Input(INPUT_SHAPE)
    x = inp
    
    x = tf.keras.layers.Masking(mask_value=0.0)(x)
    x = tf.keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x) + positional_encoding(INPUT_SHAPE[0], dim)
    x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)
    
    x = Conv1DBlock(dim, 17, drop_rate=0.4)(x)
    x = Conv1DBlock(dim, 11, drop_rate=0.4)(x)
    x = Conv1DBlock(dim,  7, drop_rate=0.4)(x)
    x = Conv1DBlock(dim,  7, drop_rate=0.4)(x)
    x = TransformerBlock(dim*2, num_heads=3, expand=2)(x)
    
    x = Conv1DBlock(dim, 17, drop_rate=0.4)(x)
    x = Conv1DBlock(dim, 11, drop_rate=0.4)(x)
    x = Conv1DBlock(dim,  5, drop_rate=0.4)(x)
    x = Conv1DBlock(dim,  3, drop_rate=0.4)(x)
    x = TransformerBlock(512, num_heads=4, expand=2)(x)
    
    x = Conv1DBlock(dim, 17, drop_rate=0.4)(x)
    x = Conv1DBlock(dim,  7, drop_rate=0.4)(x)
    x = Conv1DBlock(dim,  5, drop_rate=0.4)(x)
    x = Conv1DBlock(dim,  3, drop_rate=0.4)(x)
    x = TransformerBlock(256, num_heads=8, expand=2)(x)
    
    x = Conv1DBlock(dim, 11, drop_rate=0.45)(x)
    x = Conv1DBlock(dim,  7, drop_rate=0.45)(x)
    x = Conv1DBlock(dim,  5, drop_rate=0.45)(x)
    x = Conv1DBlock(dim,  3, drop_rate=0.45)(x)
    x = TransformerBlock(dim, expand=2)(x)
    
    x = tf.keras.layers.Dense(dim*2,activation='relu',name='top_conv')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(len(char_to_num))(x)

    model = tf.keras.Model(inp, x)

    loss = CTCLoss
    
    # Adam Optimizer
    optimizer = tfa.optimizers.RectifiedAdam(sma_threshold=4)
    optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)

    model.compile(loss=loss, optimizer=optimizer)

    return model

tf.keras.backend.clear_session()
model = get_model()

model.summary()


tf.keras.backend.clear_session()
with STRATEGY.scope():
    model = get_model()



k = model(foo[0])


model.summary()


def num_to_char_fn(y):
    return [num_to_char.get(x, "") for x in y]

@tf.function()
def decode_phrase(pred):
    x = tf.argmax(pred, axis=1)
    diff = tf.not_equal(x[:-1], x[1:])
    adjacent_indices = tf.where(diff)[:, 0]
    x = tf.gather(x, adjacent_indices)
    mask = x != pad_token_idx
    x = tf.boolean_mask(x, mask, axis=0)
    return x

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    output_text = []
    for result in pred:
        result = "".join(num_to_char_fn(decode_phrase(result).numpy()))
        output_text.append(result)
    return output_text



# A callback class to output a few transcriptions during training
class CallbackEval(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        model.save_weights("model.h5")
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = "".join(num_to_char_fn(label.numpy()))
                targets.append(label)
        print("-" * 100)
        # for i in np.random.randint(0, len(predictions), 2):
        for i in range(32):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}, len: {len(predictions[i])}")
            print("-" * 100)

# Callback function to check transcription on the val set.
validation_callback = CallbackEval(val_dataset.take(1))


N_EPOCHS = 70
N_WARMUP_EPOCHS = 10
LR_MAX = 1e-3
WD_RATIO = 0.05
WARMUP_METHOD = "exp"


def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=N_EPOCHS):
    
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max
    
def plot_lr_schedule(lr_schedule, epochs):
    fig = plt.figure(figsize=(20, 10))
    plt.plot([None] + lr_schedule + [None])
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels) # set tick step to 1 and let x axis start at 1
    
    # Increase y-limit for better readability
    plt.ylim([0, max(lr_schedule) * 1.1])
    
    # Title
    schedule_info = f'start: {lr_schedule[0]:.1E}, max: {max(lr_schedule):.1E}, final: {lr_schedule[-1]:.1E}'
    plt.title(f'Step Learning Rate Schedule, {schedule_info}', size=18, pad=12)
    
    # Plot Learning Rates
    for x, val in enumerate(lr_schedule):
        if epochs <= 40 or x % 5 == 0 or x is epochs - 1:
            if x < len(lr_schedule) - 1:
                if lr_schedule[x - 1] < val:
                    ha = 'right'
                else:
                    ha = 'left'
            elif x == 0:
                ha = 'right'
            else:
                ha = 'left'
            plt.plot(x + 1, val, 'o', color='black');
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f'{val:.1E}', xy=(x + 1, val + offset_y), size=12, ha=ha)
    
    plt.xlabel('Epoch', size=16, labelpad=5)
    plt.ylabel('Learning Rate', size=16, labelpad=5)
    plt.grid()
    plt.show()

# Learning rate for encoder
LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.50) for step in range(N_EPOCHS)]
# Plot Learning Rate Schedule
plot_lr_schedule(LR_SCHEDULE, epochs=N_EPOCHS)
# Learning Rate Callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=0)

# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio=WD_RATIO):
        self.step_counter = 0
        self.wd_ratio = wd_ratio
    
    def on_epoch_begin(self, epoch, logs=None):
        model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}')


"""history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=N_EPOCHS,
    callbacks=[
        validation_callback,
        lr_callback,
        WeightDecayCallback(),
    ]
)"""


models_path = [
    '/kaggle/input/ctc-trained-v-5/model_weights_on_epoch_30.h5',
    '/kaggle/input/ctc-trained-v-5/model_weights_on_epoch_40.h5',
    '/kaggle/input/ctc-trained-v-5/model_weights_on_epoch_50.h5',
    '/kaggle/input/ctc-trained-v-5/model_weights_on_epoch_80.h5',
    
]


models = [get_model() for _ in models_path]
for model,path in zip(models,models_path):
    model.load_weights(path)


averaged_weights = []
for w_1,w_2,w_3,w_4 in zip(models[0].get_weights(), models[1].get_weights(), models[2].get_weights(), models[3].get_weights()):
    averaged_weights.append((w_1+w_2+w_3+w_4)/4)






model.set_weights(averaged_weights)


LOAD_WEIGHTS = True
if LOAD_WEIGHTS:
    model.load_weights("/kaggle/input/ctc-trained-v-5/last_model.h5")
    print(f'Successfully Loaded Pretrained Weights')





class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(SEL_COLS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        # Preprocess Data
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, len(SEL_COLS))), lambda: tf.identity(x))
        x = x[0]
        x = pre_process0(x)
        x = pre_process1(*x)
        x = tf.reshape(x, INPUT_SHAPE)
        x = x[None]
        x = self.model(x, training=False)
        x = x[0]
        x = decode_phrase(x)
        x = tf.cond(tf.shape(x)[0] == 0, lambda: tf.zeros(1, tf.int64), lambda: tf.identity(x))
        x = tf.one_hot(x, 59)
        return {'outputs': x}

tflitemodel_base = TFLiteModel(model)
pred = tflitemodel_base(frames)["outputs"]



"".join(num_to_char_fn(decode_phrase(pred).numpy())),pred.shape


keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflitemodel_base)
keras_model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]#, tf.lite.OpsSet.SELECT_TF_OPS]
keras_model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
keras_model_converter.target_spec.supported_types = [tf.float16]
tflite_model = keras_model_converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
with open('inference_args.json', "w") as f:
    json.dump({"selected_columns" : SEL_COLS}, f)
    
!zip submission.zip  './model.tflite' './inference_args.json'


with open ("inference_args.json", "r") as f:
    SEL_COLS = json.load(f)["selected_columns"]
    
def load_relevant_data_subset(pq_path):
    return pd.read_parquet(pq_path, columns=SEL_COLS)

def create_data_gen(file_ids, y_mul=1):
    def gen():
        for file_id in file_ids:
            pqfile = f"{inpdir}/{file_id}.parquet"
            seq_refs = df.loc[df.file_id == file_id]
            seqs = load_relevant_data_subset(pqfile)

            for seq_id in seq_refs.sequence_id:
                x = seqs.iloc[seqs.index == seq_id].to_numpy()
                y = str(df.loc[df.sequence_id == seq_id].phrase.iloc[0])
                
                r_nonan = np.sum(np.sum(np.isnan(x[:, RHAND_IDX_X]), axis = 1) == 0)
                l_nonan = np.sum(np.sum(np.isnan(x[:, LHAND_IDX_X]), axis = 1) == 0)
                no_nan = max(r_nonan, l_nonan)
                
                if y_mul*len(y)<no_nan:
                    yield x, y
    return gen

pqfiles = df.file_id.unique()
val_len = int(0.05 * len(pqfiles))

test_dataset = tf.data.Dataset.from_generator(create_data_gen(pqfiles[:val_len], 0),
    output_signature=(tf.TensorSpec(shape=(None, len(SEL_COLS)), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.string))
).prefetch(buffer_size=2000)


interpreter = tf.lite.Interpreter("model.tflite")

REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

with open ("/kaggle/input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {j:i for i,j in character_map.items()}

prediction_fn = interpreter.get_signature_runner(REQUIRED_SIGNATURE)

for frame, target in test_dataset.skip(100).take(10):
    output = prediction_fn(inputs=frame)
    prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
    target = target.numpy().decode("utf-8")
    print("pred =", prediction_str, "; target =", target)


%%timeit -n 10
output = prediction_fn(inputs=frame)


from Levenshtein import distance

scores = []

for i, (frame, target) in tqdm(enumerate(test_dataset.take(1000))):
    output = prediction_fn(inputs=frame)
    prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
    target = target.numpy().decode("utf-8")
    score = (len(target) - distance(prediction_str, target)) / len(target)
    scores.append(score)
    if i % 100 == 0:
        print(np.sum(scores) / len(scores))
    
scores = np.array(scores)
print(np.sum(scores) / len(scores))
model_no=190





from Levenshtein import distance

scores = []

for i, (frame, target) in tqdm(enumerate(test_dataset.take(1000))):
    output = prediction_fn(inputs=frame)
    prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
    target = target.numpy().decode("utf-8")
    score = (len(target) - distance(prediction_str, target)) / len(target)
    scores.append(score)
    if i % 100 == 0:
        print(np.sum(scores) / len(scores))
    
scores = np.array(scores)
print(np.sum(scores) / len(scores))
model_no=190
















