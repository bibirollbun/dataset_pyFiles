import warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np, seaborn as sns
import math, json, os, random
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.model_selection import GroupKFold, KFold
from sklearn.cluster import KMeans
from tensorflow.keras import layers as L

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


debug = False


train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')


train.columns


train.shape


train.info()


train.head()


test.columns


test.shape


test.info()


#sneak peak
print(sample_sub.shape)
if ~sample_sub.isnull().values.any(): print('No missing values')
sample_sub.head()


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.kdeplot(train['signal_to_noise'], fill=True, ax=ax[0])
sns.countplot(x='SN_filter', data=train, ax=ax[1])

ax[0].set_title('Signal/Noise Distribution')
ax[1].set_title('Signal/Noise Filter Distribution');


print(f"Samples with signal_to_noise greater than 1: {len(train.loc[(train['signal_to_noise'] > 1 )])}")
print(f"Samples with SN_filter = 1: {len(train.loc[(train['SN_filter'] == 1 )])}")
print(f"Samples with signal_to_noise greater than 1, but SN_filter == 0: {len(train.loc[(train['signal_to_noise'] > 1) & (train['SN_filter'] == 0)])}")


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))
    return bpps_arr

def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))
    return bpps_arr

def read_bpps_nb(df):
    #mean and std from https://www.kaggle.com/symyksr/openvaccine-deepergcn 
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr 

train['bpps_sum'] = read_bpps_sum(train)
test['bpps_sum'] = read_bpps_sum(test)
train['bpps_max'] = read_bpps_max(train)
test['bpps_max'] = read_bpps_max(test)
train['bpps_nb'] = read_bpps_nb(train)
test['bpps_nb'] = read_bpps_nb(test)

#sanity check
train.head()


fig, ax = plt.subplots(3, figsize=(15, 10))
sns.kdeplot(np.array(train['bpps_max'].to_list()).reshape(-1),
            color="Blue", ax=ax[0], label='Train')
sns.kdeplot(np.array(test[test['seq_length'] == 107]['bpps_max'].to_list()).reshape(-1),
            color="Red", ax=ax[0], label='Public test')
sns.kdeplot(np.array(test[test['seq_length'] == 130]['bpps_max'].to_list()).reshape(-1),
            color="Green", ax=ax[0], label='Private test')
sns.kdeplot(np.array(train['bpps_sum'].to_list()).reshape(-1),
            color="Blue", ax=ax[1], label='Train')
sns.kdeplot(np.array(test[test['seq_length'] == 107]['bpps_sum'].to_list()).reshape(-1),
            color="Red", ax=ax[1], label='Public test')
sns.kdeplot(np.array(test[test['seq_length'] == 130]['bpps_sum'].to_list()).reshape(-1),
            color="Green", ax=ax[1], label='Private test')
sns.kdeplot(np.array(train['bpps_nb'].to_list()).reshape(-1),
            color="Blue", ax=ax[2], label='Train')
sns.kdeplot(np.array(test[test['seq_length'] == 107]['bpps_nb'].to_list()).reshape(-1),
            color="Red", ax=ax[2], label='Public test')
sns.kdeplot(np.array(test[test['seq_length'] == 130]['bpps_nb'].to_list()).reshape(-1),
            color="Green", ax=ax[2], label='Private test')

ax[0].set_title('Distribution of bpps_max')
ax[1].set_title('Distribution of bpps_sum')
ax[2].set_title('Distribution of bpps_nb')
plt.tight_layout();


aug_df = pd.read_csv("/kaggle/input/aug-data/aug_data.csv")
display(aug_df.head())


aug_df.columns


def aug_data(df):
    # 1. Safety Check: If data is already augmented, return it as is.
    if 'cnt' in df.columns:
        print("Dataframe already contains 'cnt'. Skipping augmentation to avoid duplicates.")
        return df

    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]
                         
    # 2. Cleanup: Remove structure/loop AND potential colliding columns from target
    # This ensures the merge doesn't create cnt_x/cnt_y
    cols_to_drop = ['structure', 'predicted_loop_type', 'cnt', 'log_gamma', 'score']
    for col in cols_to_drop:
        if col in target_df.columns:
            del target_df[col]
            
    # 3. Merge
    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')

    # 4. Map 'cnt'
    # We drop_duplicates just in case, though usually IDs are unique in aug_df
    cnt_map = new_df[['id','cnt']].drop_duplicates('id').set_index('id')['cnt'].to_dict()
    df['cnt'] = df['id'].map(cnt_map)
    
    df['log_gamma'] = 100
    df['score'] = 1.0
    
    # 5. Concatenate
    df = pd.concat([df, new_df[df.columns]], ignore_index=True)
    
    return df

train = aug_data(train)
test = aug_data(test)


if debug:
    train = train[:200]
    test = test[:200]


LOOK_BACK = 107
STEPS_AHEAD = 2
PAD_AMT = LOOK_BACK + STEPS_AHEAD
TARGET_COLS = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea = np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
    bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
    bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]
    return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea], 2)


# # --- Metrics for Model Training (Tensors) ---
# def keras_r2_score(y_true, y_pred):
#     """Calculates R^2 using TensorFlow ops."""
#     SS_res =  K.sum(K.square(y_true - y_pred)) 
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
#     return (1 - SS_res/(SS_tot + K.epsilon()))

# def keras_nse_score(y_true, y_pred):
#     """Calculates NSE using TensorFlow ops."""
#     return keras_r2_score(y_true, y_pred)


# --- Keras Metrics (For Training Progress) ---
def keras_r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def mcrmse(y_true, y_pred):
    colwise_mse = K.mean(K.square(y_true - y_pred), axis=0)
    return K.mean(K.sqrt(colwise_mse))

# --- Numpy Metrics (For Final Report) ---
def print_metrics(y_true, y_pred, label="Eval"):
    # Flatten for global stats, or keep 2D for column stats
    # Here we flatten to get a general sense of performance
    y_t_flat = y_true.flatten()
    y_p_flat = y_pred.flatten()
    
    print(f"\n--- {label} Metrics ---")
    print(f"MAE:  {mean_absolute_error(y_t_flat, y_p_flat):.5f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_t_flat, y_p_flat)):.5f}")
    print(f"R²:   {r2_score(y_t_flat, y_p_flat):.5f}")
    print("-----------------------")


def get_aligned_data(df, feature_fn, weighted=False):
    """
    TRAINING Data: Shifts targets so X[t] predicts Y[t+2].
    """
    raw_feats = feature_fn(df) # (N, 107, 6)
    raw_targets = np.array(df[TARGET_COLS].values.tolist()).transpose((0, 2, 1)) # (N, 68, 5)
    
    if weighted:
        w = np.log1p(df['signal_to_noise'].values + 0.1) / 2
    else:
        w = np.ones(len(df))

    # X: 0 to 66 (Length 66)
    valid_len = 68 - STEPS_AHEAD
    X = raw_feats[:, :valid_len, :]
    
    # Y: 2 to 68 (Length 66)
    Y = raw_targets[:, STEPS_AHEAD:68, :]
    
    return X, Y, w

def get_test_data(df, feature_fn):
    """
    TEST Data: Pads start with 2 zeros so output[0] corresponds to target[0].
    """
    raw_feats = feature_fn(df)
    
    # Pad 2 steps at the beginning of the sequence dimension (axis 1)
    # Shape becomes (N, 109, 6) for public and (N, 132, 6) for private
    padded_feats = np.pad(raw_feats, ((0,0), (STEPS_AHEAD, 0), (0,0)), mode='constant')
    
    return padded_feats


class PositionalEmbedding(L.Layer):
    def __init__(self, max_len=200, embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        # The embedding layer that holds the weights
        self.pos_emb = L.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, features)
        # 1. Get dynamic sequence length
        seq_len = tf.shape(inputs)[1]
        
        # 2. Create range [0, 1, 2, ... seq_len-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        
        # 3. Lookup embeddings
        # Output shape: (seq_len, embed_dim)
        embedded_positions = self.pos_emb(positions)
        
        # 4. Broadcast to batch size is automatic in Keras during addition
        return tf.expand_dims(embedded_positions, axis=0)


class SqueezeAndExcite1D(L.Layer):
    def __init__(self, channels, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        # Squeeze: Global Average Pooling to get a summary of the whole sequence
        self.global_pool = L.GlobalAveragePooling1D()
        # Excitation: 2 Dense layers to learn weights for each channel
        self.dense1 = L.Dense(channels // ratio, activation='relu')
        self.dense2 = L.Dense(channels, activation='sigmoid') # Output 0.0 to 1.0

    def call(self, inputs):
        # inputs shape: (Batch, Time, Channels)
        
        # 1. Squeeze: (Batch, Channels)
        x = self.global_pool(inputs)
        
        # 2. Excite: Learn importances
        x = self.dense1(x)
        x = self.dense2(x) # Shape: (Batch, Channels)
        
        # 3. Reshape for broadcasting: (Batch, 1, Channels)
        x = tf.expand_dims(x, axis=1)
        
        # 4. Scale the original input
        return inputs * x


def build_model(embed_dim=128, hidden_dim=256, dropout=0.3):
    # --- Input Definitions ---
    inputs = L.Input(shape=(None, 6), name="features")
    
    # --- Technique 1: Positional Embeddings (Fixed) ---
    # We use our custom layer here
    pos_embed = PositionalEmbedding(max_len=200, embed_dim=embed_dim)(inputs)
    
    # --- Feature Processing ---
    # Split Categorical vs Numerical using Lambda layers (Safe for Keras)
    cat_input = L.Lambda(lambda x: tf.cast(x[:, :, :3], dtype='int32'))(inputs)
    num_input = L.Lambda(lambda x: x[:, :, 3:])(inputs)

    # Embed the Sequence Bases
    feat_embed = L.Embedding(input_dim=14, output_dim=embed_dim)(cat_input)
    feat_embed = L.Reshape((-1, 3 * embed_dim))(feat_embed)
    
    # Project features to 'embed_dim' size so we can add them to position embeddings
    x = L.Concatenate()([feat_embed, num_input])
    x = L.Dense(embed_dim, activation='linear')(x) 
    
    # ADD Position Embedding to Feature Embedding
    # This fuses "What is it?" with "Where is it?"
    x = L.Add()([x, pos_embed])
    x = L.SpatialDropout1D(0.2)(x)
    
    # --- Technique 2: Residual Bi-GRU Stack ---
    
    # Layer 1: Standard Bi-GRU
    x1 = L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))(x)
    # TECHNIQUE: Recalibrate features after RNN
    x1 = SqueezeAndExcite1D(channels=hidden_dim*2)(x1) 
    
    # Layer 2: Bi-GRU with Residual Connection
    x2 = L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))(x1)
    x2 = SqueezeAndExcite1D(channels=hidden_dim*2)(x2) # TECHNIQUE
    # SKIP CONNECTION: Add x1 to x2
    x_res2 = L.Add()([x1, x2])
    x_res2 = L.LayerNormalization()(x_res2) 
    
    # Layer 3: Bi-LSTM with Residual Connection
    x3 = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))(x_res2)
    x3 = SqueezeAndExcite1D(channels=hidden_dim*2)(x3) # TECHNIQUE

    # SKIP CONNECTION: Add x_res2 to x3
    x_res3 = L.Add()([x_res2, x3])
    x_res3 = L.LayerNormalization()(x_res3)
    
    # --- Head ---
    x_out = L.Dense(128, activation='swish')(x_res3)
    x_out = L.Dropout(0.2)(x_out)
    out = L.Dense(5, activation='linear')(x_out)
    
    model = models.Model(inputs=inputs, outputs=out)
    
    # Optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.96, staircase=True
    )
    
    # Ensure you have your custom metrics defined or removed
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=mcrmse)
    
    return model


model = build_model()
model.summary()


def train_and_predict_submission(train_df, test_df, FOLDS=5, EPOCHS=25, BATCH_SIZE=64):
    
    # 1. Prepare Test Data
    # Public: Length 107, Scored 68
    public_df = test_df.query("seq_length == 107").copy()
    X_pub = get_test_data(public_df, preprocess_inputs)
    
    # Private: Length 130, Scored 91
    private_df = test_df.query("seq_length == 130").copy()
    X_priv = get_test_data(private_df, preprocess_inputs)
    
    # Accumulators
    # Note: We predict the full length, but we will slice later
    pred_pub = np.zeros((len(X_pub), X_pub.shape[1], 5)) 
    pred_priv = np.zeros((len(X_priv), X_priv.shape[1], 5))
    
    # 2. Training Loop
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n--- Fold {fold+1}/{FOLDS} ---")
        
        # Split Train/Val
        t_sub = train_df.iloc[train_idx]
        v_sub = train_df.iloc[val_idx]
        v_sub = v_sub[v_sub['SN_filter'] == 1] # Validate on high quality
        
        # Get Aligned Data (Forecasting t+2)
        X_t, Y_t, W_t = get_aligned_data(t_sub, preprocess_inputs, weighted=True)
        X_v, Y_v, _   = get_aligned_data(v_sub, preprocess_inputs, weighted=False)
        
        # Build & Train
        model = build_model()
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(X_t, Y_t, sample_weight=W_t, validation_data=(X_v, Y_v),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)
        
        # Predict on Test
        # Because we added 2 padding steps at start:
        # Input Index 0 (Pad) -> Predicts Target 0
        # Input Index 1 (Pad) -> Predicts Target 1
        # Input Index 2 (Base0) -> Predicts Target 2
        # So the output array is perfectly aligned with targets starting at 0.
        pred_pub += model.predict(X_pub, batch_size=1024) / FOLDS
        pred_priv += model.predict(X_priv, batch_size=1024) / FOLDS
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
        
    return public_df, pred_pub, private_df, pred_priv


print("Starting Training and Inference...")
public_df, pub_preds_raw, private_df, priv_preds_raw = train_and_predict_submission(
    train, test, FOLDS=5, EPOCHS=60
)


def create_valid_submission(public_df, pub_preds, private_df, priv_preds):
    """
    Creates a submission file with the correct row count (457,953).
    Generates IDs for ALL positions (0 to seq_length), not just scored ones.
    """
    ids = []
    flat_preds = []
    
    # --- Process Public Test (Length 107) ---
    print("Formatting Public Data (Length 107)...")
    for i in range(len(public_df)):
        seq_id = public_df.iloc[i]['id']
        length = public_df.iloc[i]['seq_length'] # Should be 107
        
        # Take predictions for the FULL length
        # pub_preds[i] has shape (109, 5) due to padding. We take first 107.
        sample_preds = pub_preds[i, :length, :]
        
        for t in range(length):
            ids.append(f"{seq_id}_{t}")
            flat_preds.append(sample_preds[t])
            
    # --- Process Private Test (Length 130) ---
    print("Formatting Private Data (Length 130)...")
    for i in range(len(private_df)):
        seq_id = private_df.iloc[i]['id']
        length = private_df.iloc[i]['seq_length'] # Should be 130
        
        # Take predictions for the FULL length
        # priv_preds[i] has shape (132, 5) due to padding. We take first 130.
        sample_preds = priv_preds[i, :length, :]
        
        for t in range(length):
            ids.append(f"{seq_id}_{t}")
            flat_preds.append(sample_preds[t])
            
    return ids, np.array(flat_preds)


all_ids, all_vals = create_valid_submission(public_df, pub_preds_raw, private_df, priv_preds_raw)


submission = pd.DataFrame(all_vals, columns=TARGET_COLS)
submission['id_seqpos'] = all_ids


submission = submission.groupby('id_seqpos').mean().reset_index()

print(f"Rows after averaging duplicates: {len(submission)}")


try:
    sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
    
    # Merge: valid submission rows will be filled, missing ones (if any) become NaN
    final_sub = sample_sub[['id_seqpos']].merge(submission, on='id_seqpos', how='left')
    
    # Fill any NaNs with 0 (just in case)
    final_sub = final_sub.fillna(0)
    
    # Save
    final_sub.to_csv('submission.csv', index=False)
    print(f"Success! Generated submission with {len(final_sub)} rows.")
    print(final_sub.head())

except FileNotFoundError:
    print("Sample submission file not found. Saving generated dataframe directly.")
    submission.to_csv('submission.csv', index=False)
    print(f"Generated submission with {len(submission)} rows.")




