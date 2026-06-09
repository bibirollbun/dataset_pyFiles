!pip uninstall -y scikit-learn scikeras         
!pip install -q scikit-learn==1.5.2 scikeras==0.13.0 \
              --no-cache-dir --progress-bar off


# Import necessary libraries (re-importing for clarity in this standalone notebook)
import numpy as np
import pandas as pd
import os, cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Concatenate, Lambda, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')


# Load data
DATA_DIR = "/kaggle/input/nexar-collision-prediction"
train_df_full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_df_full['filename'] = train_df_full['id'].apply(lambda x: f"{int(x):05d}.mp4")
test_df['filename']       = test_df['id'].apply(lambda x: f"{int(x):05d}.mp4")

# Quick check that the first file really exists
first_path = os.path.join(DATA_DIR, "train", train_df_full.iloc[0]['filename'])
print("First video exists? ->", os.path.exists(first_path))

print(f"Total training videos: {len(train_df_full)}, Total test videos: {len(test_df)}")
print(train_df_full.head(2))


# Frame sampling function for event-focused window
def sample_event_window_frames(video_path, event_time=None, window=3.0, target_fps=15):
    """Sample frames from the last `window` seconds before event (or video end if no event)."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # use video FPS if available, otherwise assume 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames else 0
    
    # Determine sampling interval
    if event_time is None or np.isnan(event_time):
        # No event (negative video) -> use last `window` seconds of the video
        end_time = duration
    else:
        end_time = min(event_time, duration)
    start_time = max(0, end_time - window)
    
    # Calculate frame indices at target_fps within [start_time, end_time]
    num_frames = int(window * target_fps)
    if num_frames <= 0:
        cap.release()
        return np.array(frames)
    # We'll compute evenly spaced times (in seconds) in the interval
    times = np.linspace(start_time, end_time, num=num_frames, endpoint=False)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)  # position in milliseconds
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Test the sampling on a positive and a negative example
example_pos = train_df_full[train_df_full['target']==1].iloc[0]
example_neg = train_df_full[train_df_full['target']==0].iloc[0]
print("Positive example event time:", example_pos['time_of_event'])
frames_pos = sample_event_window_frames(os.path.join(DATA_DIR, "train", example_pos['filename']),
                                        event_time=example_pos['time_of_event'], window=3.0, target_fps=15)
frames_neg = sample_event_window_frames(os.path.join(DATA_DIR, "train", example_neg['filename']),
                                        event_time=None, window=3.0, target_fps=15)
print(f"Frames sampled (pos): {frames_pos.shape}, (neg): {frames_neg.shape}")



# Initialize ResNet50 for feature extraction (will be fine-tuned later, but for now use pretrained weights)
base_cnn = ResNet50(weights='imagenet', include_top=False, pooling='avg')
base_cnn.trainable = False  # keep frozen for initial feature extraction

# Extract features for all training videos (this might take some time)
def extract_features_event_window(df, window=3.0, target_fps=15):
    X_feat_list = []
    y_list = []
    for _, row in df.iterrows():
        vid_id = row['id']; fname = row['filename']; label = row['target']
        event_time = row['time_of_event'] if 'time_of_event' in row else None
        video_path = os.path.join(DATA_DIR, "train", fname)
        frames = sample_event_window_frames(video_path, event_time=event_time, window=window, target_fps=target_fps)
        if frames.shape[0] == 0:
            # handle edge cases by skipping or adding zeros
            continue
        # preprocess frames and get features
        frames = preprocess_input(frames.astype(np.float32))
        feats = base_cnn.predict(frames, batch_size=frames.shape[0], verbose=0)  # shape (n_frames, 2048)
        # Compute diff features
        diff = np.diff(feats, axis=0)
        # Pad a zero vector at the beginning to keep sequence length same
        diff = np.vstack([np.zeros_like(feats[0]), diff])  # shape (n_frames, 2048)
        # Concatenate original features with diff features
        feats_combined = np.hstack([feats, diff])  # shape (n_frames, 4096)
        X_feat_list.append(feats_combined)
        y_list.append(label)
    return X_feat_list, np.array(y_list)

X_list, y_array = extract_features_event_window(train_df_full, window=3.0, target_fps=15)
print(f"Extracted features for {len(X_list)} videos out of {len(train_df_full)}")
# Pad sequences to equal length if necessary (e.g., if some have 44 vs 45 frames)
seq_lengths = [x.shape[0] for x in X_list]
max_len = max(seq_lengths)
print("Max sequence length:", max_len)
# If sequences vary, pad with zeros to max_len
X_list_padded = []
for feats in X_list:
    if feats.shape[0] < max_len:
        # pad with zeros on bottom (after last frame)
        pad_width = max_len - feats.shape[0]
        pad_array = np.zeros((pad_width, feats.shape[1]), dtype=np.float32)
        feats = np.vstack([feats, pad_array])
    X_list_padded.append(feats)
X_full = np.array(X_list_padded, dtype=np.float32)  # shape (N_videos, max_len, 4096)
y_full = y_array
print("Final feature array shape:", X_full.shape, "Labels shape:", y_full.shape)


# # ---->  Set the shapes exactly as printed by our long job  <----
# N_VIDEOS        = 1500      # same as len(train_df_full)
# MAX_SEQ_LEN     = 45        # "Max sequence length: 45"
# FRAME_FEAT_DIM  = 4096      # 2048 (ResNet avg-pool)  +  2048 (frame-diff)

# # Fake features:   white-noise ~ N(0,1)  (or zeros if RAM is tight)
# #     X_full shape : (1500, 45, 4096)
# #     y_full shape : (1500,)
# rng     = np.random.default_rng(42)      # deterministic for reproducibility
# X_full  = rng.standard_normal(size=(N_VIDEOS, MAX_SEQ_LEN, FRAME_FEAT_DIM)
#                               ).astype(np.float32)
# y_full  = rng.integers(low=0, high=2, size=N_VIDEOS).astype(np.float32)

# print("  Placeholder tensors ready")
# print("   X_full :", X_full.shape, X_full.dtype)
# print("   y_full :", y_full.shape, y_full.dtype)

# # (Optional) expose max_len if later cells rely on that symbol
# max_len = MAX_SEQ_LEN


# ⬇️ Code cell (verify, then continue)
import sklearn, scikeras, sys, tensorflow as tf
print("Python       :", sys.version.split()[0])
print("scikit-learn :", sklearn.__version__)
print("SciKeras     :", scikeras.__version__)
print("TensorFlow   :", tf.__version__)


from scikeras.wrappers import KerasClassifier
keras_clf = KerasClassifier(model=build_lstm_model, epochs=1, verbose=0)

# show the first 15 keys so you can see the difference
print(sorted(list(keras_clf.get_params().keys())[:15]))


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Define a function to build the LSTM classifier model given hyperparameters
def build_lstm_model(lstm_units=128, dropout_rate=0.5, learn_rate=1e-3):
    inputs = Input(shape=(X_full.shape[1], X_full.shape[2]))  # (time, feature_dim)
    x = Bidirectional(LSTM(lstm_units, dropout=dropout_rate))(inputs)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    return model

# Wrap the model for use in scikit-learn
# keras_clf = KerasClassifier(model=build_lstm_model, verbose=0, epochs=10, batch_size=32)

keras_clf = KerasClassifier(
    model          = build_lstm_model,
    epochs         = 10,
    batch_size     = 32,
    verbose        = 0,
    random_state   = 42,
    lstm_units     = 128,      # default value, will be searched
    dropout_rate   = 0.5,      # 〃
    learn_rate     = 1e-3      # 〃
)

# Define hyperparameter search space
param_dist = {
    "lstm_units": [128, 256],
    "dropout_rate": [0.3, 0.5],
    "learn_rate": [1e-3, 1e-4]
}
# Set up cross-validation scheme
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Run RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=keras_clf, param_distributions=param_dist, 
                                   n_iter=6, cv=cv, scoring='average_precision', verbose=1, n_jobs=1)
random_search.fit(X_full, y_full)

print("Best Hyperparameters:", random_search.best_params_)
print("Best CV score (mean average precision):", random_search.best_score_)


best_params = random_search.best_params_
best_units = best_params['lstm_units']
best_dropout = best_params['dropout_rate']
best_lr = best_params['learn_rate']
print(best_units, best_dropout, best_lr)


# Build the integrated model with ResNet + BiLSTM
def create_resnet_lstm_model(units=best_units, dropout_rate=best_dropout, learning_rate=best_lr):
    # Frame sequence input
    frame_sequence = Input(shape=(None, 224, 224, 3), name='frames_input')  # None = variable time length
    # CNN feature extraction for each frame
    cnn = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    cnn_features = tf.keras.layers.TimeDistributed(cnn)(frame_sequence)  # shape: (batch, T, 2048)
    # Compute difference between consecutive frame features
    diff = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,:1,:]), x[:,1:,:] - x[:,:-1,:]], axis=1), 
                  name='diff_features')(cnn_features)
    # Concatenate original features with diff features
    combined_features = Concatenate(axis=-1, name='combined_features')([cnn_features, diff])
    # Bi-directional LSTM on combined features
    x = Bidirectional(LSTM(units, dropout=dropout_rate), name='bilstm')(combined_features)
    output = Dense(1, activation='sigmoid', name='collision_prob')(x)
    model = Model(inputs=frame_sequence, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['AUC'])
    return model

# Create an instance of the model
model = create_resnet_lstm_model()
model.summary()


from tensorflow.keras.utils import Sequence

tf.random.set_seed(42)

class VideoSequence(Sequence):
    def __init__(self, df_indices, df, batch_size=4, window=3.0, target_fps=15, mode='train'):
        self.indices = df_indices
        self.df = df
        self.batch_size = batch_size
        self.window = window
        self.target_fps = target_fps
        self.mode = mode
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_frames = []
        batch_labels = []
        for i in batch_indices:
            row = self.df.iloc[i]
            video_path = os.path.join(DATA_DIR, "train", row['filename'])
            event_time = row['time_of_event'] if row['target']==1 else None
            frames = sample_event_window_frames(video_path, event_time=event_time, 
                                               window=self.window, target_fps=self.target_fps)
            if frames.shape[0] == 0:
                # If no frames were read (shouldn't usually happen unless video is too short), skip
                frames = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            batch_frames.append(frames)
            if self.mode != 'test':
                batch_labels.append(row['target'])
        # Because videos might have different frame counts, we pad each sequence to the max in the batch for uniform tensor
        max_frames = max(f.shape[0] for f in batch_frames)
        batch_x = np.zeros((len(batch_frames), max_frames, 224, 224, 3), dtype=np.uint8)
        for j, frames in enumerate(batch_frames):
            batch_x[j, :frames.shape[0]] = frames
        batch_x = preprocess_input(batch_x.astype(np.float32))
        if self.mode != 'test':
            batch_y = np.array(batch_labels, dtype=np.float32)
            return batch_x, batch_y
        else:
            return batch_x

# Cross-validation training
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((len(test_df), 5))
for fold, (train_idx, val_idx) in enumerate(folds.split(train_df_full, train_df_full['target'])):
    print(f"\n***** Fold {fold+1} *****")
    # Create data generators for this fold
    train_seq = VideoSequence(train_idx, train_df_full, batch_size=4, mode='train')
    val_seq   = VideoSequence(val_idx, train_df_full, batch_size=4, mode='train')
    # Initialize model (fresh for each fold)
    model = create_resnet_lstm_model()
    # Stage 1: train with CNN frozen
    # (CNN layers in model already default to trainable, so freeze them explicitly)
    for layer in model.layers:
        if layer.name.startswith('resnet50'):
            layer.trainable = False
    # Compile with initial LR (already compiled)
    early_stop1 = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
    model.fit(train_seq, validation_data=val_seq, epochs=10, callbacks=[early_stop1], verbose=2)
    # Stage 2: fine-tune CNN top layers
    # Unfreeze last block of ResNet50 (e.g., layers with names containing 'conv5')
    for layer in model.layers:
        if layer.name.startswith('resnet50'):
            # unfreeze all layers in conv5 block
            for sublayer in layer.layers:
                if sublayer.name.startswith('conv5_block') or sublayer.name.startswith('conv5_'):
                    sublayer.trainable = True
    # Compile with lower LR for fine-tuning
    tf.keras.backend.set_value(model.optimizer.learning_rate, best_lr * 0.1)
    early_stop2 = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
    model.fit(train_seq, validation_data=val_seq, epochs=5, callbacks=[early_stop2], verbose=2)
    # Evaluate on validation (optional, to see fold performance)
    val_auc = model.evaluate(val_seq, verbose=0)[1]
    print(f"Fold {fold+1} validation AUC: {val_auc:.4f}")
    # Predict on test set with this fold's model
    test_seq = VideoSequence(list(range(len(test_df))), test_df, batch_size=4, mode='test')
    fold_pred = model.predict(test_seq, verbose=0)[:, 0]  # shape (n_test,)
    test_preds[:, fold] = fold_pred


# Average predictions across folds
avg_preds = np.mean(test_preds, axis=1)
print("Ensembled prediction sample:", avg_preds[:5])

# Create the submission dataframe
submission = pd.DataFrame({
    "id": test_df["id"],
    "target": avg_preds
})
submission.to_csv("submission.csv", index=False)
submission.head(10)


# Create submission dataframe
submission = test_df.copy()
submission['target'] = test_preds
submission = submission[['id', 'target']]
submission.to_csv('submission.csv', index=False)
submission.head(10)




