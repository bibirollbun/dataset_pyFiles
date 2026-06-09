# ğŸŒŸ MoA Neural Network Pipeline (15000 Sample + Feature Engineering + CV + Loss Plot)
# é�©ç”¨æ–¼æœ¬æ©Ÿè™•ç�†åŸ·è¡Œï¼ŒåŒ…å�« GPUåŠ é€Ÿ èˆ‡è™›æ“¬ç’°å¢ƒæ�­å»º
# è«‹å…ˆåŸ·è¡Œä¸‹åˆ— bash æŒ‡ä»¤å•Ÿå‹•è™›æ“¬ç’°å¢ƒï¼š
# ----------------------------------------------------
# conda create -n moa-nn python=3.10 -y
# conda activate moa-nn
# pip install numpy pandas scikit-learn matplotlib joblib
# pip install tensorflow  # æˆ– pip install tensorflow-gpu å¦‚æ�œæœ‰ NVIDIA é¡¯å�¡
# pip install tqdm
# pip install iterative-stratification
# pip install notebook
# mkdir moa-nn-project
# cd moa-nn-project
# jupyter notebook
# ----------------------------------------------------

# è«‹ç¢ºèª�ä½ å·²å°‡ä»¥ä¸‹æª”æ¡ˆæ”¾å…¥ç•¶å‰�è³‡æ–™å¤¾ï¼š
# - train_features.csv
# - train_targets_scored.csv
# - train_targets_nonscored.csv
# - test_features.csv
# - sample_submission.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tqdm.auto import tqdm  # å¼•å…¥tqdmç”¨æ–¼é€²åº¦è¿½è¹¤
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Optional: ç”¨æ–¼ balanced multilabel CV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# è¨­å®šå�ƒæ•¸ - æ˜¯å�¦ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹
USE_PRETRAINED_MODELS = True  # è¨­ç‚ºTrueæ™‚ï¼Œå°‡å¾�output/paramè³‡æ–™å¤¾è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹

# ================= è·¯å¾‘è¨­å®šå�€ =================
DATA_ROOT = "/kaggle/input/lish-moa/"
MYMODELS_ROOT = "/kaggle/working/"
CSV_ROOT = "/kaggle/working/"
MODEL_OUTPUT_ROOT = "/kaggle/input/moa-all-models/output_param/nn_models"
TRAIN_FEATURES_PATH = os.path.join(DATA_ROOT, "train_features.csv")
TRAIN_TARGETS_SCORED_PATH = os.path.join(DATA_ROOT, "train_targets_scored.csv")
TRAIN_TARGETS_NONSCORED_PATH = os.path.join(DATA_ROOT, "train_targets_nonscored.csv")
TEST_FEATURES_PATH = os.path.join(DATA_ROOT, "test_features.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(CSV_ROOT, "submission.csv")
VALIDATION_OUTPUT_PATH = os.path.join(CSV_ROOT, "validation_predictions_cat.csv")
# =============================================

# æª¢æŸ¥æ˜¯å�¦æœ‰å�¯ç”¨GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # è¨­å®šè¨˜æ†¶é«”å¢�é•·é™�åˆ¶
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"ğŸš€ ä½¿ç”¨GPUåŠ é€Ÿ: {len(gpus)}å€‹GPUå�¯ç”¨")
    except RuntimeError as e:
        print(f"GPUè¨­å®šå¤±æ•—: {e}")
else:
    print("âš ï¸� æ²’æœ‰å�¯ç”¨çš„GPUï¼Œå°‡ä½¿ç”¨CPUé€²è¡Œè¨“ç·´ï¼Œé€Ÿåº¦è¼ƒæ…¢")

# 1. è®€å�–è³‡æ–™
train_features = pd.read_csv(TRAIN_FEATURES_PATH)
train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED_PATH)
train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH)
test_features = pd.read_csv(TEST_FEATURES_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# 2. æŠ½æ¨£ 15000 ç­†è¨“ç·´è³‡æ–™
df_sample = train_features.sample(n=15000, random_state=42).reset_index(drop=True)
ids = df_sample['sig_id']
Y = train_targets_scored[train_targets_scored['sig_id'].isin(ids)].reset_index(drop=True)
df_valid = train_features[~train_features['sig_id'].isin(ids)].reset_index(drop=True)
Y_valid = train_targets_scored[train_targets_scored['sig_id'].isin(df_valid['sig_id'])].reset_index(drop=True)

# 3. ç‰¹å¾µå·¥ç¨‹
def preprocess_features(df):
    df = df.drop(columns=["sig_id"])
    df = pd.get_dummies(df, columns=["cp_type", "cp_dose", "cp_time"])
    return df

X = preprocess_features(df_sample)
X_test = preprocess_features(test_features)
X_valid = preprocess_features(df_valid)
# æ¨™æº–åŒ–ç‰¹å¾µ
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns).astype(np.float32)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).astype(np.float32)
X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns).astype(np.float32)

# 4. Cross Validation è¨­å®šï¼ˆMultilabelStratifiedKFoldï¼‰
N_SPLITS = 3
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

models = {}
histories = {}

# å»ºç«‹å®Œæ•´è·¯å¾‘çš„å„²å­˜è³‡æ–™å¤¾
model_path = MODEL_OUTPUT_ROOT
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(f"{model_path}/param", exist_ok=True)  # å„²å­˜æ¨¡å�‹å�ƒæ•¸çš„è³‡æ–™å¤¾
# os.makedirs(f"{model_path}/lossplot", exist_ok=True)  # å„²å­˜lossåœ–çš„è³‡æ–™å¤¾

# é¡¯ç¤ºTensorFlowç‰ˆæœ¬
tf_version = tf.__version__
print(f"TensorFlow ç‰ˆæœ¬: {tf_version}")
start_time = time.time()

# è¨ˆç®—ç¸½ä»»å‹™æ•¸é‡�ï¼Œç”¨æ–¼é€²åº¦ä¼°ç®—
target_columns = Y.columns[1:]
total_targets = len(target_columns)
print(f"ç¸½å…±éœ€è¦�è¨“ç·´ {total_targets} å€‹ç›®æ¨™")

# è¿½è¹¤è¨“ç·´æ™‚é–“
target_times = []

class NNTracker:
    """ç”¨æ–¼è¿½è¹¤ç¥�ç¶“ç¶²çµ¡è¨“ç·´é��ç¨‹çš„é¡�åˆ¥"""
    def __init__(self):
        self.history = None
    
    def add_history(self, history):
        self.history = history

def compute_logloss(y_true, y_pred):
    """è¨ˆç®—å°�æ•¸æ��å¤±"""
    # ç¢ºä¿�é �æ¸¬å€¼åœ¨ (0, 1) ä¹‹é–“ï¼Œé�¿å…�æ•¸å€¼å•�é¡Œ
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def build_model(input_dim, learning_rate=0.001):
    """å»ºç«‹ç¥�ç¶“ç¶²çµ¡æ¨¡å�‹"""
    model = Sequential([
        # è¼¸å…¥å±¤
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # éš±è—�å±¤1
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # éš±è—�å±¤2
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # è¼¸å‡ºå±¤ (äºŒå…ƒåˆ†é¡�)
        Dense(1, activation='sigmoid')
    ])
    
    # ç·¨è­¯æ¨¡å�‹
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 5. è¨“ç·´æ¨¡å�‹æˆ–è¼‰å…¥å·²æœ‰çš„æ¨¡å�‹
if USE_PRETRAINED_MODELS:
    print("ğŸ”„ ä½¿ç”¨å·²è¨“ç·´å¥½çš„æ¨¡å�‹ï¼Œå¾�ä¿�å­˜çš„æª”æ¡ˆä¸­è¼‰å…¥...")
    
    # æª¢æŸ¥æ˜¯å�¦å­˜åœ¨æ¨¡å�‹æª”æ¡ˆ
    model_exists = all(os.path.exists(f"{model_path}/param/nn_{target}.h5") for target in target_columns)
    
    if not model_exists:
        print("â�Œ éŒ¯èª¤: æ‰¾ä¸�åˆ°æ‰€æœ‰éœ€è¦�çš„æ¨¡å�‹æª”æ¡ˆï¼Œè«‹è¨­å®š USE_PRETRAINED_MODELS=False é‡�æ–°è¨“ç·´")
        exit(1)
    
    # è¼‰å…¥å·²è¨“ç·´çš„æ¨¡å�‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='è¼‰å…¥æ¨¡å�‹'):
        model_file = f"{model_path}/param/nn_{target}.h5"
        models[target] = load_model(model_file)
        print(f"å·²è¼‰å…¥æ¨¡å�‹: {target}")
    
    print("âœ… å·²æˆ�åŠŸè¼‰å…¥æ‰€æœ‰æ¨¡å�‹")
else:
    # æ­£å¸¸è¨“ç·´æµ�ç¨‹
    for i, target in tqdm(enumerate(target_columns), total=total_targets, desc='æ•´é«”é€²åº¦'):
        target_start = time.time()
        
        print(f"\nè¨“ç·´æ¨¡å�‹ ({i+1}/{total_targets}): {target}")
        y_target = Y[target].values
        fold = 0
        fold_logloss = []
        
        # ä½¿ç”¨ train_test_split å‡½æ•¸é€²è¡Œåˆ†å‰²
        indices = range(len(X))
        
        # æ›´åš´æ ¼æª¢æŸ¥é¡�åˆ¥åˆ†å¸ƒï¼Œç¢ºä¿�æ¯�å€‹é¡�åˆ¥è‡³å°‘æœ‰ 2 ç­†è³‡æ–™
        unique_values, counts = np.unique(y_target, return_counts=True)
        min_count = counts.min() if len(counts) > 0 else 0
        
        # å¦‚æ�œä»»ä½•é¡�åˆ¥çš„æ¨£æœ¬æ•¸å°‘æ–¼ 2 æˆ–è€…å�ªæœ‰ä¸€å€‹é¡�åˆ¥ï¼Œå‰‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£
        if min_count < 2 or len(unique_values) <= 1:
            print(f"è­¦å‘Š: ç›®æ¨™ {target} çš„è³‡æ–™åˆ†å¸ƒä¸�å�‡è¡¡ï¼ŒæŸ�é¡�åˆ¥æ¨£æœ¬æ•¸é��å°‘ï¼Œå°‡ä¸�ä½¿ç”¨åˆ†å±¤æŠ½æ¨£")
            stratify_data = None
        else:
            stratify_data = y_target
            
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=stratify_data)
        
        X_train, X_val = X.iloc[train_indices].values, X.iloc[val_indices].values
        y_train, y_val = y_target[train_indices], y_target[val_indices]

        # æª¢æŸ¥é¡�åˆ¥æ˜¯å�¦ä¸�å¹³è¡¡
        pos_rate = np.mean(y_train)
        print(f"æ­£ä¾‹æ¯”ä¾‹: {pos_rate:.4f}")
        
        # é‡�å°�ä¸�å¹³è¡¡è³‡æ–™èª¿æ•´æ¬Šé‡�
        class_weight = None
        if pos_rate < 0.2 or pos_rate > 0.8:
            weight_ratio = (1 - pos_rate) / pos_rate if pos_rate < 0.5 else pos_rate / (1 - pos_rate)
            # ç‚ºkerasæº–å‚™é¡�åˆ¥æ¬Šé‡�
            class_weight = {0: 1, 1: weight_ratio} if pos_rate < 0.5 else {0: weight_ratio, 1: 1}
            print(f"è³‡æ–™ä¸�å¹³è¡¡ï¼Œèª¿æ•´æ¬Šé‡�ç‚º: {weight_ratio:.2f}")
        
        tracker = NNTracker()
        
        # è¨­å®šæ—©å�œå�ƒæ•¸
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # è¨­å®šå­¸ç¿’ç�‡æ¸›å°‘ç­–ç•¥
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
        
        try:
            # ä½¿ç”¨ç¥�ç¶“ç¶²çµ¡åˆ†é¡�å™¨
            print(f"è¨“ç·´æ¨¡å�‹ä¸­...", end=" ")
            # ç�²å�–ç‰¹å¾µæ•¸é‡�
            input_dim = X_train.shape[1]
            
            # å‰µå»ºæ¨¡å�‹
            model = build_model(input_dim)
            
            # è¨“ç·´æ¨¡å�‹
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                class_weight=class_weight,
                verbose=0  # ä¸�é¡¯ç¤ºè¨“ç·´é€²åº¦
            )
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict(X_train, verbose=0).flatten()
            val_proba = model.predict(X_val, verbose=0).flatten()
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_history(history)
            
        except Exception as e:
            print(f"ç¥�ç¶“ç¶²çµ¡ è¨“ç·´å¤±æ•—: {str(e)}")
            # å˜—è©¦ä½¿ç”¨è¼ƒç°¡å–®çš„ç¥�ç¶“ç¶²çµ¡æ¨¡å�‹
            print("å˜—è©¦ä½¿ç”¨ç°¡åŒ–çš„ç¥�ç¶“ç¶²çµ¡æ¨¡å�‹...", end=" ")
            
            # å‰µå»ºç°¡å–®æ¨¡å�‹
            model = Sequential([
                Dense(64, input_dim=X_train.shape[1], activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.01),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # ç°¡å–®è¨“ç·´
            history = model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=64,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                class_weight=class_weight,
                verbose=0
            )
            
            # è¨ˆç®—è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ��å¤±
            train_proba = model.predict(X_train, verbose=0).flatten()
            val_proba = model.predict(X_val, verbose=0).flatten()
            
            train_loss = compute_logloss(y_train, train_proba)
            val_loss = compute_logloss(y_val, val_proba)
            
            print(f"è¨“ç·´é›† Log Loss: {train_loss:.4f} | é©—è­‰é›† Log Loss: {val_loss:.4f}")
            
            tracker.add_history(history)

        # ç¹ªè£½ Loss plot
        try:
            histories[target] = tracker
            
            if hasattr(tracker, 'history') and tracker.history is not None:
                history = tracker.history.history
                
                plt.figure(figsize=(12, 5))
                
                # ç¹ªè£½æ��å¤±æ›²ç·š
                plt.subplot(1, 2, 1)
                plt.plot(history['loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.title(f'Loss Curve for {target}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # ç¹ªè£½æº–ç¢ºç�‡æ›²ç·š
                plt.subplot(1, 2, 2)
                plt.plot(history['accuracy'], label='Train Accuracy')
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
                plt.title('Accuracy Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
                plt.close()
            else:
                # å¦‚æ�œæ²’æœ‰è¨“ç·´é��ç¨‹çš„æ›²ç·šï¼Œå°±ç¹ªè£½æœ€çµ‚çµ�æ�œ
                plt.figure(figsize=(10, 6))
                plt.bar(['Train Loss', 'Valid Loss'], [train_loss, val_loss])
                plt.title(f"Final Loss for {target}")
                plt.ylabel("Log Loss")
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(f"{model_path}/lossplot/loss_{target}.png")
                plt.close()
                
        except Exception as e:
            print(f"ç„¡æ³•ç¹ªè£½æ��å¤±æ›²ç·š: {str(e)}")

        # å„²å­˜æ¨¡å�‹
        models[target] = model
        model.save(f"{model_path}/param/nn_{target}.h5")
        
        # è¨ˆç®—ä¸¦é¡¯ç¤ºé€²åº¦
        target_time = time.time() - target_start
        target_times.append(target_time)
        avg_time_per_target = np.mean(target_times)
        remaining_targets = total_targets - (i + 1)
        estimated_remaining_time = avg_time_per_target * remaining_targets
        
        # è½‰æ�›é �ä¼°å‰©é¤˜æ™‚é–“ç‚ºæ›´æ˜“è®€æ ¼å¼�
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"ç›®æ¨™ {i+1}/{total_targets} å·²å®Œæˆ�! ({target})")
        print(f"å¹³å�‡æ¯�å€‹ç›®æ¨™è¨“ç·´æ™‚é–“: {avg_time_per_target:.2f} ç§’")
        print(f"é �ä¼°å‰©é¤˜æ™‚é–“: {remaining_time_str}")
        print(f"é �ä¼°å®Œæˆ�æ™‚é–“: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

# æœ€çµ‚å®Œæˆ�è¨Šæ�¯
print("\nâœ… æ‰€æœ‰æ¨¡å�‹è¨“ç·´å·²å®Œæˆ�!")
total_time = time.time() - start_time
print(f"ç¸½è€—æ™‚: {datetime.timedelta(seconds=int(total_time))}")

# 6. æ�¨è«–èˆ‡å»ºç«‹æ��äº¤æª”æ¡ˆ
print("ğŸ”® é–‹å§‹é€²è¡Œé �æ¸¬...")
predictions = []

# å°‡æ¸¬è©¦æ•¸æ“šè½‰æ�›ç‚ºnumpyæ•¸çµ„
X_test_array = X_test.values

# é¡¯ç¤ºé �æ¸¬é€²åº¦
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='é �æ¸¬é€²åº¦'):
    model = models[target]  # å„ªå…ˆä½¿ç”¨è¨˜æ†¶é«”ä¸­çš„æ¨¡å�‹é�¿å…�é‡�è¤‡è¼‰å…¥
    pred = model.predict(X_test_array, verbose=0).flatten()
    predictions.append(pred)

predictions = np.array(predictions).T
submission = sample_submission.copy()
submission.iloc[:, 1:] = predictions
submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print("ğŸ�‰ å·²ç”¢å‡º submission_nn.csv å�¯ç›´æ�¥ä¸Šå‚³åˆ° Kaggle")

# æ·»åŠ é©—è­‰é›†çš„é �æ¸¬
print("ğŸ”� é–‹å§‹é€²è¡Œé©—è­‰é›†é �æ¸¬...")
predictions_V = []

# Show prediction progress
for i, target in tqdm(enumerate(target_columns), total=len(target_columns), desc='Validation prediction progress'):
    model = models[target]  # Use in-memory models to avoid reloading
    pred = model.predict(X_valid.values.astype(np.float32), verbose=0).flatten()  # å¼·åˆ¶è½‰å�‹ float32
    predictions_V.append(pred)

predictions_V = np.array(predictions_V).T

# Create validation submission with correct sig_ids
validation_submission = pd.DataFrame(columns=Y_valid.columns)
validation_submission['sig_id'] = df_valid['sig_id']
for col in target_columns:
    validation_submission[col] = predictions_V[:, list(target_columns).index(col)]

# Save validation predictions to a separate file
validation_submission.to_csv(VALIDATION_OUTPUT_PATH, index=False)
print(f"ğŸ�‰ NN é©—è­‰è³‡æ–™é �æ¸¬æª”æ¡ˆå·²å»ºç«‹: {VALIDATION_OUTPUT_PATH}")


