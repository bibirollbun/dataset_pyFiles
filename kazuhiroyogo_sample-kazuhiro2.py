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


# --- ä¿®æ­£: Protobufã�®ã‚¨ãƒ©ãƒ¼å›�é�¿ ---
# ã�“ã�®ã‚¨ãƒ©ãƒ¼ã�¯protobufã�®ãƒ�ãƒ¼ã‚¸ãƒ§ãƒ³ä¸�æ•´å�ˆã�Œå�Ÿå› ã�§ã�™ã€‚
# å¼·åˆ¶çš„ã�«3.20.3ã‚’ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«ã�—ã�¾ã�™ã€‚
# â€» ã�“ã‚Œã�Œå®Ÿè¡Œã�•ã‚Œã�Ÿå¾Œã€�ã‚‚ã�—ã‚¨ãƒ©ãƒ¼ã�Œç¶šã��å ´å�ˆã�¯ã€ŒKernel Restartã€�ã‚’ã�—ã�¦ã�‹ã‚‰å†�å®Ÿè¡Œã�—ã�¦ã��ã� ã�•ã�„ã€‚
os.system('pip install -U protobuf==3.20.3')


import numpy as np
import pandas as pd
import os
import gc
import random
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import KFold

# --- 0. å†�ç�¾æ€§ã�®ã�Ÿã‚�ã�®ã‚·ãƒ¼ãƒ‰å›ºå®š ---
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)


# --- 1. TPU/GPUæˆ¦ç•¥ã�®ã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ— ---
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except ValueError:
    strategy = tf.distribute.get_strategy()
    print("Running on GPU/CPU")

print(f"REPLICAS: {strategy.num_replicas_in_sync}")

# --- ğŸ’¥é‡�è¦�: æ··å�ˆç²¾åº¦ (Mixed Precision) ã�®è¨­å®š ---
# ã�“ã‚Œã�«ã‚ˆã‚Šãƒ¡ãƒ¢ãƒªä½¿ç”¨é‡�ã�Œå‰Šæ¸›ã�•ã‚Œã€�TPUã�§ã�®å­¦ç¿’ã�Œå®‰å®šãƒ»é«˜é€ŸåŒ–ã�—ã�¾ã�™
try:
    # TPUã�®å ´å�ˆã�¯ 'mixed_bfloat16' ã‚’ä½¿ç”¨
    if isinstance(strategy, tf.distribute.TPUStrategy):
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        print("Mixed precision (bfloat16) enabled.")
    else:
        # GPUã�®å ´å�ˆã�¯ 'mixed_float16' (ä»Šå›�ã�¯TPUå‰�æ��ã� ã�Œå¿µã�®ã�Ÿã‚�)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision (float16) enabled.")
except Exception as e:
    print(f"Mixed precision setting failed: {e}")

# --- è¨­å®šå€¤ ---
# å®‰å…¨ç¬¬ä¸€ã�®è¨­å®šã�«å¤‰æ›´
# BATCH_SIZE_PER_REPLICA: 64 -> 16 (ç¢ºå®Ÿã�«OOMã‚’é˜²ã��ã�Ÿã‚�)
BATCH_SIZE_PER_REPLICA = 16 
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
MAX_LEN = 192
MODEL_NAME = 'bert-base-uncased'
TARGET_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
N_FOLDS = 3
EPOCHS = 2
LEARNING_RATE = 2e-5


# --- 2. ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿ ---
print("Loading data...")
# ãƒ‘ã‚¹ã�¯ç’°å¢ƒã�«å�ˆã‚�ã�›ã�¦èª¿æ•´ã�—ã�¦ã��ã� ã�•ã�„
INPUT_DIR = "../input/jigsaw-toxic-comment-classification-challenge/"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.zip"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.zip"))

y_train_full = train_df[TARGET_COLUMNS].values


# --- 3. BERTãƒˆãƒ¼ã‚¯ãƒ³åŒ– ---
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    # ãƒ�ãƒƒãƒ�å‡¦ç�†ã�§ãƒ¡ãƒ¢ãƒªã�Œæº¢ã‚Œã‚‹ã�®ã‚’é˜²ã��ã�Ÿã‚�ã€�å°‘ã�—ã�šã�¤å‡¦ç�†ã�—ã�¦ã‚‚è‰¯ã�„ã�Œã€�
    # ã�“ã�®ãƒ‡ãƒ¼ã‚¿ã‚µã‚¤ã‚ºã�ªã‚‰ä¸€æ‹¬ã�§ã‚‚ã‚®ãƒªã‚®ãƒªã�„ã�‘ã‚‹ã�¨ä»®å®š
    encodings = tokenizer.batch_encode_plus(
        texts.tolist(), # pandas seriesã�§ã�¯ã�ªã��listã�§æ¸¡ã�™æ–¹ã�Œå®‰å…¨
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='tf' # 'np'ã�§ã�¯ã�ªã��æœ€åˆ�ã�‹ã‚‰'tf'ã�§è¿”ã�™ã�¨å¤‰æ�›ã‚³ã‚¹ãƒˆã�Œæ¸›ã‚‹å ´å�ˆã�Œã�‚ã‚‹ã�Œã€�ã‚¹ãƒ©ã‚¤ã‚¹æ“�ä½œã�®ã�Ÿã‚�ã�«npæ�¨å¥¨
    )
    return {
        'input_ids': encodings['input_ids'].numpy(), # numpyã�«å¤‰æ�›ã�—ã�¦ä¿�æŒ�ï¼ˆTF Tensorã� ã�¨ãƒ¡ãƒ¢ãƒªã‚’é£Ÿã�„ã�™ã��ã‚‹ã�Ÿã‚�ï¼‰
        'attention_mask': encodings['attention_mask'].numpy()
    }

print("Tokenizing all data...")
X_train_encoded = bert_encode(train_df['comment_text'].values, tokenizer)
X_test_encoded = bert_encode(test_df['comment_text'].values, tokenizer)

# IDã� ã�‘é€€é�¿ï¼ˆå¾Œã�§æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ä½œæˆ�ã�«ä½¿ã�†ã�Ÿã‚�ã€�ã�“ã�“ã�§ç¢ºä¿�ã�—ã�¦ã�Šã��ï¼‰
test_ids = test_df['id'].values

# ãƒ¡ãƒ¢ãƒªç¯€ç´„: ãƒ†ã‚­ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�¯ã‚‚ã�†ä¸�è¦�ã�ªã�®ã�§å‰Šé™¤
del train_df, test_df
gc.collect()


# --- 4. ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆä½œæˆ�é–¢æ•° ---
def create_dataset(encodings, labels=None, indices=None, is_train=True):
    # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹æŠ½å‡º
    if indices is not None:
        input_ids = encodings['input_ids'][indices]
        attention_mask = encodings['attention_mask'][indices]
        if labels is not None:
            labels = labels[indices]
    else:
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

    # Datasetæ§‹ç¯‰
    data = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
    
    if is_train:
        # å­¦ç¿’æ™‚ã�¯ã‚·ãƒ£ãƒƒãƒ•ãƒ«ã�¨drop_remainder=Trueæ�¨å¥¨ï¼ˆTPUã�®å½¢çŠ¶å›ºå®šã�®ã�Ÿã‚�ï¼‰
        dataset = dataset.shuffle(2048).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    else:
        # æ�¨è«–æ™‚ã�¯drop_remainder=Falseã�«ã�—ã�ªã�„ã�¨ç«¯æ•°ã�®ãƒ‡ãƒ¼ã‚¿ã�Œäºˆæ¸¬ã�•ã‚Œã�šã�«æ�¨ã�¦ã‚‰ã‚Œã‚‹
        dataset = dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=False)
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# --- 5. ãƒ¢ãƒ‡ãƒ«æ§‹ç¯‰é–¢æ•° ---
def build_model():
    bert_model = TFBertModel.from_pretrained(MODEL_NAME)
    
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    
    output = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
    cls_output = output[:, 0, :]
    
    x = tf.keras.layers.Dropout(0.1)(cls_output)

    # æ··å�ˆç²¾åº¦ã‚’ä½¿ã�†å ´å�ˆã€�å‡ºåŠ›å±¤ã�®dtypeã�¯æ˜�ç¤ºçš„ã�«float32ã�«æˆ»ã�™ã�®ã�Œå®‰å…¨
    out = tf.keras.layers.Dense(6, activation='linear', name='output_layer', dtype='float32')(x)
        
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=out)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=6, name='auc')],
        steps_per_execution=32 # TPUã�¸ã�®è»¢é€�å›�æ•°ã‚’æ¸›ã‚‰ã�—ã�¦é«˜é€ŸåŒ–
    )
    return model


# --- 6. K-Fold äº¤å·®æ¤œè¨¼ ---
print(f"Starting {N_FOLDS}-Fold Cross Validation...")

# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®ã‚µã‚¤ã‚ºã‚’å�–å¾— (X_test_encoded['input_ids'].shape[0])
n_test = X_test_encoded['input_ids'].shape[0]
final_test_preds = np.zeros((n_test, 6))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ãƒ†ã‚¹ãƒˆç”¨ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�¯å›ºå®šã�ªã�®ã�§ã�“ã�“ã�§ä½œã�£ã�¦ã�Šã�� (ãƒ¡ãƒ¢ãƒªæ³¨æ„�)
test_ds = create_dataset(X_test_encoded, is_train=False)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_encoded['input_ids'])):
    print(f"\n========== FOLD {fold + 1} / {N_FOLDS} ==========")
    
    # ã‚»ãƒƒã‚·ãƒ§ãƒ³ã‚¯ãƒªã‚¢ (é‡�è¦�)
    tf.keras.backend.clear_session()
    gc.collect()
    
    train_ds = create_dataset(X_train_encoded, y_train_full, train_idx, is_train=True)
    valid_ds = create_dataset(X_train_encoded, y_train_full, valid_idx, is_train=False)
    
    with strategy.scope():
        model = build_model()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max', patience=1, restore_best_weights=True, verbose=1
    )
    
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print(f"Predicting on test data (Fold {fold+1})...")
    # TPUã�§ã�®äºˆæ¸¬ã�¯ãƒ‡ãƒ¼ã‚¿é‡�ã�Œå¤šã�„ã�¨ä¸�å®‰å®šã�ªå ´å�ˆã�Œã�‚ã‚‹ã�Œã€�ã�“ã�®ã�¾ã�¾å®Ÿè¡Œ
    pred_logits = model.predict(test_ds, verbose=1)
    
    # Logits -> Sigmoid
    pred_proba = tf.nn.sigmoid(pred_logits).numpy()
    final_test_preds += pred_proba
    
    # ãƒ¡ãƒ¢ãƒªè§£æ”¾
    del model, train_ds, valid_ds
    gc.collect()


# --- 7. æ��å‡º ---
print("\n\nAveraging predictions...")
final_test_preds /= N_FOLDS

# CSVå†�èª­ã�¿è¾¼ã�¿ã‚’é�¿ã�‘ã€�æœ€åˆ�ã�«é€€é�¿ã�—ã�¦ã�Šã�„ã�Ÿtest_idsã‚’ä½¿ç”¨ã�—ã�¦DataFrameã‚’ä½œæˆ�
# ã�“ã‚Œã�«ã‚ˆã‚Šã€�æœ€å¾Œã�®ã‚¹ãƒ†ãƒƒãƒ—ã�§ã�®ãƒ¡ãƒ¢ãƒªä¸�è¶³(OOM)ã‚’é˜²ã��ã�¾ã�™
sub_df = pd.DataFrame(final_test_preds, columns=TARGET_COLUMNS)
sub_df.insert(0, 'id', test_ids)

filename = "submission_kazuhiro2.csv"
sub_df.to_csv(filename, index=False)
print(f"Submission file '{filename}' created successfully!")

