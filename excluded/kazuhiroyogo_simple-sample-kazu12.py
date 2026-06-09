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


!pip install protobuf==3.20.3


import numpy as np
import pandas as pd
import os
import gc # ãƒ¡ãƒ¢ãƒªè§£æ”¾ç”¨
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import KFold # K-Foldç”¨

# --- 0. TPU/GPUæˆ¦ç•¥ã�®ã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ— ---
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except ValueError:
    strategy = tf.distribute.get_strategy()
    print("Running on GPU/CPU")

print(f"REPLICAS: {strategy.num_replicas_in_sync}")

# 
GLOBAL_BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192
MODEL_NAME = 'bert-large-uncased' # Largeãƒ¢ãƒ‡ãƒ«ã‚’ä½¿ç”¨
target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
N_FOLDS = 3 # ğŸ’¥ 3åˆ†å‰²äº¤å·®æ¤œè¨¼ (æ™‚é–“ã‚’çŸ­ç¸®ã�—ã�Ÿã�„å ´å�ˆã�¯ã�“ã�“ã‚’èª¿æ•´)
EPOCHS = 2  # Largeãƒ¢ãƒ‡ãƒ«ã�ªã�®ã�§å°‘ã�ªã‚�ã�§OK (EarlyStoppingã‚‚ã�‚ã‚‹ã�Ÿã‚�)

# --- 1. ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿ ---
print("Loading data...")
train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sample_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

# å­¦ç¿’ãƒ‡ãƒ¼ã‚¿å…¨ä½“ï¼ˆåˆ†å‰²å‰�ï¼‰
y_train_full = train_df[target_columns].values


# --- 2. BERTã�®ã�Ÿã‚�ã�®ãƒˆãƒ¼ã‚¯ãƒ³åŒ– (ä¸€æ‹¬å‡¦ç�†) ---
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='np'
    )

print("Tokenizing all data...")
# ã�“ã�“ã�§å…¨ãƒ‡ãƒ¼ã‚¿ã‚’ãƒ™ã‚¯ãƒˆãƒ«åŒ–ã�—ã�¦ã�Šã��ã�¾ã�™
X_train_encoded = bert_encode(train_df['comment_text'].values, tokenizer, MAX_LEN)
X_test_encoded = bert_encode(test_df['comment_text'].values, tokenizer, MAX_LEN)


# --- 3. ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆä½œæˆ�é–¢æ•° ---
def create_dataset(encodings, labels=None, indices=None):
    # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹æŒ‡å®šã�Œã�‚ã‚‹å ´å�ˆï¼ˆK-Foldã�®åˆ†å‰²æ™‚ï¼‰ã€�ã��ã�®éƒ¨åˆ†ã� ã�‘æŠœã��å‡ºã�™
    if indices is not None:
        input_ids = encodings['input_ids'][indices]
        attention_mask = encodings['attention_mask'][indices]
    else:
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

    if labels is not None:
        # ãƒ©ãƒ™ãƒ«ã‚‚ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã�§æŠœã��å‡ºã�™
        if indices is not None:
            labels_slice = labels[indices]
        else:
            labels_slice = labels
            
        dataset = tf.data.Dataset.from_tensor_slices((
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            labels_slice
        ))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            {'input_ids': input_ids, 'attention_mask': attention_mask}
        )
    return dataset


# --- 4. ãƒ¢ãƒ‡ãƒ«æ§‹ç¯‰é–¢æ•° ---
def build_model():
    # TPUã‚¹ã‚³ãƒ¼ãƒ—å†…ã�§ãƒ¢ãƒ‡ãƒ«ã‚’ä½œã‚‹å¿…è¦�ã�Œã�‚ã‚Šã�¾ã�™
    bert_model = TFBertModel.from_pretrained(MODEL_NAME)
    
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    
    output = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
    cls_output = output[:, 0, :]
    
    x = tf.keras.layers.Dropout(0.1)(cls_output)
    out = tf.keras.layers.Dense(6, name='output_layer')(x)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=out)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Largeãƒ¢ãƒ‡ãƒ«ç”¨ã�«ä½�ã‚�ã�®å­¦ç¿’ç�‡
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=6, name='auc')]
    )
    return model


# --- 5. ğŸ’¥ K-Fold äº¤å·®æ¤œè¨¼ã�®å®Ÿè¡Œ ---
print(f"Starting {N_FOLDS}-Fold Cross Validation...")

# æœ€çµ‚çš„ã�ªäºˆæ¸¬çµ�æ�œã‚’æ ¼ç´�ã�™ã‚‹é…�åˆ—ï¼ˆã�™ã�¹ã�¦0ã�§åˆ�æœŸåŒ–ï¼‰
final_test_preds = np.zeros((len(test_df), 6))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# K-Foldã�®ãƒ«ãƒ¼ãƒ—
for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(f"\n\n========== FOLD {fold + 1} / {N_FOLDS} ==========")
    
    # TPUãƒ¡ãƒ¢ãƒªã�®ã‚¯ãƒªã‚¢ï¼ˆã�“ã‚Œã�Œã�ªã�„ã�¨OOMã‚¨ãƒ©ãƒ¼ã�«ã�ªã‚‹ã�“ã�¨ã�Œã�‚ã‚Šã�¾ã�™ï¼‰
    tf.keras.backend.clear_session()
    gc.collect()
    
    # ã�“ã�®Foldç”¨ã�®ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆä½œæˆ�
    train_ds = create_dataset(X_train_encoded, y_train_full, train_idx)
    valid_ds = create_dataset(X_train_encoded, y_train_full, valid_idx)
    
    # è¨­å®š
    train_ds = train_ds.shuffle(1000).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # ãƒ¢ãƒ‡ãƒ«ã�®æ§‹ç¯‰
    with strategy.scope():
        model = build_model()
    
    # EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max', patience=1, restore_best_weights=True
    )
    
    # å­¦ç¿’
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )
    
    # ğŸ’¥ ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®äºˆæ¸¬ (ã�“ã�®Foldã�®ãƒ¢ãƒ‡ãƒ«ã‚’ä½¿ã�£ã�¦äºˆæ¸¬)
    print(f"Predicting on test data using model from Fold {fold+1}...")
    test_ds = create_dataset(X_test_encoded) # ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿å…¨ä½“
    test_ds = test_ds.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    pred_logits = model.predict(test_ds)
    pred_proba = tf.nn.sigmoid(pred_logits).numpy()
    
    # äºˆæ¸¬çµ�æ�œã‚’åŠ ç®— (ã�‚ã�¨ã�§N_FOLDSã�§å‰²ã�£ã�¦å¹³å�‡ã‚’ã�¨ã‚‹)
    final_test_preds += pred_proba


# --- 6. å¹³å�‡åŒ–ã�¨æ��å‡º ---
print("\n\nAveraging predictions...")
# å�ˆè¨ˆã‚’Foldæ•°ã�§å‰²ã�£ã�¦å¹³å�‡ã‚’å‡ºã�™ï¼ˆã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ï¼‰
final_test_preds /= N_FOLDS

sub_df = pd.DataFrame(final_test_preds, columns=target_columns)
sub_df['id'] = test_df['id'].values
sub_df = sub_df[['id'] + target_columns]

filename = "Kazu_submission_15.csv"
sub_df.to_csv(filename, index=False)
print(f"Submission file '{filename}' created successfully!")
sub_df.head()

