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

# ğŸ’¥ã€�è¿½åŠ ã€‘ãƒ‡ã‚£ãƒ¼ãƒ—ãƒ©ãƒ¼ãƒ‹ãƒ³ã‚°ã�¨BERTã�«å¿…è¦�ã�ªãƒ©ã‚¤ãƒ–ãƒ©ãƒª
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split

# --- 0. TPU/GPUæˆ¦ç•¥ã�®ã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ— ---
# ã�“ã‚Œã�«ã‚ˆã‚Šã€�Kerasã�¯åˆ©ç”¨å�¯èƒ½ã�ªTPUã�¾ã�Ÿã�¯GPUã‚’è‡ªå‹•çš„ã�«ä½¿ç”¨ã�—ã�¾ã�™
try:
    # TPUæ¤œå‡º
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except ValueError:
    # GPUã�¾ã�Ÿã�¯CPUã�«ãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯
    strategy = tf.distribute.get_strategy()
    print("Running on GPU/CPU")

print(f"REPLICAS: {strategy.num_replicas_in_sync}")
# 
GLOBAL_BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192 # 
MODEL_NAME = 'bert-base-uncased' # 
target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# --- 1. ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿ ---
print("Loading data...")
train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sample_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

# 
train_df, valid_df = train_test_split(
    train_df, test_size=0.1, random_state=42
)
print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")


# --- 2. ğŸ’¥ BERTã�®ã�Ÿã‚�ã�®ãƒˆãƒ¼ã‚¯ãƒ³åŒ– (å‰�å‡¦ç�†) ---
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 
def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=max_len,
        padding='max_length',    # 
        truncation=True,         # 
        return_attention_mask=True,
        return_token_type_ids=False, # 
        return_tensors='np'      # 
    )

print("Tokenizing data...")
X_train_encoded = bert_encode(train_df['comment_text'].values, tokenizer, MAX_LEN)
X_valid_encoded = bert_encode(valid_df['comment_text'].values, tokenizer, MAX_LEN)
X_test_encoded = bert_encode(test_df['comment_text'].values, tokenizer, MAX_LEN)

y_train = train_df[target_columns].values
y_valid = valid_df[target_columns].values


# --- 3. ğŸ’¥ tf.data.Dataset ã�®ä½œæˆ� (åŠ¹ç�‡åŒ–) ---
# 
def create_dataset(encodings, labels=None):
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((
            {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']},
            labels
        ))
    else:
        # 
        dataset = tf.data.Dataset.from_tensor_slices(
            {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']}
        )
    return dataset

# 
train_dataset = create_dataset(X_train_encoded, y_train)
train_dataset = train_dataset.shuffle(1000).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

valid_dataset = create_dataset(X_valid_encoded, y_valid)
valid_dataset = valid_dataset.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = create_dataset(X_test_encoded)
test_dataset = test_dataset.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --- 4. ğŸ’¥ BERTãƒ¢ãƒ‡ãƒ«ã�®æ§‹ç¯‰ (ãƒ•ã‚¡ã‚¤ãƒ³ãƒ�ãƒ¥ãƒ¼ãƒ‹ãƒ³ã‚°) ---
print("Building BERT model...")

# 
with strategy.scope():
    def build_model(model_name, max_len):
        # 
        bert_model = TFBertModel.from_pretrained(model_name)
        
        # 
        input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
        
        # 
        output = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # 
        cls_output = output[:, 0, :]
        
        # 
        x = tf.keras.layers.Dropout(0.1)(cls_output)
        
        # 
        # 
        out = tf.keras.layers.Dense(6, name='output_layer')(x)
        
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=out)
        
        # 
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), # 
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # 
            metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=6, name='auc')] # 
        )
        return model

    model = build_model(MODEL_NAME, MAX_LEN)

model.summary()


# --- 5. ğŸ’¥ ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’ ---
print("Training model...")
# 
# 
EPOCHS = 1 # 
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS
)


# --- 6. ğŸ’¥ äºˆæ¸¬ã�¨ãƒ•ã‚¡ã‚¤ãƒ«ä½œæˆ� ---
print("Predicting on test data...")
# 
predictions_logits = model.predict(test_dataset)

# 
predictions_proba = tf.nn.sigmoid(predictions_logits).numpy()


# 
sub_df = pd.DataFrame(predictions_proba, columns=target_columns)
sub_df['id'] = test_df['id'].values
sub_df = sub_df[['id'] + target_columns]

sub_df.to_csv("Kazu_submission11.csv", index=False)
print("Submission file created successfully!")
sub_df.head()

