import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import TFRobertaModel
from transformers import RobertaTokenizerFast

pd.set_option('max_colwidth', None)
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')


train.head()


train.shape


train.info()


train.isnull().sum()


train.dropna(inplace=True)


train.isnull().sum()


test.head()


test.shape


test.info()


test.isnull().sum()


sns.countplot(x=train['sentiment'])
plt.title("Sentiment Distribution");


#Jaccard similarity
def jaccard(str1, str2): 
    a=set(str1.lower().split()) 
    b=set(str2.lower().split())
    c=a.intersection(b)
    return float(len(c))/(len(a)+len(b)-len(c))


train['jaccard_score']=train.apply(lambda x: jaccard(x['text'], x['selected_text']), axis=1)


train.groupby('sentiment')['jaccard_score'].describe()


plt.figure(figsize=(12,6))
sns.kdeplot(train[train['sentiment']=='positive']['jaccard_score'], label='positive', shade=True)
sns.kdeplot(train[train['sentiment']=='negative']['jaccard_score'], label='negative', shade=True)
sns.kdeplot(train[train['sentiment']=='neutral']['jaccard_score'], label='neutral', shade=True)
plt.title("Sentiment vs. Jaccard Similarity Distribution")
plt.legend();


#Spaces at the beginning/end
train['text']=train['text'].astype(str).str.replace(r'^\s+|\s+$', '', regex=True)
train['selected_text']=train['selected_text'].astype(str).str.replace(r'^\s+|\s+$', '', regex=True)
test['text']=test['text'].astype(str).str.replace(r'^\s+|\s+$', '', regex=True)


#RoBERTa-tokenizer-fast
MODEL_PATH = "/kaggle/input/roberta-base/"
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
MAX_LEN = 96


def generate_training_data(df, tokenizer, max_len):
    count = df.shape[0]
    #Inputs
    input_ids = np.zeros((count, max_len), dtype='int32')
    attention_mask = np.zeros((count, max_len), dtype='int32')
    #Outputs
    start_tokens = np.zeros((count, max_len), dtype='int32')
    end_tokens = np.zeros((count, max_len), dtype='int32')
    
    for i, row in df.iterrows():
        text = row['text']
        selected_text = row['selected_text']
        sentiment = row['sentiment']
        #Tokenize
        enc = tokenizer.encode_plus(sentiment,text,add_special_tokens=True,max_length=max_len,
            return_token_type_ids=False,padding='max_length',truncation=True,
            return_attention_mask=True,return_offsets_mapping=True)
        input_ids[i,] = enc['input_ids']
        attention_mask[i,] = enc['attention_mask']
        #Target
        mask_ids = enc.sequence_ids() 
        offsets = enc['offset_mapping']
        #Start/End character index
        start_char = text.find(selected_text)
        end_char = start_char + len(selected_text)
        #Character index to token index
        token_start_index = 0
        token_end_index = 0
        text_token_indices = [idx for idx, seq_id in enumerate(mask_ids) if seq_id == 1]
        if len(text_token_indices) > 0:
            idx_min = text_token_indices[0]
            idx_max = text_token_indices[-1]
            #Start/End token
            for idx in range(idx_min, idx_max + 1):
                if offsets[idx][0] <= start_char: 
                    token_start_index = idx
                if offsets[idx][1] >= end_char: 
                    token_end_index = idx
                    break 
        #Target index
        start_tokens[i, token_start_index] = 1
        end_tokens[i, token_end_index] = 1
    return input_ids, attention_mask, start_tokens, end_tokens


train.reset_index(drop=True, inplace=True)


input_ids, attention_mask, start_tokens, end_tokens = generate_training_data(train, tokenizer, MAX_LEN)


input_ids.shape


def build_model():
    #Input Layers
    ids = tf.keras.layers.Input((96,), dtype=tf.int32, name="input_ids")
    att = tf.keras.layers.Input((96,), dtype=tf.int32, name="attention_mask")
    #Pre-trained RoBERTa Model
    roberta = TFRobertaModel.from_pretrained(MODEL_PATH, from_pt=True, local_files_only=True)
    x = roberta(ids, attention_mask=att)[0]
    x1 = tf.keras.layers.Dropout(0.1)(x) 
    start_logits = tf.keras.layers.Dense(1)(x1)
    start_logits = tf.keras.layers.Flatten()(start_logits)
    end_logits = tf.keras.layers.Dense(1)(x1)
    end_logits = tf.keras.layers.Flatten()(end_logits)
    
    model = tf.keras.Model(inputs=[ids, att], outputs=[start_logits, end_logits])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)
    return model

model = build_model()


checkpoint = tf.keras.callbacks.ModelCheckpoint('roberta_model.h5',monitor='val_loss',verbose=1, 
    save_best_only=True,save_weights_only=True,mode='min')

history = model.fit(x=[input_ids, attention_mask],y=[start_tokens, end_tokens],epochs=3,batch_size=32, 
    verbose=1,validation_split=0.1,callbacks=[checkpoint])


model.load_weights('roberta_model.h5')


#Test data
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test_df.reset_index(drop=True, inplace=True)
test_df['text'] = test_df['text'].astype(str).str.replace(r'^\s+|\s+$', '', regex=True)

input_ids_test = np.zeros((test_df.shape[0], 96), dtype='int32')
attention_mask_test = np.zeros((test_df.shape[0], 96), dtype='int32')
for i, row in test_df.iterrows():
    try:
        enc = tokenizer.encode_plus(row['sentiment'], row['text'],add_special_tokens=True,max_length=96,
            padding='max_length',truncation=True,return_attention_mask=True)
        input_ids_test[i,] = enc['input_ids']
        attention_mask_test[i,] = enc['attention_mask']
    except:
        pass 

#Prediction
preds = model.predict([input_ids_test, attention_mask_test], verbose=1, batch_size=16)
start_preds = preds[0]
end_preds = preds[1]
final_preds = []
for i in range(test_df.shape[0]):
    try:
        sentiment = test_df.loc[i, 'sentiment']
        text = test_df.loc[i, 'text']
        #for neutral sentiment text=selected_text
        if sentiment == "neutral" or len(text.split()) < 2:
            final_preds.append(text)
            continue    
        idx_start = np.argmax(start_preds[i,])
        idx_end = np.argmax(end_preds[i,])
        if idx_start > idx_end:
            final_preds.append(text)
        else:
            text_tokens = tokenizer.decode(input_ids_test[i, idx_start:idx_end+1])
            clean_pred = text_tokens.strip()
            # Temizlik kontrolü
            if clean_pred == "" or clean_pred == "<s>" or clean_pred == "</s>":
                final_preds.append(text)
            else:
                final_preds.append(clean_pred)
    except:
        final_preds.append(test_df.loc[i, 'text'])


submission = pd.DataFrame()
submission['textID'] = test['textID']
submission['selected_text'] = final_preds
submission.to_csv('submission.csv', index=False)


submission.head(10)


import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import gc 

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(input_ids, train['sentiment'])):
    #Slicing
    train_ids = input_ids[train_idx]
    train_att = attention_mask[train_idx]
    train_start = start_tokens[train_idx]
    train_end = end_tokens[train_idx]
    #Validation
    val_ids = input_ids[val_idx]
    val_att = attention_mask[val_idx]
    val_start = start_tokens[val_idx]
    val_end = end_tokens[val_idx]   
    #Model
    tf.keras.backend.clear_session() # Önceki modelin çöpünü temizle
    model = build_model()
    #Checkpoint
    checkpoint_name = f'roberta_fold_{fold+1}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min') 
    #Training
    model.fit(x=[train_ids, train_att], y=[train_start, train_end], epochs=3, batch_size=32, verbose=1,
        validation_data=([val_ids, val_att], [val_start, val_end]),callbacks=[checkpoint])
    #RAM
    del train_ids, train_att, train_start, train_end, val_ids, val_att, val_start, val_end, model
    gc.collect()


#Test
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test_df.reset_index(drop=True, inplace=True)
test_df['text'] = test_df['text'].astype(str).str.replace(r'^\s+|\s+$', '', regex=True)

input_ids_test = np.zeros((test_df.shape[0], 96), dtype='int32')
attention_mask_test = np.zeros((test_df.shape[0], 96), dtype='int32')
for i, row in test_df.iterrows():
    try:
        enc = tokenizer.encode_plus(row['sentiment'], row['text'],add_special_tokens=True,max_length=96,
            padding='max_length',truncation=True,return_attention_mask=True)
        input_ids_test[i,] = enc['input_ids']
        attention_mask_test[i,] = enc['attention_mask']
    except:
        pass

final_start_logits = np.zeros((test_df.shape[0], 96))
final_end_logits = np.zeros((test_df.shape[0], 96))
FOLDS = 5
for fold in range(FOLDS):
    tf.keras.backend.clear_session()
    model = build_model()
    model.load_weights(f'roberta_fold_{fold+1}.h5')
    
    #Precition
    preds = model.predict([input_ids_test, attention_mask_test], verbose=1, batch_size=16)
    final_start_logits += preds[0]
    final_end_logits += preds[1]

final_start_logits /= FOLDS
final_end_logits /= FOLDS

final_preds = []
for i in range(test_df.shape[0]):
    try:
        sentiment = test_df.loc[i, 'sentiment']
        text = test_df.loc[i, 'text']
        if sentiment == "neutral" or len(text.split()) < 2:
            final_preds.append(text)
            continue    
        idx_start = np.argmax(final_start_logits[i,])
        idx_end = np.argmax(final_end_logits[i,])
        if idx_start > idx_end:
            final_preds.append(text)
        else:
            text_tokens = tokenizer.decode(input_ids_test[i, idx_start:idx_end+1])
            clean_pred = text_tokens.strip()
            if clean_pred == "" or clean_pred == "<s>" or clean_pred == "</s>":
                final_preds.append(text)
            else:
                final_preds.append(clean_pred)
    except:
        final_preds.append(test_df.loc[i, 'text'])


submission = pd.DataFrame()
submission['textID'] = test_df['textID']
submission['selected_text'] = final_preds
submission.to_csv('submission.csv', index=False)


submission.head(10)

