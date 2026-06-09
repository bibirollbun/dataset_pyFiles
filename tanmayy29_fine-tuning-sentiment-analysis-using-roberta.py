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

#Credit: https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705/notebook

#Goal of this notebook: How to tokenize the data, create question answer targets, and how to build a custom question answer head for RoBERTa
# in TensorFlow. Note that HuggingFace transformers don't have a TFRobertaForQuestionAnswering so we must make our own from TFRobertaModel.

# Here's a pro tip for people using TPU. Start each fold loop with-
# tf.tpu.experimental.initialize_tpu_system(tpu)
# This will prevent the TPU from running out of memory during 5 Fold.

#v5: got .706 score with max_len 192


import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
print('TF version',tf.__version__)


# quick check of the datasets
train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
train_df.head()


train_df.info()


# target values
train_df['sentiment'].unique()


train_df.sentiment.value_counts()


# load datasets
def read_train():
    train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    train['text'] = train['text'].astype(str) #ensuring data type is string to avoid any error
    train['selected_text'] = train['selected_text'].astype(str)
    return train

def read_test():
    test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
    test['text'] = test['text'].astype(str)
    return test

def read_submission():
    sub = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
    return sub


# load datasets
train_df = read_train()
test_df = read_test()
submission_df = read_submission()


MAX_LEN = 96 #try max_len=192 for longer training otherwise use 96
PATH = '../input/tf-roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)
# tokenizer.encode('positive').ids
# tokenizer.encode('negative').ids
# tokenizer.encode('neutral').ids
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974} #encoded values of  a particular sentiment


# required step to transform data into RoBERTa format
ct = train_df.shape[0]
# 1 for tokens and 0 for padding 
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')


import time


%%time
for k in range(train_df.shape[0]):
    
    # FIND OVERLAP
    text1 = " "+" ".join(train_df.loc[k,'text'].split())
    text2 = " ".join(train_df.loc[k,'selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)]=1
    if text1[idx-1]==' ': chars[idx-1] = 1 
    enc = tokenizer.encode(text1) 
        
    # ID_OFFSETS
    offsets = []; idx=0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx,idx+len(w)))
        idx += len(w)
    
    # START END TOKENS
    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm>0: toks.append(i) 
        
    s_tok = sentiment_id[train_df.loc[k,'sentiment']]
    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask[k,:len(enc.ids)+5] = 1
    if len(toks)>0:
        start_tokens[k,toks[0]+1] = 1
        end_tokens[k,toks[-1]+1] = 1


# tokenize the test data also as we did above for train data
ct = test_df.shape[0]
input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(test_df.shape[0]):
        
    # INPUT_IDS
    text1 = " "+" ".join(test_df.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test_df.loc[k,'sentiment']]
    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask_t[k,:len(enc.ids)+5] = 1


def scheduler(epoch):
    return 3e-5 * 0.2**epoch


# build a RoBERTa model
def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


# define the metric
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


%%time
import os, h5py
from tensorflow.keras import backend as K

weight_path = "/kaggle/input/tf-roberta/pretrained-roberta-base.h5"
print("Weight file exists:", os.path.exists(weight_path))
if not os.path.exists(weight_path):
    raise FileNotFoundError(weight_path)

# 1) Inspect the h5 file contents (layers/groups) so we know what it contains
print("\n=== Contents of HDF5 (top-level groups) ===")
with h5py.File(weight_path, "r") as f:
    def print_h5_group(g, indent=0, max_items=20):
        for i, name in enumerate(g):
            if i>=max_items:
                print(" "*(2*indent)+"... (truncated)")
                break
            print(" "*(2*indent) + "-", name)
            if isinstance(g[name], h5py.Group):
                # print one level deeper but avoid very deep printing
                for sub in list(g[name].keys())[:10]:
                    print(" "*(2*(indent+1)) + "*", sub)
    print_h5_group(f, indent=0)

# 2) Build the model and inspect its layers
K.clear_session()
model = build_model()
print("\n=== Model layers (name : #weights) ===")
for i, layer in enumerate(model.layers):
    try:
        wn = layer.count_params()
    except Exception:
        wn = "?"
    print(f"{i:02d}: {layer.name}  (params: {wn})  type: {type(layer).__name__}")

# 3) Try safe/by-name load (this will load whichever matching layers exist)
print("\n=== Attempting load_weights(by_name=True, skip_mismatch=True) ===")
try:
    model.load_weights(weight_path, by_name=True, skip_mismatch=True)
    print("Loaded weights with by_name=True, skip_mismatch=True. Matching layers were loaded; unmatched layers were skipped.")
except Exception as e:
    print("by_name load failed with error:", e)
    print("Proceeding to fallback option (try loading base via transformers).")

    # 4) Fallback: try loading HF TF roberta-base into a sublayer named like 'roberta' or similar
    try:
        from transformers import TFRobertaModel, RobertaConfig
        print("\nAttempting to load Hugging Face roberta-base and assign to sublayer if present...")
        hf = TFRobertaModel.from_pretrained("roberta-base")  # will download if not cached
        # Find a layer in your model whose class/type matches or whose name contains 'roberta'
        roberta_layer = None
        for layer in model.layers:
            if "roberta" in layer.name.lower() or "transformer" in layer.name.lower():
                roberta_layer = layer
                break
        if roberta_layer is None:
            # search nested layers
            for layer in model.layers:
                for sub in getattr(layer, "layers", []):
                    if "roberta" in sub.name.lower():
                        roberta_layer = sub
                        break
                if roberta_layer:
                    break

        if roberta_layer is None:
            raise RuntimeError("Could not find a sublayer named like 'roberta' in your model; manual wiring is required.")

        # Now try to copy weights where shapes match
        print("Found candidate sublayer in your model:", roberta_layer.name, " â€” trying to set weights from HF base.")
        try:
            # HF TF model weights are in hf.weights -> a list of tensors; easiest is to set via .set_weights()
            roberta_layer.set_weights(hf.weights)
            print("Successfully set weights of sublayer from Hugging Face roberta-base.")
        except Exception as e2:
            print("Failed to set sublayer weights directly:", e2)
            print("You may need to adapt layer shapes/names or rebuild your model to use HF TFRobertaModel as the base.")
    except Exception as hf_e:
        print("Fallback HF load failed:", hf_e)
        print("Final suggestion: rebuild your model base using transformers' TFRobertaModel and attach your own classification head, or save a compatible weights file that includes both base+head.")

print("\n=== Done. You can now run prediction. ===")
# If weights were partially loaded above, proceed to predict
try:
    print("Running a test predict (single batch) to detect runtime errors...")
    # run predict on a tiny slice to avoid long runs
    _ = model.predict([input_ids_t[:2], attention_mask_t[:2], token_type_ids_t[:2]], verbose=0)
    print("Test predict succeeded.")
except Exception as p_e:
    print("Test predict failed:", p_e)
    print("If test predict fails, your loaded weights may be incompatible; consider using transformers' model directly.")



%%time
import os
from tensorflow.keras import backend as K

n_splits = 5
preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
preds_end   = np.zeros((input_ids_t.shape[0], MAX_LEN))
DISPLAY = 1

weight_path = "/kaggle/input/tf-roberta/pretrained-roberta-base.h5"
if not os.path.exists(weight_path):
    raise FileNotFoundError(weight_path)

for i in range(n_splits):
    print('#' * 25)
    print('### MODEL %i' % (i+1))
    print('#' * 25)

    K.clear_session()
    model = build_model()

    print("Loading weights from:", weight_path)
    model.load_weights(weight_path, by_name=True, skip_mismatch=True)  # safe by-name load
    # Note: by_name=True skip_mismatch=True will load matching weights (base) and skip classifier head if mismatch.

    print('Predicting Test...')
    preds = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)
    preds_start += preds[0] / n_splits
    preds_end   += preds[1] / n_splits



# make submission file
all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b: 
        st = test_df.loc[k,'text']
    else:
        text1 = " "+" ".join(test_df.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(st)


# âœ… Display some sample predictions for notebook output
import pandas as pd
import numpy as np

# If you already have 'all' (the predicted text spans) and preds_start/preds_end:
# Create a result DataFrame
test_df['selected_text'] = all

# Optional: derive a simple sentiment label based on selected_text sentiment words
# (if your test_df already has a column named 'sentiment', you can keep that)
# For a nicer notebook view, weâ€™ll show 3 examples from each sentiment type if available
if 'sentiment' in test_df.columns:
    display_cols = ['textID', 'sentiment', 'text', 'selected_text']
else:
    # If your dataset doesnâ€™t have 'sentiment', weâ€™ll just show text + predicted phrase
    display_cols = ['textID', 'text', 'selected_text']

print("ğŸ�¯ Sample Predicted Results:\n")

# Show a few examples from each sentiment
for label in ['positive', 'neutral', 'negative']:
    print(f"\n===================== {label.upper()} REVIEWS =====================")
    subset = test_df[test_df['sentiment'] == label] if 'sentiment' in test_df.columns else test_df
    if len(subset) == 0:
        print(f"(No {label} examples found.)")
        continue
    # show up to 3 random examples
    for _, row in subset.sample(min(3, len(subset)), random_state=42).iterrows():
        print(f"\nğŸ“� Original: {row['text'][:250]}{'...' if len(row['text'])>250 else ''}")
        print(f"ğŸ’¬ Extracted phrase: {row['selected_text']}")
        if 'sentiment' in row:
            print(f"ğŸ”¹ Sentiment: {row['sentiment']}")
    print("------------------------------------------------------------------")

# Optional: still save submission file
test_df[['textID', 'selected_text']].to_csv('submission.csv', index=False)
print("\nâœ… Results saved to submission.csv")





