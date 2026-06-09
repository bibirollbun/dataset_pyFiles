import os
os.environ["KERAS_BACKEND"] = "jax"
import keras_nlp
import keras 
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


print("TenosrFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasNLP", keras_nlp.__version__)


class CFG:
    seed = 42  
    preset = "deberta_v3_extra_small_en" 
    sequence_length = 512  
    epochs = 3 
    batch_size = 16  
    scheduler = 'cosine' 
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}
    name2label = {v:k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())


keras.utils.set_random_seed(CFG.seed)


keras.mixed_precision.set_global_policy("mixed_float16")


base_path = '/kaggle/input/lmsys-chatbot-arena'


df = pd.read_csv(f'{base_path}/train.csv')


df["prompt"] = df.prompt.map(lambda x: eval(x)[0])
df["response_a"] = df.response_a.map(lambda x: eval(x.replace("null", "''"))[0])
df["response_b"] = df.response_b.map(lambda x: eval(x.replace("null","''"))[0])


df["class_name"] = df[["winner_model_a", "winner_model_b" , "winner_tie"]].idxmax(axis=1)
df["class_label"] = df.class_name.map(CFG.name2label)
df.head()


test_df = pd.read_csv(f'{base_path}/test.csv')


test_df["prompt"] = test_df.prompt.map(lambda x: eval(x)[0])
test_df["response_a"] = test_df.response_a.map(lambda x: eval(x.replace("null","''"))[0])
test_df["response_b"] = test_df.response_b.map(lambda x: eval(x.replace("null", "''"))[0])


test_df.head()


def make_pairs(row):
    row["encode_fail"] = False
    try:
        prompt = row.prompt.encode("utf-8").decode("utf-8")
    except:
        prompt = ""
        row["encode_fail"] = True

    try:
        response_a = row.response_a.encode("utf-8").decode("utf-8")
    except:
        response_a = ""
        row["encode_fail"] = True

    try:
        response_b = row.response_b.encode("utf-8").decode("utf-8")
    except:
        response_b = ""
        row["encode_fail"] = True
        
    row['options'] = [f"Prompt: {prompt}\n\nResponse: {response_a}", 
                      f"Prompt: {prompt}\n\nResponse: {response_b}"  
                     ]
    return row


df = df.apply(make_pairs, axis=1)
display(df.head(2))
test_df = test_df.apply(make_pairs, axis=1) 
display(test_df.head(2))


df.encode_fail.value_counts(normalize=False)


model_df = pd.concat([df.model_a, df.model_b])
counts = model_df.value_counts().reset_index()
counts.columns = ['LLM', 'Count']

fig = px.bar(counts, x='LLM', y='Count',
             title='Distribution of LLMs',
             color='Count', color_continuous_scale='viridis')

fig.update_layout(xaxis_tickangle=-45)  

fig.show()


counts = df['class_name'].value_counts().reset_index()
counts.columns = ['Winner', 'Win Count']

fig = px.bar(counts, x='Winner', y='Win Count',
             title='Winner distribution for Train Data',
             labels={'Winner': 'Winner', 'Win Count': 'Win Count'},
             color='Winner', color_continuous_scale='viridis')

fig.update_layout(xaxis_title="Winner", yaxis_title="Win Count")

fig.show()


from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df["class_label"])


preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
    preset=CFG.preset, 
    sequence_length=CFG.sequence_length, 
)


outs = preprocessor(df.options.iloc[0])  
for k, v in outs.items():
    print(k, ":", v.shape)


def preprocess_fn(text, label=None):
    text = preprocessor(text)
    return (text, label) if label is not None else text


def build_dataset(texts, labels=None, batch_size=32,
                  cache=True, shuffle=1024):
    AUTO = tf.data.AUTOTUNE 
    slices = (texts,) if labels is None else (texts, keras.utils.to_categorical(labels, num_classes=3)) 
    ds = tf.data.Dataset.from_tensor_slices(slices) 
    ds = ds.cache() if cache else ds 
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTO) 
    opt = tf.data.Options()
    if shuffle: 
        ds = ds.shuffle(shuffle, seed=CFG.seed) 
        opt.experimental_deterministic = False
    ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO) 
    return ds


train_texts = train_df.options.tolist() 
train_labels = train_df.class_label.tolist()  
train_ds = build_dataset(train_texts, train_labels,
                         batch_size=CFG.batch_size,
                         shuffle=True)

valid_texts = valid_df.options.tolist()  
valid_labels = valid_df.class_label.tolist() 
valid_ds = build_dataset(valid_texts, valid_labels,
                         batch_size=CFG.batch_size,
                         shuffle=False)


import math

def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 1.0e-6, 0.6e-6 * batch_size, 1e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 2, 0, 0.8

    def lrfn(epoch):  
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  


lr_cb = get_lr_callback(CFG.batch_size, plot=True)



ckpt_cb = keras.callbacks.ModelCheckpoint(f'best_model.weights.h5',
                                          monitor='val_log_loss',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          mode='min')  


log_loss = keras.metrics.CategoricalCrossentropy(name="log_loss")


inputs = {
    "token_ids": keras.Input(shape=(2, None), dtype=tf.int32, name="token_ids"),
    "padding_mask": keras.Input(shape=(2, None), dtype=tf.int32, name="padding_mask"),
}
backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
    CFG.preset,
)

response_a = {k: v[:, 0, :] for k, v in inputs.items()}
embed_a = backbone(response_a)

response_b = {k: v[:, 1, :] for k, v in inputs.items()}
embed_b = backbone(response_b)

embeds = keras.layers.Concatenate(axis=-1)([embed_a, embed_b])
embeds = keras.layers.GlobalAveragePooling1D()(embeds)
outputs = keras.layers.Dense(3, activation="softmax", name="classifier")(embeds)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(5e-6),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
    metrics=[
        log_loss,
        keras.metrics.CategoricalAccuracy(name="accuracy"),
    ],
)


model.summary()

