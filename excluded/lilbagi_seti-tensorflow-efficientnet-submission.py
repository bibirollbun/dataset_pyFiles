!pip install -q efficientnet


import os

import pandas as pd
import numpy as np
import efficientnet.tfkeras as efn
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path


def build_decoder(with_labels = True, target_size=(256, 256), ext='npy'):
    def decode(path):
        file_bytes = tf.io.read_file(path)

        if ext == 'npy':
            header_size = 128
            
            data_bytes = tf.strings.substr(file_bytes, header_size, -1)
            img = tf.io.decode_raw(data_bytes, tf.float16)
            
            img = tf.reshape(img, [6, 273, 256])
            r = tf.concat([img[0], img[1]], axis =0)
            g = tf.concat([img[2], img[3]], axis =0)
            b = tf.concat([img[4], img[5]], axis =0)
            img = tf.stack([r, g, b], axis=-1)
        else:
            if ext == 'png':
                img = tf.image.decode_png(file_bytes, channels = 3)
            elif ext in ['jpg', 'jpeg']:
                img = tf.image.decode_jpeg(file_bytes, channels = 3)
            else:
                raise ValueError("Image extension not supported")
                
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)

        return img

    def decode_with_labels(path, label):
        return decode(path), tf.cast(label, tf.float32)

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.cast(img, tf.float32)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img
    
    def augment_with_labels(img, label):
        return augment(img), tf.cast(label, tf.float32)
    
    return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None, bsize=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)
    
    return dset


data_dir = Path('../input/seti-breakthrough-listen/')

test_data_dir = data_dir / 'test'
sample_sub = data_dir / 'sample_submission.csv'
sub_df = pd.read_csv(sample_sub)

def id_to_path(file_id):
    return str(test_data_dir / file_id[0] / f"{file_id}.npy")

test_paths = sub_df["id"].apply(id_to_path)


BATCH_SIZE = 128

test_decoder = build_decoder(with_labels = False, target_size=(260, 260), ext = 'npy')
dataset = build_dataset(
    test_paths, bsize=BATCH_SIZE, repeat=False, 
    shuffle=False, augment=False, cache=False,
    decode_fn=test_decoder
)


models = []

model0 = tf.keras.models.load_model("/kaggle/input/seti-models/model_fold_0.h5")
model1 = tf.keras.models.load_model("/kaggle/input/seti-models/model_fold_1.h5")
model2 = tf.keras.models.load_model("/kaggle/input/seti-models/model_fold_2.h5")

models.append(model0)
models.append(model1)
models.append(model2)


sub_df['target'] = sum([model.predict(dataset, verbose=1) for model in models]) / len(models)


sub_df.to_csv('submission.csv',index=False)

