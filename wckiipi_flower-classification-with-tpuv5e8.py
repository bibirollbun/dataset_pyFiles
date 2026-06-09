!export PATH="${HOME}/.local/bin:${PATH}" && uv pip uninstall --system jax
!export PATH="${HOME}/.local/bin:${PATH}" && uv pip install --system tensorflow-tpu=="2.18.0" --find-links https://storage.googleapis.com/libtpu-tf-releases/index.html


import tensorflow as tf
from kaggle_datasets import KaggleDatasets
import numpy as np

print("Tensorflow version " + tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print(f"Running on TPU: {tpu.master()}")
    print("REPLICAS: ", strategy.num_replicas_in_sync)
except Exception as e:
    print(f"Error initializing TPU: {e}")
    # Fallback to default
    strategy = tf.distribute.get_strategy()
    print("Running on CPU/GPU")


# BTW running does nothing since following gets printed
# get_gcs_path is not required on TPU VMs which can directly use Kaggle datasets, using path: /kaggle/input/flower-classification-with-tpus
# GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"


# IMAGE_SIZE = [192, 192] # at this size, a GPU will run out of memory. Use the TPU
# EPOCHS = 5
# BATCH_SIZE = 16 * strategy.num_replicas_in_sync

# NUM_TRAINING_IMAGES = 12753
# NUM_TEST_IMAGES = 7382
# STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
# AUTO = tf.data.experimental.AUTOTUNE


# DATA_PATH = "/kaggle/input/flower-classification-with-tpus" 
# IMAGE_SIZE = [192, 192]
# EPOCHS = 5

# BATCH_SIZE = 16 * strategy.num_replicas_in_sync
# AUTO = tf.data.AUTOTUNE
# NUM_TRAINING_IMAGES = 12753
# STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

# def decode_image(image_data):
#     image = tf.image.decode_jpeg(image_data, channels=3)
#     image = tf.cast(image, tf.float32) / 255.0  
#     image = tf.reshape(image, [*IMAGE_SIZE, 3]) 
#     return image

# def read_labeled_tfrecord(example):
#     LABELED_TFREC_FORMAT = {
#         "image": tf.io.FixedLenFeature([], tf.string), 
#         "class": tf.io.FixedLenFeature([], tf.int64),  
#     }
#     example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
#     image = decode_image(example['image'])
#     label = tf.cast(example['class'], tf.int32)
#     return image, label 

# def read_unlabeled_tfrecord(example):
#     UNLABELED_TFREC_FORMAT = {
#         "image": tf.io.FixedLenFeature([], tf.string), 
#         "id": tf.io.FixedLenFeature([], tf.string),  
#     }
#     example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
#     image = decode_image(example['image'])
#     idnum = example['id']
#     return image, idnum 

# def load_dataset(filenames, labeled=True, ordered=False):
#     ignore_order = tf.data.Options()
#     if not ordered:
#         ignore_order.experimental_deterministic = False 
#     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 
#     dataset = dataset.with_options(ignore_order) 
#     dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
#     return dataset

# # def get_mat_dataset(filenames, labeled=True, ordered=False):
# #     dataset = load_dataset(filenames, labeled=labeled, ordered=ordered)
# #     dataset = dataset.batch(1024) # Large batch for faster loading
    
# #     print(f"Loading data into RAM... (This will take ~60 seconds)")
    
# #     all_images = []
# #     all_labels = []
    
# #     for batch_imgs, batch_lbls in dataset:
# #         all_images.append(batch_imgs.numpy())
# #         all_labels.append(batch_lbls.numpy())
        
# #     all_images = np.concatenate(all_images)
# #     all_labels = np.concatenate(all_labels)
    
# #     print(f"Successfully loaded {len(all_images)} images into RAM.")
    
# #     ram_ds = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
# #     return ram_ds

# def get_training_dataset():
#     dataset = get_mat_dataset(tf.io.gfile.glob(DATA_PATH + '/tfrecords-jpeg-192x192/train/*.tfrec'), labeled=True)
#     dataset = dataset.repeat()
#     dataset = dataset.shuffle(2048)
#     dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 
#     dataset = dataset.prefetch(AUTO)
#     return dataset

# def get_validation_dataset():
#     dataset = get_mat_dataset(tf.io.gfile.glob(DATA_PATH + '/tfrecords-jpeg-192x192/val/*.tfrec'), labeled=True, ordered=False)
#     dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
#     dataset = dataset.prefetch(AUTO)
#     return dataset

# training_dataset = get_training_dataset()
# validation_dataset = get_validation_dataset()


DATA_PATH = "/kaggle/input/flower-classification-with-tpus"
GCS_DS_PATH = DATA_PATH
IMAGE_SIZE = [192, 192]
EPOCHS = 5

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUTO = tf.data.AUTOTUNE
NUM_TRAINING_IMAGES = 12753
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_training_dataset():
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/train/*.tfrec'), labeled=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/val/*.tfrec'), labeled=True, ordered=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/test/*.tfrec'), labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()


with strategy.scope():    
    pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = False # tramsfer learning
    
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(104, activation='softmax')
    ])
        
    model.compile(
        optimizer='adam',
        loss = 'sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    
    historical = model.fit(training_dataset, 
              steps_per_epoch=STEPS_PER_EPOCH, 
              epochs=EPOCHS, 
              validation_data=validation_dataset)


