import os
os.environ["KERAS_BACKEND"] = "tensorflow" # other options: tensorflow or torch

import keras_cv
import keras
from keras import ops
import tensorflow as tf

import cv2
import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import joblib

import matplotlib.pyplot as plt


print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasCV:", keras_cv.__version__)


class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    neg_sample = 0.01 # Downsample negative calss
    pos_sample = 5.0  # Upsample positive class
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [128, 128]  # Input image size
    epochs = 8 # Training epochs
    batch_size = 16  # Batch size
    lr_mode = "cos" # LR scheduler mode from one of "cos", "step", "exp"
    class_names = ['target']
    num_classes = 1


keras.utils.set_random_seed(CFG.seed)


BASE_PATH = "/kaggle/input/isic-2024-challenge"


# Train + Valid
df = pd.read_csv(f'{BASE_PATH}/train-metadata.csv')
df = df.ffill()
display(df.head(2))

# Testing
testing_df = pd.read_csv(f'{BASE_PATH}/test-metadata.csv')
testing_df = testing_df.ffill()
display(testing_df.head(2))


print("Class Distribution Before Sampling (%):")
display(df.target.value_counts(normalize=True)*100)

# Sampling
positive_df = df.query("target==0").sample(frac=CFG.neg_sample, random_state=CFG.seed)
negative_df = df.query("target==1").sample(frac=CFG.pos_sample, replace=True, random_state=CFG.seed)
df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0)

print("\nCalss Distribution After Sampling (%):")
display(df.target.value_counts(normalize=True)*100)


from sklearn.utils.class_weight import compute_class_weight

# Assume df is your DataFrame and 'target' is the column with class labels
class_weights = compute_class_weight('balanced', classes=np.unique(df['target']), y=df['target'])
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)


import h5py

training_validation_hdf5 = h5py.File(f"{BASE_PATH}/train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"{BASE_PATH}/test-image.hdf5", 'r')


isic_id = df.isic_id.iloc[0]

# Image as Byte String
byte_string = training_validation_hdf5[isic_id][()]
print(f"Byte String: {byte_string[:20]}....")

# Convert byte string to numpy array
nparr = np.frombuffer(byte_string, np.uint8)

print("Image:")
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,::-1] # reverse last axis for bgr -> rgb
plt.imshow(image);


from sklearn.model_selection import StratifiedGroupKFold

df = df.reset_index(drop=True) # ensure continuous index
df["fold"] = -1
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
for i, (training_idx, validation_idx) in enumerate(sgkf.split(df, y=df.target, groups=df.patient_id)):
    df.loc[validation_idx, "fold"] = int(i)

# Use first fold for training and validation
training_df = df.query("fold!=0")
validation_df = df.query("fold==0")
print(f"# Num Train: {len(training_df)} | Num Valid: {len(validation_df)}")


training_df.target.value_counts()


validation_df.target.value_counts()


# Categorical features which will be one hot encoded
CATEGORICAL_COLUMNS = ["sex", "anatom_site_general",
            "tbp_tile_type","tbp_lv_location", ]

# Numeraical features which will be normalized
NUMERIC_COLUMNS = ["age_approx", "tbp_lv_nevi_confidence", "clin_size_long_diam_mm",
           "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
           "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM", ]

# Tabular feature columns
FEAT_COLS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def build_augmenter():
    # Define augmentations
    aug_layers = [
        keras_cv.layers.RandomCutout(height_factor=(0.02, 0.06), width_factor=(0.02, 0.06)),
        keras_cv.layers.RandomFlip(mode="horizontal"),
    ]
    
    # Apply augmentations to random samples
    aug_layers = [keras_cv.layers.RandomApply(x, rate=0.5) for x in aug_layers]
    
    # Build augmentation layer
    augmenter = keras_cv.layers.Augmenter(aug_layers)

    # Apply augmentations
    def augment(inp, label):
        images = inp["images"]
        aug_data = {"images": images}
        aug_data = augmenter(aug_data)
        inp["images"] = aug_data["images"]
        return inp, label
    return augment


def build_decoder(with_labels=True, target_size=CFG.image_size):
    def decode_image(inp):
        # Read jpeg image
        file_bytes = inp["images"]
        image = tf.io.decode_jpeg(file_bytes)
        
        # Resize
        image = tf.image.resize(image, size=target_size, method="area")
        
        # Rescale image
        image = tf.cast(image, tf.float32)
        image /= 255.0
        
        # Reshape
        image = tf.reshape(image, [*target_size, 3])
        
        inp["images"] = image
        return inp

    def decode_label(label, num_classes):
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [num_classes])
        return label

    def decode_with_labels(inp, label=None):
        inp = decode_image(inp)
        label = decode_label(label, CFG.num_classes)
        return (inp, label)

    return decode_with_labels if with_labels else decode_image


def build_dataset(
    isic_ids,
    hdf5,
    features,
    labels=None,
    batch_size=32,
    decode_fn=None,
    augment_fn=None,
    augment=False,
    shuffle=1024,
    cache=True,
    drop_remainder=False,
):
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter()

    AUTO = tf.data.experimental.AUTOTUNE

    images = [None]*len(isic_ids)
    for i, isic_id in enumerate(tqdm(isic_ids, desc="Loading Images ")):
        images[i] = hdf5[isic_id][()]
        
    inp = {"images": images, "features": features}
    slices = (inp, labels) if labels is not None else inp

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.cache() if cache else ds
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt = tf.data.Options()
        opt.deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds


## Train
print("# Training:")
training_features = dict(training_df[FEAT_COLS])
training_ids = training_df.isic_id.values
training_labels = training_df.target.values
training_ds = build_dataset(training_ids, training_validation_hdf5, training_features, 
                         training_labels, batch_size=CFG.batch_size,
                         shuffle=True, augment=True)

# Valid
print("# Validation:")
validation_features = dict(validation_df[FEAT_COLS])
validation_ids = validation_df.isic_id.values
validation_labels = validation_df.target.values
validation_ds = build_dataset(validation_ids, training_validation_hdf5, validation_features,
                         validation_labels, batch_size=CFG.batch_size,
                         shuffle=False, augment=False)


feature_space = keras.utils.FeatureSpace(
    features={
        # Categorical features encoded as integers
        "sex": "string_categorical",
        "anatom_site_general": "string_categorical",
        "tbp_tile_type": "string_categorical",
        "tbp_lv_location": "string_categorical",
        # Numerical features to discretize
        "age_approx": "float_discretized",
        # Numerical features to normalize
        "tbp_lv_nevi_confidence": "float_normalized",
        "clin_size_long_diam_mm": "float_normalized",
        "tbp_lv_areaMM2": "float_normalized",
        "tbp_lv_area_perim_ratio": "float_normalized",
        "tbp_lv_color_std_mean": "float_normalized",
        "tbp_lv_deltaLBnorm": "float_normalized",
        "tbp_lv_minorAxisMM": "float_normalized",
    },
    output_mode="concat",
)


training_ds_with_no_labels = training_ds.map(lambda x, _: x["features"])
feature_space.adapt(training_ds_with_no_labels)


for x, _ in training_ds.take(1):
    preprocessed_x = feature_space(x["features"])
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)


training_ds = training_ds.map(
    lambda x, y: ({"images": x["images"],
                   "features": feature_space(x["features"])}, y), num_parallel_calls=tf.data.AUTOTUNE)

validation_ds = validation_ds.map(
    lambda x, y: ({"images": x["images"],
                   "features": feature_space(x["features"])}, y), num_parallel_calls=tf.data.AUTOTUNE)


batch = next(iter(validation_ds))

print("Images:",batch[0]["images"].shape)
print("Features:", batch[0]["features"].shape)
print("Targets:", batch[1].shape)


# AUC
auc = keras.metrics.AUC()

# Loss
loss = keras.losses.BinaryCrossentropy(label_smoothing=0.02)


# Define input layers
image_input = keras.Input(shape=(*CFG.image_size, 3), name="images")
feat_input = keras.Input(shape=(feature_space.get_encoded_features().shape[1],), name="features")
inp = {"images":image_input, "features":feat_input}

# Branch for image input
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(CFG.preset)
x1 = backbone(image_input)
x1 = keras.layers.GlobalAveragePooling2D()(x1)
x1 = keras.layers.Dropout(0.2)(x1)

# Branch for tabular/feature input
x2 = keras.layers.Dense(96, activation="selu")(feat_input)
x2 = keras.layers.Dense(128, activation="selu")(x2)
x2 = keras.layers.Dropout(0.1)(x2)

# Concatenate both branches
concat = keras.layers.Concatenate()([x1, x2])

# Output layer
out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(concat)

# Build model
model = keras.models.Model(inp, out)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss,
    metrics=[auc],
)

# Model Summary
model.summary()


keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, dpi=60)


import math

def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 2.5e-5, 5e-6 * batch_size, 0.8e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback


inputs, targets = next(iter(training_ds))
images = inputs["images"]
num_images, NUMERIC_COLUMNS = 8, 4

plt.figure(figsize=(4 * NUMERIC_COLUMNS, num_images // NUMERIC_COLUMNS * 4))
for i, (image, target) in enumerate(zip(images[:num_images], targets[:num_images])):
    plt.subplot(num_images // NUMERIC_COLUMNS, NUMERIC_COLUMNS, i + 1)
    image = image.numpy().astype("float32")
    target= target.numpy().astype("int32")[0]
    
    image = (image - image.min()) / (image.max() + 1e-4)

    plt.imshow(image)
    plt.title(f"Target: {target}")
    plt.axis("off")

plt.tight_layout()
plt.show()


lr_cb = get_lr_callback(CFG.batch_size, mode="exp", plot=True)


ckpt_cb = keras.callbacks.ModelCheckpoint(
    "best_model.keras",   # Filepath where the model will be saved.
    monitor="val_auc",    # Metric to monitor (validation AUC in this case).
    save_best_only=True,  # Save only the model with the best performance.
    save_weights_only=False,  # Save the entire model (not just the weights).
    mode="max",           # The model with the maximum 'val_auc' will be saved.
)


history = model.fit(
    training_ds,
    epochs=CFG.epochs,
    callbacks=[lr_cb, ckpt_cb],
    validation_data=validation_ds,
    verbose=CFG.verbose,
    class_weight=class_weights,
)


# Extract AUC and validation AUC from history
auc = history.history['auc']
val_auc = history.history['val_auc']
epochs = range(1, len(auc) + 1)

# Find the epoch with the maximum val_auc
max_val_auc_epoch = np.argmax(val_auc)
max_val_auc = val_auc[max_val_auc_epoch]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, auc, 'o-', label='Training AUC', markersize=5, color='tab:blue')
plt.plot(epochs, val_auc, 's-', label='Validation AUC', markersize=5, color='tab:orange')

# Highlight the max val_auc
plt.scatter(max_val_auc_epoch + 1, max_val_auc, color='red', s=100, label=f'Max Val AUC: {max_val_auc:.4f}')
plt.annotate(f'Max Val AUC: {max_val_auc:.4f}', 
             xy=(max_val_auc_epoch + 1, max_val_auc), 
             xytext=(max_val_auc_epoch + 1 + 0.5, max_val_auc - 0.05),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=12,
             color='tab:red')

# Enhancing the plot
plt.title('AUC over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.xticks(epochs)

# Show the plot
plt.show()


# Best Result
best_score = max(history.history['val_auc'])
best_epoch = np.argmax(history.history['val_auc']) + 1
print("#" * 10 + " Result " + "#" * 10)
print(f"Best AUC: {best_score:.5f}")
print(f"Best Epoch: {best_epoch}")
print("#" * 28)


model.load_weights("best_model.keras")


# Testing
print("# Testing:")
testing_features = dict(testing_df[FEAT_COLS])
testing_ids = testing_df.isic_id.values
testing_ds = build_dataset(testing_ids, testing_hdf5,
                        testing_features, batch_size=CFG.batch_size,
                         shuffle=False, augment=False, cache=False)
# Apply feature space processing
testing_ds = testing_ds.map(
    lambda x: {"images": x["images"],
               "features": feature_space(x["features"])}, num_parallel_calls=tf.data.AUTOTUNE)


preds = model.predict(testing_ds).squeeze()


inputs = next(iter(testing_ds))
images = inputs["images"]

# Plotting
plt.figure(figsize=(10, 4))

for i in range(3):
    plt.subplot(1, 3, i+1)  # 1 row, 3 columns, i+1th subplot
    plt.imshow(images[i])  # Show image
    plt.title(f'Prediction: {preds[i]:.2f}')  # Set title with prediction
    plt.axis('off')  # Hide axis

plt.suptitle('Model Predictions on Testing Images', fontsize=16)
plt.tight_layout()
plt.show()


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy  # Import the necessary metrics

# Define input layers for images and tabular metadata
image_input = keras.Input(shape=(128, 128, 3), name="images")
feat_input = keras.Input(shape=(feature_space.get_encoded_features().shape[1],), name="features")
inp = {"images": image_input, "features": feat_input}

# Branch for image input using EfficientNet backbone
backbone = EfficientNetB0(weights=None, include_top=False, input_shape=(128, 128, 3))
x1 = backbone(image_input)
x1 = keras.layers.GlobalAveragePooling2D()(x1)
x1 = keras.layers.BatchNormalization()(x1)  # Adding Batch Normalization

# Branch for tabular/feature input
x2 = keras.layers.Dense(128, activation="relu")(feat_input)  # Increased units and changed activation
x2 = keras.layers.Dense(256, activation="relu")(x2)  # Increased units
x2 = keras.layers.BatchNormalization()(x2)  # Adding Batch Normalization

# Concatenate both branches
concat = keras.layers.Concatenate()([x1, x2])

# Output layer for binary classification (benign vs malignant)
out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(concat)

# Build the multi-modal model
model = keras.models.Model(inputs=inp, outputs=out)

# Compile the model
auc = AUC(name="auc")
precision = Precision(name="precision")  # Define precision metric
recall = Recall(name="recall")  # Define recall metric
accuracy = Accuracy(name="accuracy")  # Define accuracy metric
loss = keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss,
    metrics=[auc, precision, recall],  # Add precision, recall, and accuracy here
)

# Model summary
model.summary()

# Training the model
history = model.fit(
    training_ds,
    epochs=1,
    callbacks=[lr_cb, ckpt_cb],
    validation_data=validation_ds,
    verbose=CFG.verbose,
    class_weight=class_weights,
)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import Precision, Recall, AUC

# Define input layers for images and tabular metadata
image_input = keras.Input(shape=(128, 128, 3), name="images")
feat_input = keras.Input(shape=(feature_space.get_encoded_features().shape[1],), name="features")
inp = {"images": image_input, "features": feat_input}

# Branch for image input using ResNet50
backbone = ResNet50(weights=None, include_top=False, input_shape=(128, 128, 3))
x1 = backbone(image_input)
x1 = keras.layers.GlobalAveragePooling2D()(x1)
x1 = keras.layers.BatchNormalization()(x1)
x1 = keras.layers.Dropout(0.3)(x1)

# Branch for tabular/feature input
x2 = keras.layers.Dense(128, activation="relu")(feat_input)
x2 = keras.layers.BatchNormalization()(x2)
x2 = keras.layers.Dense(64, activation="relu")(x2)
x2 = keras.layers.Dropout(0.2)(x2)

# Concatenate both branches
concat = keras.layers.Concatenate()([x1, x2])

# Additional Dense layers after concatenation for deeper learning
x = keras.layers.Dense(64, activation="relu")(concat)
x = keras.layers.BatchNormalization()(x)
out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)

# Build the multi-modal model
model = keras.models.Model(inputs=inp, outputs=out)

# Compile the model with additional metrics (precision, recall, AUC)
auc = AUC(name="auc")
precision = Precision(name="precision")
recall = Recall(name="recall")
accuracy = keras.metrics.BinaryAccuracy(name="accuracy")

loss = keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss,
    metrics=[auc, precision, recall],  # Add precision, recall, and accuracy here
)

# Model summary
model.summary()

# Training the model
history = model.fit(
    training_ds,
    epochs=1,
    callbacks=[lr_cb, ckpt_cb],
    validation_data=validation_ds,
    verbose=CFG.verbose,
    class_weight=class_weights,
)


# Plot Model Architecture
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, dpi=60)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3

# Define input layers for images and tabular metadata
image_input = keras.Input(shape=(128, 128, 3), name="images")
feat_input = keras.Input(shape=(feature_space.get_encoded_features().shape[1],), name="features")
inp = {"images": image_input, "features": feat_input}

# Branch for image input using InceptionV3
backbone = InceptionV3(weights=None, include_top=False, input_shape=(128, 128, 3))
x1 = backbone(image_input)
x1 = keras.layers.GlobalAveragePooling2D()(x1)
x1 = keras.layers.BatchNormalization()(x1)  # Added BatchNormalization
x1 = keras.layers.Dropout(0.3)(x1)  # Increased Dropout rate

# Branch for tabular/feature input
x2 = keras.layers.Dense(128, activation="relu")(feat_input)  # Increased the first layer size
x2 = keras.layers.BatchNormalization()(x2)  # Added BatchNormalization
x2 = keras.layers.Dense(64, activation="relu")(x2)  # Changed second layer size
x2 = keras.layers.Dropout(0.2)(x2)  # Modified Dropout rate

# Concatenate both branches
concat = keras.layers.Concatenate()([x1, x2])

# Additional Dense layers after concatenation for deeper learning
x = keras.layers.Dense(64, activation="relu")(concat)  # New dense layer
x = keras.layers.BatchNormalization()(x)  # Added BatchNormalization
x = keras.layers.Dropout(0.3)(x)  # Increased Dropout rate
out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)

# Build the multi-modal model
model = keras.models.Model(inputs=inp, outputs=out)

# Compile the model
auc = keras.metrics.AUC(name="auc")
loss = keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss,
    metrics=[auc, precision, recall],  # Add precision, recall, and accuracy here
)

# Model summary
model.summary()

# Training the model
history = model.fit(
    training_ds,
    epochs=1,
    callbacks=[lr_cb, ckpt_cb],
    validation_data=validation_ds,
    verbose=CFG.verbose,
    class_weight=class_weights,
)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall

# ----------------- Memory-Augmented Network -----------------
class MemoryAugmentedNetwork(keras.layers.Layer):
    def __init__(self, memory_size=512, memory_dim=256, temperature=0.1, **kwargs):
        super(MemoryAugmentedNetwork, self).__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.temperature = temperature
        self.memory = self.add_weight(
            name="memory",
            shape=(memory_size, memory_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        
    def call(self, inputs):
        similarity = tf.matmul(inputs, self.memory, transpose_b=True) / self.temperature
        attention_weights = tf.nn.softmax(similarity, axis=-1)
        memory_read = tf.matmul(attention_weights, self.memory)
        return memory_read + inputs  # Residual connection

# ----------------- Image and Tabular Data Inputs -----------------
image_input = keras.Input(shape=(128, 128, 3), name="images")
feat_input = keras.Input(shape=(71,), name="features")

# ----------------- Image Branch (EfficientNetB4) -----------------
backbone = EfficientNetB4(weights=None, include_top=False, input_shape=(128, 128, 3))
for layer in backbone.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False

x1 = backbone(image_input)
x1 = layers.GlobalAveragePooling2D()(x1)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Dense(512, activation="relu")(x1)
x1 = layers.Dropout(0.5)(x1)

x1_latent = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="image_latent_projection")(x1)

# ----------------- Feature Branch -----------------
x2 = layers.Dense(256, activation="relu")(feat_input)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dense(512, activation="relu")(x2)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dropout(0.5)(x2)

x2_latent = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="feature_latent_projection")(x2)

# ----------------- Memory-Augmented Module -----------------
memory_module = MemoryAugmentedNetwork(memory_size=512, memory_dim=256, temperature=0.1)
x1_latent_projected = layers.Dense(256, activation="relu")(x1_latent)
x2_latent_projected = layers.Dense(256, activation="relu")(x2_latent)

x1_mem = memory_module(x1_latent_projected)
x2_mem = memory_module(x2_latent_projected)

# ----------------- Enhanced Contrastive Loss -----------------
def improved_contrastive_loss(y_true, y_pred, margin=1.5, alpha=0.6):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    pos_pair_distance = tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)
    neg_pair_distance = tf.maximum(margin - pos_pair_distance, 0.0)
    
    gamma = 2.0
    focal_weight = tf.pow(1. - y_pred, gamma) * y_true + tf.pow(y_pred, gamma) * (1. - y_true)
    focal = focal_weight * bce
    
    return alpha * (pos_pair_distance + neg_pair_distance) + (1 - alpha) * focal

# ----------------- Feature Aggregation -----------------
concat = layers.Concatenate(axis=-1)([x1_mem, x2_mem])

agg = layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(concat)
agg = layers.BatchNormalization()(agg)
agg = layers.Dropout(0.5)(agg)
agg = layers.Dense(512, activation="relu")(agg)
agg = layers.BatchNormalization()(agg)
agg = layers.Dropout(0.5)(agg)

# ----------------- Output Layer -----------------
out = layers.Dense(1, activation="sigmoid", dtype="float32")(agg)

# ----------------- Build and Compile Model -----------------
cmmann = keras.models.Model(inputs={"images": image_input, "features": feat_input}, outputs=out)

# Early stopping and learning rate scheduler for better performance
early_stopping_cb = EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True, verbose=1, mode='max')
lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, verbose=1, mode='max')

# Compile the model
cmmann.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=improved_contrastive_loss,  
    metrics=[AUC(), Precision(), Recall()]  
)

# ----------------- Model Summary and Plot -----------------
cmmann.summary()
keras.utils.plot_model(cmmann, show_shapes=True, show_layer_names=True, dpi=60)

# ----------------- Train the Model -----------------
history = cmmann.fit(
    training_ds,
    epochs=50,
    callbacks=[lr_scheduler, early_stopping_cb],  
    validation_data=validation_ds,
    verbose=1,
    class_weight=class_weights,
)



import numpy as np

# Find the best epoch index (0-based index)
best_epoch = np.argmax(history.history['val_auc_3'])  

# Ensure best_epoch is within valid range
if best_epoch >= len(history.history['val_auc_3']):
    best_epoch = len(history.history['val_auc_3']) - 1  

# Extract metrics at the best epoch
best_val_auc = history.history['val_auc_3'][best_epoch]
best_val_precision = history.history['val_precision_2'][best_epoch]
best_val_recall = history.history['val_recall_2'][best_epoch]

# Print results
print(f"Best Epoch: {best_epoch + 1}")  # Convert to 1-based epoch number
print(f"Best Validation AUC: {best_val_auc:.4f}")
print(f"Best Validation Precision: {best_val_precision:.4f}")
print(f"Best Validation Recall: {best_val_recall:.4f}")



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# ----------------- Multi-Head Self-Attention -----------------
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.q_dense = layers.Dense(embed_dim)
        self.k_dense = layers.Dense(embed_dim)
        self.v_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def call(self, inputs):
        query, key, value = self.q_dense(inputs), self.k_dense(inputs), self.v_dense(inputs)
        attention_scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        return self.combine_heads(tf.matmul(attention_weights, value))

# ----------------- Squeeze-and-Excitation (SE) Block -----------------
def squeeze_excite_block(inputs, ratio=16):
    filters = inputs.shape[-1]
    se = layers.GlobalAveragePooling1D()(tf.expand_dims(inputs, axis=1))  # Fix for 2D input
    se = layers.Dense(filters // ratio, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    return layers.multiply([inputs, se])

# ----------------- Vision Transformer (ViT) -----------------
def build_vit(image_size=224, patch_size=16, num_layers=8, d_model=512, num_heads=8, mlp_dim=1024):
    inputs = keras.Input(shape=(image_size, image_size, 3))
    num_patches = (image_size // patch_size) ** 2
    patch_embed = layers.Conv2D(d_model, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    patch_embed = layers.Reshape((num_patches, d_model))(patch_embed)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=d_model)(tf.range(num_patches))
    x = patch_embed + pos_embed
    
    for _ in range(num_layers):
        x = MultiHeadSelfAttention(d_model, num_heads)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(mlp_dim, activation="relu")(x)
        x = layers.Dense(d_model)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(inputs, x, name="vision_transformer")

# ----------------- Image and Tabular Data Inputs -----------------
image_input = keras.Input(shape=(224, 224, 3), name="images")
feat_input = keras.Input(shape=(71,), name="features")

# ----------------- Image Branch (Self-Contained ViT) -----------------
vit_model = build_vit()
x1 = vit_model(image_input)
x1 = layers.Dense(512, activation="relu")(x1)
x1 = squeeze_excite_block(x1)  # Fixed SE block
x1 = layers.Dropout(0.5)(x1)
x1_latent = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="image_latent_projection")(x1)

# ----------------- Feature Branch -----------------
x2 = layers.Dense(256, activation="relu")(feat_input)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dense(512, activation="relu")(x2)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dropout(0.5)(x2)
x2_latent = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="feature_latent_projection")(x2)

# ----------------- Multi-Head Attention for Feature Fusion -----------------
attention_layer = MultiHeadSelfAttention(embed_dim=512, num_heads=8)
x1_mem, x2_mem = attention_layer(x1_latent), attention_layer(x2_latent)

# ----------------- GRU for Temporal Feature Learning -----------------
gru_layer = layers.GRU(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
x1_gru, x2_gru = gru_layer(tf.expand_dims(x1_mem, axis=1)), gru_layer(tf.expand_dims(x2_mem, axis=1))

x1_gru, x2_gru = layers.Flatten()(x1_gru), layers.Flatten()(x2_gru)
concat = layers.Concatenate(axis=-1)([x1_gru, x2_gru])

# ----------------- Fully Connected Layers -----------------
agg = layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(concat)
agg = layers.BatchNormalization()(agg)
agg = layers.Dropout(0.5)(agg)
agg = layers.Dense(512, activation="relu")(agg)
agg = layers.BatchNormalization()(agg)
agg = layers.Dropout(0.5)(agg)

# ----------------- Output Layer -----------------
out = layers.Dense(1, activation="sigmoid", dtype="float32")(agg)

# ----------------- Build Model -----------------
cmmann = keras.models.Model(inputs={"images": image_input, "features": feat_input}, outputs=out)

# ----------------- Compile Model -----------------
cmmann.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()]
)

# ----------------- Callbacks -----------------
reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.4, patience=6, min_lr=1e-6, mode='max')
early_stopping = EarlyStopping(monitor='val_auc', patience=20, restore_best_weights=True, mode='max')

cmmann.summary()
keras.utils.plot_model(cmmann, show_shapes=True, show_layer_names=True, dpi=60)

# ----------------- Training -----------------
history = cmmann.fit(
    training_ds,
    epochs=15,
    callbacks=[early_stopping, reduce_lr],
    validation_data=validation_ds,
    verbose=1,
    class_weight=class_weights
)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy

# Define input layers for images and tabular metadata
image_input = keras.Input(shape=(128, 128, 3), name="images")
feat_input = keras.Input(shape=(feature_space.get_encoded_features().shape[1],), name="features")

# ----------------- Image Branch -----------------
# EfficientNet Backbone with Pretrained Weights
backbone = EfficientNetB0(weights=None, include_top=False, input_shape=(128, 128, 3))
x1 = backbone(image_input)
x1 = keras.layers.GlobalAveragePooling2D()(x1)
x1 = keras.layers.BatchNormalization()(x1)
x1 = keras.layers.Dropout(0.5)(x1)  # Increased dropout in the image branch

# First latent projection for image embeddings
x1_latent = keras.layers.Dense(512, activation="relu", name="image_latent_projection")(x1)
x1_latent = keras.layers.BatchNormalization()(x1_latent)

# ----------------- Feature Branch -----------------
# Tabular Data Feature Processing with Increased Dense Layer Sizes
x2 = keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(feat_input)
x2 = keras.layers.BatchNormalization()(x2)
x2 = keras.layers.Dropout(0.4)(x2)  # Increased dropout in the feature branch

# First latent projection for feature embeddings
x2_latent = keras.layers.Dense(512, activation="relu", name="feature_latent_projection")(x2)
x2_latent = keras.layers.BatchNormalization()(x2_latent)

# ----------------- Latent Space Alignment -----------------
# Discriminator for Alignment with Increased Network Capacity
def make_discriminator():
    d_input = keras.Input(shape=(512,))
    d_x = keras.layers.Dense(256, activation="relu")(d_input)
    d_x = keras.layers.BatchNormalization()(d_x)
    d_x = keras.layers.Dense(128, activation="relu")(d_x)
    d_x = keras.layers.BatchNormalization()(d_x)
    d_x = keras.layers.Dense(64, activation="relu")(d_x)
    d_output = keras.layers.Dense(1, activation="sigmoid", name="discriminator_output")(d_x)
    return keras.models.Model(inputs=d_input, outputs=d_output, name="Discriminator")

discriminator = make_discriminator()

# Latent space alignment
d_image = discriminator(x1_latent)
d_feature = discriminator(x2_latent)

# Loss for discriminator alignment
adversarial_loss = keras.losses.BinaryCrossentropy(from_logits=False)

# ----------------- Feature Aggregation -----------------
# Concatenate Latent Spaces
concat = keras.layers.Concatenate()([x1_latent, x2_latent])

# Hierarchical Feature Aggregation with Larger Dense Layer
agg = keras.layers.Dense(2048, activation="relu", name="aggregated_features")(concat)
agg = keras.layers.BatchNormalization()(agg)
agg = keras.layers.Dropout(0.5)(agg)  # Increased dropout

# ----------------- Output Layer -----------------
# Final Binary Classification
out = keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(agg)

# Build the MedBlendNet Model
medblendnet = keras.models.Model(inputs={"images": image_input, "features": feat_input}, outputs=out)

# ----------------- Compile the Model -----------------
# Early stopping and learning rate scheduler for better performance
early_stopping_cb = EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True, verbose=1, mode='max')
lr_scheduler = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, verbose=1, mode='max')

# Compile the model
medblendnet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[AUC(), Precision(), Recall()]  # Correctly instantiate the metrics
)

# Model summary
medblendnet.summary()
keras.utils.plot_model(medblendnet, show_shapes=True, show_layer_names=True, dpi=60)
# ----------------- Train the Model -----------------
history = medblendnet.fit(
    training_ds,
    epochs=1,
    callbacks=[lr_scheduler, early_stopping_cb, ckpt_cb],  # Add checkpoints if needed
    validation_data=validation_ds,
    verbose=1,
    class_weight=class_weights,
)


#import numpy as np

# Step 1: Find the epoch with the best validation AUC
best_epoch = np.argmax(history.history['val_auc_2'])  # Index of the epoch with the highest validation AUC

# Step 2: Extract the corresponding metrics at that epoch
best_precision = history.history['precision_1'][best_epoch]
best_recall = history.history['recall_1'][best_epoch]
best_val_auc = history.history['val_auc_2'][best_epoch]

# Step 3: Print the results
print(f"Best Precision at epoch {best_epoch + 1}: {best_precision:.5f}")
print(f"Best Recall at epoch {best_epoch + 1}: {best_recall:.5f}")
print(f"Best Validation AUC at epoch {best_epoch + 1}: {best_val_auc:.5f}")



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# Memory-Augmented Network
class MemoryAugmentedNetwork(keras.layers.Layer):
    def __init__(self, memory_size=512, memory_dim=256, temperature=0.1, **kwargs):
        super(MemoryAugmentedNetwork, self).__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.temperature = temperature
        self.memory = self.add_weight(
            name="memory",
            shape=(memory_size, memory_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        
    def call(self, inputs):
        # Scaled dot-product attention with temperature
        similarity = tf.matmul(inputs, self.memory, transpose_b=True) / self.temperature
        attention_weights = tf.nn.softmax(similarity, axis=-1)
        memory_read = tf.matmul(attention_weights, self.memory)
        # Residual connection
        return memory_read + inputs

# Image and Tabular Data Inputs
image_input = keras.Input(shape=(128, 128, 3), name="images")
feat_input = keras.Input(shape=(71,), name="features")

# ----------------- Image Branch -----------------
# Using EfficientNetB4 for better feature extraction without pre-trained weights
backbone = EfficientNetB4(weights=None, include_top=False, input_shape=(128, 128, 3))

# Freeze early layers
for layer in backbone.layers[:100]:
    layer.trainable = False

x1 = backbone(image_input)
x1 = layers.GlobalAveragePooling2D()(x1)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Dense(512, activation="relu")(x1)
x1 = layers.Dropout(0.5)(x1)  # Increased dropout

# Latent Representation for Image Features
x1_latent = layers.Dense(512, activation="relu", 
                        kernel_regularizer=regularizers.l2(1e-4),
                        name="image_latent_projection")(x1)

# ----------------- Feature Branch -----------------
# Enhanced Tabular Data Processing
x2 = layers.Dense(256, activation="relu")(feat_input)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dense(512, activation="relu")(x2)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dropout(0.5)(x2)  # Increased dropout

# Latent Representation for Tabular Features
x2_latent = layers.Dense(512, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4),
                        name="feature_latent_projection")(x2)

# ----------------- Memory-Augmented Module -----------------
memory_module = MemoryAugmentedNetwork(memory_size=512, memory_dim=256, temperature=0.1)
x1_latent_projected = layers.Dense(256, activation="relu")(x1_latent)
x2_latent_projected = layers.Dense(256, activation="relu")(x2_latent)

x1_mem = memory_module(x1_latent_projected)
x2_mem = memory_module(x2_latent_projected)

# ----------------- Enhanced Contrastive Loss -----------------
def improved_contrastive_loss(y_true, y_pred, margin=1.5, alpha=0.6):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Enhanced contrastive component
    pos_pair_distance = tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)
    neg_pair_distance = tf.maximum(margin - pos_pair_distance, 0.0)
    
    # Focal loss component
    gamma = 2.0
    focal_weight = tf.pow(1. - y_pred, gamma) * y_true + tf.pow(y_pred, gamma) * (1. - y_true)
    focal = focal_weight * bce
    
    return alpha * (pos_pair_distance + neg_pair_distance) + (1 - alpha) * focal

# ----------------- Feature Aggregation -----------------
concat = layers.Concatenate(axis=-1)([x1_mem, x2_mem])

# Enhanced Cross-Modal Feature Aggregation
agg = layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(concat)
agg = layers.BatchNormalization()(agg)
agg = layers.Dropout(0.5)(agg)  # Increased dropout
agg = layers.Dense(512, activation="relu")(agg)
agg = layers.BatchNormalization()(agg)
agg = layers.Dropout(0.5)(agg)  # Increased dropout

# ----------------- Output Layer -----------------
out = layers.Dense(1, activation="sigmoid", dtype="float32")(agg)

# Build Model
cmmann = keras.models.Model(inputs={"images": image_input, "features": feat_input}, outputs=out)

# Compile with custom metrics
cmmann.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss=improved_contrastive_loss,
    metrics=[AUC(), Precision(), Recall()]  # Correctly instantiate the metrics

)

# ----------------- Enhanced Callbacks -----------------
reduce_lr = ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.4,  # Lower factor
    patience=6,  # Adjusted patience
    min_lr=1e-6,
    mode='max'
)

early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=20,  # Increased patience
    restore_best_weights=True,
    mode='max'
)

# ----------------- Enhanced Data Augmentation -----------------
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    brightness_range=[0.7, 1.3],
    channel_shift_range=50.0
)
keras.utils.plot_model(cmmann, show_shapes=True, show_layer_names=True, dpi=60)

# ----------------- Training -----------------
history = cmmann.fit(
    training_ds,
    epochs=1,  # Increased epochs
    callbacks=[early_stopping, reduce_lr],
    validation_data=validation_ds,
    verbose=1,
    class_weight=class_weights
)


#import numpy as np

# Step 1: Find the epoch with the best validation AUC
best_epoch = np.argmax(history.history['val_auc_3'])  # Index of the epoch with the highest validation AUC

# Step 2: Extract the corresponding metrics at that epoch
best_precision = history.history['precision_2'][best_epoch]
best_recall = history.history['recall_2'][best_epoch]
best_val_auc = history.history['val_auc_3'][best_epoch]

# Step 3: Print the results
print(f"Best Precision at epoch {best_epoch + 1}: {best_precision:.5f}")
print(f"Best Recall at epoch {best_epoch + 1}: {best_recall:.5f}")
print(f"Best Validation AUC at epoch {best_epoch + 1}: {best_val_auc:.5f}")



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Attention, Add, MultiHeadAttention, Dense, Flatten, BatchNormalization, Concatenate, Dropout, GlobalAveragePooling2D, Reshape
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy
from tensorflow.keras import regularizers

# Define input layers for images and tabular metadata
image_input = keras.Input(shape=(128, 128, 3), name="images")
feat_input = keras.Input(shape=(feature_space.get_encoded_features().shape[1],), name="features")

# EfficientNetB0 Backbone with Batch Normalization
backbone = EfficientNetB0(weights=None, include_top=False, input_shape=(128, 128, 3))
x1 = backbone(image_input)
x1 = GlobalAveragePooling2D()(x1)
x1 = BatchNormalization()(x1)  # Adding Batch Normalization
x1 = Dropout(0.3)(x1)  # Dropout to prevent overfitting

# Attention Mechanism
x1_reshaped = Reshape((1, 1280))(x1)
x1_attention = MultiHeadAttention(num_heads=8, key_dim=128)(x1_reshaped, x1_reshaped)
x1 = Add()([x1_reshaped, x1_attention])
x1 = BatchNormalization()(x1)  # Batch normalization after attention
x1 = Flatten()(x1)

# Tabular Branch with DenseNet-like connections and regularization
x2 = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(feat_input)
x2 = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x2)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.3)(x2)  # Dropout for regularization

# Attention mechanism for tabular features
x2_reshaped = Reshape((1, 256))(x2)
x2_attention = MultiHeadAttention(num_heads=8, key_dim=128)(x2_reshaped, x2_reshaped)
x2 = Add()([x2_reshaped, x2_attention])
x2 = BatchNormalization()(x2)  # Batch normalization after attention
x2 = Flatten()(x2)

# Combine Image and Tabular Branches
concat = Concatenate()([x1, x2])

# Output Layer
out = Dense(1, activation="sigmoid")(concat)

# Build Model
model = keras.models.Model(inputs=[image_input, feat_input], outputs=out)

# Compile the model
auc = AUC(name="auc")
precision = Precision(name="precision")
recall = Recall(name="recall")
accuracy = Accuracy(name="accuracy")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[AUC(), Precision(), Recall()]  # Correctly instantiate the metrics
)

# Model Summary
model.summary()
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, dpi=60)

# Fit the Model
history = model.fit(
    training_ds,
    epochs=1,
    validation_data=validation_ds,
    callbacks=[lr_cb, ckpt_cb],
    class_weight=class_weights,
    verbose=CFG.verbose,
)


#import numpy as np

# Step 1: Find the epoch with the best validation AUC
best_epoch = np.argmax(history.history['val_auc_4'])  # Index of the epoch with the highest validation AUC

# Step 2: Extract the corresponding metrics at that epoch
best_precision = history.history['precision_3'][best_epoch]
best_recall = history.history['recall_3'][best_epoch]
best_val_auc = history.history['val_auc_4'][best_epoch]

# Step 3: Print the results
print(f"Best Precision at epoch {best_epoch + 1}: {best_precision:.5f}")
print(f"Best Recall at epoch {best_epoch + 1}: {best_recall:.5f}")
print(f"Best Validation AUC at epoch {best_epoch + 1}: {best_val_auc:.5f}")


