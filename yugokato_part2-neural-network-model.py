import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


import tensorflow as tf
print(tf.__version__)


import tensorflow as tf

sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print('cuda_version: ', cuda_version)


import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    ZeroPadding2D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Define the paths to your data
train_data_path = r"F:\gpr-max-deep-learning-challenge-1-gdlc-1\Training_Bscan\Training_Bscan"
test_data_path = r"F:\gpr-max-deep-learning-challenge-1-gdlc-1\Evaluation_Dataset"

# Number of training samples
num_train_samples = 4400
# Number of test samples
num_test_samples = 100

# Load the training data
labels_list = []
for i in range(num_train_samples):
    filename = os.path.join(train_data_path, f"Bscan_{i}.npy")
    data = np.load(filename)
    labels_list.append(data)

labels = np.array(labels_list)  # Shape: (4400, H, W)

# Initialize the training data with zeros
train = np.zeros_like(labels)

# Process each label to create the corresponding training data
for i, c in enumerate(labels):
    # Normalize the Bscans from -1 to 1
    c = c / np.max(np.abs(c))
    labels[i] = c
    # Fill the training data with the values of the A-scans for every 10 traces
    train[i, :, 0:-1:10] = c[:, 0:-1:10]

# Define U-net architecture
def unet_model(input_size=(None, None, 1)):
    inputs = Input(input_size)
    # Encoder
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(p4)
    c5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation="relu", padding="same")(u6)
    c6 = Conv2D(512, (3, 3), activation="relu", padding="same")(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = ZeroPadding2D(padding=((1, 0), (1, 0)))(u7)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation="relu", padding="same")(u7)
    c7 = Conv2D(256, (3, 3), activation="relu", padding="same")(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = ZeroPadding2D(padding=((1, 0), (1, 0)))(u8)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation="relu", padding="same")(u8)
    c8 = Conv2D(128, (3, 3), activation="relu", padding="same")(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation="relu", padding="same")(u9)
    c9 = Conv2D(64, (3, 3), activation="relu", padding="same")(c9)

    outputs = Conv2D(1, (1, 1), activation="linear")(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Initiate the model
model = unet_model(input_size=(labels.shape[1], labels.shape[2], 1))
model.summary()


# Add an extra dimension to train and labels
Train = train[..., np.newaxis]  # Shape: (4400, H, W, 1)
Labels = labels[..., np.newaxis]  # Shape: (4400, H, W, 1)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(Train, Labels, test_size=0.1, random_state=42)

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Define callbacks
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=30)
model_checkpoint = ModelCheckpoint(
    "Fill_Missing.h5", monitor="val_loss", mode="min", save_best_only=True
)

# Adjust learning rate
model.optimizer.learning_rate = 1e-4

# Train the model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping, model_checkpoint],
)




# Load the best model
saved_model = load_model("Fill_Missing.h5")


# (Optional) Evaluate on validation set
val_predictions = saved_model.predict(x_val)


from matplotlib import pyplot as plt


import numpy as np
import os

# Load test data and prepare for prediction
test_list = []
for i in range(num_test_samples):
    filename = os.path.join(test_data_path, f"Testing_Bscan_{i}.npy")
    data = np.load(filename)
    
    # Replace NaN with 0
    data = np.nan_to_num(data, nan=0.0)
    
    # Normalize considering only non-NaN values
    max_val = np.max(np.abs(data)) 
    if max_val > 0:  # Avoid division by zero
        data = data / max_val
    
    # Append the normalized data to the list
    test_list.append(data)

# Convert to numpy array and add extra dimension
if test_list:
    test_data = np.array(test_list)[..., np.newaxis]  # Shape: (N, H, W, 1)
else:
    print("No valid test data found.")
    test_data = np.empty((0, 0, 0, 0))  # Empty array in case of no valid data

# Generate predictions on test data
test_predictions = saved_model.predict(test_data)


# Plot some results
for i in range(min(5, len(test_predictions))):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(test_data[i, ..., 0], cmap="bone")
    plt.axis("off")
    plt.title("Corrupted Input")

    plt.subplot(1, 3, 2)
    plt.imshow(test_predictions[i, ..., 0], cmap="bone")
    plt.axis("off")
    plt.title("Model Output")

    # If ground truth is available for test data, plot it
    # Otherwise, this section can be commented out
    # plt.subplot(1, 3, 3)
    # plt.imshow(ground_truth[i, ..., 0], cmap="bone")
    # plt.axis("off")
    # plt.title("Ground Truth")

    plt.show()


test_data[i, ..., 0].shape


test_data.shape


test_predictions.shape


!pip install mlflow


!pip install "flaml[blendsearch]"


import tensorflow as tf
info = tf.sysconfig.get_build_info()
print("CUDA Version:", info["cuda_version"])
print("cuDNN Version:", info["cudnn_version"])



import tensorflow as tf
tf.config.list_physical_devices('GPU')


import tensorflow as tf
print(tf.__version__)


import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam
from flaml import tune
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.tensorflow


tf.config.list_physical_devices('GPU')


import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    concatenate, Dropout, Cropping2D, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from flaml import tune
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split

# --- å…¥åŠ›ã�¨ãƒ©ãƒ™ãƒ«ã�®ã‚µã‚¤ã‚º ---
INPUT_HEIGHT = 230
INPUT_WIDTH  = 230
LABEL_HEIGHT = 224
LABEL_WIDTH  = 224
CHANNELS     = 1

# --- TFRecord ãƒ•ã‚¡ã‚¤ãƒ«ã�®ãƒ‘ã‚¹ ---
TRAIN_TFRECORD = "/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/tfrecords/train.tfrecord"
VAL_TFRECORD   = "/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/tfrecords/val.tfrecord"

# --- ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆæƒ…å ± ---
dataset_info = {
    "Training_Bscan": {
        "input_dir": r"/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/Training_Bscan/Training_Bscan",
        "label_dir": r"/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/Training_Labels/Training_Labels",
        "count": 4400
    },
    "add_simu": {
        "input_dir": r"/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/add_simu",
        "label_dir": r"/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/add_simu_model",
        "count": 15000
    }
}

# --- TFRecord ç”¨ Feature ãƒ˜ãƒ«ãƒ‘ãƒ¼ ---
def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

# --- TFRecord ä½œæˆ�é–¢æ•° ---
def create_tfrecord(pairs, tfrecord_path):
    os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for input_path, label_path in pairs:
            # .npy ãƒ•ã‚¡ã‚¤ãƒ«èª­ã�¿è¾¼ã�¿
            input_arr = np.load(input_path)    # shape (230, 230)
            label_arr = np.load(label_path)    # shape (224, 224)

            # å…¥åŠ›ãƒ‡ãƒ¼ã‚¿æ­£è¦�åŒ– [-1, 1]
            input_norm = input_arr / np.max(np.abs(input_arr), axis=(0,1), keepdims=True)
            # ãƒ©ãƒ™ãƒ«ãƒ‡ãƒ¼ã‚¿æ­£è¦�åŒ– [-0.5, 0.5]
            label_norm = label_arr / 10.0 - 0.5

            # float32 å�‹ã�«å¤‰æ�›
            input_norm = input_norm.astype(np.float32)
            label_norm = label_norm.astype(np.float32)

            # ãƒ•ãƒ©ãƒƒãƒˆåŒ–
            input_flat = input_norm.flatten()  # length 230*230
            label_flat = label_norm.flatten()  # length 224*224

            # Example ä½œæˆ�
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': _float_feature(input_flat.tolist()),
                'label': _float_feature(label_flat.tolist()),
            }))
            writer.write(example.SerializeToString())
    print(f"Wrote {len(pairs)} records to {tfrecord_path}")

# --- TFRecord å‰�å‡¦ç�†å®Ÿè¡Œ ---
def prepare_tfrecords(test_size=0.02578, seed=42):
    # ãƒ‡ãƒ¼ã‚¿ãƒšã‚¢ãƒªã‚¹ãƒˆä½œæˆ�ï¼ˆå…¥åŠ›ãƒ•ã‚¡ã‚¤ãƒ«å��ã�®ãƒ—ãƒ¬ãƒ•ã‚£ãƒƒã‚¯ã‚¹ã‚’ dataset å��ã�§åˆ‡ã‚Šæ›¿ã�ˆï¼‰
    all_pairs = []
    for name, info in dataset_info.items():
        inp_dir = info['input_dir']
        lbl_dir = info['label_dir']
        count   = info['count']
        # add_simu ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�®ã�¿ "add_simu_Bscan_" ãƒ—ãƒ¬ãƒ•ã‚£ãƒƒã‚¯ã‚¹
        prefix = 'add_simu_Bscan_' if name == 'add_simu' else 'Bscan_'
        for i in range(count):
            inp = os.path.join(inp_dir, f"{prefix}{i}.npy")
            lbl = os.path.join(lbl_dir, f"Model_{i}.npy")
            all_pairs.append((inp, lbl))

    # è¨“ç·´/æ¤œè¨¼ åˆ†å‰²
    train_pairs, val_pairs = train_test_split(
        all_pairs, test_size=test_size, random_state=seed
    )

    # TFRecord ä½œæˆ�
    create_tfrecord(train_pairs, TRAIN_TFRECORD)
    create_tfrecord(val_pairs,   VAL_TFRECORD)


# TFRecord ä½œæˆ�ã‚’å®Ÿè¡Œ
prepare_tfrecords()


# --- MLflow ã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ— ---
mlflow.set_tracking_uri("file:///mnt/f/ml_log")
mlflow.set_experiment("gpr_unet_experiment2")

# --- TFRecord ã‚’ãƒ‘ãƒ¼ã‚¹ã�™ã‚‹é–¢æ•° ---
def _parse_function(example_proto):
    feature_description = {
        'input': tf.io.FixedLenFeature([INPUT_HEIGHT * INPUT_WIDTH], tf.float32),
        'label': tf.io.FixedLenFeature([LABEL_HEIGHT * LABEL_WIDTH], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.reshape(parsed['input'], [INPUT_HEIGHT, INPUT_WIDTH, 1])
    label = tf.reshape(parsed['label'], [LABEL_HEIGHT, LABEL_WIDTH, 1])
    return image, label

# --- U-Net ãƒ¢ãƒ‡ãƒ«æ§‹ç¯‰ ---
def build_unet_model_ver2(params, input_size=(INPUT_HEIGHT, INPUT_WIDTH, CHANNELS)):
    base_filters = params.get("filters", 32)
    kernel_size   = params.get("kernel_size", 6)
    d1 = params.get("dropout1", 0.1)
    d2 = params.get("dropout2", 0.2)
    d3 = params.get("dropout3", 0.3)

    inputs_layer = Input(input_size)
    # --- Encoder ---
    c1 = Conv2D(base_filters, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(inputs_layer)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(d1)(c1)
    c1 = Conv2D(base_filters, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(base_filters*2, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(d1)(c2)
    c2 = Conv2D(base_filters*2, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(base_filters*4, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(d2)(c3)
    c3 = Conv2D(base_filters*4, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(base_filters*8, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(d2)(c4)
    c4 = Conv2D(base_filters*8, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # --- Bottleneck ---
    c5 = Conv2D(base_filters*16, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(d3)(c5)
    c5 = Conv2D(base_filters*16, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(c5)

    # --- Decoder ---
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(base_filters*8, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(d2)(c6)
    c6 = Conv2D(base_filters*8, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(c6)

    u7 = UpSampling2D((2, 2))(c6)
    c3_cropped = Cropping2D(cropping=((0,1),(0,1)))(c3)
    u7 = concatenate([u7, c3_cropped])
    c7 = Conv2D(base_filters*4, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(d2)(c7)
    c7 = Conv2D(base_filters*4, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(c7)

    u8 = UpSampling2D((2, 2))(c7)
    c2_cropped = Cropping2D(cropping=((1,2),(1,2)))(c2)
    u8 = concatenate([u8, c2_cropped])
    c8 = Conv2D(base_filters*2, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(d1)(c8)
    c8 = Conv2D(base_filters*2, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(c8)

    u9 = UpSampling2D((2, 2))(c8)
    c1_cropped = Cropping2D(cropping=((3,3),(3,3)))(c1)
    u9 = concatenate([u9, c1_cropped])
    c9 = Conv2D(base_filters, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(d1)(c9)
    c9 = Conv2D(base_filters, (kernel_size, kernel_size), activation='relu', padding='same', kernel_initializer=he_normal())(c9)

    outputs = Conv2D(1, (1, 1), activation='linear')(c9)
    return Model(inputs=[inputs_layer], outputs=[outputs])

# --- ãƒˆãƒ¬ãƒ¼ãƒ‹ãƒ³ã‚°é–¢æ•° ---
def train_unet(config):
    with mlflow.start_run():
        mlflow.log_params(config)
        for name, info in dataset_info.items():
            mlflow.log_param(f"dataset.{name}.input_dir", info['input_dir'])
            mlflow.log_param(f"dataset.{name}.label_dir", info['label_dir'])
            mlflow.log_param(f"dataset.{name}.count",     info['count'])
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        mlflow.log_param("timestamp", ts)

        # ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆæ§‹ç¯‰
        train_ds = (
            tf.data.TFRecordDataset([TRAIN_TFRECORD],buffer_size=4*1024*1024)
            .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=1024)
            .batch(config.get("batch_size", 16))
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.TFRecordDataset([VAL_TFRECORD])
            .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(config.get("batch_size", 16))
            .prefetch(tf.data.AUTOTUNE)
        )

        # ãƒ¢ãƒ‡ãƒ«æ§‹ç¯‰ãƒ»ã‚³ãƒ³ãƒ‘ã‚¤ãƒ«
        model = build_unet_model_ver2(config)
        optimizer = Adam(learning_rate=config.get("lr", 5e-4))
        model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred)),
            metrics=['mae']
        )

        # ã‚³ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯å®šç¾©
        base_dir = '/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/tf_run8'
        log_dir = os.path.join(base_dir, ts)
        best_model_path = os.path.join(base_dir, f"{ts}_flaml_best_model.h5")
        callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=0, write_images=False, update_freq='epoch' ),
            EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1),
            ModelCheckpoint(best_model_path, monitor="val_loss", mode="min", save_best_only=True)
        ]

        # ãƒ¢ãƒ‡ãƒ«å­¦ç¿’
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            verbose=0,
            callbacks=callbacks
        )

        final_loss = history.history['val_loss'][-1]
        mlflow.log_metric("final_val_loss", final_loss)
        mlflow.log_artifact(best_model_path, artifact_path="models")
        tune.report(loss=final_loss)



# --- ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿æ�¢ç´¢è¨­å®š ---
search_space = {
    "filters": tune.choice([72]),
    "kernel_size": tune.choice([7]),
    "dropout1": tune.uniform(0.1, 0.2),
    "dropout2": tune.uniform(0.2, 0.3),
    "dropout3": tune.uniform(0.25, 0.4),
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([16])
}


from flaml import BlendSearch

blend = BlendSearch(
    metric="loss",     # æœ€é�©åŒ–ã�™ã‚‹æŒ‡æ¨™
    mode="min",        # minimize (val_loss ã‚’å°�ã�•ã��)
)


for num_run in range(100):

    analysis3 = tune.run(
        train_unet,
        config=search_space,
        metric="loss", mode="min",
        num_samples=1,
        search_alg=blend
    )
    blend.save(r"/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/blend_search_obj_LD.pkl")


import numpy as np
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    ZeroPadding2D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Load the best model
saved_model = load_model("Fill_Missing.h5")


test_data_path = r"F:\gpr-max-deep-learning-challenge-1-gdlc-1\Evaluation_Dataset"
num_test_samples = 100

# Load test data and prepare for prediction
test_list = []
nan_data_list = []
for i in range(num_test_samples):
    filename = os.path.join(test_data_path, f"Testing_Bscan_{i}.npy")
    data = np.load(filename)

    ######################
    data_nan_mask = np.isnan(data)
    
    # Replace NaN with 0
    data = np.nan_to_num(data, nan=0.0)
    
    # Normalize considering only non-NaN values
    max_val = np.max(np.abs(data))  # NaNã�¯ã�™ã�§ã�«0ã�«å¤‰æ�›ã�•ã‚Œã�¦ã�„ã‚‹
    if max_val > 0:  # Avoid division by zero
        data = data / max_val
    
    # Append the normalized data to the list
    test_list.append(data)
    nan_data_list.append(data_nan_mask)

# Convert to numpy array and add extra dimension
if test_list:
    test_data = np.array(test_list)[..., np.newaxis]  # Shape: (N, H, W, 1)
    nan_mask = np.array(nan_data_list)[..., np.newaxis]  # Shape: (N, H, W, 1)
else:
    print("No valid test data found.")
    test_data = np.empty((0, 0, 0, 0))  # Empty array in case of no valid data


# Generate predictions on test data
test_predictions = saved_model.predict(test_data)


scale_test_predictions = np.zeros_like(test_predictions)

for i in range(len(test_predictions)):
    tmp_scale = np.mean(np.abs(test_predictions[i, ..., 0]))/np.mean(np.abs(test_data[i, ..., 0]))
    scale_test_predictions[i, ..., 0] = test_predictions[i, ..., 0] / tmp_scale

scale_test_predictions[~nan_mask] = test_data[~nan_mask]


test_predictions.shape


from matplotlib import pyplot as plt

for i in range(min(5, len(test_predictions))):
#for i in range(len(test_predictions)):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    tmp_vmax = np.max(np.abs(test_data[i, ..., 0]))
    img = plt.imshow(test_data[i, ..., 0], cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    plt.axis("off")
    #plt.title("Corrupted Input")
    plt.title('Test Input {:04d}'.format(i))
    cbar = plt.colorbar(img)

    plt.subplot(1, 3, 2)
    img2 = plt.imshow(np.squeeze(scale_test_predictions[i, ...], axis=-1), cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    plt.axis("off")
    #plt.title("Model Output")
    plt.title('Output {:04d}'.format(i))
    cbar2 = plt.colorbar(img2)


    plt.subplot(1, 3, 3)
    img3 = plt.imshow(test_data[i, ..., 0] - np.squeeze(scale_test_predictions[i, ...], axis=-1), cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    #plt.imshow(ground_truth[i, ..., 0], cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    cbar3 = plt.colorbar(img3)
    
    plt.axis("off")
    plt.title("Diff")

    plt.show()


# Generate predictions on test data
test_predictions2 = saved_model.predict(scale_test_predictions)


scale_test_predictions2 = np.zeros_like(test_predictions2)

for i in range(len(test_predictions)):
    tmp_scale = np.mean(np.abs(test_predictions2[i, ..., 0]))/np.mean(np.abs(test_data[i, ..., 0]))
    scale_test_predictions2[i, ..., 0] = test_predictions2[i, ..., 0] / tmp_scale

scale_test_predictions2[~nan_mask] = test_data[~nan_mask]


from matplotlib import pyplot as plt

for i in range(min(5, len(test_predictions))):
#for i in range(len(test_predictions)):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    tmp_vmax = np.max(np.abs(test_data[i, ..., 0]))
    img = plt.imshow(test_data[i, ..., 0], cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    plt.axis("off")
    #plt.title("Corrupted Input")
    plt.title('Test Input {:04d}'.format(i))
    cbar = plt.colorbar(img)

    plt.subplot(1, 3, 2)
    img2 = plt.imshow(np.squeeze(scale_test_predictions2[i, ...], axis=-1), cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    plt.axis("off")
    #plt.title("Model Output")
    plt.title('Output {:04d}'.format(i))
    cbar2 = plt.colorbar(img2)


    plt.subplot(1, 3, 3)
    img3 = plt.imshow(test_data[i, ..., 0] - np.squeeze(scale_test_predictions2[i, ...], axis=-1), cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    #plt.imshow(ground_truth[i, ..., 0], cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    cbar3 = plt.colorbar(img3)
    
    plt.axis("off")
    plt.title("Diff")

    plt.show()


# Generate predictions on test data
test_predictions3 = saved_model.predict(scale_test_predictions2)


scale_test_predictions3 = np.zeros_like(test_predictions3)

for i in range(len(test_predictions)):
    tmp_scale = np.mean(np.abs(test_predictions3[i, ..., 0]))/np.mean(np.abs(test_data[i, ..., 0]))
    scale_test_predictions3[i, ..., 0] = test_predictions3[i, ..., 0] / tmp_scale

scale_test_predictions3[~nan_mask] = test_data[~nan_mask]


from matplotlib import pyplot as plt

for i in range(min(5, len(test_predictions))):
#for i in range(len(test_predictions)):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    tmp_vmax = np.max(np.abs(test_data[i, ..., 0]))
    img = plt.imshow(test_data[i, ..., 0], cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    plt.axis("off")
    #plt.title("Corrupted Input")
    plt.title('Test Input {:04d}'.format(i))
    cbar = plt.colorbar(img)

    plt.subplot(1, 3, 2)
    img2 = plt.imshow(np.squeeze(scale_test_predictions3[i, ...], axis=-1), cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    plt.axis("off")
    #plt.title("Model Output")
    plt.title('Output {:04d}'.format(i))
    cbar2 = plt.colorbar(img2)


    plt.subplot(1, 3, 3)
    img3 = plt.imshow(test_data[i, ..., 0] - np.squeeze(scale_test_predictions3[i, ...], axis=-1), cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    #plt.imshow(ground_truth[i, ..., 0], cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    cbar3 = plt.colorbar(img3)
    
    plt.axis("off")
    plt.title("Diff")

    plt.show()


# Generate predictions on test data
test_predictions4 = saved_model.predict(scale_test_predictions3)


scale_test_predictions4 = np.zeros_like(test_predictions4)

for i in range(len(test_predictions)):
    tmp_scale = np.mean(np.abs(test_predictions4[i, ..., 0]))/np.mean(np.abs(test_data[i, ..., 0]))
    scale_test_predictions4[i, ..., 0] = test_predictions4[i, ..., 0] / tmp_scale

scale_test_predictions4[~nan_mask] = test_data[~nan_mask]


from matplotlib import pyplot as plt

for i in range(min(5, len(test_predictions))):
#for i in range(len(test_predictions)):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    tmp_vmax = np.max(np.abs(test_data[i, ..., 0]))
    img = plt.imshow(test_data[i, ..., 0], cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    plt.axis("off")
    #plt.title("Corrupted Input")
    plt.title('Test Input {:04d}'.format(i))
    cbar = plt.colorbar(img)

    plt.subplot(1, 3, 2)
    img2 = plt.imshow(np.squeeze(scale_test_predictions4[i, ...], axis=-1), cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    plt.axis("off")
    #plt.title("Model Output")
    plt.title('Output {:04d}'.format(i))
    cbar2 = plt.colorbar(img2)


    plt.subplot(1, 3, 3)
    img3 = plt.imshow(test_data[i, ..., 0] - np.squeeze(scale_test_predictions4[i, ...], axis=-1), cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    #plt.imshow(ground_truth[i, ..., 0], cmap="bone", vmin = -tmp_vmax/20, vmax = tmp_vmax/20)
    cbar3 = plt.colorbar(img3)
    
    plt.axis("off")
    plt.title("Diff")

    plt.show()


np.save(r'F:\gpr-max-deep-learning-challenge-1-gdlc-1\scale_test_predictions4.npy', scale_test_predictions4)


import numpy as np
import h5py
from scipy.ndimage import zoom

def make_materials_txt_and_h5_lowres(Id2, src_h5_file_name ,h5_file_name, txt_file_name, is_rescale = True):
    
    #Id2.shape (224, 224)
    #domain: 50.02 26 0.02 
    # #dx_dy_dz: 0.02 0.02 0.02 

    #transpose and upside down
    Id2 = np.flipud(Id2).T

    # rescale data
    if(is_rescale):
        Id2 = (Id2 + 0.5) * 10

    # Id2 (224 x 224) ->  data(2500 x 1100)
    zoom_factors = (2500 / 224, 1100 / 224)
    data = zoom(Id2, zoom_factors, order=0)  # order=0 for nearest-neighbor

    # trim data range(1, 10)
    data = np.clip(data, 1, 10)

    # round data 0.001
    data = np.round(data, 3)

    # data to Id, apply 1.0 -> 2, 1.01 -> 3 , ..., 10.0 -> 902
    Id = (np.round((data - 1) * 100 + 2)).astype(int)

    #geometry_objects_read: 0 0 0 ../src_model/file_geo.h5 ../src_model/file_geo_materials.txt
    # 'file_geo_materials.txt'

    #material: 1 inf 1 0 pec
    #material: 1 0 1 0 free_space
    #material: 8.76539 0.000876539 1 0 |geo_1_01|
    #material: 8.88942 0.000888942 1 0 |geo_1_02|
    #material: 9.01346 0.000901346 1 0 |geo_1_03|

    #file_geo.h5
    #ID
    #32-bit unsigned integer
    #6 x 2500 x 1100 x 1
    # index 0 and 1 is 0
    # index 2-5 fill data

    #data
    #64-bit integer
    #2500 x 1100 x 1

    #rigidE
    #8-bit integer
    #12 x 2500 x 1100 x 1
    # all data is 1

    #rigidH
    #6 x 2500 x 1100 x 1
    #8-bit integer
    # all data is 1

    # Id2 round at 0.01
    # And apply number
    # Id 0 pec
    # Id 1 air
    # Id 2 = 1 
    # Id 9002 = 10

    # write file_geo_materials.txt
    with open(txt_file_name, 'w') as file:
        # ãƒ˜ãƒƒãƒ€ãƒ¼éƒ¨åˆ†ã�®æ›¸ã��è¾¼ã�¿
        file.write('#material: 1 inf 1 0 pec\n')
        file.write('#material: 1 0 1 0 free_space\n')
    
        # 1.00ã�‹ã‚‰10.000ã�¾ã�§ã‚’0.001ã�”ã�¨ã�«å‡¦ç�†
        for i, v in enumerate(np.arange(1.00, 10.01, 0.01)):
            file.write('#material: {} {} 1 0 |geo_{:04d}|\n'.format(round(v, 3), round(v / 10000, 7), i + 1))

    # write h5
    # Copy and update the HDF5 file
    # Copy and update the HDF5 file
    with h5py.File(src_h5_file_name, "r") as src_file, h5py.File(h5_file_name, "w") as dest_file:
        def copy_attrs(src_obj, dest_obj):
            for key, value in src_obj.attrs.items():
                dest_obj.attrs[key] = value

        # Copy groups, datasets, and attributes
        for name, obj in src_file.items():
            if isinstance(obj, h5py.Group):
                # Copy group
                dest_group = dest_file.create_group(name)
                copy_attrs(obj, dest_group)
            elif isinstance(obj, h5py.Dataset):
                # Copy dataset
                dest_dataset = dest_file.create_dataset(name, data=obj[...])
                copy_attrs(obj, dest_dataset)

        # Copy root attributes
        copy_attrs(src_file, dest_file)

        # Update 'ID' dataset
        if 'ID' in dest_file:
            for i in range(2, 6):
                dest_file['ID'][i, :, :, 0] = Id

        # Update 'data' dataset
        if 'data' in dest_file:
            dest_file['data'][:, :, 0] = Id

    return


predictions = np.maximum(predictions, 1.5)


for i in range(100):
    make_materials_txt_and_h5_lowres(predictions[i], r"F:\file_geo.h5" , r"F:gpr-max-deep-learning-challenge-1-gdlc-1\sub9_h5txt\lowres_test_file_geo_result_{:04d}.h5".format(i), r"F:gpr-max-deep-learning-challenge-1-gdlc-1\sub9_h5txt\lowres_test_file_geo_materials_result_{:04d}.txt".format(i), is_rescale = False)


import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

for i in range(100):

    # NumPyé…�åˆ—ã‚’ä½œæˆ�ã�¾ã�Ÿã�¯èª­ã�¿è¾¼ã�¿
    data = predictions[i]
    data = np.flipud(data)

    # ParaViewã�®ã‚¹ã‚±ãƒ¼ãƒ«ã�«å�ˆã‚�ã�›ã�¦è»¸ç¯„å›²ã‚’æŒ‡å®š
    x_min, x_max = 0, 50
    y_min, y_max = 0, 22

    # VTKé…�åˆ—ã�«å¤‰æ�›
    vtk_data = numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data.SetName("Epsilon_r")

    # VTK ImageDataã‚ªãƒ–ã‚¸ã‚§ã‚¯ãƒˆã‚’ä½œæˆ�
    image_data = vtk.vtkImageData()

    image_data.SetDimensions(data.shape[1], data.shape[0], 1)
    image_data.SetOrigin(x_min, y_min, 0)
    image_data.SetSpacing((x_max - x_min) / (data.shape[1] - 1), (y_max - y_min) / (data.shape[0] - 1), 1)
    image_data.GetPointData().SetScalars(vtk_data)

    # ãƒ•ã‚¡ã‚¤ãƒ«ã�«ä¿�å­˜
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(r"F:\gpr-max-deep-learning-challenge-1-gdlc-1\sub9vti\sub9_{:04d}_output_ud.vti".format(i))
    writer.SetInputData(image_data)
    writer.Write()





from matplotlib import pyplot as plt
import os

img_folder = os.path.join(OUT_DIR, "sub9_model_image")

for i in range(100):
    # ç”»åƒ�ã�®è¡¨ç¤º
    plt.figure(figsize=(6, 6))
    img = plt.imshow(predictions[i], cmap='viridis_r', vmin=1.5, vmax=10, extent=[0, 50, 22, 0], aspect = 'auto')  # ã‚«ãƒ©ãƒ¼ãƒ�ãƒƒãƒ—ã‚’æŒ‡å®šå�¯èƒ½ (ä¾‹: 'viridis')
    #img = plt.imshow(B2, cmap='gray_r')  # ã‚«ãƒ©ãƒ¼ãƒ�ãƒƒãƒ—ã‚’æŒ‡å®šå�¯èƒ½ (ä¾‹: 'viridis')

    # ã‚«ãƒ©ãƒ¼ãƒ�ãƒ¼ã‚’è¿½åŠ 
    cbar = plt.colorbar(img)
    cbar.set_label(r'$\varepsilon_r$')
    #cbar.set_label('Amplitude')

    # è¡¨ç¤º
    plt.title('Model {:04d}'.format(i))
    plt.xlabel("X [m]")
    plt.ylabel("Depth [m]")
    #plt.show()

    filename = os.path.join(img_folder, f"Test_model{i}.jpg")
    
    plt.savefig(filename)
    plt.close()


import numpy as np
scale_test_predictions4 = np.load(r'/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/scale_test_predictions4.npy')


from tensorflow.keras.models import load_model
import tensorflow as tf

# å��å‰�ä»˜ã��é–¢æ•°ã�¨ã�—ã�¦ã‚«ã‚¹ã‚¿ãƒ æ��å¤±ã‚’å®šç¾©
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

model_path = r"/mnt/f/gpr-max-deep-learning-challenge-1-gdlc-1/bestmodel/20250507-111642_flaml_best_model.h5"
# custom_objectsã�« {'<lambda>': custom_loss} ã‚’æ¸¡ã�™
model = load_model(model_path, custom_objects={'<lambda>': custom_loss})

# ãƒ¢ãƒ‡ãƒ«ã�®äºˆæ¸¬
result = model.predict(scale_test_predictions4)

# çµ�æ�œã�®å½¢çŠ¶ã‚’è¡¨ç¤º
print("Predicted result shape:", result.shape)


result = (result + 0.5) * 10
result_tmp = result


from matplotlib import pyplot as plt

for i in range(10):
    plt.figure(figsize=(6, 6))
    img = plt.imshow(result[i], cmap='viridis_r', vmin=1.5, vmax=10, extent=[0, 50, 22, 0], aspect = 'auto')  # ã‚«ãƒ©ãƒ¼ãƒ�ãƒƒãƒ—ã‚’æŒ‡å®šå�¯èƒ½ (ä¾‹: 'viridis')

    cbar = plt.colorbar(img)
    cbar.set_label(r'$\varepsilon_r$')


    # è¡¨ç¤º
    plt.title('Model {:04d}'.format(i))
    plt.xlabel("X [m]")
    plt.ylabel("Depth [m]")
    plt.show()


predictions = np.squeeze(result, axis=-1)


def around_plot(num):
    plt.plot(predictions[num,:,0],label='0')
    plt.plot(predictions[num,:,1],label='1')
    plt.plot(predictions[num,:,2],label='2')
    plt.plot(predictions[num,:,3],label='3')
    plt.plot(predictions[num,:,4],label='4')
    plt.legend()
    plt.title('Left')
    plt.show()

    plt.plot(predictions[num,:,-1],label='-1')
    plt.plot(predictions[num,:,-2],label='-2')
    plt.plot(predictions[num,:,-3],label='-3')
    plt.plot(predictions[num,:,-4],label='-4')
    plt.plot(predictions[num,:,-5],label='-5')
    plt.legend()
    plt.title('Right')
    plt.show()

    plt.plot(predictions[num,0,:],label='0')
    plt.plot(predictions[num,1,:],label='1')
    plt.plot(predictions[num,2,:],label='2')
    plt.plot(predictions[num,3,:],label='3')
    plt.plot(predictions[num,4,:],label='4')
    plt.legend()
    plt.title('Top')
    plt.show()

    plt.plot(predictions[num,-1,:],label='-1')
    plt.plot(predictions[num,-2,:],label='-2')
    plt.plot(predictions[num,-3,:],label='-3')
    plt.plot(predictions[num,-4,:],label='-4')
    plt.plot(predictions[num,-5,:],label='-5')
    plt.legend()
    plt.title('Bottom')
    plt.show()


around_plot(7)


predictions[:,:,-1] = predictions[:,:,-3]
predictions[:,:,-2] = predictions[:,:,-3]

predictions[:,0,:] = predictions[:,2,:]
predictions[:,1,:] = predictions[:,2,:]

predictions[:,-1,:] = predictions[:,-3,:]
predictions[:,-2,:] = predictions[:,-3,:]


np.max(predictions)


np.min(predictions)


import pandas as pd

array_2d = predictions.reshape(100, -1)
array_2d = np.maximum(array_2d, 1.5)
array_2d = np.minimum(array_2d, 10)
df = pd.DataFrame(array_2d)
df["Id"] = range(len(df))
df.set_index('Id', inplace=True)
df.to_csv(r'/mnt/f/sub12.csv')


for idx, layer in enumerate(model.layers):
    print(idx, layer.name)

