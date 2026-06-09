import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(os.path.join(dirname))


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import cv2
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
from functools import partial
import sklearn
from tqdm import tqdm_notebook as tqdm
import gc
%matplotlib inline


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()


print('Number of replicas:', strategy.num_replicas_in_sync)
print("Version of Tensorflow used : ", tf.__version__)


AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "/kaggle/input/siim-isic-melanoma-classification"
# BATCH_SIZE = 16 * strategy.num_replicas_in_sync
# IMAGE_SIZE = [1024, 1024]
# SHAPE = [256, 256] 
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
SHAPE = [384, 384] 


print("Batch Size = ", BATCH_SIZE)
print("GCS Path = ", GCS_PATH)


train = pd.DataFrame(pd.read_csv("../input/siim-isic-melanoma-classification/train.csv"))
train.head()


test = pd.DataFrame(pd.read_csv("../input/siim-isic-melanoma-classification/test.csv"))
test.head()


train.info()


test.info()


train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"


image_names = train["image_name"].values + ".jpg"
random_images = [np.random.choice(image_names) for i in range(4)] # Generates a random sample from a given 1-D array
random_images 


sample_images = []


plt.figure(figsize = (12, 8))
for i in range(4) : 
    plt.subplot(2, 2, i + 1) 
    image = cv2.imread(os.path.join(train_dir, random_images[i]))
    # cv2 reads images in BGR format. Hence we convert it to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample_images.append(image)
    plt.imshow(image, cmap = "gray")
    plt.grid(True)
# Automatically adjust subplot parameters to give specified padding.
plt.tight_layout()


from sklearn.model_selection import train_test_split 
training_files, validation_files = train_test_split(tf.io.gfile.glob(GCS_PATH + "/tfrecords/train*.tfrec"),
                                                   test_size = 0.1, random_state = 42)

testing_files = tf.io.gfile.glob(GCS_PATH + "/tfrecords/test*.tfrec")

print("Number of training files = ", len(training_files))
print("Number of validation files = ", len(validation_files))
print("Number of test files = ", len(testing_files))


def decode_image(image) : 
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return image


sample_images[0].shape


training_files


sample_picked = training_files[0]
sample_picked


file = tf.data.TFRecordDataset(sample_picked)
file


feature_description = {"image" : tf.io.FixedLenFeature([], tf.string), 
                      "target" : tf.io.FixedLenFeature([], tf.int64)}


def parse_function(example) : 
    # The example supplied is parsed based on the feature_description above.
    return tf.io.parse_single_example(example, feature_description)


parsed_dataset = file.map(parse_function)
parsed_dataset


def read_tfrecord(example, labeled) : 
    if labeled == True : 
        tfrecord_format = {"image" : tf.io.FixedLenFeature([], tf.string),
                           "target" : tf.io.FixedLenFeature([], tf.int64)}
    else:
        tfrecord_format = {"image" : tf.io.FixedLenFeature([], tf.string),
                          "image_name" : tf.io.FixedLenFeature([], tf.string)}
    
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled == True : 
        label = tf.cast(example["target"], tf.int32)
        return image, label
    else:
        image_name = example["image_name"]
        return image, image_name     


def load_dataset(filenames, labeled, ordered):
    ignore_order = tf.data.Options()
    if ordered == False: # dataset is unordered, so we ignore the order to load data quickly.
        ignore_order.experimental_deterministic = False # This disables the order and enhances the speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) 
    dataset = dataset.with_options(ignore_order) 
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return dataset


def image_augmentation(image, label) :     
    image = tf.image.resize(image, SHAPE)
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_training_dataset() : 
    dataset = load_dataset(training_files, labeled = True, ordered = False)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE) 
    return dataset


def get_validation_dataset() : 
    dataset = load_dataset(validation_files, labeled = True, ordered = False)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE) 
    return dataset


def get_test_dataset() : 
    dataset = load_dataset(testing_files, labeled = False, ordered = True)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE) 
    return dataset


training_dataset = get_training_dataset()


validation_dataset = get_validation_dataset()


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

num_training_images = count_data_items(training_files)
num_validation_images = count_data_items(validation_files)
num_testing_images = count_data_items(testing_files)

STEPS_PER_EPOCH_TRAIN = num_training_images // BATCH_SIZE
STEPS_PER_EPOCH_VAL = num_validation_images // BATCH_SIZE

print("Number of Training Images = ", num_training_images)
print("Number of Validation Images = ", num_validation_images)
print("Number of Testing Images = ", num_testing_images)
print("\n")
print("Numer of steps per epoch in Train = ", STEPS_PER_EPOCH_TRAIN)
print("Numer of steps per epoch in Validation = ", STEPS_PER_EPOCH_VAL)


image_batch, label_batch = next(iter(training_dataset))


def show_batch(image_batch, label_batch) :
    plt.figure(figsize = (20, 20))
    for n in range(8) : 
        ax = plt.subplot(2,4,n+1)
        plt.imshow(image_batch[n])
        if label_batch[n] == 0 : 
            plt.title("BENIGN")
        else:
            plt.title("MALIGNANT")
    plt.grid(False)
    plt.tight_layout()       


show_batch(image_batch.numpy(), label_batch.numpy())


del image_batch
del label_batch
gc.collect()


malignant = len(train[train["target"] == 1])
benign = len(train[train["target"] == 0 ])
total = len(train) 

print("Malignant Cases in Train Data = ", malignant)
print("Benign Cases In Train Dataset = ",benign)
print("Total Cases In Train Dataset = ",total)
print("Ratio of Malignant to Benign = ",malignant/benign)


weight_malignant = (total/malignant)/2.0
weight_benign = (total/benign)/2.0

class_weight = {0 : weight_benign , 1 : weight_malignant}

print("Weight for benign cases = ", class_weight[0])
print("Weight for malignant cases = ", class_weight[1])


callback_early_stopping = tf.keras.callbacks.EarlyStopping(patience = 15, verbose = 0, restore_best_weights = True)

callbacks_lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_auc", factor = 0.1, patience = 10, 
                                                          verbose = 0, min_lr = 1e-6)

callback_checkpoint = tf.keras.callbacks.ModelCheckpoint("melanoma_weights.h5",
                                                         save_weights_only=True, monitor='val_auc',
                                                         mode='max', save_best_only = True)


# with strategy.scope() : 
    
#     # Khá»Ÿi táº¡o bias cho lá»›p cuá»‘i
#     bias = np.log(malignant/benign)
#     bias = tf.keras.initializers.Constant(bias)
    
#     # Táº O MÃ” HÃŒNH (Giai Ä‘oáº¡n 1: Freeze)
#     base_model = tf.keras.applications.MobileNetV2(input_shape = (SHAPE[0], SHAPE[1], 3), 
#                                                    include_top = False,
#                                                    weights = "imagenet")
#     base_model.trainable = False
    
#     global model, history, history_fine_tune
    
#     model = tf.keras.Sequential([base_model,
#                                  tf.keras.layers.GlobalAveragePooling2D(),
#                                  tf.keras.layers.Dense(20, activation = "relu"), 
#                                  tf.keras.layers.Dropout(0.4),
#                                  tf.keras.layers.Dense(10, activation = "relu"),
#                                  tf.keras.layers.Dropout(0.3),
#                                  tf.keras.layers.Dense(1, activation = "sigmoid", bias_initializer = bias)
#                                ])
    
#     # COMPILE MÃ” HÃŒNH GIAI Ä�Oáº N 1 (LR = 1e-4)
#     model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), 
#                   loss = "binary_crossentropy", 
#                   metrics = [tf.keras.metrics.AUC(name = 'auc')])
    
#     print("--- Báº®T Ä�áº¦U GIAI Ä�Oáº N 1 (FREEZE) ---")

#     # HUáº¤N LUYá»†N GIAI Ä�Oáº N 1: FREEZE (8 EPOCHS)
#     EPOCHS_FREEZE = 100
#     history = model.fit(training_dataset, 
#                         epochs = EPOCHS_FREEZE, 
#                         steps_per_epoch = STEPS_PER_EPOCH_TRAIN,
#                         validation_data = validation_dataset, 
#                         validation_steps = STEPS_PER_EPOCH_VAL,
#                         callbacks = [callback_early_stopping, callbacks_lr_reduce, callback_checkpoint],
#                         class_weight = class_weight)

#     # --- GIAI Ä�Oáº N 2: FINE-TUNING ---
    
#     # Má»Ÿ Ä‘Ã³ng bÄƒng má»™t pháº§n cá»§a mÃ´ hÃ¬nh Base
#     base_model.trainable = True 
#     for layer in base_model.layers[:-20]: 
#         layer.trainable = False
    
#     # # COMPILE Láº I MÃ” HÃŒNH GIAI Ä�Oáº N 2 (LR = 1e-6)
#     # model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-6), 
#     #               loss = "binary_crossentropy", 
#     #               metrics = [tf.keras.metrics.AUC(name = 'auc')])

#     print("\n--- Báº®T Ä�áº¦U GIAI Ä�Oáº N 2 (FINE-TUNING) ---")
    
#     # HUáº¤N LUYá»†N GIAI Ä�Oáº N 2: FINE-TUNE (18 EPOCHS)
#     FINE_TUNE_EPOCHS = 200
#     TOTAL_EPOCHS = EPOCHS_FREEZE + FINE_TUNE_EPOCHS
#     START_EPOCH = history.epoch[-1] + 1 if history.epoch else 0 
    
#     history_fine_tune = model.fit(training_dataset, 
#                                   epochs = TOTAL_EPOCHS,
#                                   initial_epoch = START_EPOCH,
#                                   steps_per_epoch = STEPS_PER_EPOCH_TRAIN,
#                                   validation_data = validation_dataset, 
#                                   validation_steps = STEPS_PER_EPOCH_VAL,
#                                   callbacks = [callback_early_stopping, callbacks_lr_reduce, callback_checkpoint],
#                                   class_weight = class_weight)
    
# # Káº¾T THÃšC khá»‘i strategy.scope()


with strategy.scope():

    # --- KHá»�I Táº O bias CHO Lá»šP CUá»�I ---
    bias = np.log(malignant / benign)
    bias = tf.keras.initializers.Constant(bias)

    # --- GIAI Ä�Oáº N 1: FREEZE ---
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(SHAPE[0], SHAPE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Ä‘Ã³ng bÄƒng toÃ n bá»™

    global model, history, history_fine_tune

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=bias)
    ])

    # --- COMPILE GIAI Ä�Oáº N 1 ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    print("\n--- Báº®T Ä�áº¦U GIAI Ä�Oáº N 1 (FREEZE) ---")
    EPOCHS_FREEZE = 10

    history = model.fit(
        training_dataset,
        epochs=EPOCHS_FREEZE,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        validation_data=validation_dataset,
        validation_steps=STEPS_PER_EPOCH_VAL,
        callbacks=[callback_early_stopping, callbacks_lr_reduce, callback_checkpoint],
        class_weight=class_weight
    )

    # --- GIAI Ä�Oáº N 2: FINE-TUNING ---
    print("\n--- Báº®T Ä�áº¦U GIAI Ä�Oáº N 2 (FINE-TUNING) ---")

    # Má»Ÿ trainable cho base_model
    base_model.trainable = True

    # Giá»¯ nguyÃªn pháº§n lá»›n layer freeze (chá»‰ má»Ÿ 20 lá»›p cuá»‘i)
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Giá»¯ BatchNorm khÃ´ng trainable Ä‘á»ƒ trÃ¡nh lá»‡ch thá»‘ng kÃª
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Táº¡o optimizer má»›i, LR ráº¥t nhá»�
    fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

    # COMPILE Láº I MÃ” HÃŒNH
    model.compile(
        optimizer=fine_tune_optimizer,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    # Tiáº¿p tá»¥c train
    FINE_TUNE_EPOCHS = 10
    TOTAL_EPOCHS = EPOCHS_FREEZE + FINE_TUNE_EPOCHS
    START_EPOCH = history.epoch[-1] + 1 if history.epoch else 0

    history_fine_tune = model.fit(
        training_dataset,
        epochs=TOTAL_EPOCHS,
        initial_epoch=START_EPOCH,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        validation_data=validation_dataset,
        validation_steps=STEPS_PER_EPOCH_VAL,
        callbacks=[callback_early_stopping, callbacks_lr_reduce, callback_checkpoint],
        class_weight=class_weight
    )



n_epochs_it_ran_for = len(history.history['loss'])
n_epochs_it_ran_for


X = np.arange(0,n_epochs_it_ran_for,1)
plt.figure(1, figsize = (20, 12))
plt.subplot(1,2,1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(X, history.history["loss"], label = "Training Loss")
plt.plot(X, history.history["val_loss"], label = "Validation Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(X, history.history["auc"], label = "Training Accuracy")
plt.plot(X, history.history["val_auc"], label = "Validation Accuracy")
plt.grid(True)
plt.legend()


resulting_probabilities = model.predict(testing_dataset_images, verbose = 1)


len(resulting_probabilities)


sample_submission_file = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
sample_submission_file.head()


del sample_submission_file["target"]
sample_submission_file.head()


testing_image_names


testing_image_names = np.concatenate([x for x in testing_image_names], axis=0)
testing_image_names = np.array(testing_image_names)


decoded_test_names = []
for names in testing_image_names : 
    names = names.decode('utf-8')
    decoded_test_names.append(names)
decoded_test_names = np.array(decoded_test_names)
del testing_image_names


len(decoded_test_names), type(decoded_test_names), decoded_test_names.shape


decoded_test_names


testing_image_names = pd.DataFrame(decoded_test_names, columns=["image_name"])
testing_image_names.head()


pred_dataframe = pd.DataFrame({"image_name" : decoded_test_names, 
                               "target" : np.concatenate(resulting_probabilities)})
pred_dataframe


sample_submission_file = sample_submission_file.merge(pred_dataframe, on = "image_name")
sample_submission_file.to_csv("submission.csv", index = False)
sample_submission_file.head()


model.save("melanoma_model.h5")


# ===========================================
# ğŸ”� Ä�Ã�NH GIÃ� MÃ” HÃŒNH MELANOMA TRÃŠN Dá»® LIá»†U THáº¬T (KAGGLE)
# ===========================================
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os

print("\nğŸš€ Báº®T Ä�áº¦U Ä�Ã�NH GIÃ� MÃ” HÃŒNH TRÃŠN Dá»® LIá»†U THáº¬T")

# ====== 1ï¸�âƒ£ Cáº¤U HÃŒNH ======
DATA_DIR = "/kaggle/input/siim-isic-melanoma-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "jpeg/train")

IMG_SIZE = 384
BATCH_SIZE = 32

# ====== 2ï¸�âƒ£ Táº O DATASET KIá»‚M Ä�á»ŠNH Tá»ª Dá»® LIá»†U THáº¬T ======
print("\nÄ�ang Ä‘á»�c file train.csv...")
df = pd.read_csv(TRAIN_CSV)
print(f"Tá»•ng sá»‘ áº£nh trong dataset: {len(df)}")

# ğŸ§© Láº¥y máº«u nhá»� Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ nhanh (báº¡n cÃ³ thá»ƒ tÄƒng náº¿u muá»‘n)
df_sample = df.sample(2000, random_state=42).reset_index(drop=True)

# Ä�Æ°á»�ng dáº«n áº£nh tháº­t
image_paths = [os.path.join(TRAIN_DIR, f"{img_id}.jpg") for img_id in df_sample["image_name"]]
labels = df_sample["target"].values

# HÃ m xá»­ lÃ½ áº£nh
def decode_image(filename, label):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label

# Táº¡o dataset TensorFlow
eval_dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
eval_dataset = eval_dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
eval_dataset = eval_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

num_validation_images = len(df_sample)
print(f"Sá»‘ lÆ°á»£ng áº£nh Ä‘Æ°á»£c dÃ¹ng Ä‘á»ƒ Ä‘Ã¡nh giÃ¡: {num_validation_images}")

# ====== 3ï¸�âƒ£ Táº¢I Láº I MÃ” HÃŒNH HUáº¤N LUYá»†N ======
print("\nÄ�ang táº£i mÃ´ hÃ¬nh Ä‘áº§y Ä‘á»§ tá»« 'melanoma_model.h5'...")
model_eval = tf.keras.models.load_model("/kaggle/working/melanoma_model.h5")
print("âœ… MÃ´ hÃ¬nh táº£i thÃ nh cÃ´ng.")

# ====== 4ï¸�âƒ£ Dá»° Ä�OÃ�N ======
print("\nÄ�ang cháº¡y dá»± Ä‘oÃ¡n...")
y_true = np.array(labels)
y_pred_probs = model_eval.predict(eval_dataset.map(lambda x, y: x), verbose=1)
y_pred_labels = (y_pred_probs > 0.5).astype(int)

# ====== 5ï¸�âƒ£ Ä�Ã�NH GIÃ� HIá»†U SUáº¤T ======
print("\nğŸ“Š Káº¾T QUáº¢ Ä�Ã�NH GIÃ� MÃ” HÃŒNH")

# BÃ¡o cÃ¡o phÃ¢n loáº¡i
print("\nClassification Report:")
print(classification_report(y_true, y_pred_labels, target_names=['Benign (0)', 'Malignant (1)']))

# Ma tráº­n nháº§m láº«n
cm = confusion_matrix(y_true, y_pred_labels)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print(f"Sensitivity (Recall): {tp / (tp + fn):.4f}")
print(f"Specificity:          {specificity:.4f}")

# Váº½ Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Benign (0)', 'Pred Malignant (1)'],
            yticklabels=['True Benign (0)', 'True Malignant (1)'])
plt.title('Confusion Matrix (Evaluation Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC & AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
print(f"\nAUC (tá»« sklearn): {roc_auc:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("\nâœ… Ä�Ã¡nh giÃ¡ hoÃ n táº¥t.")


