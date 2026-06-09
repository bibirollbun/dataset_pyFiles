import os, sys
import cv2
import pandas as pd
from PIL import Image
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from collections import Counter
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet169
import scipy
from tqdm import tqdm
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

EPOCHS = 100
BATCH_SIZE = 64
SEED = 19071591
LRATE = 0.00005
MIN_LRATE = 0.00001
VERBOSE=1
DROPOUT = 0.3

np.random.seed(SEED)
tf.random.set_seed(SEED)


"""import shutil
dataset_dir = '/kaggle/working/datasets'
train_images_dir = os.path.join(dataset_dir, 'train_images')
train_csv_path = os.path.join(dataset_dir, 'train.csv')

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

if not os.path.exists(train_images_dir):
    os.makedirs(train_images_dir)

train_df = pd.DataFrame(columns=['id_code', 'diagnosis'])

train_df2015 = pd.read_csv('/kaggle/input/resized-2015-2019-blindness-detection-images/labels/trainLabels15.csv')
train_df2015 = train_df2015.rename(columns={'image': 'id_code', 'level': 'diagnosis'})
train_df2015 = train_df2015.drop(train_df2015[train_df2015['diagnosis'] == 0].sample(n=25810, replace=False).index)
train_df2015 = train_df2015.drop(train_df2015[train_df2015['diagnosis'] == 1].sample(n=1746, replace=False).index)
train_df2015 = train_df2015.drop(train_df2015[train_df2015['diagnosis'] == 2].sample(n=5224, replace=False).index)

train_df2019 = pd.read_csv('/kaggle/input/resized-2015-2019-blindness-detection-images/labels/trainLabels19.csv')
train_df2019 = train_df2019.drop(train_df2019[train_df2019['diagnosis'] == 0].sample(n=738, replace=False).index)

train_dfidrid = pd.read_csv('/kaggle/input/idrid-dataset/idrid_labels.csv')
train_dfidrid = train_dfidrid.drop(train_dfidrid[train_dfidrid['diagnosis'] == 0].sample(n=129, replace=False).index)
train_dfidrid = train_dfidrid.drop(train_dfidrid[train_dfidrid['diagnosis'] == 1].sample(n=22, replace=False).index)
train_dfidrid = train_dfidrid.drop(train_dfidrid[train_dfidrid['diagnosis'] == 2].sample(n=156, replace=False).index)
train_dfidrid = train_dfidrid.drop(train_dfidrid[train_dfidrid['diagnosis'] == 3].sample(n=83, replace=False).index)

train_df = pd.concat([train_df2015, train_df2019, train_dfidrid], ignore_index=True)

source_dirs = ['/kaggle/input/resized-2015-2019-blindness-detection-images/resized train 15',
'/kaggle/input/resized-2015-2019-blindness-detection-images/resized train 19',
'/kaggle/input/idrid-dataset/Imagenes/Imagenes']

new_rows = []
for index, row in train_df.iterrows():
    image_id = row['id_code']
    diagnosis = row['diagnosis']

    found = False  
    for source_dir in source_dirs:
        source_path_jpg = os.path.join(source_dir, image_id + '.jpg')
        source_path_png = os.path.join(source_dir, image_id + '.png')

        if os.path.exists(source_path_jpg):
            image_name = image_id
            destination_path = os.path.join(train_images_dir, image_name)
            shutil.copy(source_path_jpg, destination_path)
            new_rows.append({'id_code': image_name, 'diagnosis': diagnosis})
            found = True
        elif os.path.exists(source_path_png):
            image_name = image_id
            destination_path = os.path.join(train_images_dir, image_name)
            shutil.copy(source_path_png, destination_path)
            new_rows.append({'id_code': image_name, 'diagnosis': diagnosis})
            found = True

        if found:
            break 

    if not found:
        print(f"Image {image_id} not found in any source directory (neither .jpg .png)")

train_df = pd.DataFrame(new_rows)
train_df.to_csv(train_csv_path, index=False)

num_images = len(os.listdir(train_images_dir))
print(f"Number of images in train_images: {num_images}")"""


train_df = pd.read_csv('/kaggle/input/aptos2019/datasets/train.csv')
print(train_df.shape)
train_df.head()


train_df['diagnosis'].hist()
train_df['diagnosis'].value_counts()


def crop_image(image, tol = 7):
    if image.ndim == 2:
        mask = image > tol
        return image[np.ix_(mask.any(1), mask.any(0))]
    elif image.ndim == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = image_gray > tol
        check_shape = image[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if(check_shape == 0):
            return image
        else:
            image1 = image [:,:,0][np.ix_(mask.any(1), mask.any(0))]
            image2 = image [:,:,1][np.ix_(mask.any(1), mask.any(0))]
            image3 = image [:,:,2][np.ix_(mask.any(1), mask.any(0))]
            image = np.stack([image1, image2, image3], axis = -1)
        return image

def add_black_padding_and_resize(image, img_size):
    h, w = image.shape[:2]
    new_h, new_w = img_size, img_size
    if h > w:
        scale_factor = img_size / h
    else:
        scale_factor = img_size / w

    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h))
    top = (img_size - new_h) // 2
    bottom = img_size - new_h - top
    left = (img_size - new_w) // 2
    right = img_size - new_w - left

    padded_resized_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value= 0)
    return padded_resized_image


def preprocess_image(image, img_size=224):
    image = Image.open(image)
    image = np.array(image)
    image = crop_image(image)
    image = add_black_padding_and_resize(image, img_size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel_horizontal = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)

    kernel_vertical = np.array([[1],
                                [1],
                                [1],
                                [1],
                                [1]], dtype=np.uint8)

    blackhat_h = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_horizontal)
    blackhat_v = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_vertical)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    image_tophat = cv2.add(image, tophat)
    image_blackhat = np.maximum(np.maximum(blackhat_h, blackhat_v), blackhat)
    image = cv2.subtract(image_tophat, image_blackhat)
    image =np.stack((image,)*3, axis=-1)
    image = Image.fromarray(image)
    return image


N = train_df.shape[0]
x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)
for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(f'/kaggle/input/aptos2019/datasets/train_images/{image_id}')


test_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv')
print(test_df.shape)
test_df.head()


N = test_df.shape[0]
x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)
for i, image_id in enumerate(tqdm(test_df['id_code'])):
    x_test[i, :, :, :] = preprocess_image(f'/kaggle/input/aptos2019-blindness-detection/test_images/{image_id}.png')


y_train = pd.get_dummies(train_df['diagnosis']).values


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=SEED, shuffle=True)
print("đã chia xong")


print("Kích thước tập huấn luyện:")
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("\nKích thước tập xác thực:")
print("x_val:", x_val.shape)
print("y_val:", y_val.shape)


def print_label_counts(y_data, dataset_name=""):
  print(f"Số lượng từng nhãn trong tập {dataset_name}:")
  unique_labels, counts = np.unique(y_data, return_counts=True)
  for label, count in zip(unique_labels, counts):
    print(f"Nhãn {label}: {count} mẫu")
  print()

y_train_labels = np.argmax(y_train, axis=1)
y_val_labels = np.argmax(y_val, axis=1)
print_label_counts(y_train_labels, "huấn luyện")
print_label_counts(y_val_labels, "xác thực")


def train_datagen():
    return ImageDataGenerator(
        horizontal_flip=True,  
        vertical_flip=True,  
        rotation_range=60,
        zoom_range=0.15,  
        fill_mode='constant',
        cval=0.,
    )


data_generator = train_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=SEED)


def precision(y_true, y_pred):
    y_true_f = tf.cast(y_true, tf.float32) 
    y_pred_f = tf.cast(y_pred, tf.float32) 
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred_f, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall(y_true, y_pred):
    y_true_f = tf.cast(y_true, tf.float32) 
    y_pred_f = tf.cast(y_pred, tf.float32) 
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true_f * y_pred_f, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true_f, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    y_true_f = tf.cast(y_true, tf.float32)
    p = precision(y_true_f, y_pred)
    r = recall(y_true_f, y_pred)
    bb = beta ** 2

    def fbeta():
        num = (1 + bb) * (p * r)
        den = bb * p + r + tf.keras.backend.epsilon()
        return num / den

    fbeta_score = tf.cond(
        tf.equal(tf.reduce_sum(tf.round(tf.clip_by_value(y_true_f, 0, 1))), 0),
        lambda: 0.0, 
        fbeta         
    )

    return fbeta_score

def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

def mean_pred(y_true, y_pred):
    return tf.reduce_mean(tf.cast(y_pred, tf.float32))

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

print("Evaluation metrics defined ...")


def build_model():
    input_tensor = layers.Input(shape=(224, 224, 3))
    densenet = DenseNet169(
        weights='/kaggle/input/densenet169/DenseNet-BC-169-32-no-top.h5',
        include_top=False,
        input_shape=(224,224,3)
    )
    x = densenet(input_tensor)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    output_tensor = layers.Dense(5, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LRATE),
        metrics=['accuracy',mean_pred, precision, recall, f1_score, fbeta_score, fmeasure]
    )
    return model
print("Build model ...")


from tensorflow.keras import layers
model = build_model()
model.summary()


class KappaMetrics(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.val_kappas = []
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.validation_data

        y_pred = np.argmax(self.model.predict(x_val), axis=1)
        y_val_true = np.argmax(y_val, axis=1)

        _val_kappa = cohen_kappa_score(
            y_val_true,
            y_pred,
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)
        print(f"Epoch: {epoch+1} val_kappa: {_val_kappa:.4f}")

        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('/kaggle/working/model.h5')

        return


early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=VERBOSE, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=VERBOSE, min_lr=MIN_LRATE)

kappa_metrics = KappaMetrics(validation_data=(x_val, y_val))
history = model.fit(
    data_generator,
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    validation_steps=None,
    callbacks=[kappa_metrics, early_stopping, reduce_lr],
    verbose=VERBOSE
)


with open('/kaggle/working/history.json', 'w') as f:
    json.dump(history.history, f)

max_length = max(len(value) for value in history.history.values())
for key, value in history.history.items():
    if len(value) < max_length:
        history.history[key] = value + [None] * (max_length - len(value))
history_df = pd.DataFrame(history.history)
print(history_df.head(EPOCHS))


import json
import pandas as pd
with open('/kaggle/working/history.json', 'r') as f:
    loaded_history = json.load(f)
    
loaded_history_df = pd.DataFrame(loaded_history)
print(loaded_history_df.head())

"""print(loaded_history['loss'])
print(loaded_history['val_accuracy'])"""


ACTUAL_EPOCHS = len(history.history['loss']) if len(history.history['loss']) < EPOCHS else EPOCHS
epoch_list = list(range(1, ACTUAL_EPOCHS + 1))

f1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
t1 = f1.suptitle('CNN Performance', fontsize=12)
f1.subplots_adjust(top=0.85, wspace=0.3)

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, ACTUAL_EPOCHS + 1, 5)) 
ax1.set_ylabel('Accuracy %')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, ACTUAL_EPOCHS + 1, 5)) 
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

# Biểu đồ các metrics
ax3.plot(epoch_list, history.history['accuracy'], label='Accuracy')
ax3.plot(epoch_list, history.history['precision'], label='Precision')
ax3.plot(epoch_list, history.history['recall'], label='Recall')
ax3.plot(epoch_list, history.history['f1_score'], label='F1 score')
ax3.plot(epoch_list, history.history['fbeta_score'], label='Fbeta score')
ax3.plot(epoch_list, history.history['fmeasure'], label='FMeasure')
ax3.set_xticks(np.arange(0, ACTUAL_EPOCHS + 1, 5)) 
ax3.set_ylabel('Score')
ax3.set_xlabel('Epoch')
ax3.set_title('Performance')
l3 = ax3.legend(loc="best")

ax4.plot(epoch_list, kappa_metrics.val_kappas, label='Kappa score')
ax4.set_xticks(np.arange(0, ACTUAL_EPOCHS + 1, 5)) 
ax4.set_ylabel('Score')
ax4.set_xlabel('Epoch')
ax4.set_title('Kappa Metrics')
l4 = ax4.legend(loc="best")

print("Maximum Kappa Score: %s" % max(kappa_metrics.val_kappas))
plt.show() 


from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('/kaggle/working/model.h5', custom_objects={'precision': precision, 'recall': recall, 'f1_score': f1_score, 'fbeta_score': fbeta_score, 'fmeasure': fmeasure, 'mean_pred': mean_pred}) # Load lại model với các hàm custom metrics

y_pred_prob = model.predict(x_val)
y_val_true = np.argmax(y_val, axis=1)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_val_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=range(5), yticklabels=range(5))  # Giả sử có 5 lớp
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_val_true, y_pred, target_names=[f'Class {i}' for i in range(5)])) 

accuracy = accuracy_score(y_val_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_val_true, y_pred)
for i in range(5): 
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    print(f"Class {i}:")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5): 
    fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(5):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 5
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"],
         label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
         color='navy', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()


import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

model = load_model('/kaggle/working/model.h5', custom_objects={'precision': precision, 'recall': recall, 'f1_score': f1_score, 'fbeta_score': fbeta_score, 'fmeasure': fmeasure, 'mean_pred': mean_pred})

def crop_image(image, tol=7):
    if image.ndim == 2:
        mask = image > tol
        return image[np.ix_(mask.any(1), mask.any(0))]
    elif image.ndim == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = image_gray > tol
        check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return image
        else:
            image1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            image2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            image3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            image = np.stack([image1, image2, image3], axis=-1)
        return image
def add_black_padding_and_resize(image, img_size):
    h, w = image.shape[:2]
    new_h, new_w = img_size, img_size
    if h > w:
        scale_factor = img_size / h
    else:
        scale_factor = img_size / w

    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h))
    top = (img_size - new_h) // 2
    bottom = img_size - new_h - top
    left = (img_size - new_w) // 2
    right = img_size - new_w - left

    padded_resized_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_resized_image
def preprocess_image(image_path, img_size=224):  # Changed parameter to image_path
    image = Image.open(image_path)  # Open image from path
    image = np.array(image)
    image = crop_image(image)
    image = add_black_padding_and_resize(image, img_size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel_horizontal = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
    kernel_vertical = np.array([[1], [1], [1], [1], [1]], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat_h = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_horizontal)
    blackhat_v = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_vertical)
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    image_tophat = cv2.add(image, tophat)
    image_blackhat = np.maximum(np.maximum(blackhat_h, blackhat_v), blackhat)
    image = cv2.subtract(image_tophat, image_blackhat)
    image = np.stack((image,) * 3, axis=-1)
    return image

image_path = '/kaggle/input/aptos2019/datasets/train_images/30_right.jpg' 
img = preprocess_image(image_path)
img = img / 255.0 
img = img.reshape(1, 224, 224, 3)

prediction = model.predict(img)
print("Raw prediction:", prediction)

predicted_class = np.argmax(prediction, axis=1)[0]
print("Predicted class:", predicted_class)

class_labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}
print("Predicted condition:", class_labels[predicted_class])

