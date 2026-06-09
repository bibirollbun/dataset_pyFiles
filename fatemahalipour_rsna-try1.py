import os
import json
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydicom
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.initializers import Constant
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.python.ops import array_ops
from tqdm import tqdm
from keras import backend as K
import tensorflow as tf
import keras
from keras.applications import ResNet50
from keras.models import Model, load_model
from math import ceil, floor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import log_loss
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input


os.listdir('/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection')


Base_path = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'
train_dir = 'stage_2_train/'
test_dir = 'stage_2_test/'

train_csv = os.path.join(Base_path, 'stage_2_train.csv')
train_df = pd.read_csv(train_csv)


train_df['filename'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
train_df['type'] =train_df['ID'].str.split('_').str[2]

print(train_df.shape)
train_df.head()


np.random.seed(2025)
sample_files = np.random.choice(os.listdir(Base_path + train_dir),50000)
sample_df = train_df[train_df.filename.apply(lambda x: x.replace('.png' , '.dcm')).isin(sample_files)]


pivot_df = sample_df[['Label','filename','type']].drop_duplicates().pivot(index = 'filename' , columns = 'type' , values = 'Label').reset_index()
print(pivot_df.shape)
pivot_df


validation_df = pivot_df.sample(int(len(pivot_df)*0.15))
validation_df


y_true = []
for i in range(len(validation_df)):
    y_true.append(validation_df.iloc[i,1])

len(y_true)


full_true = []
for i in range(len(validation_df)):
    for j in range(1,7):
        full_true.append(validation_df.iloc[i,j])


training_df = pivot_df[~(pivot_df.filename.isin(validation_df.filename))]
training_df


print(training_df.head())
print(validation_df.head())


def get_pixels_hu(scan): 
    image = np.stack([scan.pixel_array])
    image = image.astype(np.int16) 
    
    image[image == -2000] = 0
    
    intercept = scan.RescaleIntercept
    slope = scan.RescaleSlope
    
    if slope != 1: 
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    
    image += np.int16(intercept) 
    
    return np.array(image, dtype=np.int16)


def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def apply_window_policy(image):

    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)

    return image


def save_and_resize(filenames, load_dir):    
    save_dir = '/kaggle/tmp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in tqdm(filenames):
        try:
            path = load_dir + filename
            new_path = save_dir + filename.replace('.dcm', '.png')
            dcm = pydicom.dcmread(path)
            image = get_pixels_hu(dcm)
            image = apply_window_policy(image[0])
            image -= image.min((0,1))
            image = (255*image).astype(np.uint8)
            image = cv2.resize(image, (224, 224)) #smaller
            res = cv2.imwrite(new_path, image)
            
        except ValueError:
            continue


save_and_resize(filenames=sample_files, load_dir=Base_path + train_dir)
#save_and_resize(filenames=sample_test, load_dir=Base_path + test_dir)


def create_model():    
    # Use ResNet50 instead of Xception
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    
    for layer in base_model.layers[-10:]:
        layer.trainable = True
        
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.15)(x)
    y_pred = Dense(6, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=y_pred)


LR = 0.00005
model = create_model()


BATCH_SIZE = 16 # had to revert back to 16 to have a comparaison point with the large model I ran locally 

def create_datagen():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,        # چرخش تصادفی ±15 درجه
        width_shift_range=0.1,    # جابجایی افقی تا 10%
        height_shift_range=0.1,   # جابجایی عمودی تا 10%
        shear_range=0.1,          # برش زاویه‌ای
        zoom_range=0.1,           # زوم تصادفی ±10%
        horizontal_flip=True,     # قرینه‌سازی افقی
        fill_mode='nearest'       # پر کردن پیکسل‌های خالی
    )

def create_train_gen(datagen):
    return datagen.flow_from_dataframe(
        training_df, 
        directory='/kaggle/tmp/',
        x_col='filename', 
        y_col=['any', 'epidural', 'intraparenchymal', 
               'intraventricular', 'subarachnoid', 'subdural'],
        class_mode='raw',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        shuffle=True   # مهم برای train
    )

def create_val_gen(datagen): 
    return datagen.flow_from_dataframe(
        validation_df, 
        directory='/kaggle/tmp/',
        x_col='filename', 
        y_col=['any', 'epidural', 'intraparenchymal', 
               'intraventricular', 'subarachnoid', 'subdural'],
        class_mode='raw',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        shuffle=False  # val نباید shuffle بشه
    )

# Using augmented generator
data_generator = create_datagen()
train_gen = create_train_gen(data_generator)
val_gen = create_val_gen(ImageDataGenerator(preprocessing_function=preprocess_input))  


# ------------------ Imports مورد نیاز ------------------
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ------------------ محاسبه class weights ------------------
label_cols = ['any', 'epidural', 'intraparenchymal',
              'intraventricular', 'subarachnoid', 'subdural']

class_weights = {}
for i, col in enumerate(label_cols):
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=training_df[col].values
    )
    class_weights[i] = weights[1]  # وزن کلاس مثبت

print("Class Weights (per-class positive weight):", class_weights)

# تبدیل به Tensor float32 برای استفاده در loss
weights = np.array(list(class_weights.values()), dtype=np.float32)
weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)  # shape (6,)

# ------------------ تعریف Weighted Binary Crossentropy (element-wise) ------------------
def weighted_binary_crossentropy(y_true, y_pred):
    """
    محاسبه BCE عنصری (shape = (batch, num_classes)) و ضرب در weights
    y_true, y_pred: shape (batch, 6)
    weights_tf: shape (6,)
    خروجی: عدد اسکالر (میانگین weighted loss)
    """
    # از K.backend.binary_crossentropy استفاده می‌کنیم تا خروجی shape=(batch,6) باشه
    bce_elementwise = tf.keras.backend.binary_crossentropy(y_true, y_pred)  # shape (batch,6)

    # اطمینان از float32
    bce_elementwise = tf.cast(bce_elementwise, tf.float32)

    # ضرب در وزن هر کلاس (broadcast: (batch,6) * (6,) => (batch,6))
    weighted_bce = bce_elementwise * weights_tf[tf.newaxis, :]

    # میانگین روی batch و کلاس‌ها
    return tf.reduce_mean(weighted_bce)

# ------------------ Compile مدل ------------------
LR = 0.00005
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss=weighted_binary_crossentropy,
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# ------------------ Callbacks ------------------
checkpoint_path = "best_model.h5"

checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss",
                             save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor="val_loss",
                          patience=5,
                          restore_best_weights=True,
                          verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=5,
                              min_lr=1e-7,
                              verbose=1)

# ------------------ Training ------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[checkpoint, earlystop, reduce_lr]
)


plt.figure(figsize=(14,5))

# Plot loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()


from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import pandas as pd
import numpy as np

label_cols = ['any', 'epidural', 'intraparenchymal',
              'intraventricular', 'subarachnoid', 'subdural']

# پیش‌بینی مدل روی validation generator
y_pred = model.predict(val_gen)

# y_true واقعی از validation_df
y_true = validation_df[label_cols].values

# Threshold 0.5 برای تبدیل به binary
y_pred_binary = (y_pred > 0.5).astype(int)

# ایجاد یک لیست برای ذخیره نتایج
results_per_class = []

for i, col in enumerate(label_cols):
    precision = precision_score(y_true[:, i], y_pred_binary[:, i])
    recall = recall_score(y_true[:, i], y_pred_binary[:, i])
    f1 = f1_score(y_true[:, i], y_pred_binary[:, i])
    try:
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
    except ValueError:
        auc = np.nan  # اگر فقط یک کلاس وجود داشته باشد (مثلاً همه 0) AUC محاسبه نمی‌شود
    results_per_class.append([col, precision, recall, f1, auc])

# نمایش به صورت جدول
df_metrics = pd.DataFrame(results_per_class, columns=['Class', 'Precision', 'Recall', 'F1', 'AUC'])
print(df_metrics)


from keras.models import load_model

best_model = load_model("best_model.h5", custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})

results = best_model.evaluate(val_gen)
print(dict(zip(best_model.metrics_names, results)))


from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

label_cols = ['any', 'epidural', 'intraparenchymal',
              'intraventricular', 'subarachnoid', 'subdural']

# پیش‌بینی مدل روی validation generator
y_pred = model.predict(val_gen)
y_true = validation_df[label_cols].values

best_thresholds = {}

plt.figure(figsize=(15,10))

for i, col in enumerate(label_cols):
    precisions, recalls, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
    
    # محاسبه F1 برای هر threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # پیدا کردن Threshold بهینه (F1-max)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_thresholds[col] = best_thresh
    
    print(f"{col}: Best threshold = {best_thresh:.3f}, Max F1 = {f1_scores[best_idx]:.3f}")
    
    # رسم PR curve
    plt.plot(recalls, precisions, label=f'{col} (best_thresh={best_thresh:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend()
plt.grid(True)
plt.show()

# نمایش جدول thresholds
df_thresholds = pd.DataFrame(list(best_thresholds.items()), columns=['Class', 'Optimal Threshold'])
print(df_thresholds)


# ایجاد y_pred_binary با Thresholdهای بهینه
y_pred_binary_opt = np.zeros_like(y_pred, dtype=int)

for i, col in enumerate(label_cols):
    y_pred_binary_opt[:, i] = (y_pred[:, i] > best_thresholds[col]).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

results_per_class_opt = []

for i, col in enumerate(label_cols):
    precision = precision_score(y_true[:, i], y_pred_binary_opt[:, i])
    recall = recall_score(y_true[:, i], y_pred_binary_opt[:, i])
    f1 = f1_score(y_true[:, i], y_pred_binary_opt[:, i])
    auc = roc_auc_score(y_true[:, i], y_pred[:, i])
    results_per_class_opt.append([col, precision, recall, f1, auc])

df_metrics_opt = pd.DataFrame(results_per_class_opt, 
                              columns=['Class', 'Precision', 'Recall', 'F1', 'AUC'])
print(df_metrics_opt)

