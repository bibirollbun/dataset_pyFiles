import os  # التعامل مع نظام التشغيل - OS operations
import cv2  # معالجة الصور - Image processing
import pandas as pd  # تحليل البيانات - Data analysis
import numpy as np  # العمليات الرياضية - Numerical operations
import seaborn as sns  # رسم بياني إحصائي - Statistical plotting
import plotly.express as px  # رسوم تفاعلية - Interactive plots
import matplotlib.pyplot as plt  # رسوم بيانية أساسية - Basic plotting
import tensorflow as tf  # تعلم عميق - Deep learning
import keras  # بناء نماذج التعلم العميق - Deep learning models
from sklearn.utils import shuffle  # خلط البيانات - Shuffle data
from tqdm import tqdm  # شريط تقدم - Progress bar
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models
from tensorflow.keras.layers import ( GlobalAveragePooling2D, Dense, Dropout,  BatchNormalization, GaussianNoise, Input, MultiHeadAttention, Reshape)
from tensorflow.keras.applications.xception import Xception, preprocess_input
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


import warnings  # التحكم في التحذيرات - Warnings control
warnings.filterwarnings("ignore")  # تجاهل التحذيرات - Ignore warnings


SEED = 22
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
IMG_SIZE = (299, 299)
Channal = 3
BATCH_SIZE = 16
epochs = 20

Number_Classes = 8

Dense_Layr_Actication = 'softmax'
Objective_Function = 'sparse_categorical_crossentropy'


Train_Path = '/kaggle/input/convolutional-neural-network-cnn-practice/COVID_IEEE/train'
Data = os.listdir(Train_Path)
Data


img_path , label = [],[]


for folder in Data:
    folder_path = os.path.join(Train_Path , folder)
    for img in os.listdir(folder_path):
         if img.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                img_path.append(os.path.join(folder_path , img))
                label.append(folder)
         else:
             print("NOT FOUND!!")


data = pd.DataFrame({"Image_Path": img_path, "label": label})


data.head()


data.info()


data['label'].value_counts()


data.duplicated().sum()


data.isna().sum()


data.shape


classes = sorted(data["label"].unique())
Number_Classes = len(classes)
print("Classes:", classes, " | num_classes:", Number_Classes)


# ============================
#  إعداد البارامترات
# ============================
CLASSES_TO_SHOW =  Number_Classes      # عدد الكلاسات اللي هتعرضها
IMAGES_PER_CLASS = 3                   # عدد الصور من كل كلاس

# ============================
#  دالة عرض الصور
# ============================
def show_images_from_dataset(Train_Path, classes_to_show, images_per_class):
    class_names = sorted(os.listdir(Train_Path))

    selected_classes = random.sample(class_names, classes_to_show)

    fig, axes = plt.subplots(classes_to_show, images_per_class, figsize=(4*images_per_class, 3*classes_to_show))

    if classes_to_show == 1:   
        axes = [axes]

    for i, cls in enumerate(selected_classes):
        class_dir = os.path.join(Train_Path, cls)
        images = os.listdir(class_dir)
        selected_images = random.sample(images, min(images_per_class, len(images)))

        for j, img_name in enumerate(selected_images):
            img_path = os.path.join(class_dir, img_name)
            img = plt.imread(img_path)

            if images_per_class == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            ax.imshow(img)
            ax.axis("off")


        fig.text(
            0.02,                              # المسافة من اليسار
            1 - (i+0.5)/classes_to_show,       # الموضع الرأسي لكل صف
            cls,                               # اسم الكلاس
            ha="left", va="center",
            fontsize=14, fontweight="bold", color="darkblue"
        )

    plt.tight_layout(rect=[0.1, 0, 1, 1])  
    plt.show()


# ============================
#  تشغيل
# ============================
show_images_from_dataset(Train_Path, CLASSES_TO_SHOW, IMAGES_PER_CLASS)



train_df, val_df = train_test_split(
    data, 
    test_size=0.15, 
    random_state=SEED, 
    stratify=data["label"]
) 


print("Train size:", len(train_df))
print("Val size:", len(val_df))


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# fit على labels كلها (train + val + test)
all_labels = pd.concat([train_df["label"], val_df["label"]])
le.fit(all_labels)

# حول كل set
train_df["label_enc"] = le.transform(train_df["label"])
val_df["label_enc"] = le.transform(val_df["label"])

print(le.classes_)  # هيطبع ["covid", "normal", "virus"]



import albumentations as A
from albumentations.pytorch import ToTensorV2



train_transform = A.Compose([
    A.Resize(299 ,299),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10),
    A.RandomBrightnessContrast(),
    A.Normalize(),
])

val_transform = A.Compose([
    A.Resize(299 ,299),
    A.Normalize(),

])



def aug_generator(df, batch_size, transform):
    while True:
        df = df.sample(frac=1).reset_index(drop=True)
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start:start+batch_size]
            images, labels = [], []
            for i, row in batch_df.iterrows():
                img = cv2.imread(row["Image_Path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                augmented = transform(image=img)
                img = augmented["image"]
                img = img.astype(np.float32)

                images.append(img)
                labels.append(row["label_enc"])

            yield np.array(images), np.array(labels)

train_gen = aug_generator(train_df, batch_size=32, transform=train_transform)
val_gen = aug_generator(val_df, batch_size=32, transform=val_transform)




model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(299,299, Channal)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(512, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    GlobalAveragePooling2D(),
    Dense(256, activation="relu", kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(Number_Classes, activation="softmax")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=Objective_Function,
    metrics=["accuracy"]
)


steps_per_epoch   = len(train_df) // 32
validation_steps  = len(val_df) // 32

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=epochs + 7
)



train_loss, train_acc = model.evaluate(train_gen, steps=200, verbose=1)

val_loss, val_acc = model.evaluate(val_gen, steps=len(val_df)//64, verbose=1)
print(f"Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")








Test_Path = "/kaggle/input/convolutional-neural-network-cnn-practice/COVID_IEEE/test"
test_files = [os.path.join(Test_Path, img) for img in os.listdir(Test_Path)]

test_transform = A.Compose([
    A.Resize(299, 299),
    A.Normalize()
])



def test_generator(file_list, batch_size, transform):
    for start in range(0, len(file_list), batch_size):
        batch_files = file_list[start:start+batch_size]
        images = []
        for file in batch_files:
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(image=img)["image"]
            img = img.astype(np.float32)
            images.append(img)
        images = np.array(images)
        yield (images,)   # ← لاحظ القوس



batch_size = 32
preds = model.predict(
    test_generator(test_files, batch_size, test_transform),
    steps=len(test_files)//batch_size + 1,
    verbose=1
)



pred_classes = np.argmax(preds, axis=1)  
print(pred_classes[:20]) 




