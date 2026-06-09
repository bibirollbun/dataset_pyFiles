import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import  tensorflow as tf
import tf_keras as tfk
import json 
import random

from sklearn.metrics import classification_report , confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow_hub import KerasLayer


from tf_keras.preprocessing import image
from tf_keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tf_keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')


def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_random_seed(42)


!unzip /kaggle/input/dogs-vs-cats/test1.zip -d /kaggle/working/
!unzip /kaggle/input/dogs-vs-cats/train.zip -d /kaggle/working/




PATH = '/kaggle/working/train'
TEST = '/kaggle/working/test1'
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32


PATH


import os

print(os.path.exists(PATH))  # Pastikan ini True
print(os.path.exists(TEST))  # Pastikan ini True


filepaths = []
labels = []

for fname in os.listdir(PATH):
    if fname.startswith("cat"):
        labels.append("cat")
    elif fname.startswith("dog"):
        labels.append("dog")
    else:
        continue
    filepaths.append(os.path.join(PATH, fname))

df = pd.DataFrame({
    'filename': filepaths,
    'class': labels
})

datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col="filename",
    y_col="class",
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col="filename",
    y_col="class",
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)



test_files = os.listdir(TEST)
df_test = pd.DataFrame({'filename': [os.path.join(TEST, fname) for fname in test_files]})

test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col="filename",
    y_col=None,  # Karena tidak ada label
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)



dict_labels = {value : key for key, value in train_generator.class_indices.items()}

for key, value in dict_labels.items():
    print(f'{key} : {value}')


train_labels = train_generator.classes

class_weight = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(train_labels),
    y = train_labels
)

class_weight = dict(enumerate(class_weight))
class_weight


class_labels = list(train_generator.class_indices.keys())

train_counts = train_generator.classes
val_counts = val_generator.classes

train_total = len(train_counts)
val_total = len(val_counts)

train_class_counts = np.bincount(train_counts)
val_class_counts = np.bincount(val_counts)


plt.figure(figsize=(18, 6))
sns.barplot(x=['Train', 'Validation'], y=[train_total, val_total], palette='Blues_d')
for i, txt in enumerate([train_total, val_total]):
    plt.annotate(txt, (i, txt), ha='center', va='bottom')
plt.title('Total Number of Samples in Train, Validation, and Test Sets')
plt.ylabel('Number of Samples')
plt.show()


images, labels = next(train_generator)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

for i, ax in enumerate(axes.flatten()):
    idx = np.random.randint(0, images.shape[0])
    img = images[idx]
    label = labels[idx]
    
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(dict_labels[np.argmax(label)])

plt.tight_layout()
plt.show()


num_classes = len(train_generator.class_indices)
num_classes


def swin(pretrained):
    tfk.backend.clear_session()

    input = tfk.layers.Input(shape=(224,224,3))

    x = KerasLayer(
        pretrained,
        trainable = False
    )(input)
    x = tfk.layers.Flatten()(x)
    x = tfk.layers.Dropout(0.2)(x)

    output = tfk.layers.Dense(num_classes, activation='softmax')(x)
    model = tfk.Model(inputs=input, outputs= output)

    model.compile(
        optimizer=tfk.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        ),
        loss="categorical_crossentropy",
        metrics=[
            tfk.metrics.AUC(name="auc"), 
            "accuracy",
            tfk.metrics.Precision(name="precision"),
            tfk.metrics.Recall(name="recall")
        ]
    )
    
    return model
        


def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH,IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    img_array = preprocess_input(img_array, mode='tf', data_format=None)
    return img, img_array


def predict_image(model):
    base_path = '/kaggle/input/brain-tumor-mri-dataset/Testing'

    with open ('/kaggle/working/labels.json','r') as label_file:
        class_names = json.load(label_file)

    selected_images = []
    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if of.path.isdir(class_path):
            img_files = os.listdir(class_path)
            for img_file in img_files:
                img_path = os.path.join(class_path, img_file)
                selected_images.append((class_name, img_path))

    max_images = min(len(selected_images),15)
    np.random.shuffle(selected_images)
    selected_images = selected_images[:max_images]

    fig, axes = plt.subplots(3,5,figsize=(20,12))
    axes = axes.flatten()

    for idx in range(max_images):
        actual_class, img_path = selected_images[idx]
        img, img_array = load_and_preprocess_images(img_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_name[str(predicted_class)]

        axes[idx].imshow(img)
        axes[idx].axis('off')
        color = 'red' if actual_class != predicred_class_name else ' black'
        axes[idx].set_title(f'Actual : {actual_class}\n Predicted : {predicted_class_name}', color = color)

    for idx in range(max_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, model_name):
    y_pred = np.argmax(model.predic(test_generator), axos = 1).tolist()
    y_true = test_generator.classes.tolist()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xtictlabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()


swin_sp4 = swin("/kaggle/input/swin/tensorflow2/small-patch4-window7-224/1")
swin_sp4.summary()


swin_sp4.fit(
    train_generator, epochs=30,
    validation_data=val_generator,
    callbacks=[tfk.callbacks.CSVLogger('swin_sp4.csv')],
    class_weight = class_weight
)


preds = swin_sp4.predict(test_generator, verbose=1)
preds


pred_labels_idx = np.argmax(preds, axis=1)

# Mapping dari index ke label:
label_map = {v: k for k, v in train_generator.class_indices.items()}

pred_labels = [label_map[idx] for idx in pred_labels_idx]
len(pred_labels)


sample_submission = pd.read_csv('/kaggle/input/dogs-vs-cats/sampleSubmission.csv')
sample_submission.head(),sample_submission.shape



sample_submission['label'] = pred_labels
sample_submission.head()



sample_submission['label'] = sample_submission['label'].replace({'dog': 1, 'cat': 0})



sample_submission


sample_submission.to_csv('submission.csv', index=False)

