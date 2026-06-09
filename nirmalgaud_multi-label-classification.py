# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import joblib


DATA_DIR = "/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/data/"


df1 = pd.read_csv('/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/train.csv', on_bad_lines='skip')
df2 = pd.read_csv('/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/test.csv', on_bad_lines='skip')


df1.head()


df1.tail()


df1.shape


df1.columns


df1.duplicated().sum()


df1.isnull().sum()


df1.info()


df1['Labels'].unique()


df1['Labels'].value_counts()


unique_labels = df1['Labels'].unique()
num_unique_labels = len(unique_labels)
print(num_unique_labels)


all_labels = df1['Labels'].dropna().astype(str).str.split().sum()
label_counts = pd.Series(all_labels).value_counts().sort_index()

print(label_counts)


def label_one_vs_all(label_string):
    if '1' in label_string.split():
        return 1
    else:
        return 0


df1['BinaryLabel'] = df1['Labels'].astype(str).apply(label_one_vs_all)

print(df1['BinaryLabel'].value_counts())


df_1 = df1[df1['BinaryLabel'] == 1].sample(n=7000, random_state=42)
df_0 = df1[df1['BinaryLabel'] == 0].sample(n=7000, random_state=42)

df_balanced = pd.concat([df_1, df_0]).sample(frac=1, random_state=42).reset_index(drop=True)


df_balanced


df_balanced = df_balanced.drop('Labels', axis =1)


df_balanced


import matplotlib.pyplot as plt
from PIL import Image

samples_1 = df_balanced[df_balanced['BinaryLabel'] == 1].sample(5, random_state=42)
samples_0 = df_balanced[df_balanced['BinaryLabel'] == 0].sample(5, random_state=42)

samples = pd.concat([samples_1, samples_0])

plt.figure(figsize=(15, 6))
for idx, row in enumerate(samples.iterrows()):
    i, data = row
    image_path = os.path.join(DATA_DIR, data['ImageID'])  
    try:
        img = Image.open(image_path)
        plt.subplot(2, 5, idx + 1)
        plt.imshow(img)
        plt.title(f"Label: {data['BinaryLabel']}")
        plt.axis('off')
    except FileNotFoundError:
        print(f"Image not found: {image_path}")

plt.tight_layout()
plt.show()


BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 10
LR_IMAGE = 1e-5
LR_TEXT = 1e-5
LR_CLASSIFIER = 1e-4
NUM_CLASSES = 1  
MAX_TEXT_LENGTH = 128


import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is set for TensorFlow")
    except RuntimeError as e:
        print(e)


class BinaryClassificationDataLoader:
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = tf.keras.Sequential([
            layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
            layers.Rescaling(1./255)
        ])

    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return self.image_transform(image)

    def tokenize_text(self, text):
        text = text.numpy().decode('utf-8')
        inputs = self.tokenizer(
            text,
            max_length=MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        return inputs['input_ids'][0], inputs['attention_mask'][0]

    def create_dataset(self):
        def generator():
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                image_path = os.path.join(DATA_DIR, row['ImageID'])
                image = self.load_image(image_path)
                text = str(row['Caption'])
                label = float(row['BinaryLabel'])
                yield (image, text), label

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=((tf.float32, tf.string), tf.float32)
        )

        def process_text(features, label):
            image, text = features
            input_ids, attention_mask = tf.py_function(
                self.tokenize_text,
                [text],
                (tf.int32, tf.int32)
            )
            input_ids.set_shape([MAX_TEXT_LENGTH])
            attention_mask.set_shape([MAX_TEXT_LENGTH])
            return (image, (input_ids, attention_mask)), label

        dataset = dataset.map(
            process_text,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class BinaryClassifier(Model):
    def __init__(self):
        super().__init__()
        self.image_encoder = ResNet50(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        self.text_encoder = TFBertModel.from_pretrained('bert-base-uncased')
        self.classifier = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1)
        ])

    def call(self, inputs):
        images, (input_ids, attention_mask) = inputs
        img_features = self.image_encoder(images)
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        combined = tf.concat([img_features, text_features], axis=1)
        return self.classifier(combined)

def train_model():
    
    train_df, val_df = train_test_split(df_balanced, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader = BinaryClassificationDataLoader(train_df, tokenizer)
    val_loader = BinaryClassificationDataLoader(val_df, tokenizer)

    train_dataset = train_loader.create_dataset()
    val_dataset = val_loader.create_dataset()

    train_steps_per_epoch = len(train_df) // BATCH_SIZE
    val_steps_per_epoch = len(val_df) // BATCH_SIZE

    print(f"Total training steps per epoch: {train_steps_per_epoch}")
    print(f"Total validation steps per epoch: {val_steps_per_epoch}")

    model = BinaryClassifier()

    image_optimizer = tf.keras.optimizers.Adam(LR_IMAGE)
    text_optimizer = tf.keras.optimizers.Adam(LR_TEXT)
    classifier_optimizer = tf.keras.optimizers.Adam(LR_CLASSIFIER)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(inputs, training=True)
            loss = loss_fn(labels, outputs)

        image_grads = tape.gradient(loss, model.image_encoder.trainable_variables)
        text_grads = tape.gradient(loss, model.text_encoder.trainable_variables)
        classifier_grads = tape.gradient(loss, model.classifier.trainable_variables)

        image_optimizer.apply_gradients(zip(image_grads, model.image_encoder.trainable_variables))
        text_optimizer.apply_gradients(zip(text_grads, model.text_encoder.trainable_variables))
        classifier_optimizer.apply_gradients(zip(classifier_grads, model.classifier.trainable_variables))

        train_acc_metric.update_state(labels, tf.sigmoid(outputs))
        return loss

    @tf.function
    def val_step(inputs, labels):
        outputs = model(inputs, training=False)
        loss = loss_fn(labels, outputs)
        val_acc_metric.update_state(labels, tf.sigmoid(outputs))
        return loss

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    num_epochs = 3
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        total_loss = 0
        train_acc_metric.reset_state()
        for step, (inputs, labels) in enumerate(train_dataset):
            loss = train_step(inputs, labels)
            total_loss += loss

            if step % 50 == 0:
                print(f"Step {step}: Loss={loss.numpy():.4f}")
            if step >= train_steps_per_epoch - 1:  
                break

        val_loss = 0
        val_acc_metric.reset_state()
        for step, (inputs, labels) in enumerate(val_dataset):
            loss = val_step(inputs, labels)
            val_loss += loss
            if step >= val_steps_per_epoch - 1:  
                break

        train_loss = total_loss / train_steps_per_epoch
        val_loss = val_loss / val_steps_per_epoch
        train_acc = train_acc_metric.result().numpy()
        val_acc = val_acc_metric.result().numpy()

        train_loss_history.append(train_loss.numpy())
        val_loss_history.append(val_loss.numpy())
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Val Accuracy')
    plt.title('Training History - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return model
    
def evaluate_model(model, dataset):
    y_true, y_pred = [], []
    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.sigmoid(outputs).numpy())

    y_pred_bin = (np.array(y_pred) > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_bin))

    cm = confusion_matrix(y_true, y_pred_bin)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


if __name__ == '__main__':
    
    model = train_model()

    _, val_df = train_test_split(df_balanced, test_size=0.2, random_state=42) 
    val_loader = BinaryClassificationDataLoader(val_df, BertTokenizer.from_pretrained('bert-base-uncased'))
    val_dataset = val_loader.create_dataset()

