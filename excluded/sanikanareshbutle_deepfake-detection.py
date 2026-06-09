import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


dataset_path = '/kaggle/input/deepfake-faces/metadata.csv'
df = pd.read_csv(dataset_path)


# Display basic dataset details
display(df.head())
display(df.tail())
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print("Duplicated Rows:", df.duplicated().sum())
print("Missing Values:")
print(df.isnull().sum())



df.info()
print("Unique Values Per Column:")
print(df.nunique())



# Classifying Features
def classify_features(df):
    categorical_features = []
    non_categorical_features = []
    discrete_features = []
    continuous_features = []
    
    for column in df.columns:
        if df[column].dtype == 'object':
            if df[column].nunique() < 10:
                categorical_features.append(column)
            else:
                non_categorical_features.append(column)
        elif df[column].dtype in ['int64', 'float64']:
            if df[column].nunique() < 10:
                discrete_features.append(column)
            else:
                continuous_features.append(column)
    
    return categorical_features, non_categorical_features, discrete_features, continuous_features

categorical, non_categorical, discrete, continuous = classify_features(df)
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)



df.fillna("Not Available", inplace=True)



for col in categorical:
    print(f"{col}: {df[col].unique()}\n")



for col in categorical:
    print(df[col].value_counts())
    print()


# Countplots for categorical features
for col in categorical:
    plt.figure(figsize=(15,6))
    sns.countplot(x=df[col], palette='hls')
    plt.title(f"Distribution of {col}")
    plt.show()



# Pie charts for categorical features
for col in categorical:
    plt.figure(figsize=(10,7))
    plt.pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.1f%%', textprops={'fontsize': 12})
    plt.title(f"Proportion of {col}")
    plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

for col in discrete + continuous:
    plt.figure(figsize=(12,5))
    sns.distplot(df[col], hist=True, kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=90)
    plt.show()



# Boxplots for numerical features
for col in discrete + continuous:
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df[col], palette='hls')
    plt.title(f"Boxplot of {col}")
    plt.xticks(rotation=90)
    plt.show()


# Violin plots for numerical features
for col in discrete + continuous:
    plt.figure(figsize=(12,5))
    sns.violinplot(x=df[col], palette='hls')
    plt.title(f"Violin Plot of {col}")
    plt.xticks(rotation=90)
    plt.show()



from sklearn.model_selection import train_test_split

real_df = df[df["label"] == "REAL"]
fake_df = df[df["label"] == "FAKE"]

sample_size = min(len(real_df), len(fake_df))
real_df = real_df.sample(sample_size, random_state=42)
fake_df = fake_df.sample(sample_size, random_state=42)

sample_meta = pd.concat([real_df, fake_df])

Train_set, Test_set = train_test_split(sample_meta, test_size=0.2, random_state=42, stratify=sample_meta['label'])
Train_set, Val_set  = train_test_split(Train_set, test_size=0.3, random_state=42, stratify=Train_set['label'])
print("Train, Validation, and Test Set Sizes:", Train_set.shape, Val_set.shape, Test_set.shape)


import cv2
import os
image_path = '/kaggle/input/deepfake-faces/faces_224/'
image_files = sorted(os.listdir(image_path))
selected_images = image_files[:9]




plt.figure(figsize=(10, 10))
for index, image_file in enumerate(selected_images):
    image = cv2.imread(os.path.join(image_path, image_file))
    plt.subplot(3, 3, index + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Image {index + 1}')
    plt.axis('off')
plt.show()




# Display image resolutions
for i, image_file in enumerate(image_files[:10]):
    image = cv2.imread(os.path.join(image_path, image_file))
    if image is not None:
        height, width, _ = image.shape
        print(f"Resolution of image {i+1}: {width} x {height}")
    else:
        print(f"Error reading image {i+1}")



# Visualizing a batch of real vs fake images
plt.figure(figsize=(15,15))
for cur, i in enumerate(Train_set.index[25:50]):
    plt.subplot(5, 5, cur+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cv2.imread(image_path + Train_set.loc[i,'videoname'][:-4] + '.jpg'))
    plt.xlabel('FAKE Image' if Train_set.loc[i,'label']=='FAKE' else 'REAL Image')
plt.show()



def retreive_dataset(set_name):
    images,labels=[],[]
    for (img, imclass) in zip(set_name['videoname'], set_name['label']):
        images.append(cv2.imread('../input/deepfake-faces/faces_224/'+img[:-4]+'.jpg'))
        if(imclass=='FAKE'):
            labels.append(1)
        else:
            labels.append(0)
    
    return np.array(images),np.array(labels)


X_train,y_train=retreive_dataset(Train_set)
X_val,y_val=retreive_dataset(Val_set)
X_test,y_test=retreive_dataset(Test_set)


import tensorflow as tf
from tensorflow.keras import layers, models
from functools import partial


tf.random.set_seed(42)


DefaultConv2D = partial(layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")

# Model Definition
model = models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[224, 224, 3]),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(units=128, activation="relu",
                 kernel_initializer="he_normal"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(units=64, activation="relu",
                 kernel_initializer="he_normal"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(units=1, activation="sigmoid")
])


initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss="binary_crossentropy", metrics=["accuracy"])


model.summary()


history = model.fit(
    X_train, y_train,
    epochs=10,  # Adjust as needed
    batch_size=32,  # Adjust as needed
    validation_data=(X_val, y_val),
    verbose=1
)


y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
y_test_pred_binary = (y_pred > 0.5).astype(int)
y_train_pred = model.predict(X_train)
y_train_pred_binary = (y_train_pred > 0.5).astype(int)


train_accuracy = accuracy_score(y_train, y_train_pred_binary)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
test_accuracy = accuracy_score(y_test, y_test_pred_binary)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
f1 = f1_score(y_test, y_test_pred_binary)
print(f"F1 Score: {f1:.4f}")

precision = precision_score(y_test, y_test_pred_binary)
print(f"Precison: {precision:.4f}")

recall = recall_score(y_test, y_test_pred_binary)
print(f"Recall: {recall:.4f}")


conf_matrix = confusion_matrix(y_test, y_test_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)


import scikitplot as skplt


skplt.metrics.plot_confusion_matrix(y_test, y_test_pred_binary, normalize=True)
plt.show()


class_report = classification_report(y_test, y_test_pred_binary)
print("Classification Report:")
print(class_report)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


from tensorflow.keras.applications import ResNet50
input_shape = (224, 224, 3)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
for layer in base_model.layers:
    layer.trainable = False
model_resnet50 = models.Sequential()
model_resnet50.add(base_model)
model_resnet50.add(layers.GlobalAveragePooling2D())
model_resnet50.add(layers.Dense(1, activation='sigmoid'))


model_resnet50.summary()


from tensorflow.keras import optimizers


model_resnet50.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model_resnet50.fit(
    X_train, y_train,
    epochs=10,  
    validation_data=(X_val, y_val),
    verbose=1
)
y_pred = model_resnet50.predict(X_test)
y_test_pred_binary = (y_pred > 0.5).astype(int)


y_train_pred = model_resnet50.predict(X_train)
y_train_pred_binary = (y_train_pred > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, y_train_pred_binary)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
test_accuracy = accuracy_score(y_test, y_test_pred_binary)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
f1 = f1_score(y_test, y_test_pred_binary)
print(f"F1 Score: {f1:.4f}")
precision = precision_score(y_test, y_test_pred_binary)
print(f"Precison: {precision:.4f}")
recall = recall_score(y_test, y_test_pred_binary)
print(f"Recall: {recall:.4f}")


conf_matrix = confusion_matrix(y_test, y_test_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)
skplt.metrics.plot_confusion_matrix(y_test, y_test_pred_binary, normalize=True)
plt.show()
class_report = classification_report(y_test, y_test_pred_binary)
print("Classification Report:")
print(class_report)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

