import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from keras.optimizers import Adam

# ignoring warnings
import warnings
warnings.simplefilter("ignore")

import os, cv2, json
from PIL import Image


WORK_DIR = '../input/cassava-leaf-disease-classification'
os.listdir(WORK_DIR)


print('Train images: %d' %len(os.listdir(os.path.join(WORK_DIR, "train_images"))))


with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
    print(json.dumps(json.loads(file.read()), indent=4))


train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_labels.head()


sns.countplot(train_labels.label, edgecolor = 'black',
              palette = sns.color_palette("viridis", 5))
plt.show()


sample = train_labels[train_labels.label == '0'].sample(6)
plt.figure(figsize=(16, 8))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    image = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    
plt.show()


sample = train_labels[train_labels.label == '1'].sample(6)
plt.figure(figsize=(16, 8))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    image = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    
plt.show()


sample = train_labels[train_labels.label == '2'].sample(6)
plt.figure(figsize=(16, 8))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    image = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    
plt.show()


sample = train_labels[train_labels.label == '3'].sample(6)
plt.figure(figsize=(16, 8))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    image = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    
plt.show()


sample = train_labels[train_labels.label == '4'].sample(6)
plt.figure(figsize=(16, 8))
for ind, (image_id, label) in enumerate(zip(sample.image_id, sample.label)):
    plt.subplot(2, 3, ind + 1)
    image = cv2.imread(os.path.join(WORK_DIR, "train_images", image_id))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    
plt.show()


y_pred = [3] * len(train_labels.label)
print('The baseline accuracy: %.3f' 
      %accuracy_score(y_pred, train_labels.label))


# The TRAIN/VALID split is performing in the generator directly.

#train, valid = train_test_split(train_labels, train_size = 0.8, shuffle = True,
#                                random_state = 0)


#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
#sns.set_style("white")
#plt.suptitle('Train vs Valid labels', size = 15)
#
#sns.countplot(train.label, edgecolor = 'black', ax = ax1,
#              palette = sns.color_palette("viridis", 5))
#sns.countplot(valid.label, edgecolor = 'black', ax = ax2,
#              palette = sns.color_palette("viridis", 5))
#plt.show()


BATCH_SIZE = 16
EPOCHS = 20
TARGET_SIZE = 224

STEPS_PER_EPOCH = len(train_df) // BATCH_SIZE
VALIDATION_STEPS = len(val_df) // BATCH_SIZE



from sklearn.model_selection import train_test_split

train_labels.label = train_labels.label.astype(str)

train_df, temp_df = train_test_split(train_labels, train_size=0.7, stratify=train_labels.label, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=(1/3), stratify=temp_df.label, random_state=42)


train_datagen = ImageDataGenerator(
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=os.path.join(WORK_DIR, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse"
)

val_datagen = ImageDataGenerator()

validation_generator = val_datagen.flow_from_dataframe(
    val_df,
    directory=os.path.join(WORK_DIR, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse"
)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=os.path.join(WORK_DIR, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)



def create_model():
    model = models.Sequential()

    model.add(EfficientNetB0(include_top = False, weights = 'imagenet',
                             input_shape = (TARGET_SIZE, TARGET_SIZE, 3)))
    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(5, activation = "softmax"))

    model.compile(optimizer = Adam(lr = 0.001),
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["acc"])
    return model


model = create_model()
model.summary()


model_save = ModelCheckpoint('./best_baseline_model.h5', 
                             save_best_only = True, 
                             save_weights_only = True,
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)
early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                              patience = 2, min_delta = 0.001, 
                              mode = 'min', verbose = 1)


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[model_save, early_stop, reduce_lr]
)


test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.set_style("white")
plt.suptitle('Train history', size = 15)

ax1.plot(epochs, acc, "bo", label = "Training acc")
ax1.plot(epochs, val_acc, "b", label = "Validation acc")
ax1.set_title("Training and validation acc")
ax1.legend()

ax2.plot(epochs, loss, "bo", label = "Training loss", color = 'red')
ax2.plot(epochs, val_loss, "b", label = "Validation loss", color = 'red')
ax2.set_title("Training and validation loss")
ax2.legend()

plt.show()


model.save('./baseline_model.h5')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import os
from math import pi

disease_map = {
    0: 'CBB',
    1: 'CBSD',
    2: 'CGM',
    3: 'CMD',
    4: 'Healthy'
}

WORK_DIR = '../input/cassava-leaf-disease-classification'
TARGET_SIZE = 224


print(f"\n Data Info:")
print(f"   â€¢ Total test images: {len(y_true)}")
print(f"   â€¢ Total predictions: {len(y_pred)}")

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"\n Test Set Breakdown (by True Label):")
for i in range(5):
    count = (y_true == i).sum()
    percentage = (count / len(y_true)) * 100
    print(f"   â€¢ {disease_map[i]:10s}: {count:4d} images ({percentage:5.2f}%)")

print(f"\n Predictions Breakdown (by Predicted Label):")
for i in range(5):
    count = (y_pred == i).sum()
    percentage = (count / len(y_pred)) * 100
    print(f"   â€¢ {disease_map[i]:10s}: {count:4d} images ({percentage:5.2f}%)")


# Confusion Matrix Heatmaps
print("\n Generating Confusion Matrices...")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Standard Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=disease_map.values(),
            yticklabels=disease_map.values(),
            ax=axes[0], cbar_kws={'label': 'Count'}, 
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=13)
axes[0].set_xlabel('Predicted Label', fontsize=13)

# Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
            xticklabels=disease_map.values(),
            yticklabels=disease_map.values(),
            ax=axes[1], cbar_kws={'label': 'Percentage'}, 
            annot_kws={'size': 12})
axes[1].set_title('Normalized Confusion Matrix (%)', fontsize=16, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=13)
axes[1].set_xlabel('Predicted Label', fontsize=13)

plt.tight_layout()
plt.show()


# Per-Class Metrics
print("\n Computing Per-Class Metrics...")

metrics_data = []
for i in range(5):
    class_mask = np.array(y_true) == i
    if class_mask.sum() > 0:
        # Calculate TP, FP, FN
        tp = np.sum((np.array(y_true) == i) & (np.array(y_pred) == i))
        fp = np.sum((np.array(y_true) != i) & (np.array(y_pred) == i))
        fn = np.sum((np.array(y_true) == i) & (np.array(y_pred) != i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = np.sum((np.array(y_true)[class_mask] == 
                          np.array(y_pred)[class_mask])) / class_mask.sum()
        
        metrics_data.append({
            'Disease': disease_map[i],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Accuracy': accuracy
        })

metrics_df = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(12, 6))
metrics_df.set_index('Disease')[['Precision', 'Recall', 'F1-Score', 'Accuracy']].plot(
    kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=1.2)
ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Disease Class', fontsize=12)
ax.set_ylim([0, 1.1])
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Distribution Comparison
print("\n Comparing True vs Predicted Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# True labels distribution
class_counts_true = pd.Series(y_true).value_counts().sort_index()
axes[0].bar([disease_map[i] for i in class_counts_true.index], 
            class_counts_true.values,
            color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
axes[0].set_title('Test Set Distribution (True Labels)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Images', fontsize=12)
axes[0].set_xlabel('Disease Class', fontsize=12)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

for i, v in enumerate(class_counts_true.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=11)

# Predictions distribution
class_counts_pred = pd.Series(y_pred).value_counts().sort_index()
axes[1].bar([disease_map[i] for i in class_counts_pred.index], 
            class_counts_pred.values,
            color='coral', edgecolor='black', linewidth=1.5, alpha=0.8)
axes[1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Images', fontsize=12)
axes[1].set_xlabel('Disease Class', fontsize=12)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

for i, v in enumerate(class_counts_pred.values):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()


# Per-Class Accuracy Radar Chart
# ========================================
print("\n Creating Radar Chart for Per-Class Accuracy...")

categories = [disease_map[i] for i in range(5)]
values = []

for i in range(5):
    class_mask = np.array(y_true) == i
    if class_mask.sum() > 0:
        acc = np.sum((np.array(y_true)[class_mask] == 
                     np.array(y_pred)[class_mask])) / class_mask.sum()
        values.append(acc)
    else:
        values.append(0)

values += values[:1]
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
ax.plot(angles, values, 'o-', linewidth=2.5, color='#2E86AB', markersize=8)
ax.fill(angles, values, alpha=0.25, color='#2E86AB')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title('Per-Class Accuracy (Radar Chart)', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()


# Sample Predictions Gallery
print("\n Generating Sample Predictions Gallery...")

sample_indices = np.random.choice(len(test_df), min(12, len(test_df)), replace=False)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, sample_idx in enumerate(sample_indices):
    img_name = test_df.iloc[sample_idx]['image_id']
    true_label = int(test_df.iloc[sample_idx]['label'])
    pred_label = y_pred[sample_idx]
    
    try:
        img_path = os.path.join(WORK_DIR, 'train_images', img_name)
        img = plt.imread(img_path)
        
        axes[idx].imshow(img)
        
        # Color code: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        title = f"True: {disease_map[true_label]}\nPred: {disease_map[pred_label]}"
        axes[idx].set_title(title, fontsize=11, fontweight='bold', color=color)
        axes[idx].axis('off')
    except:
        axes[idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')
        axes[idx].axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()


# Summary Statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

overall_accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\n Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

print(f"\n Test Set Size: {len(y_true)} images")
print(f"   â€¢ CBB:     {(y_true==0).sum()} images")
print(f"   â€¢ CBSD:    {(y_true==1).sum()} images")
print(f"   â€¢ CGM:     {(y_true==2).sum()} images")
print(f"   â€¢ CMD:     {(y_true==3).sum()} images")
print(f"   â€¢ Healthy: {(y_true==4).sum()} images")

print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_true, y_pred, 
                          target_names=list(disease_map.values()),
                          digits=4))

print("\n" + "="*70)
print(" Analysis Complete")
print("="*70)

