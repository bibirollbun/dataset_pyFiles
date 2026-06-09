


import os

# === æ•°æ�®é›†é…�ç½® (ç–Ÿç–¾ç»†èƒ�æ£€æµ‹æ•°æ�®é›†) ===
dataset_dir = '/kaggle/input/malaria-dataset'
train_dir = os.path.join(dataset_dir, 'Dataset', 'Train')
test_dir = os.path.join(dataset_dir, 'Dataset', 'Test')
num_classes = 2

print("=== æ­£åœ¨ä½¿ç”¨ Malaria Cell Images æ•°æ�®é›† ===")
print(f"è®­ç»ƒæ•°æ�®è·¯å¾„: {train_dir}")
print(f"æµ‹è¯•æ•°æ�®è·¯å¾„: {test_dir}")
print(f"ç±»åˆ«æ•°é‡�: {num_classes}")

# æ£€æŸ¥ç›®å½•ç»“æ�„
if os.path.exists(train_dir):
    subdirs = os.listdir(train_dir)
    print(f"è®­ç»ƒé›†ç±»åˆ«: {subdirs}")

if os.path.exists(test_dir):
    subdirs = os.listdir(test_dir)
    print(f"æµ‹è¯•é›†ç±»åˆ«: {subdirs}")

# Basic imports
from os import makedirs, listdir
from shutil import copyfile
from random import seed, random
import numpy as np
import pandas as pd

# visuals
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, MaxPooling2D, Dropout, Flatten, BatchNormalization, Conv2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

print("TensorFlowç‰ˆæœ¬:", tf.__version__)

# === æ•°æ�®ç”Ÿæˆ�å™¨é…�ç½® ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# äºŒåˆ†ç±»ä»»åŠ¡
class_mode = 'binary'

print("\n=== åˆ›å»ºæ•°æ�®ç”Ÿæˆ�å™¨ ===")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode=class_mode,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode=class_mode,
    subset='validation'
)

print("æ•°æ�®ç”Ÿæˆ�å™¨ç±»åˆ«ç´¢å¼•:", train_generator.class_indices)
print("è®­ç»ƒæ ·æœ¬æ•°:", train_generator.samples)
print("éªŒè¯�æ ·æœ¬æ•°:", validation_generator.samples)
print("æ•°æ�®é›†å‡†å¤‡å®Œæˆ�ï¼�")


# === ç–Ÿç–¾æ•°æ�®é›†æ–‡ä»¶è·¯å¾„å¤„ç�† ===
image_dir = "/kaggle/input/malaria-dataset/Dataset/Train"
filenames = []
labels = []

# é��å�†ä¸¤ä¸ªç±»åˆ«çš„æ–‡ä»¶å¤¹
for class_name in ['Parasite', 'Uninfected']:
    class_dir = os.path.join(image_dir, class_name)
    print(f"æ­£åœ¨å¤„ç�†ç±»åˆ«: {class_name}, è·¯å¾„: {class_dir}")
    
    if os.path.exists(class_dir):
        class_files = os.listdir(class_dir)
        print(f"æ‰¾åˆ° {len(class_files)} ä¸ªæ–‡ä»¶")
        
        # ä¿�å­˜å®Œæ•´è·¯å¾„
        full_paths = [os.path.join(class_dir, f) for f in class_files]
        filenames.extend(full_paths)
        labels.extend([class_name] * len(class_files))
    else:
        print(f"è­¦å‘Š: è·¯å¾„ä¸�å­˜åœ¨ {class_dir}")

data = pd.DataFrame({"filename": filenames, "label": labels})

# æ£€æŸ¥æ•°æ�®åˆ†å¸ƒ
print(f"æ€»æ ·æœ¬æ•°: {len(data)}")
print("ç±»åˆ«åˆ†å¸ƒ:")
print(data['label'].value_counts())

data.head()


plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=0.4)

# æ˜¾ç¤ºç–Ÿç–¾æ•°æ�®é›†çš„æ ·æœ¬å›¾ç‰‡
class_names = ['Parasite', 'Uninfected']
for i in range(10):
    plt.subplot(1,10,i+1)
    # ä»�ä¸¤ä¸ªç±»åˆ«ä¸­è½®æµ�æ˜¾ç¤ºå›¾ç‰‡
    class_name = class_names[i % 2]
    class_dir = os.path.join(train_dir, class_name)
    images = os.listdir(class_dir)
    if images:
        # ä½¿ç”¨å›ºå®šç´¢å¼•ç¡®ä¿�ä¸�è¶Šç•Œ
        img_index = min(i // 2, len(images) - 1)  # æ¯�ç±»æ˜¾ç¤º5å¼ å›¾ç‰‡
        filename = os.path.join(class_dir, images[img_index])
        image = imread(filename)
        plt.imshow(image)
        plt.title(class_name, fontsize=12)
        plt.axis('off')

plt.show()


# train test split using dataframe

labels = data['label']

X_train, X_temp = train_test_split(data, test_size=0.2, stratify=labels, random_state = 42)

label_test_val = X_temp['label']

X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state = 42)

print('The shape of train data',X_train.shape)
print('The shape of test data',X_test.shape)
print('The shape of validation data',X_val.shape)

print("\n=== æ•°æ�®åˆ†å¸ƒæ£€æŸ¥ ===")
print("è®­ç»ƒé›†ç±»åˆ«åˆ†å¸ƒ:")
print(X_train['label'].value_counts())
print("\néªŒè¯�é›†ç±»åˆ«åˆ†å¸ƒ:")
print(X_val['label'].value_counts())
print("\næµ‹è¯•é›†ç±»åˆ«åˆ†å¸ƒ:")
print(X_test['label'].value_counts())


# è�·å�–æ•°æ�®ä¸­å®�é™…çš„æ ‡ç­¾é¡ºåº�
labels = sorted(data['label'].unique())  # è‡ªåŠ¨è�·å�–æ�’åº�å��çš„æ ‡ç­¾ ['Parasite', 'Uninfected']

# ä½¿ç”¨ç»Ÿä¸€çš„æ ‡ç­¾é¡ºåº�æ�¥ç¡®ä¿�ä¸€è‡´æ€§
label1, count1 = np.unique(X_train.label, return_counts=True)
label2, count2 = np.unique(X_val.label, return_counts=True)
label3, count3 = np.unique(X_test.label, return_counts=True)

# ç¡®ä¿�ä½¿ç”¨å‰�é�¢å®šä¹‰çš„labelsé¡ºåº�
uni1 = pd.DataFrame(data=count1, index=label1, columns=['Count1'])
uni2 = pd.DataFrame(data=count2, index=label2, columns=['Count2'])
uni3 = pd.DataFrame(data=count3, index=label3, columns=['Count3'])

plt.figure(figsize=(20,6), dpi=200)
sns.set_style('darkgrid')

plt.subplot(131)
sns.barplot(data=uni1.reset_index(), x='index', y='Count1', palette='icefire', width=0.2).set_title('Class distribution in Training set', fontsize=15)
plt.xlabel('Labels', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.subplot(132)
sns.barplot(data=uni2.reset_index(), x='index', y='Count2', palette='icefire', width=0.2).set_title('Class distribution in validation set', fontsize=15)
plt.xlabel('Labels', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.subplot(133)
sns.barplot(data=uni3.reset_index(), x='index', y='Count3', palette='icefire', width=0.2).set_title('Class distribution in Testing set', fontsize=15)
plt.xlabel('Labels', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()


# ä¸ºç–Ÿç–¾æ•°æ�®é›†åˆ›å»ºç›®å½•ç»“æ�„ - ä¿®æ­£ç‰ˆæœ¬
dataset_home = 'dataset_malaria/'
subdirs = ['train/', 'test/']

for subdir in subdirs:
    labeldirs = ['Parasite/', 'Uninfected/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)

seed(1)
val_ratio = 0.2

# åˆ†åˆ«å¤„ç�†æ¯�ä¸ªç±»åˆ«ï¼Œç¡®ä¿�ç›¸å�Œçš„æµ‹è¯•æ¯”ä¾‹
for class_name in ['Parasite', 'Uninfected']:
    src_class_dir = os.path.join('/kaggle/input/malaria-dataset/Dataset/Train', class_name)
    files = listdir(src_class_dir)
    
    # å¯¹æ¯�ä¸ªç±»åˆ«çš„æ–‡ä»¶å�•ç‹¬è¿›è¡Œéš�æœºåˆ†å‰²
    class_files = listdir(src_class_dir)
    for file in class_files:
        src = os.path.join(src_class_dir, file)
        dst_dir = 'train/'
        if random() < val_ratio:
            dst_dir = 'test/'
        dst = dataset_home + dst_dir + class_name + '/' + file
        copyfile(src, dst)

# æ‰“å�°ä¿®æ­£å��çš„ç»Ÿè®¡ä¿¡æ�¯
print("ç–Ÿç–¾æ•°æ�®é›†ç»Ÿè®¡ (ä¿®æ­£å��):")
total_parasite = 0
total_uninfected = 0
for class_name in ['Parasite', 'Uninfected']:
    train_path = f"dataset_malaria/train/{class_name}"
    test_path = f"dataset_malaria/test/{class_name}"
    train_count = len(os.listdir(train_path))
    test_count = len(os.listdir(test_path))
    total = train_count + test_count
    
    if class_name == 'Parasite':
        total_parasite = total
    else:
        total_uninfected = total
        
    print(f"{class_name}: è®­ç»ƒé›† {train_count} å¼ , æµ‹è¯•é›† {test_count} å¼ , æ€»è®¡ {total} å¼ ")
    print(f"  æµ‹è¯•é›†æ¯”ä¾‹: {test_count/total:.1%}")

print(f"\nå�Ÿå§‹æ•°æ�®: Parasite {total_parasite}å¼ , Uninfected {total_uninfected}å¼ ")


# parameters
image_size = 128
image_channel = 3
bat_size = 32


# Creating image data generator - åŠ å¼ºæ•°æ�®å¢�å¼º
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=15,
                                    horizontal_flip=True,
                                    zoom_range=0.1,      # å›�åˆ°0.1
                                    shear_range=0.1,     # å›�åˆ°0.1
                                    fill_mode='nearest',
                                    width_shift_range=0.1,  # å›�åˆ°0.1
                                    height_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)


# ä¸ºç–Ÿç–¾æ•°æ�®é›†è®¾ç½®æ•°æ�®ç›®å½•
data_directory = '/kaggle/input/malaria-dataset/Dataset/Train/'

# ç¡®ä¿�æ–‡ä»¶å��æ˜¯å®Œæ•´è·¯å¾„
X_train['filename'] = X_train['filename'].apply(lambda x: os.path.join(data_directory, x) if not os.path.isabs(x) else x)
X_val['filename'] = X_val['filename'].apply(lambda x: os.path.join(data_directory, x) if not os.path.isabs(x) else x)
X_test['filename'] = X_test['filename'].apply(lambda x: os.path.join(data_directory, x) if not os.path.isabs(x) else x)

print("ä¿®å¤�å��çš„è·¯å¾„æ£€æŸ¥:")
for i in range(3):
    file_path = X_train.iloc[i]['filename']
    print(f"æ–‡ä»¶ {i}: {file_path}")
    print(f"æ–‡ä»¶å­˜åœ¨: {os.path.exists(file_path)}")
    print("---")

# ä½¿ç”¨Noneä½œä¸ºdirectoryï¼Œå› ä¸ºæ–‡ä»¶å��å·²ç»�æ˜¯å®Œæ•´è·¯å¾„
train_generator = train_datagen.flow_from_dataframe(X_train,
                                                    directory=None,  # ä½¿ç”¨Noneå› ä¸ºæ–‡ä»¶å��å·²ç»�æ˜¯å®Œæ•´è·¯å¾„
                                                    x_col='filename',
                                                    y_col='label',
                                                    batch_size=bat_size,
                                                    target_size=(image_size,image_size),
                                                    class_mode='binary'  # äºŒåˆ†ç±»ä»»åŠ¡
                                                   )
val_generator = test_datagen.flow_from_dataframe(X_val, 
                                                 directory=None,
                                                 x_col='filename',
                                                 y_col='label',
                                                 batch_size=bat_size,
                                                 target_size=(image_size,image_size),
                                                 shuffle=False,
                                                 class_mode='binary'  # äºŒåˆ†ç±»ä»»åŠ¡
                                                )

test_generator = test_datagen.flow_from_dataframe(X_test, 
                                                  directory=None,
                                                  x_col='filename',
                                                  y_col='label',
                                                  batch_size=bat_size,
                                                  target_size=(image_size,image_size),
                                                  shuffle=False,
                                                  class_mode='binary'  # äºŒåˆ†ç±»ä»»åŠ¡
                                                 )


model = Sequential()

# Input Layer - æ�¢å¤�åˆ°é€‚åº¦æ­£åˆ™åŒ–
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(image_size,image_size,image_channel), kernel_regularizer=tf.keras.regularizers.l2(0.002)))  # å›�åˆ°0.002
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))  # å›�åˆ°0.25

# Block 1
model.add(Conv2D(64,(3,3),activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(128,(3,3),activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 3 - ä¿�æŒ�256ä¸ªè¿‡æ»¤å™¨
model.add(Conv2D(256,(3,3),activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Fully Connected layers - ä¿�æŒ�512ç¥�ç»�å…ƒ
model.add(Flatten())
model.add(Dense(512,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # ä¿�æŒ�0.5

# Output layer - ä¿®æ”¹ä¸ºäºŒåˆ†ç±»
model.add(Dense(1, activation='sigmoid'))  # äºŒåˆ†ç±»ä½¿ç”¨sigmoidæ¿€æ´»å‡½æ•°


learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                            patience=3,  # ä»�2å¢�åŠ åˆ°3
                                            factor=0.5,
                                            min_lr = 0.00001,
                                            verbose = 1)

early_stoping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True,verbose=0)  # ä»�3å¢�åŠ åˆ°5


# æ ¹æ�®æ•°æ�®é›†åŠ¨æ€�è®¾ç½®æ�Ÿå¤±å‡½æ•°
if num_classes == 2:
    loss_function = 'binary_crossentropy'
else:
    loss_function = 'categorical_crossentropy'

# ä½¿ç”¨è‡ªå®šä¹‰Adamä¼˜åŒ–å™¨ï¼Œè®¾ç½®æ›´ä½�çš„å­¦ä¹ ç�‡
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])  # å›�åˆ°é»˜è®¤adam

print("æ£€æŸ¥æ¨¡å�‹é…�ç½®:")
print(f"å­¦ä¹ ç�‡: {model.optimizer.learning_rate.numpy()}")
print(f"æ�Ÿå¤±å‡½æ•°: {model.loss}")


print("æ£€æŸ¥æ•°æ�®æ ¼å¼�:")
for images, labels in train_generator:
    print(f"å›¾åƒ�å½¢çŠ¶: {images.shape}")
    print(f"æ ‡ç­¾å½¢çŠ¶: {labels.shape}")
    print(f"æ ‡ç­¾æ ¼å¼�ç¤ºä¾‹: {labels[0]}")
    break

# é‡�ç½®ç”Ÿæˆ�å™¨
train_generator.reset()
val_generator.reset()

# è®©Kerasè‡ªåŠ¨è®¡ç®—steps_per_epochï¼ˆä¸�æŒ‡å®šå�‚æ•°ï¼‰
history = model.fit(train_generator,
                    validation_data=val_generator, 
                    callbacks=[early_stoping, learning_rate_reduction],
                    epochs=30
                    # ä¸�æŒ‡å®šsteps_per_epochå’Œvalidation_stepsï¼Œè®©Kerasè‡ªåŠ¨å¤„ç�†
                   )


# plots for accuracy and Loss with epochs

error = pd.DataFrame(history.history)

plt.figure(figsize=(18,5),dpi=200)
sns.set_style('darkgrid')

plt.subplot(121)
plt.title('Cross Entropy Loss',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.plot(error['loss'], label='Train')
plt.plot(error['val_loss'], label='Validation')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.title('Classification Accuracy',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.plot(error['accuracy'], label='Train')
plt.plot(error['val_accuracy'], label='Validation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# æ·»åŠ è®­ç»ƒç»“æ�œç»Ÿè®¡
print(f"æœ€ç»ˆè®­ç»ƒå‡†ç¡®ç�‡: {error['accuracy'].iloc[-1]:.4f}")
print(f"æœ€ç»ˆéªŒè¯�å‡†ç¡®ç�‡: {error['val_accuracy'].iloc[-1]:.4f}")
print(f"æœ€ç»ˆè®­ç»ƒæ�Ÿå¤±: {error['loss'].iloc[-1]:.4f}")
print(f"æœ€ç»ˆéªŒè¯�æ�Ÿå¤±: {error['val_loss'].iloc[-1]:.4f}")


# Evaluvate for train generator
loss,acc = model.evaluate(train_generator,batch_size = bat_size, verbose = 0)

print('The accuracy of the model for training data is:',acc*100)
print('The Loss of the model for training data is:',loss)

# Evaluvate for validation generator
loss,acc = model.evaluate(val_generator,batch_size = bat_size, verbose = 0)

print('The accuracy of the model for validation data is:',acc*100)
print('The Loss of the model for validation data is:',loss)


# Save the Model
model.save("model.keras")


# prediction
result = model.predict(test_generator, batch_size=bat_size, verbose=0)

# äºŒåˆ†ç±»ä»»åŠ¡ï¼šsigmoidè¾“å‡ºç›´æ�¥å�–æ•´å¾—åˆ°é¢„æµ‹æ ‡ç­¾
y_pred = (result > 0.5).astype(int).flatten()

y_true = test_generator.labels

# Evaluate
loss, acc = model.evaluate(test_generator, batch_size=bat_size, verbose=0)

print('The accuracy of the model for testing data is:', acc*100)
print('The Loss of the model for testing data is:', loss)


# è®¾ç½®ç–Ÿç–¾æ•°æ�®é›†çš„æ ‡ç­¾
labels = ['Parasite', 'Uninfected']

print(classification_report(y_true, y_pred, target_names=labels))


# è®¾ç½®ç–Ÿç–¾æ•°æ�®é›†çš„æ ‡ç­¾
labels = ['Parasite', 'Uninfected']

confusion_mtx = confusion_matrix(y_true, y_pred) 

f,ax = plt.subplots(figsize = (8,4),dpi=200)
sns.heatmap(confusion_mtx, annot=True, linewidths=0.1, cmap = "gist_yarg_r", linecolor="black", fmt='.0f', ax=ax,cbar=False, xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted Label",fontsize=10)
plt.ylabel("True Label",fontsize=10)
plt.title("Confusion Matrix",fontsize=13)

plt.show()


# ç–Ÿç–¾æ•°æ�®é›†ä½¿ç”¨æµ‹è¯•é›†è¿›è¡Œé¢„æµ‹
print("ç–Ÿç–¾æ•°æ�®é›†æµ‹è¯•é›†é¢„æµ‹ç»“æ�œ")

# ä½¿ç”¨ä¹‹å‰�åˆ›å»ºçš„test_generatorè¿›è¡Œé¢„æµ‹
test_predict = model.predict(test_generator, verbose=0)

# äºŒåˆ†ç±»ä»»åŠ¡ï¼šsigmoidè¾“å‡ºç›´æ�¥å�–æ•´å¾—åˆ°é¢„æµ‹æ ‡ç­¾
test_predict_binary = (test_predict > 0.5).astype(int).flatten()

# åˆ›å»ºæ ‡ç­¾æ˜ å°„
label_mapping = {0: 'Parasite', 1: 'Uninfected'}

# åˆ›å»ºç»“æ�œDataFrame
test_results = pd.DataFrame({
    'true_label': test_generator.labels,
    'predicted_label': test_predict_binary
})
test_results['true_label'] = test_results['true_label'].map(label_mapping)
test_results['predicted_label'] = test_results['predicted_label'].map(label_mapping)

print("é¢„æµ‹ç»“æ�œç¤ºä¾‹:")
print(test_results.head(10))

# è®¡ç®—æµ‹è¯•é›†å‡†ç¡®ç�‡
test_accuracy = (test_results['true_label'] == test_results['predicted_label']).mean()
print(f"\næµ‹è¯•é›†å‡†ç¡®ç�‡: {test_accuracy:.2%}")


# ç–Ÿç–¾æ•°æ�®é›†çš„å�¯è§†åŒ– - ä½¿ç”¨æµ‹è¯•é›†çš„æ ·æœ¬
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

# å®šä¹‰æ ‡ç­¾æ˜ å°„
label_mapping = {0: 'Parasite', 1: 'Uninfected'}

# è�·å�–æµ‹è¯•ç”Ÿæˆ�å™¨çš„æ–‡ä»¶è·¯å¾„
test_filenames = test_generator.filenames
for idx in range(10):
    # æ�„å»ºå®Œæ•´çš„å›¾ç‰‡è·¯å¾„
    filename = test_filenames[idx]
    image_path = os.path.join('/kaggle/input/malaria-dataset/Dataset/Train', filename)
    
    image = Image.open(image_path)
    axes[idx].imshow(image)
    
    # è�·å�–çœŸå®�æ ‡ç­¾å’Œé¢„æµ‹æ ‡ç­¾
    true_label = label_mapping[test_generator.labels[idx]]
    pred_label = label_mapping[y_pred[idx]]  # ä½¿ç”¨ä¹‹å‰�è®¡ç®—çš„y_pred
    
    # è®¾ç½®æ ‡é¢˜é¢œè‰²ï¼šç»¿è‰²è¡¨ç¤ºæ­£ç¡®ï¼Œçº¢è‰²è¡¨ç¤ºé”™è¯¯
    color = 'green' if true_label == pred_label else 'red'
    axes[idx].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10, color=color)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

