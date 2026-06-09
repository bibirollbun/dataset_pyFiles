pip install packages


pip uninstall -y scikit-learn imbalanced-learn numpy



pip install --upgrade imbalanced-learn



  import sklearn
  import imblearn
  print(sklearn.__version__)
  print(imblearn.__version__)



  !pip install -U scikit-learn==0.24.0 imbalanced-learn



from imblearn.over_sampling import SMOTE



!pip uninstall -y imbalanced-learn scikit-learn



!pip install imbalanced-learn==0.10.1 scikit-learn==0.24.2



!pip install scikit-learn==1.5
!pip install imbalanced-learn==0.12.4



import sklearn
import imblearn

print("scikit-learn version:", sklearn.__version__)
print("imbalanced-learn version:", imblearn.__version__)



!pip install scikit-learn==1.5
!pip install imbalanced-learn==0.13.0



!pip uninstall -y scikit-learn imbalanced-learn
#after doing this do the next
!pip install -U scikit-learn==1.3.2
!pip install -U imbalanced-learn==0.11.0
#after doing this click on restart and clear cell output
from imblearn.over_sampling import SMOTE



!pip install -U scikit-learn==1.3.2
!pip install -U imbalanced-learn==0.11.0
#after doing this click on restart and clear cell output


from imblearn.over_sampling import SMOTE



# ------------------------------------------
# Step 1: Generate SMOTE Images and CSV
# ------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV3Large, EfficientNetV2S

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Paths
base_path = '/kaggle/working/'
train_csv_path = '/kaggle/input/aptos2019-blindness-detection/train.csv'
train_images_path = '/kaggle/input/aptos2019-blindness-detection/train_images'

# Directory to save generated images
save_dir = os.path.join(base_path, 'generated_classes_1_2_3_4')
os.makedirs(save_dir, exist_ok=True)

# Load CSV
train_df = pd.read_csv(train_csv_path)

# Target classes
target_classes = [1, 2, 3, 4]
filtered_df = train_df[train_df['diagnosis'].isin(target_classes)]

print(f"Number of images in classes 1, 2, 3, 4: {len(filtered_df)}")

# Prepare temp directory for flow_from_directory
temp_dir = os.path.join(base_path, 'temp_classes_1_2_3_4')
for cls in target_classes:
    os.makedirs(os.path.join(temp_dir, str(cls)), exist_ok=True)

# Copy images
for idx, row in filtered_df.iterrows():
    img_id = row['id_code']
    label = row['diagnosis']
    src_path = os.path.join(train_images_path, img_id + '.png')
    dst_path = os.path.join(temp_dir, str(label), img_id + '.png')
    if os.path.exists(src_path):
        os.system(f'cp "{src_path}" "{dst_path}"')

# ImageDataGenerator
imagegen = ImageDataGenerator(rescale=1./255)

train_generator = imagegen.flow_from_directory(
    temp_dir,
    class_mode="categorical",
    shuffle=False,
    batch_size=128,
    target_size=(512, 512),
    seed=42
)

# Load all images
x = np.concatenate([next(train_generator)[0] for _ in range(train_generator.__len__())])
y = np.concatenate([next(train_generator)[1] for _ in range(train_generator.__len__())])

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")

# Labels
y_labels = np.argmax(y, axis=1)

# Correct label mapping (since flow_from_directory assigns labels 0,1,2,3 for folders 1,2,3,4)
mapping = {0:1, 1:2, 2:3, 3:4}
y_labels = np.vectorize(mapping.get)(y_labels)

# Keep only target classes 1,2,3,4
mask = np.isin(y_labels, target_classes)
x = x[mask]
y = y[mask]
y_labels = y_labels[mask]

# NOW Flatten images for SMOTE
X_train = x.reshape(x.shape[0], -1)

# Count existing images per class
existing_counts = {cls: np.sum(y_labels == cls) for cls in target_classes}
print("Existing counts per class:", existing_counts)

# Find maximum number of images among the 4 target classes
max_count = max(existing_counts.values())
print(f"Max count (target for all classes): {max_count}")

# Extra needed per class
extra_needed = {cls: max_count - existing_counts.get(cls, 0) for cls in target_classes}
print(f"Extra images needed per class: {extra_needed}")


# Apply SMOTE
sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X_train, y_labels)

print(f"Shape after SMOTE - X: {X_smote.shape}, y: {y_smote.shape}")

# Store original number of images
num_original = x.shape[0]

# Collect generated samples
Xsmote_img_list = []
ys_smote_list = []
class_counts = {cls: 0 for cls in target_classes}

for idx in range(num_original, len(X_smote)):
    label = y_smote[idx]
    if label in target_classes and class_counts[label] < extra_needed[label]:
        Xsmote_img_list.append(X_smote[idx])
        ys_smote_list.append(label)
        class_counts[label] += 1
    if all(class_counts[c] >= extra_needed[c] for c in target_classes):
        break

Xsmote_img_array = np.array(Xsmote_img_list).reshape(-1, 512, 512, 3)
ys_smote_array = np.array(ys_smote_list)

print(f"Generated {len(Xsmote_img_array)} synthetic images.")

# Save generated images
for cls in target_classes:
    os.makedirs(os.path.join(save_dir, str(cls)), exist_ok=True)

# Save images and prepare CSV entries
records = []

for i in range(len(Xsmote_img_array)):
    label = ys_smote_array[i]
    img_name = f'smote_{label}_{i}.png'
    pil_img = array_to_img(Xsmote_img_array[i] * 255.0)  # Unnormalize
    img_save_path = os.path.join(save_dir, str(label), img_name)
    pil_img.save(img_save_path)
    records.append((img_name, label))


# Save CSV file
csv_df = pd.DataFrame(records, columns=["filename", "label"])
csv_path = os.path.join(base_path, "generated_images_labels.csv")
csv_df.to_csv(csv_path, index=False)

print(f"CSV file saved to: {csv_path}")

# Plot 2 generated images per class
plt.figure(figsize=(12, 8))
shown_per_class = {cls: 0 for cls in target_classes}
subplot_idx = 1

for idx in range(len(Xsmote_img_array)):
    label = ys_smote_array[idx]
    if shown_per_class[label] < 2:
        plt.subplot(4, 2, subplot_idx)
        plt.imshow(Xsmote_img_array[idx])
        plt.title(f'Class {label}')
        plt.axis('off')
        shown_per_class[label] += 1
        subplot_idx += 1
    if subplot_idx > 8:
        break

plt.suptitle('2 Generated Images from Each Class', fontsize=20)
plt.tight_layout()
plt.show()


# ------------------------------------------
# Step 2: Preprocessing Both Original and SMOTE Images
# ------------------------------------------

IMG_SIZE = 224
NB_CHANNELS = 3

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    return ((t,b), (l,r), (0,0)) if is_rgb else ((t,b), (l,r))

def standardize(x):
    x = x.astype(np.float32)
    x = x / np.max(x)
    return (x - np.mean(x)) / (np.std(x))

def normalize(img):
    img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255
    return img.astype(np.uint8)

def crop_image(img, tol=10):
    def crop_image_1(img):
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    if img.ndim == 2:
        return crop_image_1(img)
    elif img.ndim == 3:
        try:
            img_cpy = img.copy()
            h, w, _ = img.shape
            img1 = cv2.resize(crop_image_1(img[:,:,0]), (w,h))
            img2 = cv2.resize(crop_image_1(img[:,:,1]), (w,h))
            img3 = cv2.resize(crop_image_1(img[:,:,2]), (w,h))
            img[:,:,0] = img1
            img[:,:,1] = img2
            img[:,:,2] = img3
            return img
        except:
            return img_cpy

def preprocess_image(img_path):
    im = cv2.imread(img_path)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = normalize(im)
    im = crop_image(im)
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    im_lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(im_lab)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(2, 2))
    l_channel = clahe.apply(l_channel)
    im_lab = cv2.merge([l_channel, a_channel, b_channel])
    im = cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)
    im = cv2.addWeighted(im, 4, cv2.GaussianBlur(im, (0, 0), IMG_SIZE/10), -4, 128)

    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(mask, (IMG_SIZE//2, IMG_SIZE//2), IMG_SIZE//2, 255, -1)
    for c in range(3):
        im[:,:,c] = np.where(mask==255, im[:,:,c], 0)

    return im.astype(np.uint8)


#c
def augment_image(img):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True
    )
    img = img.reshape((1,) + img.shape)
    return next(datagen.flow(img, batch_size=1))[0].astype(np.uint8).squeeze()
#c

#c
train_df.rename(columns={'id_code': 'filename', 'diagnosis': 'label'})
train_df['filename'] = train_df['id_code'] + '.png'
train_df.rename(columns={'diagnosis': 'label'}, inplace=True)
#c
# Combine original and generated dataframes
original_df = train_df.rename(columns={'filename': 'filename', 'label': 'label'})
generated_df = pd.read_csv(csv_path)
final_df = pd.concat([original_df, generated_df], ignore_index=True)

# Preprocess and save
processed_dir = os.path.join(base_path, 'final_processed_images')
os.makedirs(processed_dir, exist_ok=True)

#c

new_records = []

for idx, row in train_df.iterrows():
    img_name = row['filename']
    label = row['label']
    img_path = os.path.join(train_images_path, img_name)     #img_path = os.path.join(train_images_path, img_name)
    img = preprocess_image(img_path)

    new_name = f"{img_name.split('.')[0]}_original.png"
    cv2.imwrite(os.path.join(processed_dir, new_name),img)

    new_records.append({'filename': new_name, 'label': label})    #new_records.append({'filename': new_name.replace('.jpg', '.png'), 'label': label})


for idx, row in final_df.iterrows():
    img_name = row['filename']
    label = row['label']

    # Determine if image is SMOTE or original
    if img_name.startswith('smote_'):
        img_path = os.path.join(save_dir, str(label), img_name)
        img = preprocess_image(img_path)

        new_name = f"{img_name.split('.')[0]}_aug{i}.png"
        cv2.imwrite(os.path.join(processed_dir, new_name), img)

        new_records.append({'filename': new_name, 'label': label})


final_processed_df = pd.DataFrame(new_records)
final_processed_csv = os.path.join(base_path, 'final_processed_data.csv')
final_processed_df.to_csv(final_processed_csv, index=False)

import matplotlib.pyplot as plt
import cv2

plt.figure(figsize=(12, 8))

# Define classes in desired order
target_classes_int = [1, 2, 3, 4]
shown_per_class = {cls: 0 for cls in target_classes_int}
subplot_idx = 1

for cls in target_classes_int:
    count = 0
    for record in new_records:
        label = int(record['label'])
        if label != cls:
            continue

        img_path = os.path.join(processed_dir, record['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(4, 2, subplot_idx)
            plt.imshow(img)
            plt.title(f'Class {label}')
            plt.axis('off')
            subplot_idx += 1
            count += 1
        if count >= 2:
            break

plt.suptitle('2 Preprocessed Images from Each Class (Grouped)', fontsize=20)
plt.tight_layout()
plt.show()


# ------------------------------------------
# Step 3: Split Data
# ------------------------------------------

train_df, test_val_df = train_test_split(final_processed_df, test_size=0.3, stratify=final_processed_df['label'], random_state=42)
val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df['label'], random_state=42)

# ------------------------------------------
# Step 4: Model Training
# ------------------------------------------

# Convert labels to string
train_df['label'] = train_df['label'].astype(str)
val_df['label'] = val_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)

# ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Train, validation, and test generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=processed_dir,
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=128,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=processed_dir,
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=128,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=processed_dir,
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=128,
    class_mode='categorical',
    shuffle=False
)


# Model
base_model = EfficientNetV2S(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

#base_model.trainable = False  # Freeze base for transfer learning


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])




from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=19, restore_best_weights=True, verbose=1)

checkpoint_path = os.path.join(base_path, 'best_model.keras')
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

callbacks = [early_stop, model_checkpoint]

# Training
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)


# Save model
model.save(os.path.join(base_path, 'InceptionV3_model.keras'))

# Plot training curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()

# ------------------------------------------
# Step 5: Testing
# ------------------------------------------

test_preds = model.predict(test_generator)
test_preds_classes = np.argmax(test_preds, axis=1)
true_classes = test_generator.classes

# Confusion matrix
cm = confusion_matrix(true_classes, test_preds_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Show 2 images per class with prediction
plt.figure(figsize=(12, 8))
class_counts = {i: 0 for i in range(5)}
subplot_idx = 1

for i in range(len(test_generator.filenames)):
    img = cv2.imread(os.path.join(processed_dir, test_generator.filenames[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = true_classes[i]
    pred = test_preds_classes[i]

    if class_counts[label] < 2:
        plt.subplot(5, 2, subplot_idx)
        plt.imshow(img)
        plt.title(f"True: {label} | Pred: {pred}")
        plt.axis('off')
        class_counts[label] += 1
        subplot_idx += 1

    if subplot_idx > 10:
        break

plt.tight_layout()
plt.show()

