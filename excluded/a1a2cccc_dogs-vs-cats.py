import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import zipfile

# æ£€æŸ¥ GPU
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# å®šä¹‰å…³é”®å�‚æ•° (Top 30% çš„å…³é”®è®¾ç½®)
IMAGE_SIZE = (224, 224)  # ResNet å–œæ¬¢è¿™ä¸ªå°ºå¯¸
BATCH_SIZE = 32
EPOCHS_HEAD = 5          # ç¬¬ä¸€é˜¶æ®µè½®æ•°
EPOCHS_FINE = 10         # ç¬¬äºŒé˜¶æ®µå¾®è°ƒè½®æ•°


print("ğŸ“¦ æ­£åœ¨è§£å�‹æ•°æ�®ï¼Œè¯·ç¨�å€™...")

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip","r") as z:
    z.extractall(".")

print("âœ… è§£å�‹å®Œæˆ�ï¼�")


filenames = os.listdir("./train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1) # 1 = Dog
    else:
        categories.append(0) # 0 = Cat

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
#æŠŠç±»åˆ«è½¬æˆ�å­—ç¬¦ä¸²ï¼Œå› ä¸º flow_from_dataframe éœ€è¦�å­—ç¬¦ä¸²
df['category'] = df['category'].replace({0: 'cat', 1: 'dog'}) 

print(df.head())
print(f"æ€»å›¾ç‰‡æ•°: {df.shape[0]}")


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,        # å½’ä¸€åŒ–
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.1   # åˆ’åˆ† 10% å�šéªŒè¯�
)

# éªŒè¯�é›†ä¸�éœ€è¦�å¢�å¼ºï¼Œå�ªéœ€è¦�å½’ä¸€åŒ–
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

# æ�„å»ºè®­ç»ƒé›†æµ�
train_generator = train_datagen.flow_from_dataframe(
    df, 
    "./train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='training'
)

# æ�„å»ºéªŒè¯�é›†æµ�
validation_generator = validation_datagen.flow_from_dataframe(
    df, 
    "./train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='validation'
)


print("ğŸ”¹ æ�„å»ºæ¨¡å�‹ä¸­...")

# 1. åŠ è½½é¢„è®­ç»ƒçš„ ResNet50V2 
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. å†»ç»“åŸºç¡€æ¨¡å�‹ (ç¬¬ä¸€é˜¶æ®µä¸�è®­ç»ƒéª¨å¹²)
base_model.trainable = False

# 3. æ·»åŠ è‡ªå®šä¹‰åˆ†ç±»å¤´
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x) # é˜²æ­¢è¿‡æ‹Ÿå�ˆ
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. ç¼–è¯‘æ¨¡å�‹
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


print("ğŸš€ é˜¶æ®µä¸€ï¼šè®­ç»ƒåˆ†ç±»å¤´ (Warm-up)...")
history_1 = model.fit(
    train_generator,
    epochs=EPOCHS_HEAD,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    steps_per_epoch=len(train_generator)
)


print("é˜¶æ®µäºŒï¼šè§£å†»é¡¶å±‚å¹¶å¾®è°ƒ (Fine-tuning)...")

# 1. è®¾ç½®è§£å†»ç­–ç•¥
base_model.trainable = True

#å…ˆå†»ç»“æ‰€æœ‰å±‚ï¼Œç„¶å��å�ªè§£å†»æœ€å�� 30 å±‚
for layer in base_model.layers[:-30]: 
    layer.trainable = False

# 2. é‡�æ–°ç¼–è¯‘ (å¾®è°ƒå¿…é¡»ç”¨å¾ˆå°�çš„å­¦ä¹ ç�‡)
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 3. è®¾ç½®æ—©å�œ 
callbacks = [
    EarlyStopping(patience=2, restore_best_weights=True, monitor='val_accuracy')
]

# 4. è®­ç»ƒ 
history_2 = model.fit(
    train_generator,
    epochs=5, 
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    steps_per_epoch=len(train_generator),
    callbacks=callbacks
)


import matplotlib.pyplot as plt

# ä¸»è¦�çœ‹ç¬¬äºŒé˜¶æ®µï¼ˆå¾®è°ƒï¼‰çš„æ›²çº¿ï¼Œè¿™æ‰�æ˜¯æ¨¡å�‹çœŸæ­£å�˜å¼ºçš„åœ°æ–¹
acc = history_2.history['accuracy']
val_acc = history_2.history['val_accuracy']
loss = history_2.history['loss']
val_loss = history_2.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 1. è®©éªŒè¯�é›†ç”Ÿæˆ�å™¨é‡�ç½®ï¼Œå¹¶é¢„æµ‹ä¸€é��
validation_generator.reset()
# é¢„æµ‹
print("æ­£åœ¨å¯¹éªŒè¯�é›†è¿›è¡Œè¯„ä¼°...")
val_preds = model.predict(validation_generator, verbose=1)
val_pred_classes = np.argmax(val_preds, axis=1)

# 2. è�·å�–çœŸå®�æ ‡ç­¾
val_true_classes = validation_generator.classes

# 3. ç”»å›¾
cm = confusion_matrix(val_true_classes, val_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (ResNet50V2)')
plt.show()

# æ‰“å�°æŠ¥å‘Š
print(classification_report(val_true_classes, val_pred_classes, target_names=['Cat', 'Dog']))


import matplotlib.cm as mpl_cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # æ�„å»ºä¸“é—¨ç”¨äº�Grad-CAMçš„å­�æ¨¡å�‹
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.6):
    jet = mpl_cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet_heatmap = jet_colors[heatmap_uint8]
    
    # è°ƒæ•´å¤§å°�
    jet_heatmap = tf.image.resize(jet_heatmap, (img.shape[0], img.shape[1]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # å� åŠ 
    # æ³¨æ„�ï¼šè¿™é‡Œçš„imgå·²ç»�æ˜¯0-1å½’ä¸€åŒ–çš„ï¼Œæ‰€ä»¥è¦�ä¹˜255
    superimposed_img = jet_heatmap * 255 * alpha + img * 255
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# --- æ‰§è¡Œå�¯è§†åŒ– ---
# è�·å�– ResNet50V2 çš„æœ€å��ä¸€ä¸ªå�·ç§¯å±‚å��å­—
last_conv_layer_name = "post_relu" 

print("æ­£åœ¨ç”Ÿæˆ�çƒ­åŠ›å›¾...")
validation_generator.reset()
# å�–å‡ºä¸€æ‰¹å›¾ç‰‡
x_batch, y_batch = next(validation_generator)

plt.figure(figsize=(15, 6))
num_images = 3

for i in range(num_images):
    img = x_batch[i] # è¿™æ˜¯ä¸€ä¸ª (224, 224, 3) çš„å›¾
    img_array = np.expand_dims(img, axis=0) # å¢�åŠ  batch ç»´åº¦
    
    # é¢„æµ‹
    preds = model.predict(img_array, verbose=0)
    pred_label = "Dog" if np.argmax(preds)==1 else "Cat"
    true_label = "Dog" if np.argmax(y_batch[i])==1 else "Cat"
    conf = np.max(preds)
    
    # ç”Ÿæˆ�çƒ­åŠ›å›¾
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        result_img = display_gradcam(img, heatmap)
        
        plt.subplot(1, num_images, i+1)
        plt.imshow(result_img)
        plt.title(f"True: {true_label}\nPred: {pred_label} ({conf:.1%})")
        plt.axis('off')
    except Exception as e:
        print(f"çƒ­åŠ›å›¾ç”Ÿæˆ�å¤±è´¥: {e}")
        # å¦‚æ�œ post_relu æ‰¾ä¸�åˆ°ï¼Œå°�è¯• conv5_block3_out
        print("è¯·å°�è¯•å°† last_conv_layer_name æ”¹ä¸º 'conv5_block3_out'")

plt.tight_layout()
plt.show()


print(" æ­£åœ¨é¢„æµ‹æµ‹è¯•é›†...")

# å‡†å¤‡æµ‹è¯•é›†æ–‡ä»¶å��
test_filenames = os.listdir("./test")
test_df = pd.DataFrame({'filename': test_filenames})
nb_samples = test_df.shape[0]

# æµ‹è¯•é›†ç”Ÿæˆ�å™¨ 
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
   "./test", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# é¢„æµ‹
predict = model.predict(test_generator, steps=int(np.ceil(nb_samples/BATCH_SIZE)))


# 1. æ��å�–é¢„æµ‹ç»“æ�œ

test_df['label'] = predict[:, 1] 

# 2. æ��å�–æ•°å­— ID
test_df['id'] = test_df['filename'].str.split('.').str[0]

# 3. æ•´ç�†æœ€ç»ˆæ ¼å¼� 
submission_df = test_df[['id', 'label']].copy()

# 4. æŒ‰ ID æ�’åº� 
submission_df.sort_values(by=['id'], key=lambda x: x.astype(int), inplace=True)

# 5. ä¿�å­˜ CSV
submission_df.to_csv('submission.csv', index=False)

print(" submission.csvå·²ç”Ÿæˆ�ï¼�")
print(submission_df.head())

