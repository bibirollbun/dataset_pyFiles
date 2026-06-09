import os, glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers, callbacks
from keras.models import Model
from tensorflow import keras
from tqdm.notebook import tqdm

import zipfile

print("âœ… ç�¯å¢ƒåŠ è½½å®Œæˆ�")



# é¢„å¤„ç�†

# å®šä¹‰è§£å�‹ç›®æ ‡ç›®å½•ï¼ˆå�¯è‡ªå®šä¹‰ï¼‰
train_zip_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
test_zip_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'

extract_dir = '/kaggle/working/'

# è§£å�‹ train.zip
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# è§£å�‹ test.zip
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("âœ… æ•°æ�®é›†è§£å�‹å®Œæˆ�")


# å�‚æ•°è®¾ç½®
DATA_DIR = "/kaggle/working/train"
IMG_SIZE = 299
BATCH_SIZE = 32
SEED = 42

print("âœ… å�‚æ•°é…�ç½®å®Œæˆ�")



# æ•°æ�®å‡†å¤‡ï¼ˆéªŒè¯�é›†ã€�æµ‹è¯•é›†ï¼‰

DATA_DIR = '/kaggle/working/train'
all_images = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
labels = [1 if "dog" in os.path.basename(p) else 0 for p in all_images]

# åˆ’åˆ†è®­ç»ƒ/éªŒè¯�é›† (ä¸‹æ–‡çš„å›�å½’æ¨¡å�‹éœ€è¦�è®­ç»ƒ)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_images, labels, test_size=0.4, stratify=labels, random_state=SEED
)

# print(len(val_labels))

# å›¾åƒ�è§£ç �ä¸�é¢„å¤„ç�†å‡½æ•°
def decode_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label


# âœ… æ�„å»ºéªŒè¯�é›†ï¼šç”¨äº�å¯¹éªŒè¯�é›†è¿›è¡Œæ¨¡å�‹é¢„æµ‹ã€�æ”¶é›†å †å� è¾“å…¥
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# å‡†å¤‡æµ‹è¯•é›†æ•°æ�®
test_dir = "/kaggle/working/test"
test_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpg")), key=lambda x: int(os.path.basename(x).split('.')[0]))

# æ�„å»ºæµ‹è¯•é›†
def build_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: decode_img(p, 0)[0], num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = build_test_ds(test_paths)

print("âœ… éªŒè¯�é›†ï¼ˆval_dsï¼‰åˆ†é…�å®Œæˆ�ï¼›æµ‹è¯•é›†ï¼ˆtest_dsï¼‰åŠ è½½å®Œæˆ�")


print("val_labels batchï¼š", len(val_ds))


# 3.1åŠ è½½.kerasæ¨¡å�‹

from tensorflow.keras.models import load_model

model1 = load_model("/kaggle/input/3model/ResNet50V2.keras")
model2 = load_model("/kaggle/input/3model/NASNetMobile.keras")
model3 = load_model("/kaggle/input/3model/MobileNetV2.keras")

print("âœ… æ¨¡å�‹åŠ è½½å®Œæˆ�")



# # 3.2 æ¨¡å�‹è¯„ä¼°

# # è¯„ä¼° MobileNetV2
# loss1, acc1 = model1.evaluate(val_ds, verbose=0)
# print("âœ… MobileNetV2 - Loss:", loss1, " Accuracy:", acc1)

# # è¯„ä¼° NASNetMobile
# loss2, acc2 = model2.evaluate(val_ds, verbose=0)
# print("âœ… NASNetMobile - Loss:", loss2, " Accuracy:", acc2)

# # è¯„ä¼° ResNet50V2
# loss3, acc3 = model3.evaluate(val_ds, verbose=0)
# print("âœ… ResNet50V2 - Loss:", loss3, " Accuracy:", acc3)



# 3.2 æˆªå�–æ¯�ä¸ªæ¨¡å�‹çš„å�·ç§¯è¾“å‡ºå±‚ï¼ˆç‰¹å¾�æ��å�–å™¨ï¼‰
# éœ€è¦�æŠŠæœ€å��çš„ å…¨è¿�æ�¥å±‚ï¼ˆDenseï¼‰ å�»æ�‰ï¼Œå�ªä¿�ç•™å�·ç§¯éƒ¨åˆ†
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# æ�„å»ºç‰¹å¾�æ��å�–å™¨
feature_extractor1 = Model(inputs=model1.input, outputs=model1.get_layer("ResNet50V2").output)
feature_extractor2 = Model(inputs=model2.input, outputs=model2.get_layer("NASNetMobile").output)
feature_extractor3 = Model(inputs=model3.input, outputs=model3.get_layer("MobileNetV2").output)
# feature_extractor1 = Model(inputs=model1.input, outputs=model1.get_layer("global_average_pooling2d_1").output)


print("âœ… ç‰¹å¾�æ��å�–å™¨æ�„å»ºå®Œæˆ�")



# 3.3 æ��å�–ç‰¹å¾�å¹¶æ‹¼æ�¥ç‰¹å¾�å�‘é‡�

# å®šä¹‰ç‰¹å¾�æ��å�–å‡½æ•°ï¼ˆç”¨ call æ–¹å¼�æ›´ç¨³ï¼‰
def extract_features(dataset, extractor):
    features = []
    for batch in tqdm(dataset, desc="Extracting features"):
        # batch å�¯èƒ½æ˜¯ (image,) æˆ– (image, label)
        if isinstance(batch, tuple):
            batch = batch[0]  # å�ªå�– image éƒ¨åˆ†
        preds = extractor(batch, training=False)
        features.append(preds.numpy())
    return np.vstack(features)

# æ��å�–ç‰¹å¾�
features1 = extract_features(val_ds, feature_extractor1)
features2 = extract_features(val_ds, feature_extractor2)
features3 = extract_features(val_ds, feature_extractor3)

# æ‹¼æ�¥
X_fused = np.concatenate([features1, features2, features3], axis=1)

# æ ‡ç­¾
y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)

print("âœ… ç‰¹å¾�å·²æ��å�–å¹¶æ‹¼æ�¥")



# # ç”¨ä¸€ä¸ªç®€å�•çš„æ¨¡å�‹è¿›è¡Œè��å�ˆåˆ†ç±»

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# clf = LogisticRegression(max_iter=1000)
# clf.fit(X_fused, y_true)
# y_pred = clf.predict(X_fused)

# print("è��å�ˆæ¨¡å�‹å‡†ç¡®ç�‡ï¼š", accuracy_score(y_true, y_pred))



# ä¹Ÿå�¯ä»¥æ�¢æˆ� Keras æ¨¡å�‹ï¼š

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# å®šä¹‰è��å�ˆæ¨¡å�‹
input_tensor = Input(shape=(X_fused.shape[1],))
x = Dropout(0.4)(input_tensor)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)
# x = Dense(1, activation='sigmoid')(x)
x = Dense(1, activation='sigmoid')(input_tensor)  # ç›´æ�¥Logisticå›�å½’é£�æ ¼
fusion_model = Model(inputs=input_tensor, outputs=x)

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)

# ç¼–è¯‘æ¨¡å�‹
fusion_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# æ��å‰�å�œæ­¢å›�è°ƒï¼ˆé�¿å…�è¿‡æ‹Ÿå�ˆï¼‰
early_stop = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

# è®­ç»ƒæ¨¡å�‹
history = fusion_model.fit(
    X_fused, y_true,
    epochs=20,
    batch_size=16,     # ä½¿ç”¨æ›´å°�çš„ batch_sizeï¼ˆæ��é«˜æ³›åŒ–èƒ½åŠ›ï¼‰
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# æœ€ç»ˆè¯„ä¼°ï¼ˆåœ¨è��å�ˆç‰¹å¾�ä¸Šï¼‰
loss, acc = fusion_model.evaluate(X_fused, y_true, verbose=0)
print(f"âœ… æœ€ç»ˆè��å�ˆæ¨¡å�‹åœ¨éªŒè¯�é›†ä¸Šçš„å‡†ç¡®ç�‡: {acc:.4f}ï¼Œæ�Ÿå¤±å€¼: {loss:.4f}")



# 5.1 æ��å�–æµ‹è¯•é›†ç‰¹å¾�
features1_test = extract_features(test_ds, feature_extractor1)
features2_test = extract_features(test_ds, feature_extractor2)
features3_test = extract_features(test_ds, feature_extractor3)

X_test_fused = np.concatenate([features1_test, features2_test, features3_test], axis=1)

print("âœ… æµ‹è¯•é›†ç‰¹å¾�æ��å�–å®Œæˆ�")



# # 5.2 è��å�ˆå™¨é¢„æµ‹
# y_test_pred = fusion_model.predict(X_test_fused)
# # y_test_pred = clf.predict(X_test_fused)

# print(y_test_pred[1:10])

# y_test_pred = y_test_pred.ravel()
# # y_test_pred = y_test_pred.clip(min=0.005, max=0.995)   # è£�å‰ªä¸€ä¸‹ä¸¤å¤´

# print("âœ… ")


# 5.2 è��å�ˆå™¨é¢„æµ‹
y_test_pred = fusion_model.predict(X_test_fused)

print(y_test_pred[1:10])

# æ•°å€¼æ‹‰å�‘ä¸¤è¾¹ + è£�å‰ª
alpha = 1.07
y_test_pred = ((y_test_pred - 0.5) * alpha + 0.5).clip(min=0.005, max=0.995)
y_test_pred = y_test_pred.ravel()

print("âœ… è��å�ˆé¢„æµ‹å®Œæˆ�")


print(f"é¢„æµ‹æœ€å¤§å€¼ï¼š{y_test_pred.max():.4f}")
print(f"é¢„æµ‹æœ€å°�å€¼ï¼š{y_test_pred.min():.4f}")
print(f"é¢„æµ‹å�‡å€¼ï¼š{y_test_pred.mean():.4f}")
plt.hist(y_test_pred, bins=100)
plt.title("Test Prediction Distribution")
plt.show()



# 5.3 ç”Ÿæˆ� csvæ–‡ä»¶
import pandas as pd

# æ��å�–å›¾åƒ� ID
ids = [int(os.path.basename(path).split('.')[0]) for path in test_paths]

# æ�„å»ºæ��äº¤ DataFrame
submission = pd.DataFrame({
    "id": ids,
    "label": y_test_pred
})

# æ�’åº�
submission = submission.sort_values("id").reset_index(drop=True)

# ä¿�å­˜ä¸º CSV
submission.to_csv("submission.csv", index=False)

print("âœ… å·²ä¿�å­˜ csv æ–‡ä»¶")



# åˆ é™¤å·¥ä½œåŒºæ–‡ä»¶,ä¸�ç„¶æ¯”èµ›æ��äº¤ä¼šè€—æ—¶ä¿�å­˜

import os
import shutil

def delete_folder_contents(folder_path):
    """åˆ é™¤æ–‡ä»¶å¤¹å�Šå…¶æ‰€æœ‰å†…å®¹"""
    try:
        # åˆ é™¤æ–‡ä»¶å¤¹å†…å®¹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'åˆ é™¤ {file_path} å¤±è´¥ã€‚å�Ÿå› : {e}')
        
        # åˆ é™¤ç©ºæ–‡ä»¶å¤¹
        os.rmdir(folder_path)
        print(f"æ–‡ä»¶å¤¹ '{folder_path}' å·²æˆ�åŠŸåˆ é™¤ã€‚")
    except FileNotFoundError:
        print(f"æ–‡ä»¶å¤¹ '{folder_path}' ä¸�å­˜åœ¨ã€‚")
    except PermissionError:
        print(f"æ²¡æœ‰æ�ƒé™�åˆ é™¤æ–‡ä»¶å¤¹ '{folder_path}'ã€‚")
    except OSError as e:
        print(f"åˆ é™¤æ–‡ä»¶å¤¹æ—¶å‡ºé”™: {e}")

# è¦�åˆ é™¤çš„æ–‡ä»¶å¤¹åˆ—è¡¨
folders_to_delete = ['/kaggle/working/test', '/kaggle/working/train']

# åˆ é™¤æ¯�ä¸ªæ–‡ä»¶å¤¹
for folder in folders_to_delete:
    delete_folder_contents(folder)

print("âœ… ")

