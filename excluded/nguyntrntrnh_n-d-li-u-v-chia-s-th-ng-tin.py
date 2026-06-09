import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


BASE_PATH = '../input/alaska2-image-steganalysis'
TEST_PATH = os.path.join(BASE_PATH, 'Test')
FEATURE_CACHE = '/kaggle/working/features.npy'
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
N_JOBS = 4
CLASS_WEIGHTS = {0: 1, 1: 3}


def batch_preprocess(image_paths, augment=False):
    batch = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
            
        img = cv2.resize(img, IMG_SIZE)
        
        if augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-10, 10)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        img = tf.keras.applications.efficientnet.preprocess_input(img)
        batch.append(img)
    
    return np.array(batch)


def init_feature_extractor():    
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    return Model(inputs=base_model.input, outputs=x)


def parallel_feature_extraction(paths, model):
    features = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE)):
        batch_paths = paths[i:i+BATCH_SIZE]
        batch_images = batch_preprocess(batch_paths, augment=True)
        batch_features = model.predict(batch_images, verbose=0)
        features.extend(batch_features)
    return np.array(features)


def create_dataset():
    data = []
    for folder in ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']:
        path = os.path.join(BASE_PATH, folder)
        files = [f for f in os.listdir(path) if f.endswith('.jpg')]
        files = files[:5000]
        labels = [0 if folder == 'Cover' else 1] * len(files)
        data.extend(zip([os.path.join(path, f) for f in files], labels))
    return pd.DataFrame(data, columns=['path', 'label']).sample(frac=1)

# Khởi tạo dataset
df = create_dataset()

if os.path.exists(FEATURE_CACHE):
    X = np.load(FEATURE_CACHE)
else:
    model = init_feature_extractor()
    X = parallel_feature_extraction(df['path'].values, model)
    np.save(FEATURE_CACHE, X)

y = df['label'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

classifier.compile(
    optimizer=Adam(1e-4),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        'accuracy'
    ]
)

early_stop = EarlyStopping(
    monitor='val_auc',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

history = classifier.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=256,
    class_weight=CLASS_WEIGHTS,
    callbacks=[early_stop],
    verbose=1
)


y_pred = classifier.predict(X_test)
print(f"\nĐánh giá cuối cùng:")
print(f"- AUC: {roc_auc_score(y_test, y_pred):.4f}")
print(f"- Accuracy: {accuracy_score(y_test, (y_pred > 0.5).astype(int)):.4f}")
print(f"- F1-Score: {f1_score(y_test, (y_pred > 0.5).astype(int)):.4f}")


test_paths = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]
test_features = parallel_feature_extraction(test_paths, model)

submission = pd.DataFrame({
    'Id': [os.path.basename(p) for p in test_paths],
    'Label': classifier.predict(test_features).flatten()
})
submission.to_csv('submission.csv', index=False)




