# --- BÆ¯á»šC 0: CÃ€I Ä�áº¶T MÃ”I TRÆ¯á»œNG ---
!pip install "protobuf<4"
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split

# --- 1. Cáº¤U HÃŒNH Há»† THá»�NG ---
# Tá»± Ä‘á»™ng chá»�n chiáº¿n thuáº­t phÃ¢n phá»‘i GPU Ä‘á»ƒ tá»‘i Ä‘a tá»‘c Ä‘á»™
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("ğŸš€ Ä�ang cháº¡y trÃªn TPU!")
except ValueError:
    strategy = tf.distribute.get_strategy() # Máº·c Ä‘á»‹nh lÃ  CPU hoáº·c GPU
    print(f"ğŸš€ Ä�ang cháº¡y trÃªn {strategy.num_replicas_in_sync} thiáº¿t bá»‹ (GPU/CPU)")

IMG_DIR = '/kaggle/input/microsoft-malware/processed_images' 
LABEL_FILE = '../input/malware-classification/trainLabels.csv'
IMG_SIZE = (226, 226)
BATCH_SIZE = 32 * strategy.num_replicas_in_sync # TÄƒng batch size náº¿u cÃ³ nhiá»�u GPU
EPOCHS = 30 # TÄƒng lÃªn 30 vÃ¬ model sÃ¢u cáº§n thá»�i gian há»™i tá»¥

# --- 2. LOAD Dá»® LIá»†U (Giá»¯ nguyÃªn pháº§n load cá»§a báº¡n vÃ¬ nÃ³ Ä‘Ã£ á»•n) ---
print("â�³ Ä�ang táº£i dá»¯ liá»‡u...")
labels_df = pd.read_csv(LABEL_FILE)
labels_df['Class'] = labels_df['Class'] - 1 

X = []
y = []

if os.path.exists(IMG_DIR):
    valid_files = set(os.listdir(IMG_DIR))
    for index, row in labels_df.iterrows():
        file_name = row['Id'] + '.png'
        if file_name in valid_files:
            try:
                path = os.path.join(IMG_DIR, file_name)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    X.append(img)
                    y.append(row['Class'])
            except:
                pass
else:
    print("â�Œ Lá»–I: KhÃ´ng tÃ¬m tháº¥y áº£nh!")

# Chuáº©n hÃ³a vÃ  reshape
X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0
y = np.array(y)

# Chia táº­p dá»¯ liá»‡u
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"âœ… Dá»¯ liá»‡u sáºµn sÃ ng: {X_train.shape}")

# --- 3. XÃ‚Y Dá»°NG MODEL (THEO TABLE 6 - OPTIMIZED) ---
# Má»Ÿ scope strategy Ä‘á»ƒ táº­n dá»¥ng tá»‘i Ä‘a GPU/TPU
with strategy.scope():
    model = models.Sequential()
    
    # === Input Layer ===
    model.add(layers.Input(shape=(226, 226, 1)))
    
    # === BLOCK 1: Conv 64 === (TÆ°Æ¡ng á»©ng dÃ²ng 2-4 trong báº£ng)
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization()) # ThÃªm Batch Norm Ä‘á»ƒ á»•n Ä‘á»‹nh
    model.add(layers.MaxPooling2D((2, 2))) 
    
    # === BLOCK 2: Conv 128 === (TÆ°Æ¡ng á»©ng dÃ²ng 5-8)
    # Báº£ng ghi lÃ  "Fully-connected" output 112x112 -> Thá»±c cháº¥t lÃ  Conv2D
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization()) # Table dÃ²ng 7 cÃ³ nháº¯c Ä‘áº¿n cÃ¡i nÃ y
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3)) # Table dÃ²ng 8
    
    # === BLOCK 3: Conv 256 === (TÆ°Æ¡ng á»©ng dÃ²ng 9-10)
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4)) 
    
    # === BLOCK 4: Conv 512 (Block sÃ¢u nháº¥t) === (TÆ°Æ¡ng á»©ng dÃ²ng 10-15)
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5)) # Dropout cao á»Ÿ lá»›p sÃ¢u Ä‘á»ƒ chá»‘ng overfit
    
    # === CLASSIFICATION HEAD === (TÆ°Æ¡ng á»©ng dÃ²ng 16-18)
    model.add(layers.Flatten())
    
    # Tá»�I Æ¯U QUAN TRá»ŒNG: Giáº£m 4096 xuá»‘ng 1024
    # LÃ½ do: 4096 params quÃ¡ lá»›n cho 10k áº£nh -> GÃ¢y Overfit vÃ  cháº­m.
    # 1024 lÃ  con sá»‘ "vÃ ng" cÃ¢n báº±ng giá»¯a Ä‘á»™ phá»©c táº¡p vÃ  hiá»‡u nÄƒng.
    model.add(layers.Dense(1024, activation='relu')) 
    model.add(layers.BatchNormalization()) # GiÃºp máº¡ng Dense há»�c nhanh hÆ¡n
    model.add(layers.Dropout(0.5)) # DÃ²ng 16-17: Dropout 0.5 lÃ  báº¯t buá»™c
    
    model.add(layers.Dense(9, activation='softmax')) # Output 9 lá»›p

    # DÃ¹ng Adam vá»›i learning rate nhá»� Ä‘á»ƒ model há»™i tá»¥ tá»« tá»«, chÃ­nh xÃ¡c hÆ¡n
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model.summary()

# --- 4. CALLBACKS (BÃ� KÃ�P TRAIN KHÃ”NG Cáº¦N CANH MÃ�Y) ---
# 1. LÆ°u model tá»‘t nháº¥t (khÃ´ng lo model cuá»‘i bá»‹ kÃ©m Ä‘i)
checkpoint = callbacks.ModelCheckpoint('best_cnn_model_tuned.keras', 
                                       monitor='val_accuracy', 
                                       save_best_only=True, 
                                       mode='max', verbose=1)

# 2. Giáº£m tá»‘c Ä‘á»™ há»�c náº¿u tháº¥y loss Ä‘i ngang (giÃºp nhÃ­ch thÃªm vÃ i % accuracy)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                        factor=0.2, 
                                        patience=3, 
                                        min_lr=1e-6, verbose=1)

# 3. Dá»«ng sá»›m náº¿u khÃ´ng há»�c Ä‘Æ°á»£c ná»¯a (TrÃ¡nh tá»‘n thá»�i gian 12h vÃ´ Ã­ch)
early_stop = callbacks.EarlyStopping(monitor='val_loss', 
                                     patience=7, 
                                     restore_best_weights=True, verbose=1)

print("\nğŸš€ Báº®T Ä�áº¦U TRAIN (PhiÃªn báº£n Tuned - Cháº¡y mÆ°á»£t mÃ )...")
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint, reduce_lr, early_stop])

# --- 5. LÆ¯U Káº¾T QUáº¢ ---
# LÆ°u file npy Ä‘á»ƒ Ensemble
print("\nğŸ’¾ Ä�ang lÆ°u káº¿t quáº£...")
y_pred_probs = model.predict(X_test)
np.save('cnn_probs.npy', y_pred_probs)
np.save('y_test_labels.npy', y_test)
print(f"ğŸ�† Final Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")


!pip install "protobuf<4"

