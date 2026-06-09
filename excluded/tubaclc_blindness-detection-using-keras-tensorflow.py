import os
import warnings 
import cv2 as cv
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_style(style="darkgrid")


# =============================================================================
# 1. VERÄ° YÃœKLEME VE KEÅ�F
# =============================================================================

df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}")

# SÄ±nÄ±f dengesizliÄŸi gÃ¶rselleÅŸtirme
data = df.replace({"diagnosis":{0:"No DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferative DR"}})
diagnosis_count = data.diagnosis.value_counts()

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="diagnosis", order=diagnosis_count.index, palette="viridis")
plt.xlabel("Diagnosis", weight="bold", size=15)
plt.ylabel("Frequency", weight="bold", size=15)
plt.title("Class Distribution in Dataset", weight="bold", size=16)

for i, v in enumerate(diagnosis_count.values):
    text = f"{v*100/len(data):0.2f}%"
    plt.text(s=text, x=i, y=v+50, ha="center", weight="bold", size=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# =============================================================================
# 2. GELÄ°Å�TÄ°RÄ°LMÄ°Å� VERÄ° Ã–N Ä°Å�LEME VE ARTIRMA
# =============================================================================

def preprocess_image(image_path, img_size=224):
    """GÃ¶rÃ¼ntÃ¼ Ã¶n iÅŸleme fonksiyonu"""
    img = cv.imread(image_path)
    if img is None:
        return None
    
    # Ben Graham preprocessing (retina gÃ¶rÃ¼ntÃ¼leri iÃ§in Ã¶zel)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Circular crop (retina gÃ¶rÃ¼ntÃ¼leri genelde dairesel)
    h, w = img.shape[:2]
    center = (w//2, h//2)
    radius = min(center[0], center[1], w-center[0], h-center[1])
    
    # Gaussian blur to reduce noise
    img = cv.GaussianBlur(img, (5, 5), 0)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv.cvtColor(lab, cv.COLOR_LAB2RGB)
    
    # Resize
    img = cv.resize(img, (img_size, img_size))
    
    # Normalization
    img = img.astype(np.float32) / 255.0
    
    return img

# GeliÅŸmiÅŸ veri artÄ±rma pipeline
def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.CoarseDropout(num_holes_range=(5, 10), hole_height_range=(8, 20), 
                       hole_width_range=(8, 20), p=0.3),
    ])

# Veri yÃ¼kleme ve artÄ±rma
X = []
y = []
IMG_SIZE = 224
IMG_DIR = "/kaggle/input/aptos2019-blindness-detection/train_images"
augmentation = get_augmentation_pipeline()

# SÄ±nÄ±f aÄŸÄ±rlÄ±klarÄ± hesapla
class_counts = df['diagnosis'].value_counts().sort_index()
total_samples = len(df)

print("Loading and preprocessing images...")
for target in range(5):
    image_ids = df[df["diagnosis"] == target]["id_code"]
    print(f"Processing class {target} ({len(image_ids)} images)")
    
    for image_id in tqdm(image_ids, desc=f"Class {target}"):
        path = os.path.join(IMG_DIR, f"{image_id}.png")
        img = preprocess_image(path, IMG_SIZE)
        
        if img is None:
            continue
            
        # Orijinal gÃ¶rÃ¼ntÃ¼
        X.append(img)
        y.append(target)
        
        # Az Ã¶rnekli sÄ±nÄ±flar iÃ§in daha fazla artÄ±rma
        augment_count = 0
        if target == 0:  # No DR - Ã§ok fazla Ã¶rnek var, artÄ±rma
            augment_count = 0
        elif target == 1:  # Mild
            augment_count = 1
        elif target == 2:  # Moderate  
            augment_count = 2
        elif target == 3:  # Severe
            augment_count = 4
        elif target == 4:  # Proliferative DR
            augment_count = 6
            
        # Veri artÄ±rma uygula
        for _ in range(augment_count):
            augmented = augmentation(image=img)
            aug_img = augmented['image']
            X.append(aug_img)
            y.append(target)

X = np.array(X)
y = np.array(y)

print(f"Total images after augmentation: {X.shape[0]}")
print("Class distribution after augmentation:", np.bincount(y))


# =============================================================================
# 3. SINIFLARIN DENGELENMESÄ° Ä°Ã‡Ä°N CLASS WEIGHTS
# =============================================================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)



# =============================================================================
# 4. VERÄ° BÃ–LME
# =============================================================================

# Stratified split
x_temp, x_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)

del X, y  # HafÄ±zayÄ± boÅŸalt

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")


# =============================================================================
# 5. GELÄ°Å�TÄ°RÄ°LMÄ°Å� MODEL MÄ°MARÄ°SÄ° (TRANSFER LEARNING)
# =============================================================================

def create_advanced_model(input_shape=(224, 224, 3), num_classes=5, use_transfer_learning=False):
    """GeliÅŸmiÅŸ model mimarisi"""
    
    if use_transfer_learning:
        try:
            # Ã–nce ResNet50 dene
            from tensorflow.keras.applications import ResNet50
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
            
            # Ä°lk katmanlarÄ± dondur
            base_model.trainable = False
            
            model = Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            print("Using ResNet50 Transfer Learning")
            
        except Exception as e:
            print(f"Transfer learning failed: {e}")
            print("Falling back to custom CNN...")
            use_transfer_learning = False
    
    if not use_transfer_learning:
        # GeliÅŸtirilmiÅŸ Ã¶zel CNN mimarisi
        model = Sequential([
            layers.InputLayer(input_shape=input_shape),
            
            # Block 1
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 5
            layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax")
        ])
        
        print("Using Custom CNN Architecture")
    
    return model

# Model oluÅŸtur (Transfer learning yerine Ã¶zel CNN)
model = create_advanced_model(use_transfer_learning=False)

# Optimizer
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# Compile
model.compile(
    optimizer=optimizer,
    loss=SparseCategoricalCrossentropy(),  # Label smoothing kaldÄ±rÄ±ldÄ±
    metrics=["accuracy"]  # top_2_accuracy da kaldÄ±rÄ±ldÄ± uyumluluk iÃ§in
)

model.summary()



# =============================================================================
# 6. GELÄ°Å�TÄ°RÄ°LMÄ°Å� CALLBACK'LER
# =============================================================================

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        mode="max",
        verbose=1,
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]


# =============================================================================
# 7. MODEL EÄ�Ä°TÄ°MÄ°
# =============================================================================

print("Starting model training...")
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,  # Daha kÃ¼Ã§Ã¼k batch size
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict,  # SÄ±nÄ±f aÄŸÄ±rlÄ±klarÄ±
    verbose=1
)


# =============================================================================
# 8. SONUÃ‡LARIN GÃ–RSELLEÅ�TÄ°RÄ°LMESÄ°
# =============================================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Accuracy", fontsize=14, weight='bold')
plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.title("Loss", fontsize=14, weight='bold')
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.title("Learning Rate", fontsize=14, weight='bold')
if 'lr' in history.history:
    plt.plot(history.history["lr"], linewidth=2, color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# 9. MODEL DEÄ�ERLENDÄ°RMESÄ°
# =============================================================================

print("Evaluating model on test set...")
y_predicted = model.predict(x_test, verbose=1)
y_predicted_classes = np.argmax(y_predicted, axis=1)

# Accuracy
test_accuracy = accuracy_score(y_test, y_predicted_classes)
print(f"\n{'='*50}")
print(f"TEST ACCURACY: {test_accuracy*100:.2f}%")
print(f"{'='*50}")

# Classification Report
print("\nClassification Report:")
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
print(classification_report(y_test, y_predicted_classes, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_predicted_classes)
plt.figure(figsize=(10, 8))
cmd = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=class_names
)
cmd.plot(cmap=plt.cm.Blues, values_format='d', xticks_rotation="vertical")
plt.title("Confusion Matrix", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# Per-class accuracy
print("\nPer-class Accuracy:")
for i, class_name in enumerate(class_names):
    class_mask = (y_test == i)
    if np.sum(class_mask) > 0:
        class_acc = accuracy_score(y_test[class_mask], y_predicted_classes[class_mask])
        print(f"{class_name}: {class_acc*100:.2f}%")


# =============================================================================
# 10. Ã–RNEK TAHMÄ°NLER
# =============================================================================

decode = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

plt.figure(figsize=(20, 10))
random_indices = np.random.randint(0, len(df), size=12)

for i, idx in enumerate(random_indices, 1):
    plt.subplot(3, 4, i)
    
    image_id = df.loc[idx, "id_code"]
    true_label = df.loc[idx, "diagnosis"]
    
    img_path = os.path.join(IMG_DIR, f"{image_id}.png")
    img = preprocess_image(img_path)
    
    if img is None:
        plt.text(0.5, 0.5, "Image Not Found", ha="center", va="center", color="red")
        plt.axis("off")
        continue
    
    # Tahmin
    input_img = np.expand_dims(img, axis=0)
    prediction = model.predict(input_img, verbose=0)
    pred_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # GÃ¶rsel
    plt.imshow(img)
    plt.axis("off")
    
    # Etiketler
    true_text = f"True: {decode[true_label]}"
    pred_text = f"Pred: {decode[pred_label]} ({confidence:.1f}%)"
    
    # DoÄŸru/yanlÄ±ÅŸ tahmini renklendirme
    color = "green" if true_label == pred_label else "red"
    
    plt.text(5, 15, true_text, color="white", fontsize=10, weight="bold", 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.8))
    plt.text(5, 210, pred_text, color="white", fontsize=10, weight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

plt.suptitle("Sample Predictions", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("MODEL TRAINING COMPLETED!")
print("="*70)





















































































