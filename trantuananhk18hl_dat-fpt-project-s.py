import numpy as np, pandas as pd, cv2, os, warnings, gc
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import albumentations as A


def set_seed(seed=42):
    np.random.seed(seed); tf.random.set_seed(seed); os.environ['PYTHONHASHSEED']=str(seed)
set_seed(42); set_global_policy('mixed_float16')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except Exception: pass
    print(f"GPU Enabled: {len(physical_devices)} device(s)")
else:
    print("No GPU found")
try: tf.config.optimizer.set_jit(True)
except Exception: pass

class CFG:
    data_dir = '/kaggle/input/cassava-leaf-disease-classification'
    train_dir = '/kaggle/input/cassava-leaf-disease-classification/train_images'
    img_size = 320          # 380 nếu GPU đủ mạnh
    num_classes = 5
    batch_size = 12         # giảm nếu OOM, tăng nếu dư VRAM
    # epochs
    stage1_epochs = 3
    stage2_epochs = 10
    stage3_epochs = 5
    # base learning rates
    stage1_lr = 7e-4
    stage2_lr = 5e-4
    stage3_lr = 3e-4
    # split
    val_split = 0.2
    # steps/epoch (có thể chỉnh để trade-off thời gian)
    steps_per_epoch = 600
    val_steps = 160
    # MixUp
    mixup_prob = 0.5
    mixup_alpha = 0.2

class_names = ['CBB','CBSD','CGM','CMD','Healthy']
print("Config ready")


train_df = pd.read_csv(f'{CFG.data_dir}/train.csv')
train_df['image_path'] = train_df['image_id'].apply(lambda x: f"{CFG.train_dir}/{x}")
train_df, valid_df = train_test_split(train_df, test_size=CFG.val_split,
                                      stratify=train_df['label'], random_state=42)
print("Train/Valid:", len(train_df), len(valid_df))

train_transform = A.Compose([
    A.Resize(height=CFG.img_size + 32, width=CFG.img_size + 32),
    A.RandomCrop(height=CFG.img_size, width=CFG.img_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.3),
    A.CoarseDropout(max_holes=4, max_height=CFG.img_size//12, max_width=CFG.img_size//12, p=0.25),
])
valid_transform = A.Compose([A.Resize(height=CFG.img_size, width=CFG.img_size)])

def apply_albu_then_preprocess(t, img_rgb):
    img = t(image=img_rgb)['image'].astype(np.float32)
    return preprocess_input(img)


def mixup_batch(x, y, alpha=0.2):
    if len(x) < 2: return x, y
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(x))
    return lam*x + (1-lam)*x[idx], lam*y + (1-lam)*y[idx]

class SimpleGen(keras.utils.Sequence):
    def __init__(self, df, transform, batch_size, shuffle=True, mixup=False):
        self.df = df.reset_index(drop=True)
        self.t = transform
        self.bs = batch_size
        self.shuffle = shuffle
        self.mixup = mixup
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.df)/self.bs))
    def __getitem__(self, idx):
        s,e = idx*self.bs, min((idx+1)*self.bs, len(self.df))
        batch = self.df.iloc[s:e]
        X, y = [], []
        for _, row in batch.iterrows():
            img = cv2.imread(row['image_path'])
            if img is None: img = np.zeros((CFG.img_size, CFG.img_size, 3), dtype=np.uint8)
            else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(apply_albu_then_preprocess(self.t, img))
            y.append(row['label'])
        X = np.array(X, dtype=np.float32); y = keras.utils.to_categorical(y, CFG.num_classes)
        if self.mixup and np.random.rand() < CFG.mixup_prob:
            X, y = mixup_batch(X, y, alpha=CFG.mixup_alpha)
        return X, y
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)

train_gen = SimpleGen(train_df, train_transform, CFG.batch_size, shuffle=True, mixup=True)
valid_gen = SimpleGen(valid_df, valid_transform, CFG.batch_size, shuffle=False, mixup=False)
print("Generators ready")


class CosineWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)

        # warmup: tăng tuyến tính 0 -> base_lr
        warm = tf.minimum(step / tf.maximum(warmup, 1.0), 1.0)

        # cosine sau warmup
        denom = tf.maximum(total - warmup, 1.0)
        progress = tf.clip_by_value((step - warmup) / denom, 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(np.pi * progress))

        lr = self.base_lr * (warm * cosine + (1.0 - warm) * 0.0)
        return tf.cast(lr, tf.float32)


total_steps = CFG.steps_per_epoch * (CFG.stage1_epochs + CFG.stage2_epochs + CFG.stage3_epochs)
warmup_steps = max(total_steps // 20, 1)
lr_schedule = CosineWithWarmup(CFG.stage1_lr, total_steps, warmup_steps)


from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
try:
    from tensorflow.keras.optimizers import AdamW
    USE_ADAMW = True
except Exception:
    from tensorflow.keras import optimizers
    USE_ADAMW = False

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

def build_model(trainable_layers=0):
    base = EfficientNetB4(include_top=False, weights='imagenet',
                          input_shape=(CFG.img_size, CFG.img_size, 3))
    if trainable_layers == 0:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:-trainable_layers]:
            layer.trainable = False

    inputs = keras.Input(shape=(CFG.img_size, CFG.img_size, 3))
    x = base(inputs, training=(trainable_layers > 0))
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(CFG.num_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs)

model = build_model(0)
model.summary()

# ===== Stage 1 =====
lr_schedule_1 = CosineDecayRestarts(
    initial_learning_rate=CFG.stage1_lr,
    first_decay_steps=CFG.steps_per_epoch * max(CFG.stage1_epochs, 1),
    t_mul=1.0, m_mul=1.0, alpha=0.0
)
optimizer_1 = AdamW(learning_rate=lr_schedule_1, weight_decay=1e-4) if USE_ADAMW else optimizers.Adam(learning_rate=lr_schedule_1)
model.compile(optimizer=optimizer_1, loss=loss_fn, metrics=['accuracy'])

cb1 = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)]
history1 = model.fit(
    train_gen, validation_data=valid_gen,
    epochs=CFG.stage1_epochs,
    steps_per_epoch=min(CFG.steps_per_epoch, len(train_gen)),
    validation_steps=min(CFG.val_steps, len(valid_gen)),
    callbacks=cb1, verbose=1
)

# ===== Stage 2 (unfreeze top 160 layers) =====
base = model.layers[1]
for layer in base.layers[-160:]:
    layer.trainable = True

lr_schedule_2 = CosineDecayRestarts(
    initial_learning_rate=CFG.stage2_lr,
    first_decay_steps=CFG.steps_per_epoch * max(CFG.stage2_epochs, 1),
    t_mul=1.0, m_mul=1.0, alpha=0.0
)
optimizer_2 = AdamW(learning_rate=lr_schedule_2, weight_decay=1e-4) if USE_ADAMW else optimizers.Adam(learning_rate=lr_schedule_2)
model.compile(optimizer=optimizer_2, loss=loss_fn, metrics=['accuracy'])

cb2 = [
    ModelCheckpoint('best_b4.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
]
history2 = model.fit(
    train_gen, validation_data=valid_gen,
    epochs=CFG.stage2_epochs,
    steps_per_epoch=min(CFG.steps_per_epoch, len(train_gen)),
    validation_steps=min(CFG.val_steps, len(valid_gen)),
    callbacks=cb2, verbose=1
)

# ===== Stage 3 =====
lr_schedule_3 = CosineDecayRestarts(
    initial_learning_rate=CFG.stage3_lr,
    first_decay_steps=CFG.steps_per_epoch * max(CFG.stage3_epochs, 1),
    t_mul=1.0, m_mul=1.0, alpha=0.0
)
optimizer_3 = AdamW(learning_rate=lr_schedule_3, weight_decay=1e-4) if USE_ADAMW else optimizers.Adam(learning_rate=lr_schedule_3)
model.compile(optimizer=optimizer_3, loss=loss_fn, metrics=['accuracy'])

cb3 = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)]
history3 = model.fit(
    train_gen, validation_data=valid_gen,
    epochs=CFG.stage3_epochs,
    steps_per_epoch=min(CFG.steps_per_epoch, len(train_gen)),
    validation_steps=min(CFG.val_steps, len(valid_gen)),
    callbacks=cb3, verbose=1
)


val_loss, val_acc = model.evaluate(valid_gen, verbose=0)
print(f"Validation Acc: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")

y_pred_probs = model.predict(valid_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = valid_df['label'].values

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Pred"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.show()


test_df = pd.read_csv(f'{CFG.data_dir}/sample_submission.csv')
test_df['image_path'] = test_df['image_id'].apply(lambda x: f'{CFG.data_dir}/test_images/{x}')

def predict_tta6(df, batch_size):
    t_list = [
        A.Compose([A.Resize(CFG.img_size, CFG.img_size)]),
        A.Compose([A.Resize(CFG.img_size+24, CFG.img_size+24), A.CenterCrop(CFG.img_size, CFG.img_size)]),
        A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.HorizontalFlip(p=1.0)]),
        A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.VerticalFlip(p=1.0)]),
        A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.Rotate(limit=10, p=1.0)]),
        A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.RandomBrightnessContrast(0.1,0.1,p=1.0)]),
    ]
    probs = []
    for t in t_list:
        preds_all = []
        for i in range(int(np.ceil(len(df)/batch_size))):
            s,e = i*batch_size, min((i+1)*batch_size, len(df))
            batch = df.iloc[s:e]
            X=[]
            for _, row in batch.iterrows():
                img=cv2.imread(row['image_path'])
                if img is None: img = np.zeros((CFG.img_size, CFG.img_size, 3), dtype=np.uint8)
                else: img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                arr = t(image=img)['image'].astype(np.float32)
                X.append(preprocess_input(arr))
            X = np.array(X, dtype=np.float32)
            preds_all.append(model.predict(X, verbose=0))
        probs.append(np.concatenate(preds_all, axis=0))
    return np.mean(probs, axis=0)

probs = predict_tta6(test_df, CFG.batch_size)
preds = np.argmax(probs, axis=1)

submission = pd.read_csv(f'{CFG.data_dir}/sample_submission.csv')
submission['label'] = preds
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv | head:")
print(submission.head())




