import os, gc, math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A, seaborn as sns



def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)

# Mixed precision an toàn cho mọi bản TF trên Kaggle
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("[INFO] Mixed precision enabled")
except Exception as e:
    print("[WARN] Mixed precision not available:", e)

# GPU memory growth & XLA (tuỳ bản TF)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[INFO] GPUs: {len(gpus)}")
    else:
        print("[INFO] No GPU detected")
except Exception as e:
    print("[WARN] GPU config:", e)

try:
    tf.config.optimizer.set_jit(True)
    print("[INFO] XLA JIT enabled")
except Exception:
    pass



class CFG:
    data_dir = '/kaggle/input/cassava-leaf-disease-classification'
    train_dir = '/kaggle/input/cassava-leaf-disease-classification/train_images'

    img_size = 320  # 380 nếu GPU mạnh
    num_classes = 5
    batch_size = 16

    stage1_epochs = 3
    stage2_epochs = 10
    stage3_epochs = 5

    stage1_lr = 7e-4
    stage2_lr = 5e-4
    stage3_lr = 3e-4

    val_split = 0.2

class_names = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']
print("CFG ready")



train_csv = os.path.join(CFG.data_dir, 'train.csv')
sample_sub_csv = os.path.join(CFG.data_dir, 'sample_submission.csv')


train_df = pd.read_csv(train_csv)
assert {'image_id','label'}.issubset(train_df.columns)


# image_id của Cassava ĐÃ có đuôi .jpg
train_df['image_path'] = train_df['image_id'].apply(lambda x: f"{CFG.train_dir}/{x}")


train_df, valid_df = train_test_split(
train_df, test_size=CFG.val_split, stratify=train_df['label'], random_state=42
)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
print("Train/Valid:", len(train_df), len(valid_df))


train_transform = A.Compose([
    A.Resize(height=CFG.img_size + 32, width=CFG.img_size + 32),
    A.RandomCrop(height=CFG.img_size, width=CFG.img_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.3),
    A.CoarseDropout(max_holes=4, max_height=CFG.img_size//12, max_width=CFG.img_size//12, p=0.25),
])
valid_transform = A.Compose([
    A.Resize(height=CFG.img_size, width=CFG.img_size)
])

def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def to_model_input(img_rgb, transform):
    img = transform(image=img_rgb)['image'].astype(np.float32)
    img = preprocess_input(img)  # GIỮ nguyên chuẩn hoá
    return img



def mixup_batch(x, y, alpha=0.2):
    if len(x) < 2:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(x))
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


class SimpleGen(keras.utils.Sequence):
    def __init__(self, df, transform, batch_size, shuffle=True, mixup=False, mixup_alpha=0.2, num_classes=5):
        self.df = df.reset_index(drop=True)
        self.t = transform
        self.bs = batch_size
        self.shuffle = shuffle
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.bs))

    def __getitem__(self, i):
        idxs = self.indexes[i * self.bs : (i + 1) * self.bs]
        batch = self.df.iloc[idxs]
        X, y = [], []
        for _, row in batch.iterrows():
            img = read_rgb(row['image_path'])
            X.append(to_model_input(img, self.t))
            y.append(row['label'])
        X = np.asarray(X, dtype=np.float32)
        y = keras.utils.to_categorical(y, self.num_classes)
        if self.mixup and np.random.rand() < 0.5:
            X, y = mixup_batch(X, y, self.mixup_alpha)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


train_gen = SimpleGen(train_df, train_transform, CFG.batch_size, shuffle=True, mixup=True, mixup_alpha=0.2, num_classes=CFG.num_classes)
valid_gen = SimpleGen(valid_df, valid_transform, CFG.batch_size, shuffle=False, mixup=False, num_classes=CFG.num_classes)



def se_block(x, r=8, name_prefix="se"):
    f = int(x.shape[-1])
    s = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    s = layers.Dense(max(f // r, 4), activation='relu', name=f"{name_prefix}_fc1")(s)
    s = layers.Dense(f, activation='sigmoid', name=f"{name_prefix}_fc2")(s)
    s = layers.Reshape((1, 1, f), name=f"{name_prefix}_reshape")(s)
    return layers.Multiply(name=f"{name_prefix}_scale")([x, s])


def dw_se_block(x, out_ch, s=1, drop=0.0, name_prefix='b'):
    x_in = x
    x = layers.DepthwiseConv2D(3, strides=s, padding='same', use_bias=False, name=f"{name_prefix}_dw")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = layers.Activation('swish', name=f"{name_prefix}_dw_swish")(x)

    x = layers.Conv2D(out_ch, 1, padding='same', use_bias=False, name=f"{name_prefix}_pw")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
    x = layers.Activation('swish', name=f"{name_prefix}_pw_swish")(x)

    x = se_block(x, r=8, name_prefix=f"{name_prefix}_se")
    if drop and drop > 0:
        x = layers.Dropout(drop, name=f"{name_prefix}_drop")(x)
    if s == 1 and x_in.shape[-1] == out_ch:
        x = layers.Add(name=f"{name_prefix}_add")([x_in, x])
    return x


def build_cropnet(input_shape=(320, 320, 3), num_classes=5, drop_rate=0.3):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(40, 3, strides=2, padding='same', use_bias=False, name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('swish', name='stem_swish')(x)

    x = dw_se_block(x, 48, s=1, drop=0.0,  name_prefix='stage1_b1')
    x = dw_se_block(x, 48, s=1, drop=0.0,  name_prefix='stage1_b2')

    x = dw_se_block(x, 80, s=2, drop=0.05, name_prefix='stage2_b1')
    x = dw_se_block(x, 80, s=1, drop=0.05, name_prefix='stage2_b2')

    x = dw_se_block(x, 128, s=2, drop=0.10, name_prefix='stage3_b1')
    x = dw_se_block(x, 128, s=1, drop=0.10, name_prefix='stage3_b2')
    x = dw_se_block(x, 128, s=1, drop=0.10, name_prefix='stage3_b3')

    x = dw_se_block(x, 192, s=2, drop=0.15, name_prefix='stage4_b1')
    x = dw_se_block(x, 192, s=1, drop=0.15, name_prefix='stage4_b2')

    x = layers.GlobalAveragePooling2D(name='head_gap')(x)
    if drop_rate and drop_rate > 0:
        x = layers.Dropout(drop_rate, name='head_drop')(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='pred')(x)
    return keras.Model(inputs, outputs, name='CropNet')


model = build_cropnet(input_shape=(CFG.img_size, CFG.img_size, 3),
                      num_classes=CFG.num_classes, drop_rate=0.3)
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)


def make_adamw(lr):
    try:
        from tensorflow.keras.optimizers import AdamW
        return AdamW(learning_rate=lr, weight_decay=1e-4)
    except Exception:
        try:
            import tensorflow_addons as tfa
            return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
        except Exception:
            return optimizers.Adam(learning_rate=lr)



# Stage 1 — train toàn bộ (vì CropNet không có pretrain)
opt1 = make_adamw(CFG.stage1_lr)
model.compile(optimizer=opt1, loss=loss_fn, metrics=['accuracy'])
cb = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)]
history1 = model.fit(
train_gen,
validation_data=valid_gen,
epochs=CFG.stage1_epochs,
verbose=1
)


# Stage 2 — tiếp tục huấn luyện với LR nhỏ hơn
opt2 = make_adamw(CFG.stage2_lr)
model.compile(optimizer=opt2, loss=loss_fn, metrics=['accuracy'])
history2 = model.fit(
train_gen,
validation_data=valid_gen,
epochs=CFG.stage2_epochs,
verbose=1,
callbacks=cb
)


# Stage 3 — fine-tune thêm, LR thấp nhất
opt3 = make_adamw(CFG.stage3_lr)
model.compile(optimizer=opt3, loss=loss_fn, metrics=['accuracy'])
history3 = model.fit(
train_gen,
validation_data=valid_gen,
epochs=CFG.stage3_epochs,
verbose=1,
callbacks=cb
)


val_probs = model.predict(valid_gen, verbose=0)
val_pred = val_probs.argmax(axis=1)
val_true = valid_df['label'].values

print("\n[Classification Report]\n")
print(classification_report(val_true, val_pred, target_names=class_names, digits=4))

cm = confusion_matrix(val_true, val_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Pred')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()



sub_df = pd.read_csv(sample_sub_csv)
assert 'image_id' in sub_df.columns
sub_df['image_path'] = sub_df['image_id'].apply(lambda x: f"{CFG.data_dir}/test_images/{x}")

TTA_LIST = [
    A.Compose([A.Resize(CFG.img_size, CFG.img_size)]),
    A.Compose([A.Resize(CFG.img_size+24, CFG.img_size+24), A.CenterCrop(CFG.img_size, CFG.img_size)]),
    A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.HorizontalFlip(p=1.0)]),
    A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.VerticalFlip(p=1.0)]),
    A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.Rotate(limit=10, p=1.0)]),
    A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.RandomBrightnessContrast(0.1,0.1,p=1.0)]),
]

def predict_tta6(df, batch_size=CFG.batch_size):
    probs_all = []
    for t in TTA_LIST:
        local_probs = []
        for i in range(int(np.ceil(len(df)/batch_size))):
            s, e = i*batch_size, min((i+1)*batch_size, len(df))
            batch = df.iloc[s:e]
            X = []
            for _, row in batch.iterrows():
                img = read_rgb(row['image_path'])
                X.append(to_model_input(img, t))
            X = np.asarray(X, dtype=np.float32)
            local_probs.append(model.predict(X, verbose=0))
        probs_all.append(np.concatenate(local_probs, axis=0))
    return np.mean(probs_all, axis=0)

probs = predict_tta6(sub_df, CFG.batch_size)
preds = probs.argmax(axis=1)

submission = pd.read_csv(sample_sub_csv)
submission['label'] = preds
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv | head:\n", submission.head())



# Cell 11 — Dọn bộ nhớ (tuỳ chọn)
# =============================
del train_gen, valid_gen, train_df, valid_df; gc.collect();

