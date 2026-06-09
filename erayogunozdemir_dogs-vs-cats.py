# ===============================
# 1) Kurulum, Yol YapÄ±sÄ±, Veri Ã‡Ä±kartma (ZIP uyumlu)
# ===============================

import os, sys, glob, random, shutil, zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf

# --- SÃ¼rÃ¼m bilgisi ---
print("Python :", sys.version)
print("TensorFlow:", tf.__version__)

# --- Deterministiklik ve seed ---
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# --- GPU bellek ayarÄ± ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU aktif:", [tf.config.experimental.get_device_details(g)['device_name'] for g in gpus])
else:
    print("GPU bulunamadÄ±; CPU ile devam.")

# --- Veri kaynaÄŸÄ± kÃ¶k klasÃ¶r ---
INPUT_ROOT = Path("/kaggle/input/dogs-vs-cats")
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
DATA_ROOT = WORK_ROOT / "data" / "dogs-vs-cats"

RAW_TRAIN_DIR = DATA_ROOT / "train"
RAW_TEST_DIR  = DATA_ROOT / "test1"

if not RAW_TRAIN_DIR.exists() or not RAW_TEST_DIR.exists():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    train_zip = INPUT_ROOT / "train.zip"
    test_zip  = INPUT_ROOT / "test1.zip"

    assert train_zip.exists(), f"train.zip bulunamadÄ±: {INPUT_ROOT}"
    assert test_zip.exists(), f"test1.zip bulunamadÄ±: {INPUT_ROOT}"

    with zipfile.ZipFile(train_zip, 'r') as z:
        z.extractall(DATA_ROOT)   # â†’ data/dogs-vs-cats/train/...
    with zipfile.ZipFile(test_zip, 'r') as z:
        z.extractall(DATA_ROOT)   # â†’ data/dogs-vs-cats/test1/...

    print("Zip dosyalarÄ± aÃ§Ä±ldÄ±:", DATA_ROOT)

# --- Eski modelleri temizle (opsiyonel) ---
old_models = WORK_ROOT / "models"
if old_models.exists():
    shutil.rmtree(old_models)
    print("Eski modeller temizlendi.")

# --- Veri kontrolÃ¼ ---
all_files = sorted(glob.glob(str(RAW_TRAIN_DIR / '*.jpg')))
print("Ã–rnek dosya yolu:", all_files[0] if all_files else "BulunamadÄ±")
assert len(all_files) > 0, f"HiÃ§ .jpg bulunamadÄ±: {RAW_TRAIN_DIR}"

labels = np.array([1 if Path(f).name.startswith('dog') else 0 for f in all_files])
print(f"Toplam: {len(labels)}  |  Cats: {(labels==0).sum()}  Dogs: {(labels==1).sum()}")
print("Ã–rnek dosya:", Path(all_files[0]).name)



# ===============================
# 2) Veri HazÄ±rlama Pipelineâ€™Ä±
# ===============================

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2.1 Etiket Ã§Ä±karma ---
all_files = sorted(glob.glob(str(RAW_TRAIN_DIR / '*.jpg')))
labels = np.array([1 if Path(f).name.startswith('dog') else 0 for f in all_files], dtype=np.int32)
class_names = ['cat','dog']

print(f"EÄŸitim datasÄ± toplam: {len(all_files)} | Cats: {(labels==0).sum()} | Dogs: {(labels==1).sum()}")

# --- 2.2 Stratified split (80/20) ---
train_files, val_files, y_train, y_val = train_test_split(
    all_files, labels, test_size=0.2, random_state=SEED, stratify=labels
)
print(f"Train: {len(train_files)} | Val: {len(val_files)}")

# --- 2.3 Ã–n iÅŸleme fonksiyonu ---
def decode_img(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# --- 2.4 Augmentasyon ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
])

def augment_img(img, label):
    return data_augmentation(img, training=True), label

# --- 2.5 tf.data pipeline ---
def make_dataset(files, labels, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(lambda f,l: decode_img(f,l), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_img, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_files, y_train, augment=True, shuffle=True)
val_ds   = make_dataset(val_files, y_val, augment=False, shuffle=False)

# --- 2.6 Kontrol iÃ§in birkaÃ§ gÃ¶rsel gÃ¶sterelim ---
sample_imgs, sample_labels = next(iter(train_ds))
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(sample_imgs[i].numpy())
    plt.title(class_names[sample_labels[i].numpy()])
    plt.axis("off")
plt.show()



from tensorflow.keras import layers, models, regularizers

def build_cnn(input_shape=(224,224,3), num_classes=1, dropout_rate=0.3, l2_reg=1e-4):
    inputs = layers.Input(shape=input_shape)

    # Blok 1
    x = layers.Conv2D(32, (3,3), padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Blok 2
    x = layers.Conv2D(64, (3,3), padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Blok 3
    x = layers.Conv2D(128, (3,3), padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Fully connected
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="baseline_cnn_bn")
    return model

model = build_cnn()
model.summary()



# ===============================
# 4) EÄŸitim Stratejisi â€” Compile, Callback, Fit
# ===============================

from tensorflow import keras

# --- Derleme ---
loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.0)   # smoothing kapalÄ±
optimizer = keras.optimizers.Adam(learning_rate=3e-4)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        'accuracy',
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

# --- Callback'ler ---
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=6, restore_best_weights=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        patience=3, factor=0.5, min_lr=1e-6, monitor="val_loss", verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="cnn_best.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

# --- EÄŸitim ---
EPOCHS = 30

# KÃ¶pek recall'unu artÄ±rmak iÃ§in class_weight
class_weight = {0: 1.0, 1: 1.3}

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)



# ===============================
# 5A) EÄŸitim EÄŸrileri â€“ GÃ¶rselleÅŸtirme
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) History'yi DataFrame'e Ã§evir
assert 'history' in globals(), "history bulunamadÄ±. Ã–nce model.fit(...) Ã§alÄ±ÅŸmÄ±ÅŸ olmalÄ±."
hist = pd.DataFrame(history.history)
hist.index = np.arange(1, len(hist) + 1)  # epoch 1'den baÅŸlasÄ±n
print("EÄŸitim geÃ§miÅŸi sÃ¼tunlarÄ±:", list(hist.columns))
display(hist.tail())

# 2) YardÄ±mcÄ± Ã§izim fonksiyonu
def plot_metric(df, metric, val_metric=None, title=None, best_on='auto'):
    if metric not in df.columns:
        return
    plt.figure(figsize=(7,4))
    plt.plot(df.index, df[metric].values, label=metric)
    if val_metric and val_metric in df.columns:
        plt.plot(df.index, df[val_metric].values, label=val_metric)
        # En iyi epoch (val_metric'e gÃ¶re) â€“ loss ise min, diÄŸerleri max
        if best_on == 'auto':
            best_on = 'min' if 'loss' in val_metric else 'max'
        if best_on == 'min':
            best_epoch = int(np.argmin(df[val_metric].values) + 1)
        else:
            best_epoch = int(np.argmax(df[val_metric].values) + 1)
        plt.axvline(best_epoch, linestyle='--', alpha=0.5)
        plt.text(best_epoch + 0.1, plt.ylim()[0], f"best@{best_epoch}", rotation=90, va='bottom')
    plt.title(title or metric)
    plt.xlabel("Epoch"); plt.ylabel(metric)
    plt.grid(True, alpha=0.25); plt.legend()
    plt.show()

# 3) Ã‡izimler
plot_metric(hist, 'loss', 'val_loss', 'Loss (train vs val)', best_on='min')
plot_metric(hist, 'accuracy', 'val_accuracy', 'Accuracy (train vs val)', best_on='max')

if 'precision' in hist.columns or 'val_precision' in hist.columns:
    plot_metric(hist, 'precision', 'val_precision', 'Precision (train vs val)', best_on='max')

if 'recall' in hist.columns or 'val_recall' in hist.columns:
    plot_metric(hist, 'recall', 'val_recall', 'Recall (train vs val)', best_on='max')

if 'auc' in hist.columns or 'val_auc' in hist.columns:
    plot_metric(hist, 'auc', 'val_auc', 'AUC (train vs val)', best_on='max')

# Learning rate tarihÃ§esi bazÄ± TF versiyonlarÄ±nda 'learning_rate' bazÄ±larÄ±nda 'lr' olarak geÃ§er
lr_key = 'learning_rate' if 'learning_rate' in hist.columns else ('lr' if 'lr' in hist.columns else None)
if lr_key is not None:
    plot_metric(hist, lr_key, None, 'Learning Rate', best_on='max')



# ===============================
# 5B) Otomatik Yorum â€“ HÄ±zlÄ± TanÄ± & Ã–neriler
# ===============================
import numpy as np

def safe_get(col, default=None):
    return hist[col].values if col in hist.columns else default

loss = safe_get('loss'); val_loss = safe_get('val_loss')
acc  = safe_get('accuracy'); val_acc = safe_get('val_accuracy')
prec = safe_get('precision'); vprec = safe_get('val_precision')
rec  = safe_get('recall');    vrec  = safe_get('val_recall')
auc  = safe_get('auc');       vauc  = safe_get('val_auc')

n = len(hist)
suggestions = []

# 1) En iyi epoch'lar
best_loss_epoch = int(np.argmin(val_loss) + 1) if val_loss is not None else None
best_auc_epoch  = int(np.argmax(vauc) + 1)     if vauc is not None else None

print("â€” En iyi epoch bilgisi â€”")
if best_loss_epoch: print(f"  â€¢ Min val_loss epoch: {best_loss_epoch} (val_loss={val_loss[best_loss_epoch-1]:.4f})")
if best_auc_epoch:  print(f"  â€¢ Max val_auc  epoch: {best_auc_epoch}  (val_auc ={vauc[best_auc_epoch-1]:.4f})")

# 2) Genelizasyon farkÄ± (gap)
if loss is not None and val_loss is not None:
    gap_final = val_loss[-1] - loss[-1]
    gap_best  = val_loss[best_loss_epoch-1] - loss[best_loss_epoch-1]
    print(f"\nâ€” Genelizasyon farkÄ± (val_loss - loss) â€”")
    print(f"  â€¢ Final epoch gap: {gap_final:+.4f}")
    print(f"  â€¢ Best-loss epoch gap: {gap_best:+.4f}")

    # Heuristik: Overfit/Underfit
    # Overfit sinyali: val_loss son 3 epokta artarken train_loss dÃ¼ÅŸmeye devam ediyorsa
    if n >= 4:
        tr_drop = loss[-4] - loss[-1]   # son 3 epokta train dÃ¼ÅŸÃ¼ÅŸÃ¼
        val_rise = val_loss[-1] - val_loss[-4]  # son 3 epokta val artÄ±ÅŸÄ±
        if tr_drop > 0.02 and val_rise > 0.02:
            suggestions.append("Overfit sinyali: Son epoklarda train loss dÃ¼ÅŸerken val loss artÄ±yor. Patience/ES iyi; Dropout/L2 biraz artÄ±rÄ±labilir veya augmentasyon hafifÃ§e gÃ¼Ã§lendirilebilir.")
    # Underfit sinyali: hem train hem val loss yÃ¼ksek ve aralarÄ± Ã§ok yakÄ±n
    if val_loss[-1] > 0.60 and loss[-1] > 0.60 and abs(gap_final) < 0.05:
        suggestions.append("Underfit sinyali: Train/Val loss yÃ¼ksek ve birbirine Ã§ok yakÄ±n. Kapasite artÄ±r (daha fazla filtre/blok) veya transfer learning dene.")

# 3) Precisionâ€“Recall dengesi
if vprec is not None and vrec is not None:
    diff = float(vrec[-1] - vprec[-1])
    print(f"\nâ€” PR Dengesi â€”\n  â€¢ Val recall - precision (final): {diff:+.3f}")
    if diff > 0.15:
        suggestions.append("Recall â‰« Precision: EÅŸik (threshold) Ã§ok dÃ¼ÅŸÃ¼k veya class_weight yÃ¼ksek. EÄŸitim sonrasÄ± threshold kalibrasyonu yap; gerekiyorsa class_weight'i 1.3 â†’ 1.15-1.25 aralÄ±ÄŸÄ±na Ã§ek.")
    elif diff < -0.15:
        suggestions.append("Precision â‰« Recall: Model fazla tutucu. Threshold'u dÃ¼ÅŸÃ¼r veya augmentasyonu Ã§eÅŸitlendir; class_weight ile dog sÄ±nÄ±fÄ±nÄ± biraz aÄŸÄ±rlÄ±klandÄ±r.")

# 4) AUC trendi
if vauc is not None and n >= 5:
    # son 5 epokta geliÅŸim
    delta_auc = vauc[-1] - vauc[max(-5, -n)]
    print(f"\nâ€” AUC Trend â€”\n  â€¢ Son 5 epok delta (val_auc): {delta_auc:+.4f}")
    if delta_auc < 0.005:
        suggestions.append("Val AUC son epoklarda plato yapmÄ±ÅŸ gÃ¶rÃ¼nÃ¼yor. LR yarÄ±ya dÃ¼ÅŸÃ¼rme (ReduceLROnPlateau), augmentasyon hafifletme veya daha bÃ¼yÃ¼k batch ile tekrar dene.")

# 5) LR kaydÄ± varsa raporla
lr_key = 'learning_rate' if 'learning_rate' in hist.columns else ('lr' if 'lr' in hist.columns else None)
if lr_key:
    lr_last = float(hist[lr_key].iloc[-1])
    print(f"\nâ€” Learning Rate â€”\n  â€¢ Son LR: {lr_last:.2e}")

# 6) Son Ã¶nerileri yazdÄ±r
print("\nâ€” Ã–neriler â€”")
if suggestions:
    for s in suggestions:
        print("  â€¢", s)
else:
    print("  â€¢ EÄŸilimler saÄŸlÄ±klÄ± gÃ¶rÃ¼nÃ¼yor. Bir sonraki adÄ±m: EÅŸik (threshold) kalibrasyonu ve karÄ±ÅŸÄ±klÄ±k matrisi ile doÄŸrulama.")



# ===============================
# 6) DoÄŸrulama DeÄŸerlendirmesi â€” Rapor & KarÄ±ÅŸÄ±klÄ±k Matrisi
# ===============================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# 1) En iyi modeli yÃ¼kle ve doÄŸrulama iÃ§in olasÄ±lÄ±klarÄ± al
best = keras.models.load_model("cnn_best.keras")
val_probs = best.predict(val_ds, verbose=0).ravel()  # p(dog)

# 2) EÅŸik (threshold) kalibrasyonu: F1 ve dog-recall iÃ§in en iyi eÅŸik
ts = np.linspace(0.20, 0.80, 121)

def dog_recall_for_threshold(t):
    y_hat = (val_probs >= t).astype(int)
    cm = confusion_matrix(y_val, y_hat, labels=[0,1])
    tp = cm[1,1]; fn = cm[1,0]
    return tp / (tp + fn + 1e-12)

scores_f1  = [(t, f1_score(y_val, (val_probs >= t).astype(int))) for t in ts]
scores_rec = [(t, dog_recall_for_threshold(t)) for t in ts]

best_t_f1  = max(scores_f1,  key=lambda x: x[1])[0]
best_t_rec = max(scores_rec, key=lambda x: x[1])[0]

# KullanÄ±lacak eÅŸiÄŸi seÃ§ (hedefine gÃ¶re). Ä°stersen manuel override yap.
USE_OBJECTIVE = "f1"   # "f1" veya "dog_recall"
OVERRIDE_T = None      # Ã–rn: 0.47 verirsen bu kullanÄ±lÄ±r

use_t = (OVERRIDE_T if OVERRIDE_T is not None 
         else (best_t_f1 if USE_OBJECTIVE=="f1" else best_t_rec))

print(f"SeÃ§ilen eÅŸik: {use_t:.3f}  |  En iyi F1 eÅŸiÄŸi: {best_t_f1:.3f}  |  En iyi dog-recall eÅŸiÄŸi: {best_t_rec:.3f}")
print("ROC AUC (val):", roc_auc_score(y_val, val_probs))

# 3) SÄ±nÄ±flandÄ±rma raporu
y_pred = (val_probs >= use_t).astype(int)
report = classification_report(y_val, y_pred, target_names=class_names, digits=4)
print("\n=== SÄ±nÄ±flandÄ±rma Raporu (val) ===")
print(report)

# 4) KarÄ±ÅŸÄ±klÄ±k matrisi (ham ve normalize)
cm = confusion_matrix(y_val, y_pred, labels=[0,1])
cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

def plot_confusion_matrix(cm_, labels, normalize=False, title="Confusion Matrix"):
    plt.figure(figsize=(5,4))
    im = plt.imshow(cm_, interpolation='nearest')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels); plt.yticks(tick_marks, labels)
    fmt = ".2f" if normalize else "d"
    thresh = cm_.max() / 2. if cm_.size > 0 else 0.5
    for i in range(cm_.shape[0]):
        for j in range(cm_.shape[1]):
            plt.text(j, i, format(cm_[i, j], fmt),
                     ha="center", va="center")
    plt.ylabel('GerÃ§ek SÄ±nÄ±f'); plt.xlabel('Tahmin SÄ±nÄ±f')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm,      class_names, normalize=False, title="CM (Ham SayÄ±lar)")
plot_confusion_matrix(cm_norm, class_names, normalize=True,  title="CM (Normalize, satÄ±r %)")

# 5) Ã–zet metrikler (Ã¶zellikle DOG sÄ±nÄ±fÄ± iÃ§in)
tp, fp = cm[1,1], cm[0,1]
fn, tn = cm[1,0], cm[0,0]
dog_precision = tp / (tp + fp + 1e-12)
dog_recall    = tp / (tp + fn + 1e-12)

print(f"\nDOG precision: {dog_precision:.4f} | DOG recall: {dog_recall:.4f}")
print(f"Accuracy: {(tn+tp)/cm.sum():.4f}")

# 6) (Opsiyonel) Rapor ve CMâ€™i diske kaydet
with open("val_classification_report.txt", "w") as f:
    f.write(f"Threshold: {use_t:.3f}\n")
    f.write(f"ROC AUC: {roc_auc_score(y_val, val_probs):.4f}\n\n")
    f.write(report)

plt.imsave("val_cm_counts.png", cm, format='png')
plt.imsave("val_cm_norm.png", cm_norm, format='png')
print("\nKaydedildi: val_classification_report.txt, val_cm_counts.png, val_cm_norm.png")



# ===============================
# 7) AÃ§Ä±klanabilirlik â€” Grad-CAM (son konv: last_conv)
# ===============================
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow import keras
import matplotlib.cm as cm

# 1) En iyi modeli yÃ¼kle
best = keras.models.load_model("cnn_best.keras")

# 2) Son konv katmanÄ±nÄ± bul (Ã¶ncelik: 'last_conv', yoksa en sondaki Conv2D)
def find_last_conv_layer_name(model, preferred_name='last_conv'):
    # Ã–nce ismen ara
    try:
        layer = model.get_layer(preferred_name)
        if isinstance(layer, layers.Conv2D):
            return preferred_name
    except Exception:
        pass
    # Bulamazsa, sondan baÅŸlayarak ilk Conv2D'yi getir
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("Modelde Conv2D katmanÄ± bulunamadÄ±.")

last_conv_name = find_last_conv_layer_name(best, 'last_conv')
print("Grad-CAM iÃ§in kullanÄ±lacak son konv katmanÄ±:", last_conv_name)

# 3) Grad-CAM Ä±sÄ± haritasÄ±nÄ± Ã¼ret
def make_gradcam_heatmap(img_tensor, model, conv_layer_name, pred_index=None):
    """
    img_tensor: (H, W, 3) float32 [0,1]
    model: tf.keras.Model
    conv_layer_name: son konv katman adÄ±
    pred_index: Ã§ok sÄ±nÄ±flÄ± ise hedef sÄ±nÄ±f; binary ise None (pozitif sÄ±nÄ±f/dog)
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )

    img_batch = tf.expand_dims(img_tensor, 0)  # (1,H,W,3)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        # Hedef skor
        if predictions.shape[-1] == 1:
            # Binary: dog olasÄ±lÄ±ÄŸÄ± (pozitif sÄ±nÄ±f)
            class_channel = predictions[:, 0]
        else:
            # Ã‡ok sÄ±nÄ±flÄ± ise tahmin edilen sÄ±nÄ±fÄ± seÃ§ (veya pred_index parametresi)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    # Gradyanlar
    grads = tape.gradient(class_channel, conv_outputs)        # (1,h,w,c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))      # (c,)

    # Kanal aÄŸÄ±rlÄ±klarÄ± ile feature map'leri tart
    conv_outputs = conv_outputs[0]                            # (h,w,c)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (h,w)

    # ReLU ve normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-12)
    return heatmap.numpy(), float(predictions.numpy().squeeze())

# 4) IsÄ± haritasÄ±nÄ± orijinal gÃ¶rsel Ã¼zerine bind et
def overlay_heatmap(img_tensor, heatmap, alpha=0.35, cmap='jet'):
    """
    img_tensor: (H,W,3) float32 [0,1]
    heatmap: (h,w) float32 [0,1]
    """
    H, W = img_tensor.shape[:2]
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (H, W)).numpy().squeeze()
    colormap = cm.get_cmap(cmap)
    heatmap_color = colormap(heatmap_resized)[..., :3]  # RGBA->RGB
    overlay = heatmap_color * alpha + img_tensor * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    return heatmap_resized, overlay

# 5) EÅŸik (threshold): Ã¶nceki adÄ±mdan 'use_t' varsa onu kullan, yoksa 0.50
use_t = globals().get('use_t', 0.50)
print(f"KullanÄ±lan eÅŸik (threshold): {use_t:.3f}")

# 6) Val setinden birkaÃ§ Ã¶rnekle Grad-CAM gÃ¶rselleÅŸtir
n_show = 6  # kaÃ§ gÃ¶rsel gÃ¶stermek istersin
shown = 0
plt.figure(figsize=(12, 4 * ((n_show + 2)//3)))

for batch_imgs, batch_labels in val_ds:
    bs = batch_imgs.shape[0]
    for i in range(bs):
        if shown >= n_show:
            break

        img = batch_imgs[i].numpy()        # [0,1]
        true_label = int(batch_labels[i].numpy())
        heatmap, prob = make_gradcam_heatmap(img, best, last_conv_name)
        _, overlay = overlay_heatmap(img, heatmap, alpha=0.35, cmap='jet')

        pred_label = int(prob >= use_t)
        title = f"y_true={true_label} | p(dog)={prob:.3f} | y_pred={pred_label}"

        # 3 sÃ¼tun: Orijinal â€“ Heatmap â€“ Overlay
        col = (shown % 3)
        row = (shown // 3)
        # Orijinal
        plt.subplot(((n_show + 2)//3), 3, shown + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

        # Heatmap (tek baÅŸÄ±na)
        plt.figure(figsize=(12, 4 * ((n_show + 2)//3))) if False else None  # tek fig iÃ§inde kal
        # AynÄ± subplotta gÃ¶stermek iÃ§in kÃ¼Ã§Ã¼k hile: alt satÄ±rdaki iki plot'u kombine edeceÄŸiz
        # (Basit tutmak adÄ±na 3 gÃ¶rseli ardÄ±ÅŸÄ±k satÄ±rlarda gÃ¶stereceÄŸiz)
        # -> Pratikte tek satÄ±rda 3 gÃ¶rsel istenirse ayrÄ± grid kurulabilir.

        # Heatmap gÃ¶rseli
        plt.subplot(((n_show + 2)//3), 3, min(shown + 2, n_show))
        plt.imshow(heatmap, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')

        # Overlay gÃ¶rseli
        plt.subplot(((n_show + 2)//3), 3, min(shown + 3, n_show))
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis('off')

        shown += 3  # 3 gÃ¶rsel birden yer kapladÄ±ÄŸÄ± iÃ§in sayacÄ± 3 artÄ±rÄ±yoruz

    if shown >= n_show:
        break

plt.tight_layout()
plt.show()

# 7) (Opsiyonel) YanlÄ±ÅŸ sÄ±nÄ±flanan Ã¶rneklerden bir batch seÃ§ip sadece onlarÄ± gÃ¶rselleÅŸtir
mis_examples = []
for batch_imgs, batch_labels in val_ds.take(5):  # ilk 5 batch'i tara (isteÄŸe baÄŸlÄ± bÃ¼yÃ¼t)
    probs = best.predict(batch_imgs, verbose=0).ravel()
    preds = (probs >= use_t).astype(int)
    y_true = batch_labels.numpy().astype(int)
    mis_mask = preds != y_true
    for i in np.where(mis_mask)[0]:
        mis_examples.append((batch_imgs[i].numpy(), y_true[i], float(probs[i])))
    if len(mis_examples) >= 6:
        break

if mis_examples:
    plt.figure(figsize=(12, 8))
    for k, (img, y_true, p) in enumerate(mis_examples[:6]):
        heatmap, prob = make_gradcam_heatmap(img, best, last_conv_name)
        _, overlay = overlay_heatmap(img, heatmap, alpha=0.35)
        plt.subplot(2, 3, k+1)
        plt.imshow(overlay)
        plt.title(f"YANLIÅ�: y={y_true}, p(dog)={p:.3f}")
        plt.axis('off')
    plt.suptitle("YanlÄ±ÅŸ SÄ±nÄ±flanan Ã–rnekler â€” Grad-CAM Overlay", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("Ä°lk taranan batch'lerde yanlÄ±ÅŸ sÄ±nÄ±flanan Ã¶rnek bulunamadÄ± (harika!).")



USE_TL = True

if USE_TL:
    IMG_TL = 224
    tl_input = layers.Input(shape=(IMG_TL, IMG_TL, 3))
    base = keras.applications.MobileNetV2(
        input_tensor=tl_input, include_top=False, weights='imagenet')
    base.trainable = False

    tl_model = keras.Sequential([
        layers.Resizing(IMG_TL, IMG_TL),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name="mobilenetv2_transfer")

    tl_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss='binary_crossentropy', metrics=['accuracy'])

    tl_history = tl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(MODELS_DIR/'mobilenetv2_tl.h5'),
                                            monitor='val_accuracy', save_best_only=True)
        ]
    )

    plot_training_curves(tl_history)


import keras_tuner as kt

def build_model(hp):
    base_filters = hp.Choice("base_filters", [16, 32, 64])
    kernel_size  = hp.Choice("kernel_size",  [3, 5])
    dropout_rate = hp.Choice("dropout_rate", [0.3, 0.5])
    dense_units  = hp.Choice("dense_units",  [64, 128, 256])
    lr           = hp.Choice("lr",           [1e-3, 5e-4, 1e-4])

    m = build_simple_cnn(base_filters=base_filters,
                         kernel_size=kernel_size,
                         dropout_rate=dropout_rate,
                         dense_units=dense_units)
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss='binary_crossentropy', metrics=['accuracy'])
    return m

HPO_TRAIN_SAMPLES = 800
HPO_VAL_SAMPLES   = 200
HPO_BATCH_SIZE    = 32

tr_small = (tf.data.Dataset.from_tensor_slices((train_files, y_train))
            .shuffle(2048, seed=SEED)
            .map(load_image, num_parallel_calls=AUTOTUNE)
            .map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            .take(HPO_TRAIN_SAMPLES)
            .batch(HPO_BATCH_SIZE)
            .repeat()
            .prefetch(AUTOTUNE))

va_small = (tf.data.Dataset.from_tensor_slices((val_files, y_val))
            .map(load_image, num_parallel_calls=AUTOTUNE)
            .take(HPO_VAL_SAMPLES)
            .batch(HPO_BATCH_SIZE)
            .repeat()
            .prefetch(AUTOTUNE))

STEPS_PER_EPOCH    = HPO_TRAIN_SAMPLES // HPO_BATCH_SIZE
VAL_STEPS_PER_EPOCH = HPO_VAL_SAMPLES // HPO_BATCH_SIZE

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=6,        
    factor=3,
    directory=str(MODELS_DIR / 'kt'),
    project_name='dogs_vs_cats'
)

early = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
print(f"[KT] Arama baÅŸlÄ±yorâ€¦ steps/epoch={STEPS_PER_EPOCH}, val_steps={VAL_STEPS_PER_EPOCH}")
tuner.search(
    tr_small,
    validation_data=va_small,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS_PER_EPOCH,
    epochs=6,
    callbacks=[early],
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_cfg = {k: best_hp.get(k) for k in ["base_filters","kernel_size","dropout_rate","dense_units","lr"]}
print("[KT] En iyi hiperparametreler:", best_cfg)

(MODELS_DIR / 'kt').mkdir(parents=True, exist_ok=True)
with open(MODELS_DIR / 'kt' / 'best_hparams.json', 'w') as f:
    json.dump(best_cfg, f, indent=2)
print(f"[KT] Hiperparametreler kaydedildi: {MODELS_DIR / 'kt' / 'best_hparams.json'}")

best_model = tuner.hypermodel.build(best_hp)
best_model.summary()




# ===============================
# 10) DoÄŸrulama Metrikleri â€” Tek Nokta (thresh=0.50) & Raporlama
# ===============================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report
)

# 1) Modeli seÃ§: varsa mbv2'yi, yoksa baseline CNN'i yÃ¼kle
MODEL_CANDIDATES = ["mbv2_best.keras", "cnn_best.keras", "best_model.keras"]
model_path = next((m for m in MODEL_CANDIDATES if Path(m).exists()), None)
assert model_path is not None, "DeÄŸerlendirilecek model bulunamadÄ± (mbv2_best.keras/cnn_best.keras)."
model = keras.models.load_model(model_path)
print(f"DeÄŸerlendirilen model: {model_path}")

# 2) Val olasÄ±lÄ±klarÄ± ve sabit eÅŸik (0.50)
val_probs = model.predict(val_ds, verbose=0).ravel().astype(np.float64)  # p(dog)
THRESH = 0.50
y_pred = (val_probs >= THRESH).astype(int)

# 3) Temel metrikler
acc  = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, zero_division=0)
rec  = recall_score(y_val, y_pred, zero_division=0)            # dog recall
f1   = f1_score(y_val, y_pred, zero_division=0)
roc  = roc_auc_score(y_val, val_probs)                          # ROC-AUC (threshold baÄŸÄ±msÄ±z)
prc  = average_precision_score(y_val, val_probs)                # PR-AUC (AP)
brier = float(np.mean((val_probs - y_val)**2))
lloss = log_loss(y_val, np.clip(val_probs, 1e-7, 1-1e-7))

cm = confusion_matrix(y_val, y_pred, labels=[0,1])
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp + 1e-12)
balanced_acc = 0.5 * (rec + specificity)

# 4) Ã–zet tablo
summary = pd.DataFrame({
    "threshold":   [THRESH],
    "accuracy":    [acc],
    "precision":   [prec],
    "recall_dog":  [rec],
    "f1":          [f1],
    "specificity": [specificity],
    "balanced_acc":[balanced_acc],
    "roc_auc":     [roc],
    "pr_auc":      [prc],
    "brier":       [brier],
    "log_loss":    [lloss],
    "tp":[tp], "fp":[fp], "fn":[fn], "tn":[tn],
    "model":[model_path]
})
print("\nâ€” Tek Nokta (thresh=0.50) Ã–zet â€”")
display(summary.round(4))

# 5) SÄ±nÄ±flandÄ±rma raporu
print("\n=== SÄ±nÄ±flandÄ±rma Raporu (thresh=0.50) ===")
print(classification_report(y_val, y_pred, target_names=class_names, digits=4))

# 6) KarÄ±ÅŸÄ±klÄ±k matrisleri (ham & normalize)
def plot_conf_mat(M, labels, normalize=False, title="Confusion Matrix"):
    mat = (M.astype(np.float64) / M.sum(axis=1, keepdims=True)) if normalize else M
    plt.figure(figsize=(5,4))
    im = plt.imshow(mat, interpolation='nearest')
    plt.title(title); plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels); plt.yticks(ticks, labels)
    fmt = ".2f" if normalize else "d"
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, format(mat[i, j], fmt), ha="center", va="center")
    plt.ylabel("GerÃ§ek"); plt.xlabel("Tahmin"); plt.tight_layout(); plt.show()

plot_conf_mat(cm, class_names, normalize=False, title="CM (Ham)")
plot_conf_mat(cm, class_names, normalize=True,  title="CM (Normalize, satÄ±r %)")

# 7) KayÄ±t
summary.to_csv("val_metrics_thresh_0.50.csv", index=False)
summary.to_json("val_metrics_thresh_0.50.json", orient="records", indent=2)
with open("val_report_thresh_0.50.txt","w") as f:
    f.write(f"Model: {model_path}\nThreshold: {THRESH:.2f}\n")
    f.write(f"ROC AUC: {roc:.4f} | PR AUC: {prc:.4f} | Brier: {brier:.6f} | LogLoss: {lloss:.6f}\n\n")
    f.write(classification_report(y_val, y_pred, target_names=class_names, digits=4))

print("\nKaydedildi: val_metrics_thresh_0.50.csv, val_metrics_thresh_0.50.json, val_report_thresh_0.50.txt")


