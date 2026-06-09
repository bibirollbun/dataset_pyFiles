import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, callbacks

# paths                                                          
TRAIN_DIR = "../input/plant-seedlings-classification/train"

# hyperparams                                                    
BATCH_SIZE    = 32
IMG_SIZE      = (224, 224)
EPOCHS        = 50
VALID_SPLIT   = 0.2
LR            = 1e-3

# data generators with train/val split                           
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=VALID_SPLIT,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2,
)
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

# compute class weights                                           
labels = train_gen.classes
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# build model                                                    
base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(train_gen.num_classes, activation="softmax")(x)
model = models.Model(base.input, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

# custom callback to compute macro-F1 and confusion each epoch     
class MetricsCallback(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.history = {"f1": []}

    def on_epoch_end(self, epoch, logs=None):
        val_preds = np.argmax(self.model.predict(val_gen, verbose=0), axis=1)
        val_trues = val_gen.classes
        report = classification_report(val_trues, val_preds, output_dict=True, zero_division=0)
        f1 = report["macro avg"]["f1-score"]
        self.history["f1"].append(f1)
        print(f" — val_macro_f1: {f1:.4f}")
        # optionally, save per-epoch confusion:
        # cm = confusion_matrix(val_trues, val_preds)
        return

metrics_cb = MetricsCallback()

# other callbacks                                                
es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
ckp = callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss")

# train                                                           
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[es, rlr, ckp, metrics_cb],
)

# plot & save curves                                             
os.makedirs("plots", exist_ok=True)
# 1) accuracy & loss
plt.figure(); plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc"); plt.legend()
plt.title("Accuracy"); plt.savefig("plots/accuracy.png")

plt.figure(); plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss"); plt.legend()
plt.title("Loss"); plt.savefig("plots/loss.png")

# 2) precision & recall
plt.figure(); plt.plot(history.history["precision"], label="train_precision")
plt.plot(history.history["val_precision"], label="val_precision"); plt.legend()
plt.title("Precision"); plt.savefig("plots/precision.png")

plt.figure(); plt.plot(history.history["recall"], label="train_recall")
plt.plot(history.history["val_recall"], label="val_recall"); plt.legend()
plt.title("Recall"); plt.savefig("plots/recall.png")

# 3) macro-F1
plt.figure(); plt.plot(metrics_cb.history["f1"], label="val_macro_f1"); plt.legend()
plt.title("Val macro-F1"); plt.savefig("plots/f1.png")

# final confusion matrix                                       
val_preds = np.argmax(model.predict(val_gen, verbose=0), axis=1)
val_trues = val_gen.classes
cm = confusion_matrix(val_trues, val_preds)
plt.figure(figsize=(10,10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix"); plt.colorbar()
tick_marks = np.arange(train_gen.num_classes)
plt.xticks(tick_marks, list(train_gen.class_indices.keys()), rotation=90)
plt.yticks(tick_marks, list(train_gen.class_indices.keys()))
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")

print("Done! All plots saved under /kaggle/working/plots/, best model in best_model.keras")





