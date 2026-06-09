# to hidden any warrrnign 
import warnings
warnings.filterwarnings("ignore")


import os, sys, warnings

def init_env():
    """
    Suppress TensorFlow, protobuf, and general runtime warnings.
    Ensures cleaner notebook logs and avoids unnecessary noise.
    """
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Silence annoying protobuf warnings in Kaggle environment
    sys.stderr = open(os.devnull, 'w')
    try:
        import google.protobuf.message_factory as mf
        if not hasattr(mf.MessageFactory, "GetPrototype"):
            mf.MessageFactory.GetPrototype = lambda self, desc: None
    except:
        pass
init_env()
init_env()


import os
import json
import cv2
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger,
    LearningRateScheduler
)


from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import LayerNormalization



from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score

from sklearn.model_selection import train_test_split



print("TensorFlow version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))


#  determin raandom 
tf.random.set_seed(42)
np.random.seed(42)



ROOT = "/kaggle/input/"
DATASET = "cassava-leaf-disease-classification"


DATA_DIR = os.path.join(ROOT, DATASET)
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
LABEL_JSON = os.path.join(DATA_DIR, "label_num_to_disease_map.json")
SUBMISSION_TEMPLATE = os.path.join(DATA_DIR, "sample_submission.csv")





# FOCAL LOSS (for imbalanced data)
NUM_CLASSES = 5  # we have 5 cassava classes

def sparse_focal_loss(gamma=2.0):
    """
    Sparse Categorical Focal Loss for multi-class problems.
    y_true: integer class labels (0..4)
    y_pred: softmax probabilities from the model
    """
    def loss_fn(y_true, y_pred):
        # make sure y_true is int
        y_true = tf.cast(y_true, tf.int32)

        # one-hot encode: (batch,) -> (batch, NUM_CLASSES)
        y_true_oh = tf.one_hot(y_true, depth=NUM_CLASSES)

        # avoid log(0)
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # standard cross-entropy
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred_clipped), axis=-1)

        # probability of the true class
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)

        # focal factor => samples that are already easy (p_t high) get lower weight
        focal_factor = tf.pow(1.0 - p_t, gamma)

        return focal_factor * ce

    return loss_fn



train_df = pd.read_csv(TRAIN_CSV)



with open(LABEL_JSON, "r") as f:
    label_map = json.load(f)



train_df





train_df[["label"]].value_counts()



# LABEL DISTRIBUTION 
plt.figure(figsize=(9,5))
sns.countplot(x=train_df["label"], palette="viridis")
plt.title("Label Distribution Across Training Set")
plt.xlabel("Label ID")
plt.ylabel("Number of Samples")
plt.show()



# Percentage distribution
label_counts = train_df["label"].value_counts().sort_index()
label_percent = (label_counts / len(train_df)) * 100


plt.figure(figsize=(9,5))
sns.barplot(x=label_percent.index, y=label_percent.values, palette="mako")
plt.title("Percentage Distribution (%)")
plt.xlabel("Label ID")
plt.ylabel("Percentage of Dataset (%)")
plt.show()



#  summary
label_summary = pd.DataFrame({
    "Class ID": label_counts.index,
    "Class Name": [label_map[str(i)] for i in label_counts.index],
    "Count": label_counts.values,
    "Percentage": label_percent.values
})


print("Class-Level Statistical Summary:")
label_summary


max_class = label_counts.max()
min_class = label_counts.min()
imbalance_ratio = max_class / min_class



print("Class Balance Evaluation:")
print("Maximum class count:", max_class)
print("Minimum class count:", min_class)
print("Imbalance ratio (max/min):", round(imbalance_ratio, 3))





def get_resolution(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return img.shape[1], img.shape[0]  # width, height


sample_list = train_df.sample(500, random_state=42).image_id.values




resolutions = []
for img_id in sample_list:
    img_path = os.path.join(TRAIN_IMG_DIR, img_id)
    res = get_resolution(img_path)
    if res:
        resolutions.append(res)


res_df = pd.DataFrame(resolutions, columns=["Width", "Height"])



res_df


def brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:,:,2].mean()



def contrast(img):
    return img.std()



brightness_vals = []
contrast_vals = []



for img_id in sample_list[:200]:   # reduce compute cost
    img_path = os.path.join(TRAIN_IMG_DIR, img_id)
    im = cv2.imread(img_path)
    if im is not None:
        brightness_vals.append(brightness(im))
        contrast_vals.append(contrast(im))



plt.figure(figsize=(7,5))
sns.histplot(brightness_vals, bins=30, kde=True, color="orange")
plt.title("Brightness Distribution (Sampled Images)")
plt.xlabel("Brightness Level")
plt.ylabel("Frequency")
plt.show()



plt.figure(figsize=(7,5))
sns.histplot(contrast_vals, bins=30, kde=True, color="purple")
plt.title("Contrast Distribution (Sampled Images)")
plt.xlabel("Contrast Level")
plt.ylabel("Frequency")
plt.show()



print("Brightness Statistics:\n")
display( pd.Series(brightness_vals).describe())


print("Contrast Statistics:\n")
display(pd.Series(contrast_vals).describe())


print("****--")





# # SHOW 3 SAMPLES PER CLASS


plt.figure(figsize=(12, 18))

unique_labels = sorted(train_df["label"].astype(int).unique())

for row_idx, lbl in enumerate(unique_labels):
    # Sample 3 images from this class
    subset = train_df[train_df["label"].astype(int) == lbl].sample(3, random_state=42)

    for col_idx, img_id in enumerate(subset.image_id.values):
        img_path = os.path.join(TRAIN_IMG_DIR, img_id)
        img = plt.imread(img_path)

        plt.subplot(len(unique_labels), 3, row_idx * 3 + col_idx + 1)
        plt.imshow(img)
        plt.axis("off")

        # Title for the FIRST column of each row
        if col_idx == 0:
            plt.title(f"Class {lbl}: {label_map[str(lbl)]}", fontsize=10)

plt.tight_layout()
plt.show()



def rgb_histogram(path):
    img = cv2.imread(path)
    if img is None:
        return None
    colors = ("b","g","r")
    hist_data = {}
    for i,col in enumerate(colors):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        hist_data[col] = hist.flatten()
    return hist_data


img_sample = os.path.join(TRAIN_IMG_DIR, train_df.sample(1).image_id.values[0])
hist = rgb_histogram(img_sample)


plt.figure(figsize=(8,5))
plt.plot(hist["r"], color="red")
plt.plot(hist["g"], color="green")
plt.plot(hist["b"], color="blue")
plt.title("RGB Histogram Example")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()






def edge_detection_visual(path):
    img = cv2.imread(path, 0)
    edges = cv2.Canny(img, 100, 200)
    plt.figure(figsize=(6,4))
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Detection Visualization")
    plt.axis("off")
    plt.show()



print("**")


edge_detection_visual(img_sample)







img_id = train_df.sample(1).image_id.values[0]
img_path = os.path.join(TRAIN_IMG_DIR, img_id)
img = plt.imread(img_path)


plt.figure(figsize=(12,6))

plt.subplot(1,4,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(np.rot90(img))
plt.title("Rotation Example")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(np.fliplr(img))
plt.title("Horizontal Flip")
plt.axis("off")





noise = img + np.random.normal(0, 10, img.shape)
plt.subplot(1,4,4)
plt.imshow(np.clip(noise, 0, 255).astype(np.uint8))
plt.title("Noise Injection")
plt.axis("off")

plt.tight_layout()
plt.show()





def green_segmentation(img):
    """Simple HSV green-mask segmentation to isolate leaf region"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Green range in HSV (tuned for cassava leaves)
    lower = np.array([25, 40, 40])
    upper = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    return segmented


# Show example of segmentation
example_id = train_df.sample(1).image_id.values[0]
example_path = os.path.join(TRAIN_IMG_DIR, example_id)



img_original = cv2.imread(example_path)
img_segmented = green_segmentation(img_original)



plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB))
plt.title("Leaf Segmentation (Green Mask)")
plt.axis("off")

plt.tight_layout()
plt.show()








full_df = train_df.copy()



train_df, val_df = train_test_split(
    full_df,
    test_size=0.2,
    stratify=full_df["label"],
    random_state=42
)



classes = np.unique(train_df["label"].astype(int))
class_weights_raw = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].astype(int)
)


class_weights = dict(zip(classes, class_weights_raw))



print("Class weights:", class_weights)



print("Train size:", len(train_df))
print("Validation size:", len(val_df))



train_dist = train_df["label"].value_counts().sort_index()
val_dist = val_df["label"].value_counts().sort_index()


dist_df = pd.DataFrame({
    "Class ID": train_dist.index,
    "Disease": [label_map[str(i)] for i in train_dist.index],
    "Train Count": train_dist.values,
    "Val Count": val_dist.values,
    "Train %": (train_dist / len(train_df) * 100).values,
    "Val %": (val_dist / len(val_df) * 100).values
})


print("Class distribution in Train vs Validation:")
display(dist_df)


plt.figure(figsize=(10,5))
bar_width = 0.35
idx = np.arange(len(train_dist))

plt.bar(idx - bar_width/2, train_dist.values, width=bar_width, label="Train")
plt.bar(idx + bar_width/2, val_dist.values, width=bar_width, label="Validation")

plt.xticks(idx, train_dist.index)
plt.xlabel("Class ID")
plt.ylabel("Sample Count")
plt.title("Class Distribution: Train vs Validation")
plt.legend()
plt.tight_layout()
plt.show()






IMG_SIZE = 224
BATCH_SIZE = 32



# Convert labels to strings 
# requires labels to be string when using sparse mode
train_df["label"] = train_df["label"].astype(str)
val_df["label"] = val_df["label"].astype(str)



train_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=20,
    zoom_range=0.20,
    shear_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.05,
    height_shift_range=0.05,

    brightness_range=[0.75, 1.25]

)


val_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)



train_flow = train_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_IMG_DIR,
    x_col="image_id",
    y_col="label",
    class_mode="sparse",       
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)



val_flow = val_gen.flow_from_dataframe(
    dataframe=val_df,
    directory=TRAIN_IMG_DIR,
    x_col="image_id",
    y_col="label",
    class_mode="sparse",      
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)





def build_model(img_size):
    inp = Input(shape=(img_size, img_size, 3))
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inp)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = LayerNormalization()(x)
    x = Dropout(0.45)(x)
    x = Dense(512, activation="swish",kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.45)(x)

    out = Dense(5, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    return model



model = build_model(IMG_SIZE)


lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=5e-4,
    decay_steps=1000
)




model.compile(
    optimizer = Adam(lr_schedule),
    loss=sparse_focal_loss(gamma=2.0), 
    metrics=["accuracy"]
)



model.summary()






callbacks_stage1 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ModelCheckpoint("stage1_best.h5", save_best_only=True),
    CSVLogger("stage1_log.csv")
]






history_1 = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=70,
    callbacks=callbacks_stage1,
    class_weight=class_weights     
)





for layer in model.layers:
    layer.trainable = True

for layer in model.layers[:200]:
    layer.trainable = False






model.compile(
    optimizer = Adam(lr_schedule),
    loss=sparse_focal_loss(gamma=2.0),   # 
    metrics=["accuracy"]
)


callbacks_stage2 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ModelCheckpoint("stage2_best.h5", save_best_only=True),
    CSVLogger("stage2_log.csv")
]


#from sklearn.utils.class_weight import compute_class_weight

# Proper class weights
#classes = np.unique(train_df["label"].astype(int))
#class_weights_raw = compute_class_weight("balanced", classes=classes, y=train_df["label"].astype(int))
#class_weights = dict(zip(classes, class_weights_raw))






history_2 = model.fit(
    train_flow,
    validation_data=val_flow,
    class_weight=class_weights,
    epochs=70,
    callbacks=callbacks_stage2
)





def plot_training(h1, h2):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss")
    plt.show()


plot_training(history_1, history_2)



acc_s1 = history_1.history["val_accuracy"][-1]
acc_s2 = history_2.history["val_accuracy"][-1]



print(f"Stage 1 Final Validation Accuracy: {acc_s1:.4f}")
print(f"Stage 2 Final Validation Accuracy: {acc_s2:.4f}")
print(f"Improvement: {(acc_s2 - acc_s1)*100:.2f}%")


print("___")


loss_s1 = history_1.history["val_loss"][-1]
loss_s2 = history_2.history["val_loss"][-1]


# Loss Comparison
print(f"Stage 1 Final Validation Loss: {loss_s1:.4f}")
print(f"Stage 2 Final Validation Loss: {loss_s2:.4f}")






def plot_stage_history(history, stage_name="Stage"):
    epochs = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, history.history["accuracy"], label="Train")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation")
    plt.title(f"{stage_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history.history["loss"], label="Train")
    plt.plot(epochs, history.history["val_loss"], label="Validation")
    plt.title(f"{stage_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_stage_history(history_1, stage_name="Stage 1 (Frozen Backbone)")



plot_stage_history(history_2, stage_name="Stage 2 (Fine-tuning)")






val_flow_eval = val_gen.flow_from_dataframe(
    val_df,
    directory=TRAIN_IMG_DIR,
    x_col="image_id",
    y_col="label",
    class_mode="sparse",
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False
)



pred_probs = model.predict(val_flow_eval)
y_pred = pred_probs.argmax(axis=1)
y_true = val_df["label"].values.astype(int)



cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()



print(classification_report(y_true, y_pred, digits=3))



bal_acc = balanced_accuracy_score(y_true, y_pred)
w_acc = np.average(y_true == y_pred, weights=None)


# gives you fair measurements despite the imbalance
print(f"Balanced Accuracy: {bal_acc:.4f}")


print(f"Weighted Accuracy (macro-weighted): {np.average(y_true == y_pred):.4f}")








from sklearn.metrics import classification_report

# get classification report 
target_names = [f"{i} - {label_map[str(i)]}" for i in sorted(np.unique(y_true))]
report_dict = classification_report(
    y_true, y_pred, output_dict=True, target_names=target_names, digits=4
)



# convert to DataFrame (only class rows)
per_class_metrics = pd.DataFrame(report_dict).T
per_class_metrics = per_class_metrics.iloc[:len(target_names)]  # remove 'accuracy', 'macro avg', ...
per_class_metrics = per_class_metrics[["precision", "recall", "f1-score", "support"]]



print("Per-class metrics:")
display(per_class_metrics)


# plot precision / recall / f1
plt.figure(figsize=(12,5))
x = np.arange(len(per_class_metrics))
w = 0.25

plt.bar(x - w, per_class_metrics["precision"], width=w, label="Precision")
plt.bar(x,       per_class_metrics["recall"],    width=w, label="Recall")
plt.bar(x + w, per_class_metrics["f1-score"], width=w, label="F1-score")

plt.xticks(x, per_class_metrics.index, rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Per-class Precision / Recall / F1")
plt.legend()
plt.tight_layout()
plt.show()


# Normalized confusion matrix
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(7,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
plt.title("Normalized Confusion Matrix (Row-wise)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()





val_df_eval = val_df.reset_index(drop=True)

mis_idx = np.where(y_true != y_pred)[0]
print("Number of misclassified samples:", len(mis_idx))


if len(mis_idx) > 0:
    n_show = min(16, len(mis_idx))
    chosen = np.random.choice(mis_idx, size=n_show, replace=False)

    plt.figure(figsize=(14,14))
    for i, idx in enumerate(chosen):
        row = val_df_eval.iloc[idx]
        img_path = os.path.join(TRAIN_IMG_DIR, row.image_id)
        img = plt.imread(img_path)

        true_lbl = int(y_true[idx])
        pred_lbl = int(y_pred[idx])

        plt.subplot(4,4,i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"True: {true_lbl} ({label_map[str(true_lbl)]})\n"
            f"Pred: {pred_lbl} ({label_map[str(pred_lbl)]})",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()
else:
    print("No misclassified samples to display (perfect accuracy on validation).")





def generate_grad_cam(model, img_path, layer_name="top_conv", alpha=0.4):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    arr = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(rgb.astype("float32"), axis=0)
    )


    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    cam = np.zeros(conv_out.shape[1:3], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_out[0,:,:,i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() + 1e-9)
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(rgb, 1-alpha, heatmap, alpha, 0)

    plt.figure(figsize=(5,5))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM")
    plt.show()



# example 
example = val_df.sample(1).image_id.values[0]
generate_grad_cam(model, os.path.join(TRAIN_IMG_DIR, example))





#  Grad-CAM for Multiple Validation Samples

def generate_grad_cam_multi(model, df, num_images=12, last_conv="top_conv", alpha=0.4):
    """
    Generates Grad-CAM heatmaps for multiple images.
    df: DataFrame containing image_id and label columns (use validation dataframe)
    num_images: number of samples to visualize
    """

    sample_df = df.sample(num_images, random_state=42).reset_index(drop=True)

    plt.figure(figsize=(14, 14))

    for i in range(num_images):
        img_id = sample_df.loc[i, "image_id"]
        true_lbl = sample_df.loc[i, "label"]

        img_path = os.path.join(TRAIN_IMG_DIR, img_id)

        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
       # arr = np.expand_dims(rgb.astype("float32") / 255.0, axis=0)
        arr = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(rgb.astype("float32"), axis=0)
)

        # Build Grad-CAM model
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(arr)
            pred_idx = tf.argmax(preds[0])
            loss = preds[:, pred_idx]

        grads = tape.gradient(loss, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = np.zeros(conv_out.shape[1:3], dtype=np.float32)

        for j, w in enumerate(weights):
            cam += w * conv_out[0, :, :, j]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam / (cam.max() + 1e-9)
        cam = np.uint8(255 * cam)

        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb, 1 - alpha, heatmap, alpha, 0)

        # Plot
        plt.subplot(4, 3, i + 1)
        plt.imshow(overlay)
        pred_lbl = int(pred_idx.numpy())
        title_str = f"True: {true_lbl} ({label_map[str(true_lbl)]})\nPred: {pred_lbl} ({label_map[str(pred_lbl)]})"
        plt.title(title_str, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()





# Multi-GradCAM
generate_grad_cam_multi(model, val_df, num_images=12, last_conv="top_conv", alpha=0.4)






# FINAL_MODEL_PATH = "cassava_efficientnetb0_final.keras"
model.save("cassava_final_model.keras")
print("is save model ")





sample_sub = pd.read_csv(SUBMISSION_TEMPLATE)
test_df = pd.DataFrame({"image_id": sample_sub["image_id"].values})


test_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_flow = test_gen.flow_from_dataframe(
    test_df,
    directory=TEST_IMG_DIR,
    x_col="image_id",
    y_col=None,
    class_mode=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
    batch_size=32
)


test_pred = model.predict(test_flow).argmax(axis=1)
submission = test_df.copy()
submission["label"] = test_pred


submission.to_csv("submission.csv", index=False)





