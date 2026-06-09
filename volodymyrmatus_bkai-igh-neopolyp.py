!pip install keras-cv gdown


import tensorflow as tf
print(tf.__version__)

cur_dir = %env PWD
is_kaggle = cur_dir == '/kaggle/working'
try:
    tpu = 'local' if is_kaggle else ''    
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print('Using TPU')
    # print("TPU devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)
except:
    GPU_list = tf.config.list_physical_devices('GPU')
    GPU_count = len(GPU_list) if type(GPU_list) is list else 0
    if GPU_count == 0:
        print('Using CPU')
        strategy = tf.distribute.get_strategy()
    else:
        print('Using GPU')
        # print("GPU devices: ", tf.config.list_logical_devices('GPU'))
        if len(GPU_list) == 1:
            strategy = tf.distribute.get_strategy()
        else:
            # strategy = tf.distribute.MirroredStrategy() # Швидше, але використовує більше пам'яті GPU
            strategy = tf.distribute.experimental.CentralStorageStrategy()

print('DEVICES AVAILABLE:', strategy.num_replicas_in_sync)


import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import keras_cv
from tensorflow.keras.metrics import Metric
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display, Image
import pandas as pd
from tensorflow.keras import mixed_precision
import albumentations as A

# print('GPU:', tf.config.list_physical_devices('GPU'))


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


X_TRAIN_DIR = "/kaggle/input/bkai-igh-neopolyp/train/train"
Y_TRAIN_DIR = "/kaggle/input/bkai-igh-neopolyp/train_gt/train_gt"


IMAGE_SIZE = (512, 512)
NUM_CLASSES = 3
seed = 42
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 4 * strategy.num_replicas_in_sync


def pad_and_resize_image(image, target_size=(512, 512)):
    # image: tf.Tensor, [H,W,3], float32 [0,1] or uint8 [0,255]
    original_shape = tf.shape(image)
    height, width = original_shape[0], original_shape[1]
    max_dim = tf.maximum(height, width)
    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) // 2
    image = tf.image.pad_to_bounding_box(image, pad_height, pad_width, max_dim, max_dim)
    image = tf.image.resize(image, target_size)
    return image

def pad_and_resize_mask(mask, target_size=(512, 512)):
    original_shape = tf.shape(mask)
    height, width = original_shape[0], original_shape[1]
    max_dim = tf.maximum(height, width)
    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) // 2
    mask = tf.image.pad_to_bounding_box(mask, pad_height, pad_width, max_dim, max_dim)
    mask = tf.image.resize(mask, target_size, method='nearest')
    return mask


def rgb_mask_to_class(mask, tolerance=40):
    mask = tf.cast(mask, tf.uint8)
    def color_match(channel, value, tolerance=tolerance):
        return tf.abs(tf.cast(mask[..., channel], tf.int32) - value) < tolerance
    # Red → class 0 (neoplastic)
    neoplastic = tf.reduce_all(
        tf.stack([color_match(0, 255), color_match(1, 0), color_match(2, 0)], axis=-1), axis=-1)
    # Green → class 1 (non-neoplastic)
    non_neoplastic = tf.reduce_all(
        tf.stack([color_match(0, 0), color_match(1, 255), color_match(2, 0)], axis=-1), axis=-1)
    # Black → class 2 (background)
    background = tf.reduce_all(
        tf.stack([color_match(0, 0), color_match(1, 0), color_match(2, 0)], axis=-1), axis=-1)
    mask_out = tf.where(neoplastic, 0, tf.where(non_neoplastic, 1, tf.where(background, 2, 2)))
    return tf.cast(mask_out, tf.int32)



def load_image_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=3)
    image = pad_and_resize_image(image, target_size=IMAGE_SIZE)
    mask = pad_and_resize_mask(mask, target_size=IMAGE_SIZE)
    mask = rgb_mask_to_class(mask)
    image = tf.cast(image, tf.float32) / 255.0  # Ensure [0,1]
    return image, mask


def build_dataset(image_paths, mask_paths, augment_fn=None, batch_size=BATCH_SIZE, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image_mask, num_parallel_calls=AUTOTUNE)
    if augment_fn is not None:
        dataset = dataset.map(augment_fn, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def show_image_mask_pairs(
    dataset, 
    batch_idx=0, 
    cols=4, 
    class_cmap='jet'
):
    """
    Display a batch of image/mask pairs from a tf.data.Dataset, 
    with each mask under its image in a vertical column layout.

    Args:
        dataset: tf.data.Dataset
        batch_idx: int, batch number to display
        cols: int, number of columns (pairs per row)
        class_cmap: str, colormap for masks
    """
    # Fetch desired batch
    batch = None
    for i, (images, masks) in enumerate(dataset):
        if i == batch_idx:
            batch = (images, masks)
            break
    if batch is None:
        print(f"Batch {batch_idx} does not exist.")
        return

    images, masks = batch
    n = min(cols, images.shape[0])
    plt.figure(figsize=(cols * 4, 8))
    for i in range(n):
        # Image on top
        plt.subplot(2, n, i + 1)
        img = images[i].numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        plt.imshow(img)
        plt.title(f"Image {i}")
        plt.axis('off')
        # Mask below
        plt.subplot(2, n, n + i + 1)
        plt.imshow(masks[i].numpy(), cmap=class_cmap, vmin=0, vmax=2)
        plt.title(f"Mask {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


image_paths = sorted(glob(os.path.join(X_TRAIN_DIR, "*.jpeg"), recursive=True))
mask_paths = sorted(glob(os.path.join(Y_TRAIN_DIR, "*.jpeg"), recursive=True))
assert len(image_paths) == len(mask_paths), "Image and mask counts do not match"
len(image_paths)


train_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]


train_imgs = [image_paths[i] for i in train_index]
train_masks = [mask_paths[i] for i in train_index]


albumentations_transform = A.Compose([
    # Spatial transforms
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.25),
    A.Rotate(limit=15, interpolation=0, border_mode=0, value=0, mask_value=2, p=0.5),

    # Elastic/affine warps (simulate camera movement)
    A.ElasticTransform(alpha=50, sigma=4, alpha_affine=20, interpolation=1, border_mode=0, value=0, mask_value=2, p=0.12),
    A.GridDistortion(num_steps=5, distort_limit=0.05, interpolation=1, border_mode=0, value=0, mask_value=2, p=0.10),

    # Random resized cropping & scaling (simulate zooms)
    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.1),

    # Intensity/contrast augmentation
    A.RandomBrightnessContrast(brightness_limit=0.16, contrast_limit=0.16, p=0.32),
    A.RandomGamma(gamma_limit=(85, 120), p=0.18),
    A.CLAHE(clip_limit=2, tile_grid_size=(8,8), p=0.10),

    # Blur, noise, sharpening (simulate endoscope variability)
    A.GaussianBlur(blur_limit=(3, 7), p=0.09),
    A.GaussNoise(var_limit=(2, 10), p=0.12),
    A.MotionBlur(blur_limit=5, p=0.05),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.07),

    # Cutout/hole (simulate occlusions, debris, bubbles)
    A.CoarseDropout(max_holes=4, max_height=64, max_width=64, min_holes=1, min_height=24, min_width=24, fill_value=0, mask_fill_value=2, p=0.11),
],
    additional_targets={'mask': 'mask'}
)

def albumentations_augment_numpy(image, mask):
    # Convert EagerTensors to NumPy arrays!
    image = image.numpy() if hasattr(image, 'numpy') else image
    mask = mask.numpy() if hasattr(mask, 'numpy') else mask

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    augmented = albumentations_transform(image=image, mask=mask)
    aug_image = augmented['image'].astype(np.float32) / 255.0
    aug_mask = augmented['mask'].astype(np.int32)
    return aug_image, aug_mask

def albumentations_augment(image, mask):
    aug_img, aug_mask = tf.py_function(
        func=albumentations_augment_numpy,
        inp=[image, mask],
        Tout=[tf.float32, tf.int32]
    )
    # Set shapes so tf.data knows them
    aug_img.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    aug_mask.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1]])
    return aug_img, aug_mask


train_ds = build_dataset(train_imgs, train_masks, augment_fn=albumentations_augment, batch_size=BATCH_SIZE, shuffle=True)


show_image_mask_pairs(train_ds, batch_idx=1, cols=4)


class MeanPolypDice(Metric):
    """
    Computes the mean Dice coefficient for neoplastic and non-neoplastic classes
    (competition metric), averaged across the batch.
    Assumes y_true: (batch, H, W), ints in [0,1,2]
           y_pred: (batch, H, W, num_classes), softmax probabilities.
    """
    def __init__(self, smooth=1e-5, name="mean_polyp_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.dice_total = self.add_weight(name="dice_total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)  # (batch, H, W), int
        
        dice_scores = []
        for class_id in [0, 1]:  # Only polyp classes
            y_true_c = tf.cast(tf.equal(y_true, class_id), tf.float32)
            y_pred_c = tf.cast(tf.equal(y_pred, class_id), tf.float32)
            
            # FIX: Use tf.shape not .shape
            batch_dim = tf.shape(y_true_c)[0]
            y_true_c_f = tf.reshape(y_true_c, [batch_dim, -1])
            y_pred_c_f = tf.reshape(y_pred_c, [batch_dim, -1])
            
            intersection = tf.reduce_sum(y_true_c_f * y_pred_c_f, axis=1)
            union = tf.reduce_sum(y_true_c_f, axis=1) + tf.reduce_sum(y_pred_c_f, axis=1)
            
            dice = tf.where(
                union > 0,
                (2. * intersection + self.smooth) / (union + self.smooth),
                tf.ones_like(union)
            )
            dice_scores.append(dice)   # (batch,)

        mean_dice = tf.reduce_mean(tf.stack(dice_scores, axis=1), axis=1)  # mean over classes, shape (batch,)
        self.dice_total.assign_add(tf.reduce_sum(mean_dice))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        
    def result(self):
        return tf.math.divide_no_nan(self.dice_total, self.count)
        
    def reset_states(self):
        self.dice_total.assign(0.0)
        self.count.assign(0.0)


def multiclass_dice_loss(y_true, y_pred, num_classes=3, smooth=1e-5):
    # y_true: (batch, H, W), int32; y_pred: (batch, H, W, num_classes), softmax
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    intersection = tf.reduce_sum(y_true_onehot * y_pred, axis=[1,2])
    union = tf.reduce_sum(y_true_onehot + y_pred, axis=[1,2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)

def combo_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    dice = multiclass_dice_loss(y_true, y_pred)
    return ce + dice


kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []


for fold, (train_index, val_index) in enumerate(kf.split(image_paths)):
    print(f"Fold {fold + 1}")

    train_imgs = [image_paths[i] for i in train_index]
    train_masks = [mask_paths[i] for i in train_index]
    val_imgs = [image_paths[i] for i in val_index]
    val_masks = [mask_paths[i] for i in val_index]

    train_ds = build_dataset(train_imgs, train_masks, augment_fn=albumentations_augment, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = build_dataset(val_imgs, val_masks, augment_fn=None, batch_size=BATCH_SIZE, shuffle=False)

    dice_metric = MeanPolypDice(name="dice_coef")
    
    with strategy.scope():
        model = keras_cv.models.DeepLabV3Plus.from_preset(
            "yolo_v8_m_backbone_coco", load_weights=True,
            num_classes=NUM_CLASSES, input_shape=[*IMAGE_SIZE, 3]
        )
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.00001),
            loss=combo_loss,
            metrics=[keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False, name="mean_iou"),
                     dice_metric],
        )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', patience=10, restore_best_weights=True, mode='max',start_from_epoch=5)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=350,
                        callbacks=[early_stop, reduce_lr])
    
    best_val_dice = max(history.history['val_dice_coef'])
    fold_metrics.append(best_val_dice)
 
    model.save(f"model_fold_{fold+1}.keras")


fold_metrics = np.array(fold_metrics)
print("Mean IoU across 5 folds: ", fold_metrics.mean())
print("Std deviation of IoU: ", fold_metrics.std())


# all_histories_df = pd.DataFrame()

# for fold_idx, hist in enumerate(global_history):
#     fold_df = pd.DataFrame(hist.history)
#     fold_df["epoch"] = fold_df.index
#     fold_df["fold"] = fold_idx + 1
#     all_histories_df = pd.concat([all_histories_df, fold_df], ignore_index=True)


# def display_predictions(dataset, model, num_samples=3):
#     plt.figure(figsize=(10, num_samples * 4))

#     for images, masks in dataset.take(1):
#         preds = model.predict(images, verbose=0)
#         preds = tf.argmax(preds, axis=-1).numpy().astype(np.uint8)
#         gt_masks = masks.numpy().astype(np.uint8)

#         for j in range(num_samples):
#             img = images[j].numpy()
#             if img.dtype == np.float16:
#                 img = img.astype(np.float32)
#             # Optionally scale to 0-255 for display
#             # img = (img * 255).astype(np.uint8)

#             plt.subplot(num_samples, 3, j * 3 + 1)
#             plt.imshow(img)
#             plt.axis("off")

#             plt.subplot(num_samples, 3, j * 3 + 2)
#             plt.imshow(gt_masks[j], cmap='jet', vmin=0, vmax=2)
#             plt.axis("off")

#             plt.subplot(num_samples, 3, j * 3 + 3)
#             plt.imshow(preds[j], cmap='jet', vmin=0, vmax=2)
#             plt.axis("off")

#     plt.suptitle("Input images     Ground truth masks   Predicted masks", fontsize=18, fontweight='bold')
#     plt.subplots_adjust(top=0.93, hspace=0.1, wspace=0.05)
#     plt.tight_layout(rect=[0, 0, 1, 1])
#     plt.show()


# display_predictions(train_ds, model, num_samples=3)




