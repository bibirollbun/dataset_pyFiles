import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from glob import glob
from tqdm.notebook import tqdm
import joblib
import matplotlib.pyplot as plt 
import os
os.environ["KERAS_BACKEND"] = "tensorflow" 
import keras_cv
import keras
import cv2
import pandas as pd
import numpy as np
from keras import ops
import tensorflow as tf
from keras import ops
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold


class CFG:
    verbose = 1
    seed = 42
    pos_ratio = 0.25
    preset = "efficientnetv2_b2_imagenet"
    image_size = [128, 128]
    epochs = 2
    batch_size = 256
    lr_mode = "cos"
    class_names = ['target']
    num_classes = 1

keras.utils.set_random_seed(CFG.seed)


print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasCV:", keras_cv.__version__)


keras.utils.set_random_seed(42)


BASE_PATH = "/kaggle/input/isic-2024-challenge"




training_validation_hdf5 = h5py.File(f"{BASE_PATH}/train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"{BASE_PATH}/test-image.hdf5", 'r')

print("HDF5 files loaded")
print(f"Number of training images: {len(training_validation_hdf5.keys())}")


df = pd.read_csv(f'{BASE_PATH}/train-metadata.csv', low_memory=False)
df = df.ffill()

print(f"Dataset shape: {df.shape}")
print(f"Positive cases: {df['target'].sum()}")
print(f"Negative cases: {len(df) - df['target'].sum()}")


#df= df.ffill()


CATEGORICAL_COLUMNS = [
    "sex", 
    "anatom_site_general",
    "tbp_tile_type",
    "tbp_lv_location"
]

NUMERIC_COLUMNS = [
    "age_approx", 
    "tbp_lv_nevi_confidence", 
    "clin_size_long_diam_mm",
    "tbp_lv_areaMM2", 
    "tbp_lv_area_perim_ratio", 
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaLBnorm", 
    "tbp_lv_minorAxisMM"
]

FEAT_COLS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

print(f"Total features: {len(FEAT_COLS)}")
print(f"Features: {FEAT_COLS}")


#display(df.head(5))


def build_decoder(with_labels=True, target_size=CFG.image_size):
    def decode_image(inp):
        file_bytes = inp["images"]
        image = tf.io.decode_jpeg(file_bytes)
        image = tf.image.resize(image, size=target_size, method="area")
        image = tf.cast(image, tf.float32)
        image /= 255.0
        image = tf.reshape(image, [*target_size, 3])
        inp["images"] = image
        return inp

    def decode_label(label, num_classes):
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [num_classes])
        return label

    def decode_with_labels(inp, label=None):
        inp = decode_image(inp)
        label = decode_label(label, CFG.num_classes)
        return (inp, label)

    return decode_with_labels if with_labels else decode_image


def build_augmenter():
    aug_layers = [
        keras_cv.layers.RandomCutout(height_factor=(0.02, 0.06), width_factor=(0.02, 0.06)),
        keras_cv.layers.RandomFlip(mode="horizontal"),
    ]
    aug_layers = [keras_cv.layers.RandomApply(x, rate=0.5) for x in aug_layers]
    augmenter = keras_cv.layers.Augmenter(aug_layers)

    def augment(inp, label):
        images = inp["images"]
        aug_data = {"images": images}
        aug_data = augmenter(aug_data)
        inp["images"] = aug_data["images"]
        return inp, label
    
    return augment


#display(df.shape)


#testing_df = pd.read_csv(f'{BASE_PATH}/test-metadata.csv', low_memory=False)
#testing_df = testing_df.ffill()
#display(testing_df.head(2))


#display(df['target'].value_counts(normalize=True) * 100)


def build_balanced_dataset(
    df,
    hdf5,
    feature_cols,
    batch_size=256,
    pos_ratio=0.25,
    decode_fn=None,
    augment_fn=None,
    augment=False,
):
    if decode_fn is None:
        decode_fn = build_decoder(with_labels=True)
    
    if augment_fn is None:
        augment_fn = build_augmenter()
    
    AUTO = tf.data.experimental.AUTOTUNE
    
    pos_per_batch = int(batch_size * pos_ratio)
    neg_per_batch = batch_size - pos_per_batch
    
    pos_df = df[df['target'] == 1].reset_index(drop=True)
    neg_df = df[df['target'] == 0].reset_index(drop=True)
    
    print(f"Positives: {len(pos_df)} | Negatives: {len(neg_df)}")
    print(f"Per batch: {pos_per_batch} positive + {neg_per_batch} negative")
    
    pos_images = []
    for isic_id in tqdm(pos_df['isic_id'].values, desc="Loading Positives"):
        pos_images.append(hdf5[isic_id][()])
    
    neg_images = []
    for isic_id in tqdm(neg_df['isic_id'].values, desc="Loading Negatives"):
        neg_images.append(hdf5[isic_id][()])
    
    pos_features = {col: pos_df[col].values for col in feature_cols}
    pos_labels = pos_df['target'].values
    
    neg_features = {col: neg_df[col].values for col in feature_cols}
    neg_labels = neg_df['target'].values
    
    pos_ds = tf.data.Dataset.from_tensor_slices((
        {"images": pos_images, "features": pos_features},
        pos_labels
    ))
    pos_ds = pos_ds.map(decode_fn, num_parallel_calls=AUTO)
    pos_ds = pos_ds.shuffle(len(pos_df))
    pos_ds = pos_ds.repeat()
    pos_ds = pos_ds.batch(pos_per_batch)
    
    neg_ds = tf.data.Dataset.from_tensor_slices((
        {"images": neg_images, "features": neg_features},
        neg_labels
    ))
    neg_ds = neg_ds.map(decode_fn, num_parallel_calls=AUTO)
    neg_ds = neg_ds.shuffle(min(len(neg_df), 10000))
    neg_ds = neg_ds.repeat()
    neg_ds = neg_ds.batch(neg_per_batch)
    
    balanced_ds = tf.data.Dataset.zip((pos_ds, neg_ds))
    
    def combine_batches(pos_batch, neg_batch):
        pos_inp, pos_lbl = pos_batch
        neg_inp, neg_lbl = neg_batch
        
        combined_images = tf.concat([pos_inp["images"], neg_inp["images"]], axis=0)
        
        combined_features = {}
        for key in pos_inp["features"].keys():
            combined_features[key] = tf.concat(
                [pos_inp["features"][key], neg_inp["features"][key]], 
                axis=0
            )
        
        combined_labels = tf.concat([pos_lbl, neg_lbl], axis=0)
        
        batch_size = tf.shape(combined_images)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        
        shuffled_images = tf.gather(combined_images, indices)
        shuffled_features = {k: tf.gather(v, indices) for k, v in combined_features.items()}
        shuffled_labels = tf.gather(combined_labels, indices)
        
        return ({"images": shuffled_images, "features": shuffled_features}, shuffled_labels)
    
    balanced_ds = balanced_ds.map(combine_batches, num_parallel_calls=AUTO)
    
    if augment:
        balanced_ds = balanced_ds.map(augment_fn, num_parallel_calls=AUTO)
    
    balanced_ds = balanced_ds.prefetch(AUTO)
    
    return balanced_ds





check_ds = build_balanced_dataset(
    df=df, 
    hdf5=training_validation_hdf5, 
    feature_cols=FEAT_COLS, 
    batch_size=256, 
    pos_ratio=0.25,
    augment=False
)

images_and_feats, labels = next(iter(check_ds))

total_images = len(labels)
positive_cases = np.sum(labels.numpy())
negative_cases = total_images - positive_cases

print(f"Total images in batch: {total_images}")
print(f"Cancer cases (Positives): {int(positive_cases)}")
print(f"Safe cases (Negatives): {int(negative_cases)}")
print(f"Actual Ratio: {positive_cases/total_images:.2%}")


class FocalLoss(tf.keras.losses.Loss):
    
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        loss = -alpha_t * focal_weight * tf.math.log(p_t)
        
        return tf.reduce_mean(loss)

print("now we defined focal loss")


focal_loss = FocalLoss(gamma=2.0, alpha=0.25)

y_true_test = tf.constant([[1], [0], [1], [0]], dtype=tf.float32)
y_pred_test = tf.constant([[0.9], [0.1], [0.3], [0.8]], dtype=tf.float32)

loss_value = focal_loss(y_true_test, y_pred_test)

print("Focal Loss Verification ")
print(f"Test labels:      {y_true_test.numpy().flatten()}")
print(f"Test predictions: {y_pred_test.numpy().flatten()}")
print(f"Focal Loss value: {loss_value.numpy():.4f}")

if loss_value.numpy() > 0:
    print("\n Focal Loss is working ")
else:
    print("\n Focal Loss is not working :( ")


def build_augmenter():
    
    def augment(inp, label):
        images = inp["images"]
        
        images = tf.image.random_flip_left_right(images)
        
        images = tf.image.random_flip_up_down(images)
        
        images = tf.image.random_brightness(images, max_delta=0.2)
        
        images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
        
        images = tf.image.random_saturation(images, lower=0.8, upper=1.2)
        
        images = tf.image.random_hue(images, max_delta=0.02)
        
        images = tf.clip_by_value(images, 0.0, 1.0)
        
        inp["images"] = images
        
        return inp, label
    
    return augment

print(" we have Augmenter :D ")


augment_fn = build_augmenter()

test_image = tf.random.uniform([4, 128, 128, 3], 0, 1)
test_features = {"age": tf.constant([50, 60, 70, 80])}
test_labels = tf.constant([[1], [0], [1], [0]])

test_input = {"images": test_image, "features": test_features}

augmented_input, augmented_labels = augment_fn(test_input, test_labels)


print(f"Input image shape:  {test_image.shape}")
print(f"Output image shape: {augmented_input['images'].shape}")
print(f"Labels preserved:   {tf.reduce_all(test_labels == augmented_labels).numpy()}")

original_mean = tf.reduce_mean(test_image).numpy()
augmented_mean = tf.reduce_mean(augmented_input['images']).numpy()

print(f"\nOriginal mean pixel value:  {original_mean:.4f}")
print(f"Augmented mean pixel value: {augmented_mean:.4f}")

if augmented_input['images'].shape == test_image.shape:
    print("\n  Augmentation is ok ")
else:
    print("\n Shape mismatch :( ")


def visualize_augmentation(balanced_ds, num_samples=4):
    
    augment_fn = build_augmenter()
    
    for inputs, labels in balanced_ds.take(1):
        
        original_images = inputs["images"][:num_samples]
        augmented_inputs, _ = augment_fn(
            {"images": original_images, "features": inputs["features"]}, 
            labels[:num_samples]
        )
        augmented_images = augmented_inputs["images"]
        
        fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
        
        for i in range(num_samples):
            
            orig_img = original_images[i].numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f"Original\nLabel: {int(labels[i].numpy()[0])}")
            axes[0, i].axis('off')
            
            aug_img = augmented_images[i].numpy()
            aug_img = (aug_img - aug_img.min()) / (aug_img.max() - aug_img.min() + 1e-8)
            axes[1, i].imshow(aug_img)
            axes[1, i].set_title("Augmented")
            axes[1, i].axis('off')
        
        plt.suptitle("Original vs Augmented Images", fontsize=16)
        plt.tight_layout()
        plt.show()
        break





df = df.reset_index(drop=True)

df["fold"] = -1

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)

for fold_num, (train_idx, valid_idx) in enumerate(sgkf.split(df, y=df.target, groups=df.patient_id)):
    df.loc[valid_idx, "fold"] = fold_num

training_df = df[df["fold"] != 0].reset_index(drop=True)
validation_df = df[df["fold"] == 0].reset_index(drop=True)

print(f"Training samples:   {len(training_df)}")
print(f"Validation samples: {len(validation_df)}")



print("data split verif")


print(f"   Training:   {len(training_df)} samples ({len(training_df)/len(df)*100:.1f}%)")
print(f"   Validation: {len(validation_df)} samples ({len(validation_df)/len(df)*100:.1f}%)")

train_pos = training_df['target'].sum()
train_neg = len(training_df) - train_pos
valid_pos = validation_df['target'].sum()
valid_neg = len(validation_df) - valid_pos

print(f"\n2. check for class dist ")
print(f"   Training   - Positive: {train_pos} ({train_pos/len(training_df)*100:.3f}%)")
print(f"   Training   - Negative: {train_neg} ({train_neg/len(training_df)*100:.3f}%)")
print(f"   Validation - Positive: {valid_pos} ({valid_pos/len(validation_df)*100:.3f}%)")
print(f"   Validation - Negative: {valid_neg} ({valid_neg/len(validation_df)*100:.3f}%)")

train_patients = set(training_df['patient_id'].unique())
valid_patients = set(validation_df['patient_id'].unique())
overlap = train_patients.intersection(valid_patients)

print(f"\n3. check for overlap in patients :")
print(f"   Training patients:   {len(train_patients)}")
print(f"   Validation patients: {len(valid_patients)}")
print(f"   Overlapping patients: {len(overlap)}")

if len(overlap) == 0:
    print("\n No overlap haha ")
else:
    print(f"\n We have an error: {len(overlap)} patients appear in both set")


CATEGORICAL_COLUMNS = [
    "sex",
    "anatom_site_general",
    "tbp_tile_type",
    "tbp_lv_location",
]

NUMERIC_COLUMNS = [
    "age_approx",
    "tbp_lv_nevi_confidence",
    "clin_size_long_diam_mm",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_minorAxisMM",
]

FEAT_COLS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

print(f"Categorical features: {len(CATEGORICAL_COLUMNS)}")
print(f"Numerical features:   {len(NUMERIC_COLUMNS)}")
print(f"Total features:       {len(FEAT_COLS)}")


feature_space = keras.utils.FeatureSpace(
    features={
        "sex": "string_categorical",
        "anatom_site_general": "string_categorical",
        "tbp_tile_type": "string_categorical",
        "tbp_lv_location": "string_categorical",
        "age_approx": "float_normalized",
        "tbp_lv_nevi_confidence": "float_normalized",
        "clin_size_long_diam_mm": "float_normalized",
        "tbp_lv_areaMM2": "float_normalized",
        "tbp_lv_area_perim_ratio": "float_normalized",
        "tbp_lv_color_std_mean": "float_normalized",
        "tbp_lv_deltaLBnorm": "float_normalized",
        "tbp_lv_minorAxisMM": "float_normalized",
    },
    output_mode="concat",
)

print("now we have FeatureSpace")


adapt_data = {col: training_df[col].values for col in FEAT_COLS}

adapt_ds = tf.data.Dataset.from_tensor_slices(adapt_data)

feature_space.adapt(adapt_ds)

print("FeatureSpace adapted")


print("FEATURESPACE VERIFICATION")


sample_data = {col: training_df[col].values[:5] for col in FEAT_COLS}

sample_ds = tf.data.Dataset.from_tensor_slices(sample_data).batch(5)

for batch in sample_ds:
    processed = feature_space(batch)
    break

print(f"\n1. INPUT (Raw features):")
print(f"   Number of features: {len(FEAT_COLS)}")
for col in FEAT_COLS[:3]:
    print(f"   {col}: {sample_data[col][:3]}")
print("   ...")

print(f"\n2. OUTPUT (Processed features):")
print(f"   Shape: {processed.shape}")
print(f"   First sample: {processed[0].numpy()[:10]}...")

print(f"\n3. SIZE CHANGE:")
print(f"   Input:  {len(FEAT_COLS)} raw features")
print(f"   Output: {processed.shape[1]} processed features")

if processed.shape[1] > len(FEAT_COLS):
    print(f"\n Features expanded from {len(FEAT_COLS)} to {processed.shape[1]}")
else:
    print("\n error :( ")


def build_model():
    
    image_input = keras.Input(shape=(128, 128, 3), name="images")
    
    feature_input = keras.Input(shape=(feature_space.get_encoded_features().shape[1],), name="features")
    
    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(CFG.preset)
    x1 = backbone(image_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    
    x2 = keras.layers.Dense(96, activation="selu")(feature_input)
    x2 = keras.layers.Dense(128, activation="selu")(x2)
    x2 = keras.layers.Dropout(0.1)(x2)
    
    concat = keras.layers.Concatenate()([x1, x2])
    
    output = keras.layers.Dense(1, activation="sigmoid")(concat)
    
    model = keras.Model(inputs={"images": image_input, "features": feature_input}, outputs=output)
    
    return model

model = build_model()
print(" now we have the model ")


focal_loss = FocalLoss(gamma=2.0, alpha=0.25)

auc = keras.metrics.AUC(name='auc')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss,
    metrics=[auc],
)

print(" model compiled haha ")


model.summary()


test_images = tf.random.uniform([2, 128, 128, 3], 0, 1)
test_features = tf.random.uniform([2, feature_space.get_encoded_features().shape[1]], 0, 1)

test_output = model.predict({"images": test_images, "features": test_features}, verbose=0)

print(f"\n1. INPUT SHAPES:")
print(f"   Images:   {test_images.shape}")
print(f"   Features: {test_features.shape}")

print(f"\n2. OUTPUT SHAPE:")
print(f"   Predictions: {test_output.shape}")

print(f"\n3. OUTPUT VALUES:")
print(f"   Sample predictions: {test_output.flatten()}")

print(f"\n4. OUTPUT RANGE:")
print(f"   Min: {test_output.min():.4f}")
print(f"   Max: {test_output.max():.4f}")

if test_output.min() >= 0 and test_output.max() <= 1:
    print("\n our model outputs valid probabilities (0 to 1)")
else:
    print("\n Output values are outside 0-1 range. that's bad :( ")


def build_training_dataset_with_features(
    df,
    hdf5,
    feature_cols,
    feature_space,
    batch_size=256,
    pos_ratio=0.25,
):
    AUTO = tf.data.experimental.AUTOTUNE
    
    pos_per_batch = int(batch_size * pos_ratio)
    neg_per_batch = batch_size - pos_per_batch
    
    pos_df = df[df['target'] == 1].reset_index(drop=True)
    neg_df = df[df['target'] == 0].reset_index(drop=True)
    
    print(f"Positives: {len(pos_df)} | Negatives: {len(neg_df)}")
    print(f"Per batch: {pos_per_batch} positive + {neg_per_batch} negative")
    
    print("Loading positive images...")
    pos_images = []
    for isic_id in tqdm(pos_df['isic_id'].values, desc="Positives"):
        pos_images.append(hdf5[isic_id][()])
    
    print("Loading negative images...")
    neg_images = []
    for isic_id in tqdm(neg_df['isic_id'].values, desc="Negatives"):
        neg_images.append(hdf5[isic_id][()])
    
    pos_features_dict = {col: pos_df[col].values for col in feature_cols}
    pos_labels = pos_df['target'].values
    
    neg_features_dict = {col: neg_df[col].values for col in feature_cols}
    neg_labels = neg_df['target'].values
    
    print("Processing positive features with FeatureSpace...")
    pos_features_ds = tf.data.Dataset.from_tensor_slices(pos_features_dict)
    pos_features_processed = []
    for batch in pos_features_ds.batch(1000):
        processed = feature_space(batch)
        pos_features_processed.append(processed.numpy())
    pos_features_tensor = np.concatenate(pos_features_processed, axis=0)
    
    print("Processing negative features with FeatureSpace...")
    neg_features_ds = tf.data.Dataset.from_tensor_slices(neg_features_dict)
    neg_features_processed = []
    for batch in neg_features_ds.batch(1000):
        processed = feature_space(batch)
        neg_features_processed.append(processed.numpy())
    neg_features_tensor = np.concatenate(neg_features_processed, axis=0)
    
    print(f"Processed features shape: {pos_features_tensor.shape[1]} dimensions")
    
    def decode_image(image_bytes):
        image = tf.io.decode_jpeg(image_bytes)
        image = tf.image.resize(image, CFG.image_size, method="area")
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    def augment_image(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
    
    pos_ds = tf.data.Dataset.from_tensor_slices((pos_images, pos_features_tensor, pos_labels))
    pos_ds = pos_ds.shuffle(len(pos_df))
    pos_ds = pos_ds.repeat()
    pos_ds = pos_ds.batch(pos_per_batch)
    
    neg_ds = tf.data.Dataset.from_tensor_slices((neg_images, neg_features_tensor, neg_labels))
    neg_ds = neg_ds.shuffle(min(len(neg_df), 10000))
    neg_ds = neg_ds.repeat()
    neg_ds = neg_ds.batch(neg_per_batch)
    
    balanced_ds = tf.data.Dataset.zip((pos_ds, neg_ds))
    
    def combine_and_process(pos_batch, neg_batch):
        pos_img, pos_feat, pos_lbl = pos_batch
        neg_img, neg_feat, neg_lbl = neg_batch
        
        all_img = tf.concat([pos_img, neg_img], axis=0)
        all_feat = tf.concat([pos_feat, neg_feat], axis=0)
        all_lbl = tf.concat([pos_lbl, neg_lbl], axis=0)
        
        batch_size = tf.shape(all_img)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        
        all_img = tf.gather(all_img, indices)
        all_feat = tf.gather(all_feat, indices)
        all_lbl = tf.gather(all_lbl, indices)
        
        all_lbl = tf.cast(all_lbl, tf.float32)
        all_lbl = tf.reshape(all_lbl, [-1, 1])
        
        return (all_img, all_feat, all_lbl)
    
    balanced_ds = balanced_ds.map(combine_and_process, num_parallel_calls=AUTO)
    
    def decode_batch(images, features, labels):
        decoded_images = tf.map_fn(decode_image, images, fn_output_signature=tf.TensorSpec([128, 128, 3], tf.float32))
        return (decoded_images, features, labels)
    
    balanced_ds = balanced_ds.map(decode_batch, num_parallel_calls=AUTO)
    
    def augment_batch(images, features, labels):
        augmented = tf.map_fn(augment_image, images)
        return (augmented, features, labels)
    
    balanced_ds = balanced_ds.map(augment_batch, num_parallel_calls=AUTO)
    
    def format_output(images, features, labels):
        return ({"images": images, "features": features}, labels)
    
    balanced_ds = balanced_ds.map(format_output, num_parallel_calls=AUTO)
    balanced_ds = balanced_ds.prefetch(AUTO)
    
    return balanced_ds

print("\n now we have the function ")


training_ds = build_training_dataset_with_features(
    df=training_df,
    hdf5=training_validation_hdf5,
    feature_cols=FEAT_COLS,
    feature_space=feature_space,
    batch_size=CFG.batch_size,
    pos_ratio=0.25,
)

print("\n now we have training dataset ")


val_images = []
for isic_id in tqdm(validation_df['isic_id'].values, desc="Loading Validation Images"):
    val_images.append(training_validation_hdf5[isic_id][()])

val_features_dict = {col: validation_df[col].values for col in FEAT_COLS}
val_labels = validation_df['target'].values


val_features_ds = tf.data.Dataset.from_tensor_slices(val_features_dict)
val_features_processed = []
for batch in val_features_ds.batch(1000):
    processed = feature_space(batch)
    val_features_processed.append(processed.numpy())
val_features_tensor = np.concatenate(val_features_processed, axis=0)

def decode_image(image_bytes):
    image = tf.io.decode_jpeg(image_bytes)
    image = tf.image.resize(image, CFG.image_size, method="area")
    image = tf.cast(image, tf.float32) / 255.0
    return image

validation_ds = tf.data.Dataset.from_tensor_slices((val_images, val_features_tensor, val_labels))
validation_ds = validation_ds.batch(CFG.batch_size)

def process_val_batch(images, features, labels):
    decoded = tf.map_fn(decode_image, images, fn_output_signature=tf.TensorSpec([128, 128, 3], tf.float32))
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, [-1, 1])
    return ({"images": decoded, "features": features}, labels)

validation_ds = validation_ds.map(process_val_batch, num_parallel_calls=tf.data.AUTOTUNE)
validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)

print("\n Validation dataset is ok ")



for inputs, labels in training_ds.take(1):
    print(f"Images shape:   {inputs['images'].shape}")
    print(f"Features shape: {inputs['features'].shape}")
    print(f"Labels shape:   {labels.shape}")
    
    pos_count = int(tf.reduce_sum(labels).numpy())
    total = len(labels)
    print(f"Positive ratio: {pos_count}/{total} = {pos_count/total*100:.1f}%")


for inputs, labels in validation_ds.take(1):
    print(f"Images shape:   {inputs['images'].shape}")
    print(f"Features shape: {inputs['features'].shape}")
    print(f"Labels shape:   {labels.shape}")

print("\n Datasets are ok ")


num_negatives = len(training_df[training_df['target'] == 0])
neg_per_batch = int(CFG.batch_size * 0.75)
steps_per_epoch = num_negatives // neg_per_batch

print(f"Negative samples:    {num_negatives}")
print(f"Negatives per batch: {neg_per_batch}")
print(f"Steps per epoch:     {steps_per_epoch}")


import math

def get_lr_callback(epochs=10):
    lr_start = 1e-5
    lr_max = 1e-4
    lr_min = 1e-6
    lr_ramp_ep = 2
    
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        else:
            decay_epochs = epochs - lr_ramp_ep
            decay_index = epoch - lr_ramp_ep
            phase = math.pi * decay_index / decay_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr
    
    plt.figure(figsize=(10, 4))
    plt.plot([lrfn(e) for e in range(epochs)])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.show()
    
    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

lr_callback = get_lr_callback(epochs=CFG.epochs)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_auc",
    save_best_only=True,
    mode="max",
    verbose=1,
)

print(" ok ")



print(f"Epochs:          {CFG.epochs}")
print(f"Batch size:      {CFG.batch_size}")
print(f"Steps per epoch: {steps_per_epoch}")


history = model.fit(
    training_ds,
    epochs=CFG.epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,
    callbacks=[lr_callback, checkpoint_callback],
    verbose=1,
)



fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss', marker='o')
axes[0].plot(history.history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Over Epochs')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['auc'], label='Train AUC', marker='o')
axes[1].plot(history.history['val_auc'], label='Val AUC', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')
axes[1].set_title('AUC Over Epochs')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

best_epoch = np.argmax(history.history['val_auc']) + 1
best_auc = max(history.history['val_auc'])


print("BEST RESULT")

print(f"Best Epoch:          {best_epoch}")
print(f"Best Validation AUC: {best_auc:.4f}")




model.load_weights("best_model.keras")
print("Best model loaded ")




testing_df = pd.read_csv(f'{BASE_PATH}/test-metadata.csv')
testing_df = testing_df.ffill()

testing_hdf5 = h5py.File(f"{BASE_PATH}/test-image.hdf5", 'r')

print(f"Test samples: {len(testing_df)}")
print("Test data loaded ")



test_images = []
for isic_id in tqdm(testing_df['isic_id'].values, desc="Loading Test Images"):
    test_images.append(testing_hdf5[isic_id][()])

test_features_dict = {col: testing_df[col].values for col in FEAT_COLS}

test_features_ds = tf.data.Dataset.from_tensor_slices(test_features_dict)
test_features_processed = []
for batch in test_features_ds.batch(1000):
    processed = feature_space(batch)
    test_features_processed.append(processed.numpy())
test_features_tensor = np.concatenate(test_features_processed, axis=0)

def decode_image(image_bytes):
    image = tf.io.decode_jpeg(image_bytes)
    image = tf.image.resize(image, CFG.image_size, method="area")
    image = tf.cast(image, tf.float32) / 255.0
    return image

test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_features_tensor))
test_ds = test_ds.batch(CFG.batch_size)

def process_test_batch(images, features):
    decoded = tf.map_fn(decode_image, images, fn_output_signature=tf.TensorSpec([128, 128, 3], tf.float32))
    return {"images": decoded, "features": features}

test_ds = test_ds.map(process_test_batch, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print("Test dataset is ok ")


def predict_with_tta(model, dataset, num_augmentations=4):
    
    print(f"performing TTA with {num_augmentations} augmentations ")
    
    print("  [1/4] Original")
    preds_original = model.predict(dataset, verbose=1)
    
    def flip_horizontal(inputs):
        return {"images": tf.image.flip_left_right(inputs["images"]), 
                "features": inputs["features"]}
    
    def flip_vertical(inputs):
        return {"images": tf.image.flip_up_down(inputs["images"]), 
                "features": inputs["features"]}
    
    def flip_both(inputs):
        flipped = tf.image.flip_left_right(inputs["images"])
        flipped = tf.image.flip_up_down(flipped)
        return {"images": flipped, "features": inputs["features"]}
    
    print("  [2/4] Horizontal flip ")
    ds_h = dataset.map(flip_horizontal, num_parallel_calls=tf.data.AUTOTUNE)
    preds_h = model.predict(ds_h, verbose=1)
    
    print("  [3/4] Vertical flip")
    ds_v = dataset.map(flip_vertical, num_parallel_calls=tf.data.AUTOTUNE)
    preds_v = model.predict(ds_v, verbose=1)
    
    print("  [4/4] Both flips")
    ds_both = dataset.map(flip_both, num_parallel_calls=tf.data.AUTOTUNE)
    preds_both = model.predict(ds_both, verbose=1)
    
    final_preds = (preds_original + preds_h + preds_v + preds_both) / 4.0
    
    print(f" TTA is ok ")
    
    return final_preds





predictions = predict_with_tta(model, test_ds, num_augmentations=4)
predictions = predictions.squeeze()

print(f"\nPredictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:5]}")
print(f"Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")


submission_df = testing_df[["isic_id"]].copy()
submission_df["target"] = predictions

sample_submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
final_submission = sample_submission[["isic_id"]].merge(submission_df, on="isic_id", how="left")

final_submission.to_csv("submission.csv", index=False)


print("SUBMISSION FILE CREATED!")
print(f"\nFile saved: submission.csv")
print(f"Total predictions: {len(final_submission)}")
print("\nFirst 5 rows:")
print(final_submission.head())
print("\nPrediction statistics:")
print(final_submission['target'].describe())


print("sample test images with predictions:")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, (inputs) in enumerate(test_ds.take(1)):
    images = inputs["images"].numpy()
    
    for j in range(min(3, len(images))):
        ax = axes[j]
        img = images[j]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title(f"Prediction: {predictions[j]:.4f}")
        ax.axis('off')

plt.suptitle("Test Images with Model Predictions", fontsize=14)
plt.tight_layout()
plt.show()

