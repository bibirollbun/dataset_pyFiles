import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ----------------- Data Loading and Preprocessing ------------------
train = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv')
test = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')
sub = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')

train_path = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train/'
test_path = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/'

# label binary encoding
cls_to_idx = {'editada': 0, 'real': 1}

train['label'] = [cls_to_idx[x] for x in train['label'].values]
train['image'] = train_path + train['image']
test['image'] = test_path + test['image']


print(f"Training set size: {len(train)}")


def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0  # normalized
    image = tf.image.per_image_standardization(image)
    return image, label


# create tf dataset
train_ds = tf.data.Dataset.from_tensor_slices((train['image'].values, train['label'].values))
test_ds = tf.data.Dataset.from_tensor_slices(test['image'].values)


BATCH_SIZE = 32
train_ds = train_ds.map(preprocess_image).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#preprocess test data
def preprocess_test_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0  # normalized
    image = tf.image.per_image_standardization(image)
    return image

test_ds= test_ds.map(preprocess_test_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



# Add rotation layer
rotation_layer = layers.RandomRotation(factor=0.3)


def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = rotation_layer(image)  # apply rotation
    return image, label


train_ds = train_ds.map(augment_image)

# sanity check
for images, labels in train_ds.take(1):
    print('image batch shape: ', images.shape)
    print('labels batch shape', labels.shape)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f'label {labels[i]}')
    plt.show()


# ------------------Model Building and Training------------------------

IMG_SIZE = (224, 224)
# Load pre-trained InceptionV3 (or other model)
base_model = keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze the initial layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# Add custom layers for classification
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.7)(x)  # Increased dropout
x = layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.02))(x)  # Dense layer before activation
x = layers.BatchNormalization()(x)  # Added Batch Normalization
x = layers.ReLU()(x)  # Activation function after Batch Normalization
x = layers.Dropout(0.4)(x)  # Increased dropout
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

# Learning rate schedule and optimizer
initial_learning_rate = 1e-4
optimizer = keras.optimizers.AdamW(learning_rate=initial_learning_rate)


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return (lr * tf.math.exp(-0.1)).numpy()


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


# Compile
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])

# Train the model
EPOCHS = 50  # Increased Epoch
history = model.fit(train_ds, epochs=EPOCHS, callbacks=[callback])


# Prediction
preds = model.predict(test_ds)
sub['label'] = preds

sub.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")




