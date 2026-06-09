import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, GaussianNoise
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1_l2, l2  # Fix: Import both regularizers
from tensorflow.keras.optimizers import SGD, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import Policy, set_global_policy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight


# Enable mixed precision
set_global_policy(Policy('mixed_float16'))

# Data loading and preprocessing
x_train = np.load('X_train.npy')
x_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')

# Enhanced PCA with more components
pca = PCA(n_components=768)
x_train = pca.fit_transform(StandardScaler().fit_transform(x_train))
x_test = pca.transform(StandardScaler().fit_transform(x_test))

# Class weighting with boost
classes, counts = np.unique(y_train, return_counts=True)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train.flatten())
class_weights = dict(enumerate(1.5 * class_weights))



# Fixed model architecture with proper imports
def create_model():
    inputs = Input(shape=(768,))
    x = GaussianNoise(0.02)(inputs)
    
    x = Dense(384, activation='gelu', kernel_regularizer=l1_l2(0.001, 0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    
    x = Dense(256, activation='gelu', kernel_regularizer=l1_l2(0.001, 0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.55)(x)
    
    # Fixed: Use l2 regularizer now properly imported
    outputs = Dense(20, activation='softmax', kernel_regularizer=l2(0.005))(x)
    return Model(inputs, outputs)

model = create_model()

# Optimizer configuration
optimizer_sgd = SGD(
    learning_rate=0.025,
    momentum=0.95,
    nesterov=True,
    clipnorm=1.0
)
optimizer_adamw = AdamW(
    learning_rate=0.0005,
    weight_decay=0.005,
    clipnorm=1.0
)

# Callbacks with accuracy focus
lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    mode='max',
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=12,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

# Training Phase 1: SGD
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer_sgd, metrics=['accuracy'])
history = model.fit(
    x_train, y_train,
    epochs=120,
    batch_size=192,
    validation_split=0.18,
    class_weight=class_weights,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# Training Phase 2: AdamW
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer_adamw, metrics=['accuracy'])
history = model.fit(
    x_train, y_train,
    epochs=60,
    batch_size=192,
    validation_split=0.18,
    class_weight=class_weights,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)


# Enhanced predictions
tta_steps = 7
predictions = [model.predict(x_test, verbose=0) for _ in range(tta_steps)]
y_test_hat = np.argmax(np.mean(predictions, axis=0), axis=1)



# After you make your predictions, you should submit them on the Kaggle webpage for our competition.
# You may also (and I recommend you do it) send your code to me (at tsdj@sam.sdu.dk).
# Then I can provide feecback if you'd like (so ask away!).

# Below is a small check that your output has the right type and shape
# Create submission file
pd.DataFrame({'Id': np.arange(len(y_test_hat)), 'Predicted': y_test_hat}).to_csv('submission2.csv', index=False)

