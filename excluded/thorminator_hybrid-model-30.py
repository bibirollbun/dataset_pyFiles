import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load data
x_train = np.load('/kaggle/input/ml-ef-nn-intro/X_train.npy')
x_test = np.load('/kaggle/input/ml-ef-nn-intro/X_test.npy')
y_train = np.load('/kaggle/input/ml-ef-nn-intro/y_train.npy')

# Normalize data 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# Revert to original architecture with adjustments
def create_model():
    inputs = Input(shape=(64,))
    
    x = Dense(768, activation='swish', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(1024, activation='swish', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(512, activation='swish', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='swish', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(20, activation='softmax')(x)
    return Model(inputs, outputs)

model = create_model()

# Hybrid optimizer strategy 
optimizer_sgd = SGD(learning_rate=0.007, momentum=0.9, nesterov=True)
optimizer_adamw = AdamW(learning_rate=0.0003, weight_decay=0.0001)

# Original learning rate schedule
def lr_schedule(epoch):
    if epoch < 30:
        return 0.007  
    elif epoch < 80:
        return 0.0015  
    else:
        return 0.0005

# Compile first with SGD
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer_sgd,
    metrics=['accuracy']
)

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
clr_scheduler = LearningRateScheduler(lr_schedule)

# Phase 1: SGD training
history = model.fit(
    x_train, y_train,
    epochs=80,
    batch_size=128,  
    validation_split=0.1,
    callbacks=[lr_scheduler, clr_scheduler],
    verbose=1
)

# Switch to AdamW
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer_adamw,
    metrics=['accuracy']
)

# Phase 2: AdamW fine-tuning
history = model.fit(
    x_train, y_train,
    epochs=60,
    batch_size=128,
    validation_split=0.1,
    callbacks=[lr_scheduler],
    verbose=1
)


# Generate predictions
y_test_hat = model.predict(x_test)
y_test_hat = tf.argmax(y_test_hat, axis=1).numpy()

# Create submission
submission = pd.DataFrame({
    'Id': np.arange(len(y_test_hat)),
    'Predicted': y_test_hat
})


# Validation
assert isinstance(submission, pd.DataFrame)
assert all(submission.columns == ['Id', 'Predicted'])
assert len(submission) == 5000

submission.to_csv('submission.csv', index=False)

